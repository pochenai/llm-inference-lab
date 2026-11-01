use super::cache_tree::CacheTree;
use crate::radix_tree::{NodeId, SplitPrefix};
use crate::runner::kv_pool::{KVContext, KVContextError, KVPool, PrefixLease};
use std::{
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};

/// Engine-local token/KV cache; unrelated to the gateway's bounded text hints.
#[derive(Clone, Copy, Debug)]
pub struct PrefixCacheConfig {
    /// Maximum tokens indexed from each committed history; default 4096.
    pub max_prefix_tokens: NonZeroUsize,
    /// Unique tree-owned KV token budget; evict unpinned LRU leaves when needed.
    pub max_cached_tokens: NonZeroUsize,
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self {
            max_prefix_tokens: NonZeroUsize::new(4096).unwrap(),
            max_cached_tokens: NonZeroUsize::new(4096).unwrap(),
        }
    }
}

/// Controller for find -> reuse -> publish; row/GPU leases share its tree through hits.
pub(in crate::runner) struct PrefixCache<P: KVPool> {
    tree: Arc<Mutex<CacheTree<P::PrefixState>>>,
    pool: Arc<P>,
    config: PrefixCacheConfig,
}

impl<P: KVPool + 'static> PrefixCache<P> {
    pub(in crate::runner) fn new(config: PrefixCacheConfig, pool: Arc<P>) -> Self {
        Self {
            tree: Arc::new(Mutex::new(CacheTree::default())),
            pool,
            config,
        }
    }

    /// Match/split the prefix and pin its node-to-root path before returning.
    /// Reuse shares this protection with the context; admission failure or an unused
    /// result unpins automatically when the last lease drops, without evicting KV.
    pub(in crate::runner) fn find(
        &self,
        tokens: &[u32],
    ) -> Option<Arc<PrefixMatch<P::PrefixState>>> {
        // KV has no final hidden/logits: leave one input token to compute even for an exact hit.
        let limit = tokens
            .len()
            .saturating_sub(1)
            .min(self.config.max_prefix_tokens.get());
        let found = {
            let mut tree = self.tree.lock().unwrap();
            // May split a compressed edge so the matched prefix ends at a node boundary.
            let found = tree.match_prefix(&tokens[..limit]);
            if found.matched_len == 0 {
                return None;
            }
            tree.pin(found.node);
            found
        };
        // Construct/drop path guards outside the mutex: final Drop locks the tree again.
        Some(self.pinned_match(found.node, found.matched_len))
    }

    pub(in crate::runner) fn reuse(
        &self,
        hit: &Arc<PrefixMatch<P::PrefixState>>,
        kv: &mut KVContext<P>,
    ) -> Result<(), KVContextError> {
        if !Arc::ptr_eq(&hit.tree, &self.tree)
            || !kv.uses_pool(&self.pool)
            || kv.allocated_len() != 0
        {
            return Err(KVContextError::InvalidState(
                "prefix reuse needs the same cache/pool and an empty context".into(),
            ));
        }
        let lease = hit.lease();
        let old = {
            let tree = self.tree.lock().unwrap();
            kv.bind_prefix(&tree.segments(hit.node), &lease)?
        };
        drop(old);
        Ok(())
    }

    /// Transfer completed private KV into tree edges, deduplicate an already-cached
    /// path, then re-anchor the context. No slot counts and no K/V copies are involved.
    pub(in crate::runner) fn publish(
        &self,
        tokens: &[u32],
        kv: &mut KVContext<P>,
    ) -> Result<(), KVContextError> {
        if !kv.has_owned_pool_state() || !kv.uses_pool(&self.pool) || kv.is_pending() {
            return Err(KVContextError::InvalidState(
                "prefix publication needs idle committed KV from this pool".into(),
            ));
        }
        let limit = tokens
            .len()
            .min(kv.ready_len())
            .min(self.config.max_prefix_tokens.get())
            .min(self.config.max_cached_tokens.get());
        if limit == 0 {
            return Ok(());
        }
        let (node, len, added) = {
            let mut tree = self.tree.lock().unwrap();
            let found = tree.match_prefix(&tokens[..limit]);
            let missing = limit - found.matched_len;
            let shortage = tree
                .cached_tokens()
                .saturating_add(missing)
                .saturating_sub(self.config.max_cached_tokens.get());
            if shortage > 0 {
                // Keep the match alive while pruning; the tree mutex covers the final re-pin.
                tree.pin(found.node);
                tree.evict_until(|freed| freed >= shortage);
                tree.unpin(found.node);
            }
            let mut node = found.node;
            let mut len = found.matched_len;
            let mut added = 0;
            if missing > 0
                && missing
                    <= self
                        .config
                        .max_cached_tokens
                        .get()
                        .saturating_sub(tree.cached_tokens())
            {
                let slots = kv.take_prefix(found.matched_len..limit)?;
                node = tree.append(node, &tokens[found.matched_len..limit], slots);
                len = limit;
                added = missing;
            }
            if len == 0 {
                return Ok(());
            }
            tree.pin(node);
            (node, len, added)
        };
        let lease = PrefixLease::from_shared(len, self.pinned_match(node, len));
        let result = {
            let tree = self.tree.lock().unwrap();
            kv.bind_prefix(&tree.segments(node), &lease)
        };
        // Old pins may destroy cache resources. Do not drop them under its mutex.
        match result {
            Ok(old) => drop(old),
            Err(error) => {
                kv.invalidate();
                return Err(error);
            }
        }
        if added > 0 {
            metrics::counter!("engine_prefix_cache_published_tokens_total").increment(added as u64);
            let tree = self.tree.lock().unwrap();
            tracing::debug!(target: "engine::prefix_cache", event = "prefix_published", tokens = added,
                cached_tokens = tree.cached_tokens(), nodes = tree.node_count() - 1);
        }
        Ok(())
    }

    /// Reclaim cached KV for admission; pins protect the selected hit and GPU work.
    /// Rows belong to contexts, so evicting prefixes cannot resolve row exhaustion.
    pub(in crate::runner) fn evict_for_capacity(&self, tokens: usize, rows: usize) {
        if self.pool.can_append(tokens, rows) || !self.pool.can_append(0, rows) {
            return;
        }
        self.tree
            .lock()
            .unwrap()
            .evict_until(|_| self.pool.can_append(tokens, rows));
    }

    /// Wrap a path already pinned under the tree mutex; construct/drop guards outside it.
    fn pinned_match(&self, node: NodeId, token_len: usize) -> Arc<PrefixMatch<P::PrefixState>> {
        Arc::new(PrefixMatch {
            tree: self.tree.clone(),
            node,
            token_len,
        })
    }
}

/// A protected match is also its RAII guard. Its final shared owner unpins the path.
pub(in crate::runner) struct PrefixMatch<S: SplitPrefix> {
    pub(super) tree: Arc<Mutex<CacheTree<S>>>,
    pub(super) node: NodeId,
    pub(super) token_len: usize,
}

impl<S: SplitPrefix + Send + Sync + 'static> PrefixMatch<S> {
    pub(in crate::runner) fn token_len(&self) -> usize {
        self.token_len
    }
    fn lease(self: &Arc<Self>) -> PrefixLease {
        PrefixLease::from_shared(self.token_len, self.clone())
    }
}

impl<S: SplitPrefix> Drop for PrefixMatch<S> {
    fn drop(&mut self) {
        self.tree.lock().unwrap().unpin(self.node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::kv_pool::{NaiveKVPool, NaiveKVPoolConfig};
    use cuda_async::{
        device_context::global_policy,
        device_operation::{value, with_context, DeviceOp},
    };
    use std::{hint::black_box, time::Instant};

    #[tokio::test]
    async fn eviction_reclaims_pool_slots_for_publication_and_admission() -> anyhow::Result<()> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = global_policy(0)?;
        let pool = NaiveKVPool::<f32>::new(
            NaiveKVPoolConfig {
                num_slots: 6,
                max_rows: 2,
                max_seq_len: 8,
                head_dim: 4,
                ..Default::default()
            },
            &policy,
        )
        .await?;
        let idle = pool.metrics();
        let cache = PrefixCache::new(
            PrefixCacheConfig {
                max_prefix_tokens: NonZeroUsize::new(3).unwrap(),
                max_cached_tokens: NonZeroUsize::new(3).unwrap(),
            },
            pool.clone(),
        );
        let mut source = KVContext::new(pool.clone())?;
        with_context(|exec| value(source.prepare_append(3, exec)))
            .schedule(&policy)?
            .await??;
        source.commit_append()?;
        cache.publish(&[1, 2, 3], &mut source)?;
        source.release();
        let hit = cache.find(&[1, 2, 3, 4]).unwrap();
        let retained = pool.metrics();
        let mut cold = KVContext::new(pool.clone())?;
        with_context(|exec| value(cold.prepare_append(3, exec)))
            .schedule(&policy)?
            .await??;
        cold.commit_append()?;
        cache.publish(&[7, 8, 9], &mut cold)?;
        assert!(cache.find(&[7, 8, 9, 0]).is_none());
        assert_eq!(cache.tree.lock().unwrap().cached_tokens(), 3);
        drop(hit);
        cache.publish(&[7, 8, 9], &mut cold)?;
        cold.release();
        assert!(cache.find(&[1, 2, 3, 4]).is_none());
        assert_eq!(pool.metrics(), retained);
        let replacement = cache.find(&[7, 8, 9, 0]).unwrap();
        cache.evict_for_capacity(6, 1);
        assert!(!pool.can_append(6, 1));
        drop(replacement);
        // Prefix eviction cannot supply context rows, so do not discard hits for that case.
        let row_a = KVContext::new(pool.clone())?;
        let row_b = KVContext::new(pool.clone())?;
        cache.evict_for_capacity(6, 1);
        assert_eq!(cache.tree.lock().unwrap().cached_tokens(), 3);
        drop((row_a, row_b));
        cache.evict_for_capacity(6, 1);
        assert!(pool.can_append(6, 1));
        assert_eq!(pool.metrics(), idle);
        assert!(cache.find(&[7, 8, 9, 0]).is_none());
        Ok(())
    }

    /// Run manually with cargo test --release -p engine --lib prefix_cache_cpu_probe -- --ignored --nocapture.
    /// GPU setup is excluded; measurements include CPU allocations/locks and zero GPU submissions.
    #[tokio::test]
    #[ignore = "manual CPU ownership microbenchmark with GPU pool setup"]
    async fn prefix_cache_cpu_probe() -> anyhow::Result<()> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = global_policy(0)?;
        let pool = NaiveKVPool::<f32>::new(
            NaiveKVPoolConfig {
                num_slots: 32768,
                max_seq_len: 32769,
                max_rows: 2,
                head_dim: 4,
                ..Default::default()
            },
            &policy,
        )
        .await?;
        let idle = pool.metrics();
        for len in [1, 4096, 32768] {
            let mut source = KVContext::new(pool.clone())?;
            with_context(|exec| value(source.prepare_append(len, exec)))
                .schedule(&policy)?
                .await??;
            source.commit_append()?;
            let tokens = vec![1u32; len + 1];
            let cache = PrefixCache::new(
                PrefixCacheConfig {
                    max_prefix_tokens: NonZeroUsize::new(len).unwrap(),
                    max_cached_tokens: NonZeroUsize::new(len).unwrap(),
                },
                pool.clone(),
            );
            cache.publish(&tokens[..len], &mut source)?;
            source.release();
            let hit = cache.find(&tokens).unwrap();
            let reuse = median_ns(|| {
                let mut kv = KVContext::new(pool.clone()).unwrap();
                cache.reuse(black_box(&hit), &mut kv).unwrap();
                drop(black_box(kv));
            });
            drop(hit);
            let lookup = median_ns(|| {
                drop(black_box(cache.find(black_box(&tokens))));
            });
            let full = median_ns(|| {
                let hit = cache.find(black_box(&tokens)).unwrap();
                let mut kv = KVContext::new(pool.clone()).unwrap();
                cache.reuse(&hit, &mut kv).unwrap();
                drop(hit);
                drop(black_box(kv));
            });
            println!("{{\"tokens\":{len},\"lookup_pin_release_ns\":{lookup},\"alloc_reuse_release_ns\":{reuse},\"lookup_alloc_reuse_release_ns\":{full}}}");
            drop(cache);
            assert_eq!(pool.metrics(), idle);
        }
        Ok(())
    }

    fn median_ns(mut operation: impl FnMut()) -> u128 {
        for _ in 0..100 {
            operation();
        }
        let mut samples = Vec::new();
        for _ in 0..5 {
            let started = Instant::now();
            for _ in 0..20_000 {
                operation();
            }
            samples.push(started.elapsed().as_nanos() / 20_000);
        }
        samples.sort();
        samples[samples.len() / 2]
    }
}
