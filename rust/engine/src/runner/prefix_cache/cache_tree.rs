use crate::radix_tree::{NodeId, RadixTree, SplitPrefix, ROOT};

/// Cache-specific metadata stays outside the reusable radix index.
struct CacheEntry<S> {
    slots: S,
    lock_ref: usize,
}

impl<S: SplitPrefix> SplitPrefix for CacheEntry<S> {
    fn split_prefix(&mut self, len: usize) -> Self {
        // Existing suffix pins now traverse the new parent, so inherit their count.
        Self {
            slots: self.slots.split_prefix(len),
            lock_ref: self.lock_ref,
        }
    }
}

/// Cache ownership and path protection around a reusable radix index.
/// The index handles topology; this wrapper handles KV segments and pin counts.
pub(super) struct CacheTree<S> {
    index: RadixTree<u32, CacheEntry<S>>,
}

impl<S> Default for CacheTree<S> {
    fn default() -> Self {
        Self {
            index: RadixTree::default(),
        }
    }
}

impl<S: SplitPrefix> CacheTree<S> {
    /// Match a token prefix, splitting a compressed edge at the boundary if needed.
    pub(super) fn match_prefix(&mut self, tokens: &[u32]) -> crate::radix_tree::PrefixMatch {
        self.index.match_prefix(tokens)
    }

    pub(super) fn append(&mut self, parent: NodeId, tokens: &[u32], slots: S) -> NodeId {
        self.index
            .append(parent, tokens, CacheEntry { slots, lock_ref: 0 })
    }

    pub(super) fn cached_tokens(&self) -> usize {
        self.index.key_elements()
    }

    pub(super) fn node_count(&self) -> usize {
        self.index.node_count()
    }

    pub(super) fn pin(&mut self, mut node: NodeId) {
        while node != ROOT {
            self.index.value_mut(node).unwrap().lock_ref += 1;
            node = self.index.parent(node).unwrap();
        }
    }
    pub(super) fn unpin(&mut self, mut node: NodeId) {
        while node != ROOT {
            let value = self.index.value_mut(node).unwrap();
            assert!(value.lock_ref > 0, "cached path protection underflow");
            value.lock_ref -= 1;
            node = self.index.parent(node).unwrap();
        }
    }
    pub(super) fn segments(&self, node: NodeId) -> Vec<&S> {
        let mut parts: Vec<_> = self
            .index
            .ancestors(node)
            .filter_map(|id| self.index.value(id).map(|entry| &entry.slots))
            .collect();
        parts.reverse();
        parts
    }
}

#[cfg(test)]
mod tests {
    use super::super::prefix_cache::PrefixMatch;
    use super::*;
    use crate::runner::kv_pool::PrefixLease;
    use std::sync::{Arc, Mutex};

    #[test]
    fn split_paths_inherit_protection_and_clones_release_only_once() {
        let mut tree = CacheTree::default();
        let tokens: Vec<u32> = (0..100).collect();
        let a = tree.append(ROOT, &tokens, tokens.clone());
        tree.pin(a);
        let tree = Arc::new(Mutex::new(tree));
        let lease_a = PrefixLease::from_shared(
            100,
            Arc::new(PrefixMatch {
                tree: tree.clone(),
                node: a,
                token_len: 100,
            }),
        );
        let b = {
            let mut s = tree.lock().unwrap();
            let b = s.match_prefix(&tokens[..50]).node;
            s.pin(b);
            b
        };
        let lease_b = PrefixLease::from_shared(
            50,
            Arc::new(PrefixMatch {
                tree: tree.clone(),
                node: b,
                token_len: 50,
            }),
        );
        let c = {
            let mut s = tree.lock().unwrap();
            let c = s.match_prefix(&tokens[..20]).node;
            s.pin(c);
            c
        };
        let lease_c = PrefixLease::from_shared(
            20,
            Arc::new(PrefixMatch {
                tree: tree.clone(),
                node: c,
                token_len: 20,
            }),
        );
        {
            let s = tree.lock().unwrap();
            assert_eq!(s.index.parent(a), Some(b));
            assert_eq!(s.index.parent(b), Some(c));
            assert_eq!(s.index.value(a).unwrap().lock_ref, 1);
            assert_eq!(s.index.value(b).unwrap().lock_ref, 2);
            assert_eq!(s.index.value(c).unwrap().lock_ref, 3);
            let mut slots: Vec<_> = s
                .segments(a)
                .into_iter()
                .flat_map(|v| v.iter())
                .copied()
                .collect();
            slots.sort();
            assert_eq!(slots, tokens);
        }
        let gpu = lease_a.clone();
        drop(lease_a);
        assert_eq!(tree.lock().unwrap().index.value(a).unwrap().lock_ref, 1);
        drop(gpu);
        assert_eq!(tree.lock().unwrap().index.value(a).unwrap().lock_ref, 0);
        assert_eq!(tree.lock().unwrap().index.value(b).unwrap().lock_ref, 1);
        assert_eq!(tree.lock().unwrap().index.value(c).unwrap().lock_ref, 2);
        drop(lease_b);
        drop(lease_c);
        let s = tree.lock().unwrap();
        assert!(s
            .index
            .ancestors(a)
            .filter_map(|id| s.index.value(id))
            .all(|v| v.lock_ref == 0));
    }
}
