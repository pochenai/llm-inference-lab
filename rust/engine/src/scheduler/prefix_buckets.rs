use super::routing::AccessState;
use patricia_tree::PatriciaMap;
use std::{
    collections::BTreeMap,
    ops::Bound::{Excluded, Included, Unbounded},
};

/// Bound the upstream recursive walks even for adversarial, deeply nested keys.
const KEYS_PER_BUCKET: usize = 128;
pub(super) const MAX_KEY_BYTES: usize = 64 * 1024;
const INDEX_ROOT_BYTES: usize = 512;
const INDEX_BUCKET_BYTES: usize = 192;

/// Standard BTreeMap fences select at most three small PatriciaMaps for exact LCP.
/// Empty first fence is permanent; other fences remain lower bounds after removals.
#[derive(Default)]
pub(super) struct PrefixBuckets {
    buckets: BTreeMap<Vec<u8>, PatriciaMap<AccessState>>,
    len: usize,
    fence_bytes: usize,
}

impl PrefixBuckets {
    pub(super) fn len(&self) -> usize {
        self.len
    }
    pub(super) fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(super) fn index_bytes(&self) -> usize {
        if self.buckets.is_empty() {
            0
        } else {
            INDEX_ROOT_BYTES + self.buckets.len() * INDEX_BUCKET_BYTES + self.fence_bytes
        }
    }

    pub(super) fn longest_common_prefix_len(&self, input: &[u8]) -> usize {
        let mut before = self.buckets.range::<[u8], _>((Unbounded, Included(input)));
        let current = before.next_back().map(|(_, bucket)| bucket);
        let previous = before.next_back().map(|(_, bucket)| bucket);
        let next = self
            .buckets
            .range::<[u8], _>((Excluded(input), Unbounded))
            .next()
            .map(|(_, bucket)| bucket);
        let mut matched = 0;
        // Fences can outlive their first key: both neighbors cover the nearest keys.
        for bucket in [current, previous, next].into_iter().flatten() {
            matched = matched.max(bucket.longest_common_prefix_len(input));
            if matched == input.len() {
                break;
            }
        }
        matched
    }

    pub(super) fn insert(&mut self, input: &[u8], access: AccessState) -> Option<AccessState> {
        assert!(
            input.len() <= MAX_KEY_BYTES,
            "routing key exceeds bounded trie limit"
        );
        if self.buckets.is_empty() {
            self.buckets.insert(Vec::new(), PatriciaMap::new());
        }
        let (fence, bucket) = self
            .buckets
            .range_mut::<[u8], _>((Unbounded, Included(input)))
            .next_back()
            .unwrap();
        let previous = bucket.insert(input, access);
        if previous.is_none() {
            self.len += 1;
        }
        if bucket.len() > KEYS_PER_BUCKET {
            let fence = fence.clone();
            let bucket = self.buckets.get_mut(&fence).unwrap();
            let keys: Vec<_> = bucket.keys().skip(bucket.len() / 2).collect();
            let pivot = keys[0].clone();
            let mut right = PatriciaMap::new();
            for key in keys {
                right.insert(&key, bucket.remove(&key).unwrap());
            }
            self.fence_bytes += pivot.len();
            self.buckets.insert(pivot, right);
        }
        previous
    }

    pub(super) fn remove(&mut self, input: &[u8]) -> Option<AccessState> {
        let (fence, bucket) = self
            .buckets
            .range_mut::<[u8], _>((Unbounded, Included(input)))
            .next_back()?;
        let removed = bucket.remove(input)?;
        self.len -= 1;
        if bucket.is_empty() && !fence.is_empty() {
            let fence = fence.clone();
            self.fence_bytes -= fence.len();
            self.buckets.remove(&fence);
        }
        Some(removed)
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (Vec<u8>, &AccessState)> {
        self.buckets.values().flat_map(PatriciaMap::iter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    #[test]
    fn bucket_fences_and_removals_preserve_exact_lcp() {
        let mut buckets = PrefixBuckets::default();
        let mut keys = Vec::new();
        let mut rng = StdRng::seed_from_u64(47);
        for tick in 0..1000 {
            let key: Vec<_> = (0..rng.random_range(1..64))
                .map(|_| b'a' + rng.random_range(0..8))
                .collect();
            buckets.insert(
                &key,
                AccessState {
                    last_access_tick: tick,
                },
            );
            keys.push(key);
        }
        keys.sort();
        keys.dedup();
        for key in keys.iter().step_by(2) {
            buckets.remove(key);
        }
        let retained: Vec<_> = keys
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 1)
            .map(|(_, k)| k)
            .collect();
        for _ in 0..500 {
            let query: Vec<_> = (0..rng.random_range(0..70))
                .map(|_| b'a' + rng.random_range(0..8))
                .collect();
            let expected = retained
                .iter()
                .map(|key| key.iter().zip(&query).take_while(|(a, b)| a == b).count())
                .max()
                .unwrap_or(0);
            assert_eq!(buckets.longest_common_prefix_len(&query), expected);
        }
        assert_eq!(buckets.len(), retained.len());
        assert_eq!(buckets.iter().count(), retained.len());
    }

    #[test]
    fn deeply_nested_and_branching_keys_fit_a_small_stack() {
        std::thread::Builder::new()
            .stack_size(2 * 1024 * 1024)
            .spawn(|| {
                let input = vec![b'a'; 4096];
                let mut buckets = PrefixBuckets::default();
                for length in 1..=4096 {
                    buckets.insert(
                        &input[..length],
                        AccessState {
                            last_access_tick: length as u64,
                        },
                    );
                }
                assert_eq!(buckets.longest_common_prefix_len(&input), 4096);
                for length in 1..=4096 {
                    assert!(buckets.remove(&input[..length]).is_some());
                }
                assert!(buckets.is_empty());
                for length in 0..4096 {
                    let mut key = input.clone();
                    key[length] = b'b';
                    buckets.insert(
                        &key,
                        AccessState {
                            last_access_tick: length as u64,
                        },
                    );
                }
                assert_eq!(buckets.longest_common_prefix_len(&input), 4095);
                // Exercise maximum supported key size with bucket splits as well.
                let mut large = PrefixBuckets::default();
                for prefix in 0..300 {
                    let mut key = vec![b'a'; MAX_KEY_BYTES];
                    key[prefix] = b'b';
                    large.insert(
                        &key,
                        AccessState {
                            last_access_tick: prefix as u64,
                        },
                    );
                }
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
