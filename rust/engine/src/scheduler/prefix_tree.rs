use super::routing::{AccessState, DestinationId};
use crate::radix_tree::RadixMap;
use std::{
    collections::{BinaryHeap, HashMap},
    mem::size_of,
};

/// Reclaim one eighth of the directory per pressure cycle, not one entry per insert.
pub(super) const EVICTION_HEADROOM_DIVISOR: usize = 8;
/// Recheck this charge if AccessState gains larger or heap-backed fields.
const NODE_ALLOWANCE_BYTES: usize = 64;
const ENTRY_ALLOWANCE_BYTES: usize = 128;

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct EvictionCandidate {
    tick: u64,
    destination: DestinationId,
    key: Vec<u8>,
}

#[derive(Default)]
struct DestinationPrefixes {
    /// u8 is one UTF-8 byte of a capped prompt key: "abc" -> [97, 98, 99].
    /// A full key is &[u8], not a node/slot index; AccessState stores its recency.
    entries: RadixMap<u8, AccessState>,
    charged_bytes: usize,
}

/// One bounded radix index per next hop: exact LCP queries never scan prompt records.
/// Budget charges conservatively ignore prefix sharing.
#[derive(Default)]
pub(super) struct PrefixTree {
    destinations: HashMap<DestinationId, DestinationPrefixes>,
    charged_bytes: usize,
}

/// Approximate allocation charge, including spare buckets and control metadata.
fn map_bytes<K, V>(map: &HashMap<K, V>) -> usize {
    if map.capacity() == 0 {
        0
    } else {
        map.capacity() * (size_of::<(K, V)>() + 1) * 2 + 64
    }
}

impl PrefixTree {
    /// Conservative full-key charge; actual arena/label/child capacity is added separately.
    pub(super) fn entry_charge(len: usize) -> usize {
        len + ENTRY_ALLOWANCE_BYTES
    }

    pub(super) fn bytes(&self) -> usize {
        self.charged_bytes
            + map_bytes(&self.destinations)
            + self.destinations.len() * NODE_ALLOWANCE_BYTES
            + self
                .destinations
                .values()
                .map(|p| p.entries.allocated_bytes())
                .sum::<usize>()
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.destinations
            .values()
            .map(|entry| entry.entries.len())
            .sum()
    }

    #[cfg(test)]
    pub(super) fn is_empty(&self) -> bool {
        self.destinations.is_empty()
    }

    pub(super) fn matches(&self, input: &[u8]) -> HashMap<DestinationId, usize> {
        self.destinations
            .iter()
            .filter_map(|(&destination, prefixes)| {
                let matched = prefixes.entries.longest_common_prefix_len(input);
                (matched > 0).then_some((destination, matched))
            })
            .collect()
    }

    pub(super) fn insert(&mut self, input: &[u8], destination: DestinationId, tick: u64) {
        if input.is_empty() {
            return;
        }
        let prefixes = self.destinations.entry(destination).or_default();
        if prefixes
            .entries
            .insert(
                input,
                AccessState {
                    last_access_tick: tick,
                },
            )
            .is_none()
        {
            let charge = Self::entry_charge(input.len());
            prefixes.charged_bytes += charge;
            self.charged_bytes += charge;
        }
    }

    /// Scan once per batch, retaining only the oldest eighth of keys as scratch.
    /// The heap and reconstructed keys are temporary and do not retain full inputs.
    pub(super) fn oldest_batch(&self) -> Vec<EvictionCandidate> {
        let entries: usize = self.destinations.values().map(|p| p.entries.len()).sum();
        let limit = entries.div_ceil(EVICTION_HEADROOM_DIVISOR);
        let mut oldest = BinaryHeap::with_capacity(limit);
        for (&destination, prefixes) in &self.destinations {
            for (key, access) in prefixes.entries.iter() {
                let candidate = EvictionCandidate {
                    tick: access.last_access_tick,
                    destination,
                    key,
                };
                if oldest.len() < limit {
                    oldest.push(candidate);
                } else if oldest.peek().is_some_and(|newest| candidate < *newest) {
                    *oldest.peek_mut().unwrap() = candidate;
                }
            }
        }
        oldest.into_sorted_vec()
    }

    pub(super) fn evict(&mut self, candidate: EvictionCandidate) {
        let Some(prefixes) = self.destinations.get_mut(&candidate.destination) else {
            return;
        };
        if prefixes.entries.remove(&candidate.key).is_some() {
            let charge = Self::entry_charge(candidate.key.len());
            prefixes.charged_bytes -= charge;
            self.charged_bytes -= charge;
        }
        if prefixes.entries.is_empty() {
            self.destinations.remove(&candidate.destination);
        }
        if self.destinations.is_empty() {
            self.destinations = HashMap::new();
        }
    }

    pub(super) fn remove_destination(&mut self, destination: DestinationId) {
        if let Some(prefixes) = self.destinations.remove(&destination) {
            self.charged_bytes -= prefixes.charged_bytes;
        }
        if self.destinations.is_empty() {
            self.destinations = HashMap::new();
        } else {
            self.destinations.shrink_to_fit();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_partial_match_and_shared_destinations() {
        let mut tree = PrefixTree::default();
        tree.insert(b"SYS:aaa", DestinationId(1), 1);
        tree.insert(b"SYS:bbb", DestinationId(2), 2);
        tree.insert(b"SYS:aab", DestinationId(3), 3);
        assert_eq!(
            tree.matches(b"SYS:ccc"),
            HashMap::from([
                (DestinationId(1), 4),
                (DestinationId(2), 4),
                (DestinationId(3), 4)
            ])
        );
        assert_eq!(tree.matches(b"SYS:aax")[&DestinationId(1)], 6);
        assert_eq!(tree.matches(b"SYS:aax")[&DestinationId(2)], 4);
        assert_eq!(
            tree.matches(b"SY"),
            HashMap::from([
                (DestinationId(1), 2),
                (DestinationId(2), 2),
                (DestinationId(3), 2)
            ])
        );
        assert!(tree.matches(b"other").is_empty());
        assert!(tree.matches(b"").is_empty());
    }

    #[test]
    fn eviction_preserves_other_branches_and_reclaims_nodes() {
        let mut tree = PrefixTree::default();
        let empty_bytes = tree.bytes();
        tree.insert(b"SYS:aaa", DestinationId(1), 1);
        tree.insert(b"SYS:bbb", DestinationId(1), 2);
        tree.insert(b"SYS:aaa", DestinationId(2), 3);
        let candidate = tree.oldest_batch().into_iter().next().unwrap();
        assert_eq!(candidate.destination, DestinationId(1));
        tree.evict(candidate);
        assert_eq!(tree.matches(b"SYS:bbb")[&DestinationId(1)], 7);
        assert_eq!(tree.matches(b"SYS:aaa")[&DestinationId(2)], 7);
        tree.remove_destination(DestinationId(1));
        assert!(!tree.matches(b"SYS:bbb").contains_key(&DestinationId(1)));
        tree.remove_destination(DestinationId(2));
        assert!(tree.is_empty());
        assert_eq!(tree.bytes(), empty_bytes);
    }

    #[test]
    fn compressed_matches_agree_with_independent_prefix_scan() {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(23);
        let mut tree = PrefixTree::default();
        let mut records = Vec::new();
        for tick in 1..300 {
            let input: Vec<u8> = (0..rng.random_range(1..30))
                .map(|_| b'a' + rng.random_range(0..4))
                .collect();
            let destination = DestinationId(rng.random_range(0..5));
            tree.insert(&input, destination, tick);
            records.push((input, destination));
        }
        for _ in 0..200 {
            let input: Vec<u8> = (0..rng.random_range(0..35))
                .map(|_| b'a' + rng.random_range(0..4))
                .collect();
            let mut expected = HashMap::<DestinationId, usize>::new();
            for (record, destination) in &records {
                let common = input.iter().zip(record).take_while(|(a, b)| a == b).count();
                if common > 0 {
                    expected
                        .entry(*destination)
                        .and_modify(|n| *n = (*n).max(common))
                        .or_insert(common);
                }
            }
            assert_eq!(tree.matches(&input), expected);
        }
        for destination in 0..5 {
            tree.remove_destination(DestinationId(destination));
        }
        assert!(tree.is_empty());
    }
}
