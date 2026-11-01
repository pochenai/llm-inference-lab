use super::{NodeId, RadixTree, SplitPrefix, ROOT};
use std::{hash::Hash, mem::size_of};

struct Terminal<V>(Option<V>);
impl<V> SplitPrefix for Terminal<V> {
    fn split_prefix(&mut self, _len: usize) -> Self {
        Self(None)
    }
}

/// Ordinary key/value map on the same compressed tree. K is one key element;
/// complete keys are &[K]. Values belong to complete keys, not implicit prefix nodes.
/// No cache/resource policy is involved.
pub struct RadixMap<K, V> {
    tree: RadixTree<K, Terminal<V>>,
    len: usize,
}

impl<K, V> Default for RadixMap<K, V> {
    fn default() -> Self {
        Self {
            tree: RadixTree::default(),
            len: 0,
        }
    }
}

impl<K: Clone + Eq + Hash, V> RadixMap<K, V> {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn insert(&mut self, key: &[K], value: V) -> Option<V> {
        let found = self.tree.match_prefix(key);
        let old = if found.matched_len == key.len() {
            if found.node == ROOT {
                self.tree
                    .node_mut(ROOT)
                    .value
                    .get_or_insert(Terminal(None))
                    .0
                    .replace(value)
            } else {
                self.tree.value_mut(found.node).unwrap().0.replace(value)
            }
        } else {
            self.tree
                .append(found.node, &key[found.matched_len..], Terminal(Some(value)));
            None
        };
        if old.is_none() {
            self.len += 1;
        }
        old
    }

    /// Longest path overlap, including matches inside edges and before stored keys.
    pub fn longest_common_prefix_len(&self, key: &[K]) -> usize {
        self.tree.longest_common_prefix_len(key)
    }

    pub fn remove(&mut self, key: &[K]) -> Option<V> {
        // Do not split on misses: repeated removals must not accumulate empty boundaries.
        let mut node = ROOT;
        let mut offset = 0;
        while offset < key.len() {
            let child = *self.tree.node(node).children.get(&key[offset])?;
            let label = &self.tree.node(child).label;
            if !key[offset..].starts_with(label) {
                return None;
            }
            offset += label.len();
            node = child;
        }
        let value = self.tree.value_mut(node)?.0.take()?;
        self.len -= 1;
        self.compact(node);
        if self.len == 0 {
            self.tree = RadixTree::default();
        }
        Some(value)
    }

    /// Remove empty leaves and merge valueless unary nodes. This map owns all node
    /// handles privately; the resource tree never merges nodes behind active pins.
    fn compact(&mut self, mut id: NodeId) {
        while id != ROOT && self.tree.value(id).unwrap().0.is_none() {
            let parent = self.tree.parent(id).unwrap();
            let count = self.tree.node(id).children.len();
            if count > 1 {
                break;
            }
            let node = self.tree.nodes[id.0].take().unwrap();
            self.tree.free.push(id);
            if let Some(&child) = node.children.values().next() {
                let edge = self.tree.node_mut(child);
                let mut label = node.label;
                label.extend_from_slice(&edge.label);
                edge.label = label;
                edge.parent = Some(parent);
                let first = edge.label[0].clone();
                self.tree.node_mut(parent).children.insert(first, child);
            } else {
                self.tree.key_elements -= node.label.len();
                self.tree.node_mut(parent).children.remove(&node.label[0]);
            }
            id = parent;
        }
    }

    /// Reconstruct keys only for iteration/eviction; lookup never scans stored keys.
    pub fn iter(&self) -> impl Iterator<Item = (Vec<K>, &V)> {
        self.tree
            .nodes
            .iter()
            .enumerate()
            .filter_map(move |(index, node)| {
                let value = node.as_ref()?.value.as_ref()?.0.as_ref()?;
                let mut labels: Vec<_> = self
                    .tree
                    .ancestors(NodeId(index))
                    .map(|id| self.tree.node(id).label.as_slice())
                    .collect();
                labels.reverse();
                let mut key = Vec::with_capacity(labels.iter().map(|s| s.len()).sum());
                for label in labels {
                    key.extend_from_slice(label);
                }
                Some((key, value))
            })
    }

    /// Conservative retained-allocation charge, including spare arena/child capacity.
    pub fn allocated_bytes(&self) -> usize {
        self.tree.nodes.capacity() * size_of::<Option<super::Node<K, Terminal<V>>>>()
            + self.tree.free.capacity() * size_of::<NodeId>()
            + self
                .tree
                .nodes
                .iter()
                .flatten()
                .map(|node| {
                    node.label.capacity() * size_of::<K>()
                        + if node.children.capacity() == 0 {
                            0
                        } else {
                            node.children.capacity() * (size_of::<(K, NodeId)>() + 1) * 2 + 64
                        }
                })
                .sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use std::collections::BTreeMap;

    #[test]
    fn map_updates_removes_and_merges_match_an_independent_oracle() {
        let mut map = RadixMap::default();
        let mut oracle = BTreeMap::new();
        let mut rng = StdRng::seed_from_u64(819);
        for _ in 0..3000 {
            let key: Vec<u8> = (0..rng.random_range(0..40))
                .map(|_| rng.random_range(0..8))
                .collect();
            if rng.random_range(0..3) == 0 {
                assert_eq!(map.remove(&key), oracle.remove(&key));
            } else {
                let value = rng.random::<u32>();
                assert_eq!(map.insert(&key, value), oracle.insert(key.clone(), value));
            }
            let expected = oracle
                .keys()
                .map(|k| k.iter().zip(&key).take_while(|(a, b)| a == b).count())
                .max()
                .unwrap_or(0);
            assert_eq!(map.longest_common_prefix_len(&key), expected);
            assert_eq!(map.len(), oracle.len());
            assert_eq!(
                map.tree.key_elements(),
                map.tree
                    .nodes
                    .iter()
                    .flatten()
                    .map(|node| node.label.len())
                    .sum::<usize>()
            );
        }
        assert_eq!(
            map.iter().collect::<BTreeMap<_, _>>(),
            oracle.iter().map(|(k, v)| (k.clone(), v)).collect()
        );
        for key in oracle.keys() {
            assert_eq!(map.remove(key), oracle.get(key).copied());
        }
        assert!(map.is_empty());
        assert_eq!(map.tree.node_count(), 1);
        assert_eq!(map.tree.key_elements(), 0);
    }

    #[test]
    fn remove_compacts_paths_and_reuses_arena_slots() {
        let mut map = RadixMap::default();
        map.insert(&[1; 4096], 1);
        for len in (1..4096).rev() {
            map.insert(&vec![1; len], len);
        }
        for len in 1..4096 {
            map.remove(&vec![1; len]);
        }
        assert_eq!(map.tree.node_count(), 2);
        let capacity = map.tree.nodes.capacity();
        for i in 2..200 {
            map.insert(&[i], i);
            map.remove(&[i]);
        }
        assert_eq!(map.tree.nodes.capacity(), capacity);
        assert_eq!(map.longest_common_prefix_len(&[1; 4096]), 4096);
    }
}
