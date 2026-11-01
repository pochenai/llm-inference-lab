//! Compressed sequence index with splittable values on edges and stable node IDs.
//! Path compression/splitting adapts patricia_tree 0.10.2's Node::split_at:
//! https://github.com/sile/patricia_tree/blob/f959e447722792af0bd891a16376e1cd591f4bb9/src/node.rs
//! MIT, Copyright (c) 2017 Takeru Ohta; see LICENSE in this directory.
//! Local changes use a safe, iterative arena, preserve suffix IDs, and split values.
//! Resource ownership, protection counts and eviction policy belong to the caller.

mod map;

pub use map::RadixMap;

use std::{collections::HashMap, hash::Hash};

/// Move the first len logical elements into a new value, leaving the suffix in self.
/// Results partition the logical elements and preserve their resource lifetimes.
/// Physical backing may remain shared when an allocator has coarser granularity.
/// The tree calls this only strictly inside an edge.
pub trait SplitPrefix {
    fn split_prefix(&mut self, len: usize) -> Self;
}

impl<T> SplitPrefix for Vec<T> {
    fn split_prefix(&mut self, len: usize) -> Self {
        // Allocate the smaller half; keep existing storage with the larger half.
        let mut prefix = if len <= self.len() / 2 {
            self.drain(..len).collect::<Vec<_>>()
        } else {
            let suffix = self.split_off(len);
            std::mem::replace(self, suffix)
        };
        // Arbitrary split order must not retain a large allocation per tiny edge.
        if prefix.capacity() > prefix.len().saturating_mul(2) {
            prefix.shrink_to_fit();
        }
        if self.capacity() > self.len().saturating_mul(2) {
            self.shrink_to_fit();
        }
        prefix
    }
}

/// An ID belongs to its originating tree. Splits preserve the original suffix ID.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(usize);

pub const ROOT: NodeId = NodeId(0);

struct Node<K, V> {
    label: Vec<K>,
    value: Option<V>,
    parent: Option<NodeId>,
    children: HashMap<K, NodeId>,
}

/// Nodes reference children by ID: traversal and destruction never recurse.
pub struct RadixTree<K, V> {
    nodes: Vec<Option<Node<K, V>>>,
    free: Vec<NodeId>,
    key_elements: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefixMatch {
    pub node: NodeId,
    pub matched_len: usize,
}

impl<K, V> Default for RadixTree<K, V> {
    fn default() -> Self {
        Self {
            nodes: vec![Some(Node {
                label: Vec::new(),
                value: None,
                parent: None,
                children: HashMap::new(),
            })],
            free: Vec::new(),
            key_elements: 0,
        }
    }
}

impl<K: Clone + Eq + Hash, V: SplitPrefix> RadixTree<K, V> {
    pub fn parent(&self, node: NodeId) -> Option<NodeId> {
        self.node(node).parent
    }
    pub fn value(&self, node: NodeId) -> Option<&V> {
        self.node(node).value.as_ref()
    }
    pub fn value_mut(&mut self, node: NodeId) -> Option<&mut V> {
        self.node_mut(node).value.as_mut()
    }
    pub fn node_count(&self) -> usize {
        self.nodes.len() - self.free.len()
    }

    /// Total key elements across compressed edges, counting shared paths once.
    /// This counts logical edge lengths, not distinct symbols or allocated capacity.
    pub fn key_elements(&self) -> usize {
        self.key_elements
    }

    /// Walk from node to root, including both endpoints, without allocating a path.
    pub fn ancestors(&self, node: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        std::iter::successors(Some(node), |&id| self.parent(id))
    }

    /// Iterate live non-root leaves without reconstructing their keys.
    pub fn leaves(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes.iter().enumerate().filter_map(|(id, node)| {
            node.as_ref()
                .filter(|node| id != ROOT.0 && node.children.is_empty())
                .map(|_| NodeId(id))
        })
    }

    pub fn is_leaf(&self, node: NodeId) -> bool {
        self.node(node).children.is_empty()
    }

    /// Remove a non-root leaf and return its payload; other node IDs stay stable.
    pub fn remove_leaf(&mut self, id: NodeId) -> Option<V> {
        if id == ROOT || !self.is_leaf(id) {
            return None;
        }
        let node = self.nodes[id.0].take().unwrap();
        self.node_mut(node.parent.unwrap())
            .children
            .remove(&node.label[0]);
        self.key_elements -= node.label.len();
        self.free.push(id);
        node.value
    }

    /// Materialize a boundary inside an edge when necessary. No match returns ROOT.
    /// A split preserves existing IDs: existing users acquire the new parent in their path.
    pub fn match_prefix(&mut self, key: &[K]) -> PrefixMatch {
        let mut node = ROOT;
        let mut matched_len = 0;
        while matched_len < key.len() {
            let Some(&child) = self.node(node).children.get(&key[matched_len]) else {
                break;
            };
            let label = &self.node(child).label;
            let common = label
                .iter()
                .zip(&key[matched_len..])
                .take_while(|(a, b)| a == b)
                .count();
            matched_len += common;
            if common < label.len() {
                node = self.split(child, common);
                break;
            }
            node = child;
        }
        PrefixMatch { node, matched_len }
    }

    /// Append one unmatched edge. Match the full key first; suffix must be nonempty
    /// and its first element must not already name a child of parent.
    pub fn append(&mut self, parent: NodeId, suffix: &[K], value: V) -> NodeId {
        assert!(!suffix.is_empty());
        assert!(!self.node(parent).children.contains_key(&suffix[0]));
        let id = self.allocate(Node {
            label: suffix.to_vec(),
            value: Some(value),
            parent: Some(parent),
            children: HashMap::new(),
        });
        self.node_mut(parent).children.insert(suffix[0].clone(), id);
        self.key_elements += suffix.len();
        id
    }

    /// Read-only LCP queries do not split edges or allocate nodes.
    pub fn longest_common_prefix_len(&self, key: &[K]) -> usize {
        let mut node = ROOT;
        let mut matched = 0;
        while matched < key.len() {
            let Some(&child) = self.node(node).children.get(&key[matched]) else {
                break;
            };
            let label = &self.node(child).label;
            let common = label
                .iter()
                .zip(&key[matched..])
                .take_while(|(a, b)| a == b)
                .count();
            matched += common;
            if common < label.len() {
                break;
            }
            node = child;
        }
        matched
    }

    fn node(&self, id: NodeId) -> &Node<K, V> {
        self.nodes[id.0].as_ref().expect("removed radix node")
    }
    fn node_mut(&mut self, id: NodeId) -> &mut Node<K, V> {
        self.nodes[id.0].as_mut().expect("removed radix node")
    }
    fn allocate(&mut self, node: Node<K, V>) -> NodeId {
        if let Some(id) = self.free.pop() {
            self.nodes[id.0] = Some(node);
            id
        } else {
            let id = NodeId(self.nodes.len());
            self.nodes.push(Some(node));
            id
        }
    }

    fn split(&mut self, child: NodeId, at: usize) -> NodeId {
        let edge = self.node_mut(child);
        assert!(at > 0 && at < edge.label.len());
        let parent = edge.parent.unwrap();
        let first = edge.label[0].clone();
        let value = edge.value.as_mut().unwrap().split_prefix(at);
        let label = edge.label.split_prefix(at);
        let suffix_first = edge.label[0].clone();
        let middle = self.allocate(Node {
            label,
            value: Some(value),
            parent: Some(parent),
            children: HashMap::from([(suffix_first, child)]),
        });
        self.node_mut(child).parent = Some(middle);
        self.node_mut(parent).children.insert(first, middle);
        middle
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    #[test]
    fn remove_leaf_preserves_parents_accounting_and_reuses_ids() {
        let mut tree = RadixTree::default();
        let suffix = tree.append(ROOT, &[1, 2, 3], vec![1, 2, 3]);
        let parent = tree.match_prefix(&[1]).node;
        assert!(tree.remove_leaf(ROOT).is_none());
        assert!(tree.remove_leaf(parent).is_none());
        assert_eq!(tree.leaves().collect::<Vec<_>>(), vec![suffix]);
        assert_eq!(tree.remove_leaf(suffix), Some(vec![2, 3]));
        assert_eq!(tree.key_elements(), 1);
        assert!(tree.is_leaf(parent));
        assert_eq!(tree.remove_leaf(parent), Some(vec![1]));
        assert_eq!(tree.key_elements(), 0);
        assert_eq!(tree.node_count(), 1);
        assert_eq!(tree.append(ROOT, &[9], vec![9]), parent);
        assert_eq!(tree.match_prefix(&[9]).matched_len, 1);
    }

    #[test]
    fn descending_splits_keep_linear_label_and_value_capacity() {
        let mut tree = RadixTree::default();
        let key = vec![3u32; 4096];
        tree.append(ROOT, &key, key.clone());
        for len in (1..key.len()).rev() {
            tree.match_prefix(&key[..len]);
        }
        assert_eq!(tree.node_count(), key.len() + 1);
        assert!(
            tree.nodes
                .iter()
                .flatten()
                .map(|node| node.label.capacity())
                .sum::<usize>()
                <= 2 * key.len()
        );
        assert!(
            tree.nodes
                .iter()
                .flatten()
                .filter_map(|node| node.value.as_ref())
                .map(Vec::capacity)
                .sum::<usize>()
                <= 2 * key.len()
        );
    }

    #[test]
    fn split_preserves_suffix_ids_and_disjoint_edge_values() {
        let mut tree = RadixTree::default();
        let a = tree.append(ROOT, &[1, 2, 3, 4], vec![10, 11, 12, 13]);
        let hit = tree.match_prefix(&[1, 2, 9]);
        assert_eq!(hit.matched_len, 2);
        assert_eq!(tree.parent(a), Some(hit.node));
        assert_eq!(tree.value(a).unwrap(), &[12, 13]);
        assert_eq!(tree.value(hit.node).unwrap(), &[10, 11]);
        tree.append(hit.node, &[9], vec![99]);
        assert_eq!(tree.key_elements(), 5);
        assert_eq!(tree.match_prefix(&[1, 2, 3, 4]).node, a);
        assert_eq!(tree.match_prefix(&[]).node, ROOT);
    }

    #[test]
    fn randomized_paths_match_an_independent_prefix_oracle() {
        let mut rng = StdRng::seed_from_u64(271);
        let mut tree = RadixTree::default();
        let mut keys = Vec::new();
        for _ in 0..500 {
            let key: Vec<u32> = (0..rng.random_range(1..80))
                .map(|_| rng.random_range(0..16))
                .collect();
            let found = tree.match_prefix(&key);
            if found.matched_len < key.len() {
                tree.append(
                    found.node,
                    &key[found.matched_len..],
                    key[found.matched_len..].to_vec(),
                );
            }
            keys.push(key);
        }
        for _ in 0..1000 {
            let query: Vec<u32> = (0..rng.random_range(0..90))
                .map(|_| rng.random_range(0..16))
                .collect();
            let expected = keys
                .iter()
                .map(|key| key.iter().zip(&query).take_while(|(a, b)| a == b).count())
                .max()
                .unwrap();
            let found = tree.match_prefix(&query);
            assert_eq!(found.matched_len, expected);
            let mut parts: Vec<_> = tree
                .ancestors(found.node)
                .filter_map(|id| tree.value(id))
                .collect();
            parts.reverse();
            assert_eq!(
                parts.into_iter().flatten().copied().collect::<Vec<_>>(),
                query[..expected]
            );
        }
    }

    #[test]
    fn nested_paths_and_drop_fit_a_small_stack() {
        std::thread::Builder::new()
            .stack_size(2 * 1024 * 1024)
            .spawn(|| {
                let mut tree = RadixTree::default();
                let key = vec![7u32; 8192];
                tree.append(ROOT, &key, key.clone());
                for len in 1..=key.len() {
                    assert_eq!(tree.match_prefix(&key[..len]).matched_len, len);
                }
                assert_eq!(tree.node_count(), key.len() + 1);
                assert!(
                    tree.nodes
                        .iter()
                        .flatten()
                        .map(|node| node.label.capacity())
                        .sum::<usize>()
                        <= 2 * key.len()
                );
                assert!(
                    tree.nodes
                        .iter()
                        .flatten()
                        .filter_map(|node| node.value.as_ref())
                        .map(Vec::capacity)
                        .sum::<usize>()
                        <= 2 * key.len()
                );
                drop(tree);
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
