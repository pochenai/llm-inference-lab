use super::prefix_tree::{PrefixTree, EVICTION_HEADROOM_DIVISOR};
use crate::{error::EngineError, types::SessionId};
use rand::{seq::IteratorRandom, Rng};
use std::{collections::HashMap, num::NonZeroUsize};

/// Keep routing key work bounded independently of full forwarded prompt size.
const MAX_KEY_BYTES: usize = 64 * 1024;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DestinationId(pub u64);

/// Isolates prefix hints and destinations whose model/input semantics are incompatible.
/// Identical text under different weights, LoRA adapters or position encodings may
/// produce different KV. All endpoints in one namespace must agree on these semantics.
/// Use CacheNamespace(0) for the current single-model setup; this is not a request/session ID.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CacheNamespace(pub u64);

#[derive(Clone, Copy, Debug)]
pub struct Destination {
    pub id: DestinationId,
    pub namespace: CacheNamespace,
}

#[derive(Clone, Debug)]
pub struct RoutingConfig {
    /// Namespace served by this Scheduler; requests use its existing model-agnostic API.
    pub namespace: CacheNamespace,
    /// Byte cap for routing hints, at most 64 KiB; never truncates forwarded prompts.
    pub max_prefix_bytes: NonZeroUsize,
    /// Accounted retained allocations, not allocator RSS; temporary splits may exceed it.
    pub max_index_bytes: NonZeroUsize,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            namespace: CacheNamespace::default(),
            max_prefix_bytes: NonZeroUsize::new(4096).unwrap(),
            max_index_bytes: NonZeroUsize::new(64 * 1024 * 1024).unwrap(),
        }
    }
}

/// Read-only routing snapshot; the policy returns its selected candidate unchanged.
#[derive(Clone, Copy, Debug)]
pub struct DestinationCandidate {
    pub destination: DestinationId,
    pub matched_prefix_bytes: usize,
}

pub trait RoutingPolicy: Send + Sync + 'static {
    /// Read-only selection; framework owns placement, accounting and forwarding.
    fn select(&self, candidates: &[DestinationCandidate]) -> Option<DestinationCandidate>;
}

#[derive(Default)]
pub struct RandomPrefixPolicy;

impl RandomPrefixPolicy {
    /// Accept an injected RNG so policy tests can use a fixed seed.
    #[inline]
    fn select_with(
        &self,
        candidates: &[DestinationCandidate],
        rng: &mut impl Rng,
    ) -> Option<DestinationCandidate> {
        let longest = candidates.iter().map(|c| c.matched_prefix_bytes).max()?;
        candidates
            .iter()
            .filter(|c| c.matched_prefix_bytes == longest)
            .choose(rng)
            .copied()
    }
}

impl RoutingPolicy for RandomPrefixPolicy {
    #[inline]
    fn select(&self, candidates: &[DestinationCandidate]) -> Option<DestinationCandidate> {
        self.select_with(candidates, &mut rand::rng())
    }
}

#[derive(Clone)]
pub(super) struct AccessState {
    pub(super) last_access_tick: u64,
}

struct DestinationState {
    namespace: CacheNamespace,
    available: bool,
}

pub(super) struct RoutingState {
    destinations: HashMap<DestinationId, DestinationState>,
    session_routes: HashMap<SessionId, DestinationId>,
    config: RoutingConfig,
    prefixes: PrefixTree,
    access_tick: u64,
}

impl RoutingState {
    pub(super) fn new(
        destinations: &[Destination],
        config: RoutingConfig,
    ) -> Result<Self, EngineError> {
        if config.max_prefix_bytes.get() > MAX_KEY_BYTES {
            return Err(EngineError::InvalidInput(
                "routing prefix cap exceeds supported 64 KiB limit".into(),
            ));
        }
        let mut states = HashMap::new();
        for destination in destinations {
            if states
                .insert(
                    destination.id,
                    DestinationState {
                        namespace: destination.namespace,
                        available: true,
                    },
                )
                .is_some()
            {
                return Err(EngineError::InvalidInput(
                    "duplicate routing destination".into(),
                ));
            }
        }
        if !states.values().any(|s| s.namespace == config.namespace) {
            return Err(EngineError::InvalidInput(
                "no destination in Scheduler namespace".into(),
            ));
        }
        Ok(Self {
            destinations: states,
            session_routes: HashMap::new(),
            config,
            prefixes: PrefixTree::default(),
            access_tick: 0,
        })
    }

    pub(super) fn session_destination(
        &self,
        session: SessionId,
    ) -> Result<DestinationId, EngineError> {
        let &destination = self
            .session_routes
            .get(&session)
            .ok_or(EngineError::SessionLost(session))?;
        self.available(destination)
            .then_some(destination)
            .ok_or(EngineError::ChannelClosed)
    }

    pub(super) fn prefix_matches(&self, input: &str) -> HashMap<DestinationId, usize> {
        if !self.multiple_destinations() {
            return HashMap::new();
        }
        self.prefixes.matches(self.prefix(input))
    }

    fn multiple_destinations(&self) -> bool {
        self.available_destinations().take(2).count() == 2
    }

    pub(super) fn available_destinations(&self) -> impl Iterator<Item = DestinationId> + '_ {
        self.destinations
            .keys()
            .copied()
            .filter(|&destination| self.available(destination))
    }

    fn prefix<'a>(&self, input: &'a str) -> &'a [u8] {
        &input.as_bytes()[..input.len().min(self.config.max_prefix_bytes.get())]
    }

    /// Borrow the full input; only new capped tree labels are copied.
    /// Staged before ownership transfer; immediate send failure invalidates this destination.
    pub(super) fn record_dispatch(&mut self, input: &str, destination: DestinationId) {
        let input = input.as_bytes();
        if input.is_empty() || !self.multiple_destinations() {
            return;
        }
        let input = &input[..input.len().min(self.config.max_prefix_bytes.get())];
        if PrefixTree::entry_charge(input.len()) > self.config.max_index_bytes.get() {
            return;
        }
        self.access_tick = self
            .access_tick
            .checked_add(1)
            .expect("routing recency counter exhausted");
        self.prefixes.insert(input, destination, self.access_tick);
        self.prune();
    }

    pub(super) fn index_bytes(&self) -> usize {
        self.prefixes.bytes()
    }

    fn prune(&mut self) {
        let budget = self.config.max_index_bytes.get();
        if self.index_bytes() <= budget {
            return;
        }
        let target = budget - budget / EVICTION_HEADROOM_DIVISOR;
        while self.index_bytes() > target {
            let candidates = self.prefixes.oldest_batch();
            if candidates.is_empty() {
                break;
            }
            for candidate in candidates {
                self.prefixes.evict(candidate);
                if self.index_bytes() <= target {
                    return;
                }
            }
        }
    }

    pub(super) fn available(&self, destination: DestinationId) -> bool {
        self.destinations
            .get(&destination)
            .is_some_and(|s| s.available && s.namespace == self.config.namespace)
    }

    pub(super) fn any_available(&self) -> bool {
        self.destinations.keys().any(|&id| self.available(id))
    }

    pub(super) fn disconnected(&mut self, destination: DestinationId) {
        if let Some(state) = self.destinations.get_mut(&destination) {
            state.available = false;
        }
        self.prefixes.remove_destination(destination);
        if !self.multiple_destinations() {
            self.prefixes = PrefixTree::default();
        }
    }

    pub(super) fn opened(
        &mut self,
        session: SessionId,
        destination: DestinationId,
    ) -> Result<(), EngineError> {
        if !self.available(destination) {
            return Err(EngineError::ChannelClosed);
        }
        self.session_routes.insert(session, destination);
        Ok(())
    }

    /// An old endpoint's terminal reply must not erase another endpoint's placement.
    pub(super) fn closed(&mut self, session: SessionId, destination: DestinationId) {
        if self.session_routes.get(&session) == Some(&destination) {
            self.session_routes.remove(&session);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    fn state(config: RoutingConfig) -> RoutingState {
        RoutingState::new(
            &[
                Destination {
                    id: DestinationId(1),
                    namespace: CacheNamespace(0),
                },
                Destination {
                    id: DestinationId(2),
                    namespace: CacheNamespace(0),
                },
                Destination {
                    id: DestinationId(3),
                    namespace: CacheNamespace(1),
                },
            ],
            config,
        )
        .unwrap()
    }

    #[test]
    fn random_policy_selects_only_longest_matches() {
        let candidates = [
            DestinationCandidate {
                destination: DestinationId(1),
                matched_prefix_bytes: 4,
            },
            DestinationCandidate {
                destination: DestinationId(2),
                matched_prefix_bytes: 4,
            },
            DestinationCandidate {
                destination: DestinationId(3),
                matched_prefix_bytes: 2,
            },
        ];
        let mut rng = StdRng::seed_from_u64(17);
        let mut counts = [0; 2];
        for _ in 0..1000 {
            let decision = RandomPrefixPolicy
                .select_with(&candidates, &mut rng)
                .unwrap();
            assert_eq!(decision.matched_prefix_bytes, 4);
            counts[decision.destination.0 as usize - 1] += 1;
        }
        assert!(counts.iter().all(|&n| n > 400 && n < 600), "{counts:?}");
        assert_eq!(
            RandomPrefixPolicy
                .select(&candidates[..1])
                .unwrap()
                .destination,
            DestinationId(1)
        );
        let misses = candidates.map(|candidate| DestinationCandidate {
            matched_prefix_bytes: 0,
            ..candidate
        });
        let fallback = RandomPrefixPolicy.select(&misses).unwrap();
        assert!(misses
            .iter()
            .any(|candidate| candidate.destination == fallback.destination));
        assert_eq!(fallback.matched_prefix_bytes, 0);
        assert!(RandomPrefixPolicy.select(&[]).is_none());
    }

    #[test]
    fn caps_utf8_bytes_and_isolates_namespace_and_sessions() {
        assert!(RoutingState::new(
            &[Destination {
                id: DestinationId(1),
                namespace: CacheNamespace(0)
            }],
            RoutingConfig {
                max_prefix_bytes: NonZeroUsize::new(MAX_KEY_BYTES + 1).unwrap(),
                ..Default::default()
            },
        )
        .is_err());
        let mut state = state(RoutingConfig {
            max_prefix_bytes: NonZeroUsize::new(4).unwrap(),
            ..Default::default()
        });
        state.record_dispatch("你好吗", DestinationId(1));
        let matches = state.prefix_matches("你好世界");
        assert_eq!(matches[&DestinationId(1)], 4);
        assert!(!state
            .available_destinations()
            .any(|id| id == DestinationId(3)));
        state.opened(7, DestinationId(2)).unwrap();
        assert_eq!(state.session_destination(7).unwrap(), DestinationId(2));
        state.disconnected(DestinationId(2));
        assert!(state.session_destination(7).is_err());
        state.closed(7, DestinationId(2));
        assert!(matches!(
            state.session_destination(7),
            Err(EngineError::SessionLost(7))
        ));
        assert_eq!(RoutingConfig::default().max_prefix_bytes.get(), 4096);
    }

    #[test]
    fn budget_and_endpoint_invalidation_do_not_leave_unbounded_hints() {
        let mut state = state(RoutingConfig {
            max_index_bytes: NonZeroUsize::new(4096).unwrap(),
            ..Default::default()
        });
        for n in 0..1000 {
            state.record_dispatch(
                &format!("system:{n:04}:{}", "x".repeat(200)),
                DestinationId(1),
            );
            assert!(state.index_bytes() <= 4096);
        }
        state.disconnected(DestinationId(1));
        assert_eq!(state.index_bytes(), 0);
        assert!(state.prefix_matches("system:0999").is_empty());
        assert_eq!(
            state.available_destinations().collect::<Vec<_>>(),
            vec![DestinationId(2)]
        );
        let mut tiny = self::state(RoutingConfig {
            max_index_bytes: NonZeroUsize::new(1).unwrap(),
            ..Default::default()
        });
        tiny.record_dispatch("oversized", DestinationId(1));
        assert_eq!(tiny.index_bytes(), 0);
    }

    #[test]
    fn singleton_namespace_never_allocates_prefix_hints() {
        let mut state = RoutingState::new(
            &[
                Destination {
                    id: DestinationId(1),
                    namespace: CacheNamespace(0),
                },
                Destination {
                    id: DestinationId(2),
                    namespace: CacheNamespace(1),
                },
            ],
            RoutingConfig::default(),
        )
        .unwrap();
        for i in 0..1000 {
            state.record_dispatch(&format!("{i}:{}", "x".repeat(4096)), DestinationId(1));
        }
        assert_eq!(state.index_bytes(), 0);
        assert!(state.prefix_matches("anything").is_empty());
    }

    #[test]
    fn pressure_reclaims_headroom_and_session_cleanup_checks_owner() {
        let config = RoutingConfig {
            max_index_bytes: NonZeroUsize::new(64 * 1024).unwrap(),
            ..Default::default()
        };
        let budget = config.max_index_bytes.get();
        let mut state = state(config);
        let mut saw_batch = false;
        for i in 0..100 {
            let input = format!("{i:08x}:{}", "x".repeat(4087));
            let before = state.prefixes.len();
            state.record_dispatch(&input, DestinationId(1));
            // Detect actual eviction, including pressure from arena-capacity growth.
            if state.prefixes.len() <= before {
                assert!(state.index_bytes() <= budget - budget / EVICTION_HEADROOM_DIVISOR);
                saw_batch = true;
            }
            assert!(state.index_bytes() <= budget);
        }
        assert!(saw_batch);
        state.opened(9, DestinationId(2)).unwrap();
        state.closed(9, DestinationId(1));
        assert_eq!(state.session_destination(9).unwrap(), DestinationId(2));
        state.closed(9, DestinationId(2));
        assert!(matches!(
            state.session_destination(9),
            Err(EngineError::SessionLost(9))
        ));
        state.disconnected(DestinationId(2));
        assert_eq!(state.index_bytes(), 0);
    }
}
