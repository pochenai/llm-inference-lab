use super::kernels;

// Supported architecture dimensions, distinct from execution tile sizes.
pub(super) const ATTENTION_HEAD_DIM: usize = 256;
pub(super) const GATED_QUERY_HEAD_DIM: usize = 2 * ATTENTION_HEAD_DIM;
pub(super) const ROTARY_DIM: usize = 64;
pub(super) const GDN_KEY_HEAD_DIM: usize = 128;
pub(super) const GDN_VALUE_HEAD_DIM: usize = 128;
pub(super) const GDN_QK_CHANNELS_PER_HEAD: usize = 2 * GDN_KEY_HEAD_DIM;
pub(super) const MAX_ROUTED_EXPERTS: usize = 256;
pub(super) const MAX_VOCAB_SIZE: usize = 262144;

// GPU-owned constants keep host launch shapes aligned with kernel tile shapes.
pub(super) const CONV_KERNEL_SIZE: usize = kernels::CONV_KERNEL_SIZE as usize;
pub(super) const CONV_HISTORY_LEN: usize = kernels::CONV_HISTORY_LEN as usize;
pub(super) const POINTWISE_COLUMNS: usize = kernels::POINTWISE_COLUMNS as usize;
pub(super) const DELTA_VALUE_COLUMNS: usize = kernels::DELTA_VALUE_COLUMNS as usize;
pub(super) const EXPERT_OUTPUT_COLUMNS: usize = kernels::EXPERT_OUTPUT_COLUMNS as usize;

// The shared Qwen2 pointwise kernels use a fixed 256-column tile.
pub(super) const SHARED_POINTWISE_COLUMNS: usize = 256;
pub(super) const LINEAR_TILE_ROWS: usize = 16;
pub(super) const LINEAR_TILE_COLUMNS: usize = 64;
pub(super) const LINEAR_TILE_REDUCTION: usize = 64;
