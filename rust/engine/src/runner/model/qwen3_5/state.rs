use std::{ops::Range, sync::Arc};

use anyhow::{ensure, Context, Result};
use cuda_async::{
    device_buffer::DevicePointer,
    device_operation::{value, with_context, DeviceOp, ExecutionContext},
    scheduling_policies::SchedulingPolicy,
};
use cutile::{
    api,
    core::{bf16, ElementType},
    tensor::Tensor,
    tile_kernel::TileKernel,
    DType,
};

use super::{
    config::{Qwen3_5LayerType, Qwen3_5TextConfig},
    constants::*,
    kernels,
};
use crate::runner::kv_pool::{KVPool, NaiveContextState, NaiveKVPool};

enum LayerState<T: DType> {
    Delta {
        conv: Tensor<T>,
        recurrent: Tensor<f32>,
    },
    Attention {
        k: Tensor<T>,
        v: Tensor<T>,
    },
    PooledAttention {
        index: usize,
    },
}

/// Exclusive, append-only logical history backed by mutable hybrid state.
/// Failed/cancelled updates cannot be rolled back without a recurrent snapshot.
pub struct Qwen3_5Context<T: DType = bf16> {
    owner: Arc<()>,
    layers: Vec<LayerState<T>>,
    history: Vec<u32>,
    pending: Option<Vec<u32>>,
    max_seq_len: usize,
    valid: bool,
}

impl<T: DType + ElementType> Qwen3_5Context<T> {
    pub fn ready_len(&self) -> usize {
        self.history.len()
    }
    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// Caller validates dimensions and retains all allocations on the submission.
    pub(super) unsafe fn allocate(
        config: &Qwen3_5TextConfig,
        owner: Arc<()>,
        max_seq_len: usize,
        pooled: bool,
        exec: &ExecutionContext,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let mut attention_index = 0;
        for kind in &config.layer_types {
            layers.push(match kind {
                Qwen3_5LayerType::LinearAttention => LayerState::Delta {
                    conv: api::zeros(&[
                        CONV_HISTORY_LEN,
                        config.linear_num_key_heads * GDN_QK_CHANNELS_PER_HEAD
                            + config.linear_num_value_heads * GDN_VALUE_HEAD_DIM,
                    ])
                    .execute(exec)?,
                    recurrent: api::zeros(&[
                        config.linear_num_value_heads * GDN_KEY_HEAD_DIM,
                        GDN_VALUE_HEAD_DIM,
                    ])
                    .execute(exec)?,
                },
                Qwen3_5LayerType::FullAttention if pooled => {
                    let index = attention_index;
                    attention_index += 1;
                    LayerState::PooledAttention { index }
                }
                Qwen3_5LayerType::FullAttention => LayerState::Attention {
                    k: api::zeros(&[max_seq_len, config.num_key_value_heads * ATTENTION_HEAD_DIM])
                        .execute(exec)?,
                    v: api::zeros(&[max_seq_len, config.num_key_value_heads * ATTENTION_HEAD_DIM])
                        .execute(exec)?,
                },
            });
        }
        Ok(Self {
            owner,
            layers,
            history: Vec::with_capacity(max_seq_len),
            pending: None,
            max_seq_len,
            valid: true,
        })
    }

    pub(super) fn validate(&self, owner: &Arc<()>, tokens: &[u32]) -> Result<()> {
        ensure!(
            Arc::ptr_eq(owner, &self.owner),
            "context belongs to another engine"
        );
        ensure!(
            self.valid && self.pending.is_none(),
            "context was invalidated by a cancelled or failed forward"
        );
        ensure!(
            tokens.len() > self.history.len()
                && tokens.len() <= self.max_seq_len
                && tokens.starts_with(&self.history),
            "expected an extending history within the maximum session sequence length"
        );
        Ok(())
    }

    pub(super) fn begin<'a>(
        &'a mut self,
        tokens: &[u32],
        positions: Range<usize>,
        pooled: Option<(&'a NaiveKVPool<T>, &'a mut NaiveContextState)>,
        exec: &'a ExecutionContext,
    ) -> Result<Qwen3_5ExecutionView<'a, T>> {
        self.validate(&self.owner, tokens)?;
        ensure!(
            positions.start == self.history.len() && positions.end == tokens.len(),
            "execution range differs from hybrid history"
        );
        self.pending = Some(tokens[positions.clone()].to_vec());
        self.valid = false;
        Ok(Qwen3_5ExecutionView {
            layers: &mut self.layers,
            positions,
            pooled,
            exec,
        })
    }

    /// Called only after the corresponding GPU completion succeeds.
    pub(super) fn commit(&mut self, positions: Range<usize>) -> Result<()> {
        let pending = self.pending.as_ref().context("no pending hybrid update")?;
        ensure!(
            !self.valid
                && positions.start == self.history.len()
                && positions.len() == pending.len(),
            "hybrid commit range mismatch"
        );
        self.history.extend(self.pending.take().unwrap());
        self.valid = true;
        Ok(())
    }

    pub(super) async fn execute<O: Send, R>(
        &mut self,
        owner: &Arc<()>,
        tokens: &[u32],
        policy: &Arc<dyn SchedulingPolicy>,
        submit: impl FnOnce(&mut Qwen3_5ExecutionView<'_, T>, &ExecutionContext) -> Result<O> + Send,
        complete: impl FnOnce(O) -> Result<R>,
    ) -> Result<R> {
        self.validate(owner, tokens)?;
        let positions = self.ready_len()..tokens.len();
        let output = with_context(|exec| {
            value((|| {
                let mut view = self.begin(tokens, positions.clone(), None, exec)?;
                submit(&mut view, exec)
            })())
        })
        .schedule(policy)?
        .await??;
        let output = complete(output)?;
        self.commit(positions)?;
        Ok(output)
    }
}

/// One pending append's resources. It cannot change history, validity or allocation ownership.
pub struct Qwen3_5ExecutionView<'a, T: DType = bf16> {
    layers: &'a mut [LayerState<T>],
    positions: Range<usize>,
    pooled: Option<(&'a NaiveKVPool<T>, &'a mut NaiveContextState)>,
    exec: &'a ExecutionContext,
}

pub(super) struct DeltaView<T: DType> {
    pub conv: DevicePointer<T>,
    pub recurrent: DevicePointer<f32>,
}

pub(super) struct AttentionView<T: DType> {
    pub k: Arc<Tensor<T>>,
    pub v: Arc<Tensor<T>>,
    pub table: DevicePointer<i32>,
    pub table_offset: usize,
    pub pooled: bool,
}

impl<T: DType + ElementType> Qwen3_5ExecutionView<'_, T> {
    pub(super) fn start_pos(&self) -> usize {
        self.positions.start
    }

    pub(super) fn delta(&mut self, layer: usize) -> Result<DeltaView<T>> {
        let Some(LayerState::Delta { conv, recurrent }) = self.layers.get_mut(layer) else {
            anyhow::bail!("expected DeltaNet state");
        };
        // SAFETY: this view exclusively borrows state; exec retains allocations through completion.
        unsafe {
            Ok(DeltaView {
                conv: retained_pointer(conv, self.exec)?,
                recurrent: retained_pointer(recurrent, self.exec)?,
            })
        }
    }

    pub(super) unsafe fn attention(
        &mut self,
        layer: usize,
        k: &Tensor<T>,
        v: &Tensor<T>,
    ) -> Result<AttentionView<T>> {
        ensure!(
            k.shape() == v.shape()
                && k.shape().len() == 2
                && k.shape()[0] as usize == self.positions.len(),
            "KV input differs from execution range"
        );
        match self.layers.get_mut(layer).context("invalid state layer")? {
            LayerState::Attention {
                k: cache_k,
                v: cache_v,
            } => {
                ensure!(
                    k.shape()[1] == cache_k.shape()[1],
                    "KV width differs from cache"
                );
                let heads = k.shape()[1] as usize / ATTENTION_HEAD_DIM;
                for (dst, src) in [(&mut *cache_k, k), (&mut *cache_v, v)] {
                    kernels::cache_v(
                        retained_pointer(dst, self.exec)?,
                        src,
                        self.positions.start as i32,
                    )
                    .generics(vec![T::DTYPE.to_string(), ATTENTION_HEAD_DIM.to_string()])
                    .grid((self.positions.len() as u32, heads as u32, 1))
                    .execute(self.exec)?;
                }
                // The contiguous specialization never dereferences this mapping pointer.
                Ok(AttentionView {
                    k: cache_k.into_shared_alias(),
                    v: cache_v.into_shared_alias(),
                    table: DevicePointer::from_cu_deviceptr(0),
                    table_offset: 0,
                    pooled: false,
                })
            }
            LayerState::PooledAttention { index } => {
                let (pool, state) = self.pooled.as_mut().context("missing GQA state pool")?;
                let dst = pool.write_view(state, *index, self.positions.clone(), self.exec)?;
                pool.store(dst, k.slice(&[])?, v.slice(&[])?)?;
                let view = pool.read_view(state, *index, 0..self.positions.end)?;
                view.retain(self.exec)?;
                let (k, v, table) = view.tensors();
                Ok(AttentionView {
                    k: k.clone(),
                    v: v.clone(),
                    table: table.device_pointer(),
                    table_offset: view.row() * view.table_width(),
                    pooled: true,
                })
            }
            _ => anyhow::bail!("expected attention state"),
        }
    }
}

/// Retain raw-pointer destinations through GPU completion. The caller must
/// exclusively own the tensor and order every access on this execution context.
pub(super) unsafe fn retained_pointer<T: DType>(
    tensor: &mut Tensor<T>,
    exec: &ExecutionContext,
) -> Result<DevicePointer<T>> {
    exec.retain(tensor.into_shared_alias())?;
    Ok(tensor.device_pointer())
}
