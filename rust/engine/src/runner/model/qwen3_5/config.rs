use anyhow::{ensure, Result};
use cutile::core::bf16;
use serde::Deserialize;

use super::constants::*;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
pub enum Qwen3_5ModelType {
    #[serde(rename = "qwen3_5_moe")]
    Moe,
    #[serde(rename = "qwen3_5_moe_text")]
    MoeText,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Qwen3_5LayerType {
    LinearAttention,
    FullAttention,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Qwen3_5HiddenAct {
    Silu,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Qwen3_5Dtype {
    Bfloat16,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Qwen3_5RopeType {
    Default,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Qwen3_5Config {
    pub model_type: Qwen3_5ModelType,
    pub tie_word_embeddings: bool,
    pub text_config: Qwen3_5TextConfig,
    #[serde(default)]
    quantization_config: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Qwen3_5TextConfig {
    pub model_type: Qwen3_5ModelType,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    pub eos_token_id: u32,
    pub layer_types: Vec<Qwen3_5LayerType>,
    pub rope_parameters: RopeParameters,
    pub attention_bias: bool,
    pub attn_output_gate: bool,
    pub hidden_act: Qwen3_5HiddenAct,
    pub mlp_only_layers: Vec<usize>,
    pub dtype: Qwen3_5Dtype,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RopeParameters {
    pub rope_type: Qwen3_5RopeType,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
}

impl Qwen3_5Config {
    pub fn validate(&self) -> Result<()> {
        let c = &self.text_config;
        ensure!(
            [
                c.hidden_size,
                c.vocab_size,
                c.num_hidden_layers,
                c.num_attention_heads,
                c.num_key_value_heads,
                c.linear_num_key_heads,
                c.linear_num_value_heads,
                c.num_experts,
                c.num_experts_per_tok,
                c.moe_intermediate_size,
                c.shared_expert_intermediate_size
            ]
            .into_iter()
            .all(|n| n > 0 && n <= i32::MAX as usize),
            "dimensions must be positive i32 values"
        );
        ensure!(
            self.model_type == Qwen3_5ModelType::Moe && c.model_type == Qwen3_5ModelType::MoeText,
            "expected Qwen3.5 MoE configuration"
        );
        ensure!(
            self.quantization_config.is_none(),
            "only the unquantized BF16 checkpoint is supported"
        );
        ensure!(
            !self.tie_word_embeddings
                && !c.attention_bias
                && c.attn_output_gate
                && c.mlp_only_layers.is_empty(),
            "unsupported decoder configuration"
        );
        ensure!(
            c.num_hidden_layers > 0 && c.layer_types.len() == c.num_hidden_layers,
            "invalid layer schedule"
        );
        ensure!(
            c.hidden_size > 0
                && c.hidden_size % POINTWISE_COLUMNS == 0
                && c.hidden_size.is_power_of_two()
                && c.vocab_size > 0
                && c.vocab_size <= MAX_VOCAB_SIZE
                && c.eos_token_id < c.vocab_size as u32,
            "invalid hidden/vocabulary dimensions"
        );
        ensure!(
            c.head_dim == ATTENTION_HEAD_DIM
                && c.linear_key_head_dim == GDN_KEY_HEAD_DIM
                && c.linear_value_head_dim == GDN_VALUE_HEAD_DIM
                && c.linear_conv_kernel_dim == CONV_KERNEL_SIZE,
            "unsupported attention/convolution dimensions"
        );
        ensure!(
            c.num_key_value_heads > 0
                && c.num_attention_heads > 0
                && c.num_attention_heads % c.num_key_value_heads == 0
                && c.linear_num_key_heads > 0
                && c.linear_num_value_heads > 0
                && c.linear_num_value_heads % c.linear_num_key_heads == 0,
            "invalid grouped head counts"
        );
        ensure!(
            c.num_experts_per_tok.is_power_of_two()
                && c.num_experts.is_power_of_two()
                && c.num_experts <= MAX_ROUTED_EXPERTS
                && (1..=c.num_experts).contains(&c.num_experts_per_tok)
                && c.moe_intermediate_size > 0
                && c.moe_intermediate_size % POINTWISE_COLUMNS == 0
                && c.shared_expert_intermediate_size > 0
                && c.shared_expert_intermediate_size % SHARED_POINTWISE_COLUMNS == 0,
            "unsupported expert dimensions"
        );
        ensure!(
            c.rms_norm_eps.is_finite()
                && c.rms_norm_eps > 0.0
                && c.max_position_embeddings > 0
                && c.max_position_embeddings <= i32::MAX as usize,
            "invalid norm/position configuration"
        );
        ensure!(
            c.rope_parameters.partial_rotary_factor
                == ROTARY_DIM as f32 / ATTENTION_HEAD_DIM as f32
                && c.rope_parameters.rope_theta.is_finite()
                && c.rope_parameters.rope_theta > 0.0,
            "unsupported RoPE configuration"
        );
        for dimensions in [
            vec![c.num_experts, 2 * c.moe_intermediate_size, c.hidden_size],
            vec![c.num_experts, c.hidden_size, c.moe_intermediate_size],
            vec![c.num_attention_heads, 2, c.head_dim],
            vec![
                c.linear_num_value_heads,
                GDN_KEY_HEAD_DIM,
                GDN_VALUE_HEAD_DIM,
            ],
            vec![c.vocab_size, c.hidden_size],
            vec![
                c.linear_num_key_heads * GDN_QK_CHANNELS_PER_HEAD
                    + c.linear_num_value_heads * GDN_VALUE_HEAD_DIM,
                c.hidden_size,
            ],
            vec![c.num_key_value_heads, ATTENTION_HEAD_DIM, c.hidden_size],
        ] {
            ensure!(
                dimensions
                    .iter()
                    .try_fold(1usize, |n, d| n.checked_mul(*d))
                    .is_some_and(|n| n <= i32::MAX as usize),
                "tensor exceeds i32 indexing"
            );
        }
        Ok(())
    }
}

impl Qwen3_5TextConfig {
    /// Session-state bytes for max_seq_len total tokens (prompt plus generation).
    /// Includes BF16 K/V, FP32 recurrence and BF16 convolution history.
    pub fn state_bytes(&self, max_seq_len: usize) -> Option<usize> {
        let full = self
            .layer_types
            .iter()
            .filter(|&&kind| kind == Qwen3_5LayerType::FullAttention)
            .count();
        let linear = self.num_hidden_layers.checked_sub(full)?;
        let kv = max_seq_len
            .checked_mul(self.num_key_value_heads)?
            .checked_mul(self.head_dim)?
            .checked_mul(2)?
            .checked_mul(size_of::<bf16>())?
            .checked_mul(full)?;
        let recurrent = self
            .linear_num_value_heads
            .checked_mul(self.linear_key_head_dim)?
            .checked_mul(self.linear_value_head_dim)?
            .checked_mul(size_of::<f32>())?;
        let channels = self
            .linear_num_key_heads
            .checked_mul(self.linear_key_head_dim)?
            .checked_mul(2)?
            .checked_add(
                self.linear_num_value_heads
                    .checked_mul(self.linear_value_head_dim)?,
            )?;
        kv.checked_add(
            linear.checked_mul(
                recurrent.checked_add(
                    channels
                        .checked_mul(CONV_HISTORY_LEN)?
                        .checked_mul(size_of::<bf16>())?,
                )?,
            )?,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_json() -> serde_json::Value {
        serde_json::json!({
            "model_type": "qwen3_5_moe",
            "tie_word_embeddings": false,
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "vocab_size": 248320,
                "hidden_size": 2048,
                "num_hidden_layers": 40,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 32,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "num_experts": 256,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 512,
                "shared_expert_intermediate_size": 512,
                "rms_norm_eps": 1e-06,
                "max_position_embeddings": 262144,
                "eos_token_id": 248044,
                "layer_types": (["linear_attention", "linear_attention", "linear_attention", "full_attention"].repeat(10)),
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 10000000,
                    "partial_rotary_factor": 0.25
                },
                "attention_bias": false,
                "attn_output_gate": true,
                "hidden_act": "silu",
                "mlp_only_layers": [],
                "dtype": "bfloat16"
            }
        })
    }

    fn config() -> Qwen3_5Config {
        serde_json::from_value(config_json()).unwrap()
    }

    #[test]
    fn checkpoint_and_state_budget() {
        let c = config();
        c.validate().unwrap();
        assert_eq!(
            c.text_config
                .layer_types
                .iter()
                .filter(|&&kind| kind == Qwen3_5LayerType::FullAttention)
                .count(),
            10
        );
        assert_eq!(c.text_config.state_bytes(4096), Some(148275200));
    }

    #[test]
    fn rejects_unsupported_configs() {
        let mut c = config();
        c.model_type = Qwen3_5ModelType::MoeText;
        assert!(c.validate().is_err());
        let mut c = config();
        c.text_config.model_type = Qwen3_5ModelType::Moe;
        assert!(c.validate().is_err());
        let mut c = config();
        c.text_config.linear_num_key_heads = 0;
        assert!(c.validate().is_err());
        let mut c = config();
        c.quantization_config = Some(serde_json::json!({"quant_method":"fp8"}));
        assert!(c.validate().is_err());
        let mut c = config();
        c.text_config.layer_types.pop();
        assert!(c.validate().is_err());
        let mut c = config();
        c.text_config.num_experts_per_tok = 257;
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_unknown_config_tags_during_deserialization() {
        for (path, value) in [
            ("/model_type", "qwen2"),
            ("/text_config/model_type", "unknown_model"),
            ("/text_config/layer_types/0", "unknown_attention"),
            ("/text_config/hidden_act", "gelu"),
            ("/text_config/dtype", "float32"),
            ("/text_config/rope_parameters/rope_type", "linear"),
        ] {
            let mut json = config_json();
            *json.pointer_mut(path).unwrap() = serde_json::json!(value);
            let error = serde_json::from_value::<Qwen3_5Config>(json).unwrap_err();
            assert!(error.to_string().contains(value), "{path}: {error}");
        }
    }
}
