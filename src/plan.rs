//! Llama decoder graph construction, independent of the WIT host.
//!
//! One pre-norm decoder block: RMSNorm, the Q, K and V projections (with an
//! additive bias when the checkpoint has one), RoPE, grouped-query attention,
//! the output projection and a residual add, then RMSNorm, the SwiGLU-gated
//! MLP and a second residual add. Every dimension, and whether a bias term
//! exists at all, comes from the [`Config`] the Runtime bound to the Model
//! Instance: the same code serves every Llama-compatible checkpoint, and a
//! Qwen2 checkpoint's bias-bearing projections the same way.

/// The graph IR and the sink the builder writes to are the SDK's: the same
/// types every Model Component builds its graph with.
pub use magnetar_component_sdk::graph::{Attr, GraphSink, Kv, KvRole, Node, Phase, Value};

const ROPE_POSITION_MODE: &str = "sequential";
const ROPE_ROTATION_PAIRING: &str = "half_split";

/// Llama 3's per-frequency RoPE scaling (Hugging Face `rope_type = "llama3"`).
#[derive(Clone, Debug, PartialEq)]
pub struct BandedRopeScaling {
    pub factor: f64,
    pub low_freq_factor: f64,
    pub high_freq_factor: f64,
    pub original_max_position_embeddings: u64,
}

/// The frequency scaling the checkpoint declares.
#[derive(Clone, Debug, PartialEq)]
pub enum RopeScaling {
    /// Llama 3's, which this Component declares on its `rope` nodes.
    Banded(BandedRopeScaling),
    /// YaRN's, which it does not apply: a checkpoint that declares it is refused.
    Yarn,
}

/// The architecture the Runtime bound to the Model Instance.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub hidden_size: u64,
    pub intermediate_size: u64,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dimension: u64,
    pub vocabulary_size: u64,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub tie_word_embeddings: bool,
    /// Whether `self_attn.{q,k,v}_proj` carry an additive bias.
    pub attention_bias: bool,
    /// The activation the checkpoint declares. Only `silu` is supported.
    pub hidden_activation: Option<String>,
    pub rope_scaling: Option<RopeScaling>,
}

/// Which graph to build: its phase, its rows, and the layers `[start_layer,
/// end_layer)` it covers.
#[derive(Clone, Copy, Debug)]
pub struct Segment {
    pub phase: Phase,
    pub sequence_length: u64,
    pub position_offset: u64,
    pub start_layer: u32,
    pub end_layer: u32,
}

/// Maps a per-layer weight to the canonical Model Artifact tensor name
/// `weight_edge` is called with. Both Llama's and Qwen2's Hugging Face
/// checkpoints share this `self_attn`/`mlp` naming convention, so the same
/// canonical names resolve real weights for either family.
fn weight_tensor_name(layer: u64, suffix: &str) -> String {
    match suffix {
        "input_norm" | "post_attn_norm" => format!("layers.{layer}.{suffix}"),
        "q_proj" | "k_proj" | "v_proj" | "o_proj" | "q_bias" | "k_bias" | "v_bias" => {
            format!("layers.{layer}.self_attn.{suffix}")
        }
        "gate_proj" | "up_proj" | "down_proj" => format!("layers.{layer}.mlp.{suffix}"),
        other => unreachable!("weight_tensor_name: unknown weight suffix '{other}'"),
    }
}

/// A node's tensor-parallel role (`column`, `row` or `local`): how a Runtime
/// may split it across Devices.
fn tensor_parallel(kind: &'static str) -> Attr {
    Attr::new("tensor_parallel", kind)
}

struct Builder<'a, S> {
    sink: &'a mut S,
    config: &'a Config,
    rope_scaling: Option<&'a BandedRopeScaling>,
    sequence: u64,
    append: bool,
}

impl<S: GraphSink> Builder<'_, S> {
    #[allow(clippy::too_many_arguments)]
    fn node(
        &mut self,
        id: String,
        operator: &'static str,
        family: &'static str,
        attrs: Vec<Attr>,
        inputs: Vec<String>,
        width: u64,
        kv: Option<Kv>,
    ) -> String {
        self.sink.add_node(Node {
            id,
            operator,
            family,
            attrs,
            inputs,
            shape: vec![self.sequence, width],
            kv,
        })
    }

    fn kv(&mut self, layer: u64, role: KvRole) -> Kv {
        Kv {
            resource: self.sink.kv_resource(layer, role),
            role,
            append: self.append,
        }
    }

    fn rope_attrs(&self, head_count: u64) -> Vec<Attr> {
        let mut attrs = vec![
            Attr::new("base", self.config.rope_theta),
            Attr::new("dimension", self.config.head_dimension as i64),
            Attr::new("position_mode", ROPE_POSITION_MODE),
            // Hugging Face checkpoints rotate (i, i + dimension/2)
            // (`rotate_half`), not adjacent pairs; declared explicitly so the
            // Runtime never infers it (`fix-rope-rotation-pairing`).
            Attr::new("rotation_pairing", ROPE_ROTATION_PAIRING),
            Attr::new("position_offset", 0_i64),
            Attr::new("head_count", head_count as i64),
        ];
        if let Some(scaling) = self.rope_scaling {
            attrs.extend([
                Attr::new("frequency_scaling", "banded"),
                Attr::new("frequency_scaling_factor", scaling.factor),
                Attr::new("low_frequency_factor", scaling.low_freq_factor),
                Attr::new("high_frequency_factor", scaling.high_freq_factor),
                Attr::new(
                    "original_context_length",
                    scaling.original_max_position_embeddings as i64,
                ),
            ]);
        }
        attrs
    }

    /// One projection: the weight's edge, then the node reading it.
    #[allow(clippy::too_many_arguments)]
    fn projection(
        &mut self,
        prefix: &str,
        layer: u64,
        name: &'static str,
        attrs: Vec<Attr>,
        input: &str,
        width: u64,
        kv: Option<Kv>,
    ) -> String {
        let weight = self.sink.weight_edge(&weight_tensor_name(layer, name));
        self.node(
            format!("{prefix}.{name}"),
            "matmul",
            "LinearAlgebra",
            attrs,
            vec![input.to_string(), weight],
            width,
            kv,
        )
    }

    /// The bias added to a projection's output: the bias weight's edge, then
    /// the `add` node.
    fn bias(
        &mut self,
        prefix: &str,
        layer: u64,
        name: &'static str,
        input: String,
        width: u64,
        kv: Option<Kv>,
    ) -> String {
        let weight = self.sink.weight_edge(&weight_tensor_name(layer, name));
        self.node(
            format!("{prefix}.{name}"),
            "add",
            "Tensor",
            vec![],
            vec![input, weight],
            width,
            kv,
        )
    }

    fn norm(&mut self, id: String, weight: &str, input: String, width: u64) -> String {
        let weight = self.sink.weight_edge(weight);
        self.node(
            id,
            "rmsnorm",
            "Normalization",
            vec![Attr::new("epsilon", self.config.rms_norm_eps)],
            vec![input, weight],
            width,
            None,
        )
    }

    /// Builds one pre-norm decoder layer from the hidden state `residual_in`.
    fn layer(&mut self, layer: u64, residual_in: &str) -> String {
        let config = self.config;
        let prefix = format!("layer{layer}");
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;
        let attention_head_count = u64::from(config.num_attention_heads);
        let kv_head_count = u64::from(config.num_key_value_heads);
        let head_dimension = config.head_dimension;
        let q_dim = attention_head_count * head_dimension;
        let kv_dim = kv_head_count * head_dimension;

        let normed = self.norm(
            format!("{prefix}.input_norm"),
            &weight_tensor_name(layer, "input_norm"),
            residual_in.to_string(),
            hidden_size,
        );

        // A real Llama checkpoint declares no additive bias term for
        // self_attn.{q,k,v}_proj; a Qwen2/2.5 one does, and `attention_bias`
        // reflects what the ingested checkpoint declared. When present, the
        // bias is added to the raw projection output *before* RoPE and KV
        // caching, so a bias-add node's output (not the matmul's) becomes
        // `q`/`k`/`v` below.
        let q = self.projection(&prefix, layer, "q_proj", vec![], &normed, q_dim, None);
        let q = if config.attention_bias {
            self.bias(&prefix, layer, "q_bias", q, q_dim, None)
        } else {
            q
        };
        let k = self.projection(&prefix, layer, "k_proj", vec![], &normed, kv_dim, None);
        let k = if config.attention_bias {
            self.bias(&prefix, layer, "k_bias", k, kv_dim, None)
        } else {
            k
        };
        // The V projection's weight is read before its KV resource is asked
        // for, and the resource is attached to the bias-add node when there is
        // one, to the matmul when there is not.
        let v_weight = self.sink.weight_edge(&weight_tensor_name(layer, "v_proj"));
        let v_kv = self.kv(layer, KvRole::V);
        let v = self.node(
            format!("{prefix}.v_proj"),
            "matmul",
            "LinearAlgebra",
            vec![],
            vec![normed, v_weight],
            kv_dim,
            if config.attention_bias {
                None
            } else {
                Some(v_kv.clone())
            },
        );
        let v = if config.attention_bias {
            self.bias(&prefix, layer, "v_bias", v, kv_dim, Some(v_kv))
        } else {
            v
        };

        let rope_q_attrs = self.rope_attrs(attention_head_count);
        let q_rope = self.node(
            format!("{prefix}.rope_q"),
            "rope",
            "PositionEncoding",
            rope_q_attrs,
            vec![q],
            q_dim,
            None,
        );
        let k_kv = self.kv(layer, KvRole::K);
        let rope_k_attrs = self.rope_attrs(kv_head_count);
        let k_rope = self.node(
            format!("{prefix}.rope_k"),
            "rope",
            "PositionEncoding",
            rope_k_attrs,
            vec![k],
            kv_dim,
            Some(k_kv),
        );

        let attn_out = self.node(
            format!("{prefix}.attention"),
            "attention",
            "Attention",
            vec![
                Attr::new("causal", true),
                Attr::new("head_count", attention_head_count as i64),
                Attr::new("kv_head_count", kv_head_count as i64),
                Attr::new("head_dimension", head_dimension as i64),
                Attr::new("attention_mask_kind", "causal"),
            ],
            vec![q_rope, k_rope, v],
            q_dim,
            None,
        );
        let attn_proj = self.projection(
            &prefix,
            layer,
            "o_proj",
            vec![],
            &attn_out,
            hidden_size,
            None,
        );
        let post_attn = self.node(
            format!("{prefix}.residual1"),
            "residual-add",
            "Tensor",
            vec![],
            vec![residual_in.to_string(), attn_proj],
            hidden_size,
            None,
        );

        let mlp_normed = self.norm(
            format!("{prefix}.post_attn_norm"),
            &weight_tensor_name(layer, "post_attn_norm"),
            post_attn.clone(),
            hidden_size,
        );
        let gate = self.projection(
            &prefix,
            layer,
            "gate_proj",
            vec![tensor_parallel("column")],
            &mlp_normed,
            intermediate_size,
            None,
        );
        let up = self.projection(
            &prefix,
            layer,
            "up_proj",
            vec![tensor_parallel("column")],
            &mlp_normed,
            intermediate_size,
            None,
        );
        let activated = self.node(
            format!("{prefix}.silu"),
            "silu",
            "Activation",
            vec![tensor_parallel("local")],
            vec![gate],
            intermediate_size,
            None,
        );
        let mlp_hidden = self.node(
            format!("{prefix}.mul"),
            "mul",
            "Tensor",
            vec![tensor_parallel("local")],
            vec![activated, up],
            intermediate_size,
            None,
        );
        let mlp_out = self.projection(
            &prefix,
            layer,
            "down_proj",
            vec![tensor_parallel("row")],
            &mlp_hidden,
            hidden_size,
            None,
        );

        self.node(
            format!("{prefix}.residual2"),
            "residual-add",
            "Tensor",
            vec![],
            vec![post_attn, mlp_out],
            hidden_size,
            None,
        )
    }
}

/// Builds one *segment* of the decoder stack's graph: the real decoder layers
/// `[segment.start_layer, segment.end_layer)` only. The whole graph is the
/// segment from layer 0 to `config.num_hidden_layers`.
///
/// A segment that starts at layer 0 begins from a declared `token_ids` input
/// with the token embedding lookup; one that starts later begins from a
/// declared `hidden_states_in` input of shape `[sequence_length, hidden_size]`,
/// a prior segment's output. A segment that ends at the last layer ends with
/// the final norm and the LM head; an earlier one ends with the raw post-layer
/// hidden state for a following segment to consume.
///
/// A prefill graph does not append to the KV history and a decode graph does.
///
/// # Errors
///
/// When the checkpoint's activation is not `silu`, or is unknown, or declares
/// YaRN frequency scaling: before any call to `sink`, so a refused checkpoint
/// leaves no half-built graph in the Runtime.
pub fn build_graph<S: GraphSink>(
    sink: &mut S,
    config: &Config,
    segment: Segment,
) -> Result<String, String> {
    match config.hidden_activation.as_deref() {
        Some("silu") => {}
        Some(other) => {
            return Err(format!(
                "the Llama Component does not support activation {other:?}"
            ));
        }
        None => return Err("the Llama Component requires a known checkpoint activation".into()),
    }
    // Llama 3 checkpoints rescale RoPE frequencies per pair; the Runtime only
    // forwards what this Component declares on its `rope` nodes.
    let rope_scaling = match &config.rope_scaling {
        None => None,
        Some(RopeScaling::Banded(banded)) => Some(banded),
        Some(RopeScaling::Yarn) => {
            return Err("the Llama Component does not apply YaRN".into());
        }
    };

    let sequence_length = segment.sequence_length;
    sink.begin_graph(segment.phase, sequence_length, segment.position_offset);
    let mut b = Builder {
        sink,
        config,
        rope_scaling,
        sequence: sequence_length,
        append: segment.phase == Phase::Decode,
    };

    let mut hidden = if segment.start_layer == 0 {
        let token_ids = b.sink.declare_input("token_ids", &[sequence_length]);
        let token_embedding = b.sink.weight_edge("token_embedding");
        b.node(
            "embedding".into(),
            "embedding",
            "Tensor",
            vec![],
            vec![token_ids, token_embedding],
            config.hidden_size,
            None,
        )
    } else {
        b.sink
            .declare_input("hidden_states_in", &[sequence_length, config.hidden_size])
    };

    for layer in u64::from(segment.start_layer)..u64::from(segment.end_layer) {
        hidden = b.layer(layer, &hidden);
    }

    if segment.end_layer != config.num_hidden_layers {
        return Ok(b.sink.finish_graph(&hidden));
    }

    let hidden_final = b.norm(
        "final_norm".into(),
        "final_norm",
        hidden,
        config.hidden_size,
    );

    let lm_head_weight = b.sink.weight_edge("lm_head");
    if config.tie_word_embeddings {
        // `alias-weight-edge` requires its target edge to exist in this
        // graph; a segment that does not start at layer 0 never looked up the
        // embedding, so register it first (idempotent otherwise).
        b.sink.weight_edge("token_embedding");
        b.sink.alias_weight_edge(&lm_head_weight, "token_embedding");
    }
    let logits = b.node(
        "lm_head".into(),
        "matmul",
        "LinearAlgebra",
        vec![],
        vec![hidden_final, lm_head_weight],
        config.vocabulary_size,
        None,
    );

    Ok(b.sink.finish_graph(&logits))
}
