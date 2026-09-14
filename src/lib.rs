//! Production, configurable Llama Model Component
//! (`magnetar:model-component-graph@1.2.0`).
//!
//! Builds prefill and decode Execution Graphs for whatever Llama-compatible
//! architecture the active Model Instance was loaded with, via the
//! Runtime-owned `graph-builder` and `model-config` Capabilities -- this
//! Component never touches `magnetar-runtime`'s Rust types at all, only the
//! WIT contract.
//!
//! # Relationship to `components/qwen`
//!
//! This Component's graph-building logic is structurally identical to
//! [`components/qwen`](https://github.com/astorise/Magnetar-component-Qwen)'s
//! own, deliberately: Llama (1/2/3) and Qwen2/2.5 are both real,
//! well-documented instances of the same pre-norm, RoPE, grouped-query-
//! attention, SwiGLU-gated decoder block -- Qwen2's own architecture is
//! Llama's with an added QKV bias term, not a different block shape. Every
//! axis that actually differs between the two families (whether `self_attn.
//! {q,k,v}_proj` carry an additive bias, whether the input/output embedding
//! is tied, head/layer/hidden dimensions, RoPE base, GQA group size) is
//! already expressed generically by `model-config`'s `architecture-config`
//! record -- there is no Llama-specific graph shape to add beyond what that
//! contract already models. A real Llama checkpoint's own bias-free
//! `self_attn.{q,k,v}_proj` and (for most sizes) untied embeddings drive
//! this at configuration time, the same way a real Qwen2 checkpoint's own
//! bias-bearing projections do -- neither family is hardcoded here.
//!
//! Architecture dimensions (hidden size, layer count, head counts, head
//! dimension, vocabulary size, RMSNorm epsilon, RoPE parameters, tied
//! embeddings, attention bias) are queried from `model-config` at the start
//! of each `build-prefill-graph`/`build-decode-graph` call, not compiled-in
//! constants: the same Component binary produces a configuration-correct
//! graph for any compatible Llama checkpoint the Runtime has bound
//! configuration for.

wit_bindgen::generate!({
    path: "wit/model-component-graph.wit",
    world: "model-component-graph-producer",
});

use magnetar::model_component_graph::graph_builder::{
    Attribute, AttributeValue, GraphPhase, KvCacheRole, KvInvolvement, TensorShape, add_node,
    alias_weight_edge, begin_graph, declare_input, finish_graph, kv_resource, weight_edge,
};
use magnetar::model_component_graph::model_config::{self, ArchitectureConfig};

const ROPE_POSITION_MODE: &str = "sequential";

fn shape(dimensions: Vec<u64>) -> TensorShape {
    TensorShape { dimensions }
}

fn attr(name: &str, value: AttributeValue) -> Attribute {
    Attribute {
        name: name.to_string(),
        value,
    }
}

/// Maps a per-layer weight to the canonical Model Artifact tensor name
/// `weight_edge` is called with. Identical shape to `components/qwen`'s own
/// mapping -- both families' real Hugging Face checkpoints share this exact
/// `self_attn`/`mlp` naming convention (Qwen2's checkpoint format was
/// itself modeled on Llama's), so the same canonical names resolve real
/// weights for either family without this Component carrying any
/// Llama-specific suffix table.
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

/// Builds one pre-norm decoder layer: RMSNorm, QKV matmul, RoPE, grouped-
/// query attention, output projection, residual-add, RMSNorm, SwiGLU-gated
/// MLP, residual-add -- through `graph-builder` calls instead of direct
/// `ExecutionGraph` construction. Every dimension, and whether a bias term
/// exists at all, comes from `config`, queried once per
/// `build-prefill-graph`/`build-decode-graph` call -- not a compiled-in
/// constant. A real Llama checkpoint's own `architecture-config` (no QKV
/// bias) produces a graph with no bias-add nodes at all; a real Qwen2
/// checkpoint's own `architecture-config` (QKV bias present) produces one
/// with them -- the same code path serves both, driven only by what was
/// actually ingested.
fn build_layer(
    config: &ArchitectureConfig,
    layer: u64,
    residual_in: &str,
    sequence_length: u64,
    kv_append: bool,
) -> String {
    let prefix = format!("layer{layer}");
    let hidden_size = config.hidden_size;
    let intermediate_size = config.intermediate_size;
    let attention_head_count = config.num_attention_heads as u64;
    let kv_head_count = config.num_key_value_heads as u64;
    let head_dimension = config.head_dimension;
    let q_dim = attention_head_count * head_dimension;
    let kv_dim = kv_head_count * head_dimension;

    let input_norm_weight = weight_edge(&weight_tensor_name(layer, "input_norm"));
    let normed = add_node(
        &format!("{prefix}.input_norm"),
        "rmsnorm",
        "Normalization",
        &[attr(
            "epsilon",
            AttributeValue::FloatValue(config.rms_norm_eps),
        )],
        &[residual_in.to_string(), input_norm_weight],
        &shape(vec![sequence_length, hidden_size]),
        None,
    );

    // A real Llama checkpoint declares no additive bias term for
    // self_attn.{q,k,v}_proj (unlike Qwen2/2.5's own architectural
    // default) -- `config.attention_bias` reflects whatever the ingested
    // checkpoint actually declared, so this branch is dead code for a real
    // Llama Model Artifact today, not a Llama-specific special case: the
    // same conditional this Component shares with `components/qwen`
    // handles whichever family's checkpoint provided the bias tensors (or
    // did not). When present, bias is added to the raw projection output
    // *before* RoPE/KV caching, so a bias-add node's output (not the
    // matmul's) becomes `q`/`k`/`v` below.
    let q_weight = weight_edge(&weight_tensor_name(layer, "q_proj"));
    let q = add_node(
        &format!("{prefix}.q_proj"),
        "matmul",
        "LinearAlgebra",
        &[],
        &[normed.clone(), q_weight],
        &shape(vec![sequence_length, q_dim]),
        None,
    );
    let q = if config.attention_bias {
        let q_bias = weight_edge(&weight_tensor_name(layer, "q_bias"));
        add_node(
            &format!("{prefix}.q_bias"),
            "add",
            "Tensor",
            &[],
            &[q, q_bias],
            &shape(vec![sequence_length, q_dim]),
            None,
        )
    } else {
        q
    };
    let k_weight = weight_edge(&weight_tensor_name(layer, "k_proj"));
    let k = add_node(
        &format!("{prefix}.k_proj"),
        "matmul",
        "LinearAlgebra",
        &[],
        &[normed.clone(), k_weight],
        &shape(vec![sequence_length, kv_dim]),
        None,
    );
    let k = if config.attention_bias {
        let k_bias = weight_edge(&weight_tensor_name(layer, "k_bias"));
        add_node(
            &format!("{prefix}.k_bias"),
            "add",
            "Tensor",
            &[],
            &[k, k_bias],
            &shape(vec![sequence_length, kv_dim]),
            None,
        )
    } else {
        k
    };
    let v_weight = weight_edge(&weight_tensor_name(layer, "v_proj"));
    let v_kv = kv_resource(layer, KvCacheRole::V);
    let v_kv_involvement = KvInvolvement {
        resource_id: v_kv,
        role: KvCacheRole::V,
        append: kv_append,
    };
    let v = add_node(
        &format!("{prefix}.v_proj"),
        "matmul",
        "LinearAlgebra",
        &[],
        &[normed, v_weight],
        &shape(vec![sequence_length, kv_dim]),
        if config.attention_bias {
            None
        } else {
            Some(&v_kv_involvement)
        },
    );
    let v = if config.attention_bias {
        let v_bias = weight_edge(&weight_tensor_name(layer, "v_bias"));
        add_node(
            &format!("{prefix}.v_bias"),
            "add",
            "Tensor",
            &[],
            &[v, v_bias],
            &shape(vec![sequence_length, kv_dim]),
            Some(&v_kv_involvement),
        )
    } else {
        v
    };

    let rope_attrs = |offset: i64, head_count: u64| {
        vec![
            attr("base", AttributeValue::FloatValue(config.rope_theta)),
            attr("dimension", AttributeValue::Integer(head_dimension as i64)),
            attr(
                "position_mode",
                AttributeValue::Text(ROPE_POSITION_MODE.to_string()),
            ),
            attr("position_offset", AttributeValue::Integer(offset)),
            attr("head_count", AttributeValue::Integer(head_count as i64)),
        ]
    };
    let q_rope = add_node(
        &format!("{prefix}.rope_q"),
        "rope",
        "PositionEncoding",
        &rope_attrs(0, attention_head_count),
        &[q],
        &shape(vec![sequence_length, q_dim]),
        None,
    );
    let k_kv = kv_resource(layer, KvCacheRole::K);
    let k_rope = add_node(
        &format!("{prefix}.rope_k"),
        "rope",
        "PositionEncoding",
        &rope_attrs(0, kv_head_count),
        &[k],
        &shape(vec![sequence_length, kv_dim]),
        Some(&KvInvolvement {
            resource_id: k_kv,
            role: KvCacheRole::K,
            append: kv_append,
        }),
    );

    let attn_out = add_node(
        &format!("{prefix}.attention"),
        "attention",
        "Attention",
        &[
            attr("causal", AttributeValue::Boolean(true)),
            attr(
                "head_count",
                AttributeValue::Integer(attention_head_count as i64),
            ),
            attr(
                "kv_head_count",
                AttributeValue::Integer(kv_head_count as i64),
            ),
            attr(
                "head_dimension",
                AttributeValue::Integer(head_dimension as i64),
            ),
            attr(
                "attention_mask_kind",
                AttributeValue::Text("causal".to_string()),
            ),
        ],
        &[q_rope, k_rope, v],
        &shape(vec![sequence_length, q_dim]),
        None,
    );

    let o_weight = weight_edge(&weight_tensor_name(layer, "o_proj"));
    let attn_proj = add_node(
        &format!("{prefix}.o_proj"),
        "matmul",
        "LinearAlgebra",
        &[],
        &[attn_out, o_weight],
        &shape(vec![sequence_length, hidden_size]),
        None,
    );

    let post_attn = add_node(
        &format!("{prefix}.residual1"),
        "residual-add",
        "Tensor",
        &[],
        &[residual_in.to_string(), attn_proj],
        &shape(vec![sequence_length, hidden_size]),
        None,
    );

    let post_attn_norm_weight = weight_edge(&weight_tensor_name(layer, "post_attn_norm"));
    let mlp_normed = add_node(
        &format!("{prefix}.post_attn_norm"),
        "rmsnorm",
        "Normalization",
        &[attr(
            "epsilon",
            AttributeValue::FloatValue(config.rms_norm_eps),
        )],
        &[post_attn.clone(), post_attn_norm_weight],
        &shape(vec![sequence_length, hidden_size]),
        None,
    );

    let gate_weight = weight_edge(&weight_tensor_name(layer, "gate_proj"));
    let gate = add_node(
        &format!("{prefix}.gate_proj"),
        "matmul",
        "LinearAlgebra",
        &[],
        &[mlp_normed.clone(), gate_weight],
        &shape(vec![sequence_length, intermediate_size]),
        None,
    );
    let up_weight = weight_edge(&weight_tensor_name(layer, "up_proj"));
    let up = add_node(
        &format!("{prefix}.up_proj"),
        "matmul",
        "LinearAlgebra",
        &[],
        &[mlp_normed, up_weight],
        &shape(vec![sequence_length, intermediate_size]),
        None,
    );
    let activated = add_node(
        &format!("{prefix}.silu"),
        "silu",
        "Activation",
        &[],
        &[gate],
        &shape(vec![sequence_length, intermediate_size]),
        None,
    );
    let mlp_hidden = add_node(
        &format!("{prefix}.mul"),
        "mul",
        "Tensor",
        &[],
        &[activated, up],
        &shape(vec![sequence_length, intermediate_size]),
        None,
    );
    let down_weight = weight_edge(&weight_tensor_name(layer, "down_proj"));
    let mlp_out = add_node(
        &format!("{prefix}.down_proj"),
        "matmul",
        "LinearAlgebra",
        &[],
        &[mlp_hidden, down_weight],
        &shape(vec![sequence_length, hidden_size]),
        None,
    );

    add_node(
        &format!("{prefix}.residual2"),
        "residual-add",
        "Tensor",
        &[],
        &[post_attn, mlp_out],
        &shape(vec![sequence_length, hidden_size]),
        None,
    )
}

fn build_graph(
    phase: GraphPhase,
    sequence_length: u64,
    position_offset: u64,
    kv_append: bool,
) -> String {
    let config = model_config::model_config();
    begin_graph(phase, sequence_length, position_offset);

    let token_ids = declare_input("token_ids", &shape(vec![sequence_length]));
    let token_embedding = weight_edge("token_embedding");
    let mut hidden = add_node(
        "embedding",
        "embedding",
        "Tensor",
        &[],
        &[token_ids, token_embedding],
        &shape(vec![sequence_length, config.hidden_size]),
        None,
    );

    for layer in 0..config.num_hidden_layers as u64 {
        hidden = build_layer(&config, layer, &hidden, sequence_length, kv_append);
    }

    let final_norm_weight = weight_edge("final_norm");
    let hidden_final = add_node(
        "final_norm",
        "rmsnorm",
        "Normalization",
        &[attr(
            "epsilon",
            AttributeValue::FloatValue(config.rms_norm_eps),
        )],
        &[hidden, final_norm_weight],
        &shape(vec![sequence_length, config.hidden_size]),
        None,
    );

    let lm_head_weight = weight_edge("lm_head");
    if config.tie_word_embeddings {
        alias_weight_edge(&lm_head_weight, "token_embedding");
    }
    let logits = add_node(
        "lm_head",
        "matmul",
        "LinearAlgebra",
        &[],
        &[hidden_final, lm_head_weight],
        &shape(vec![sequence_length, config.vocabulary_size]),
        None,
    );

    finish_graph(&logits)
}

struct Component;

impl Guest for Component {
    fn build_prefill_graph(sequence_length: u64) -> String {
        build_graph(GraphPhase::Prefill, sequence_length, 0, false)
    }

    fn build_decode_graph(cached_token_count: u64) -> String {
        build_graph(GraphPhase::Decode, 1, cached_token_count, true)
    }
}

export!(Component);
