//! Golden graphs: every call the builder makes, recorded as text and compared
//! with the files in `tests/golden/`.
//!
//! These files were recorded from the builder before it moved onto
//! `magnetar-component-sdk` and `magnetar-component-api`: the unmodified graph
//! code of `src/lib.rs` at `10a68c3`, run on the host against a recording
//! stand-in for the `graph-builder` and `model-config` imports its generated
//! bindings provide. They pin the graph a Model Instance receives: a change that
//! alters one node, attribute, input order, shape or KV involvement fails here,
//! whoever made it. Regenerate them deliberately with
//! `UPDATE_GOLDEN=1 cargo test --test golden_graphs`.
//!
//! The test uses only the crate's public `plan` API, so it holds across the
//! change of the types behind it.

use magnetar_component_Llama::plan::{
    BandedRopeScaling, Config, GraphSink, KvRole, Node, Phase, RopeScaling, Segment, Value,
    build_graph,
};
use std::{fmt::Write as _, path::PathBuf};

/// Records every call, one line each, in call order.
#[derive(Default)]
struct TextSink {
    text: String,
}

impl TextSink {
    fn line(&mut self, line: std::fmt::Arguments<'_>) {
        writeln!(self.text, "{line}").unwrap();
    }
}

impl GraphSink for TextSink {
    fn begin_graph(&mut self, phase: Phase, sequence_length: u64, position_offset: u64) {
        self.line(format_args!(
            "begin {phase:?} sequence={sequence_length} offset={position_offset}"
        ));
    }
    fn declare_input(&mut self, name: &str, shape: &[u64]) -> String {
        self.line(format_args!("input {name} {shape:?}"));
        format!("input:{name}")
    }
    fn weight_edge(&mut self, name: &str) -> String {
        self.line(format_args!("weight {name}"));
        format!("weight:{name}")
    }
    fn alias_weight_edge(&mut self, edge: &str, target: &str) {
        self.line(format_args!("alias {edge} -> {target}"));
    }
    fn kv_resource(&mut self, layer: u64, role: KvRole) -> String {
        self.line(format_args!("kv {layer} {role:?}"));
        format!("kv:{layer}:{role:?}")
    }
    fn add_node(&mut self, node: Node) -> String {
        let mut line = format!(
            "node {} op={} family={} inputs={:?} shape={:?}",
            node.id, node.operator, node.family, node.inputs, node.shape
        );
        for attr in &node.attrs {
            let value = match &attr.value {
                Value::Bool(v) => format!("bool:{v}"),
                Value::Int(v) => format!("int:{v}"),
                Value::Float(v) => format!("float:{v:?}"),
                Value::Text(v) => format!("text:{v}"),
            };
            write!(line, " {}={value}", attr.name).unwrap();
        }
        if let Some(kv) = &node.kv {
            write!(
                line,
                " kv=({}, {:?}, append={})",
                kv.resource, kv.role, kv.append
            )
            .unwrap();
        }
        self.line(format_args!("{line}"));
        format!("out:{}", node.id)
    }
    fn finish_graph(&mut self, output: &str) -> String {
        self.line(format_args!("finish {output}"));
        output.into()
    }
}

fn tiny(layers: u32, attention_bias: bool, tie_word_embeddings: bool) -> Config {
    Config {
        hidden_size: 8,
        intermediate_size: 12,
        num_hidden_layers: layers,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dimension: 4,
        vocabulary_size: 17,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        tie_word_embeddings,
        attention_bias,
        hidden_activation: Some("silu".into()),
        rope_scaling: None,
    }
}

/// `meta-llama/Llama-3.2-1B-Instruct`, as its `config.json` publishes it
/// (`tests/fixtures/llama-3.2-1b.config.json`).
fn llama_3_2_1b() -> Config {
    let raw: serde_json::Value =
        serde_json::from_slice(include_bytes!("fixtures/llama-3.2-1b.config.json")).unwrap();
    let scaling = &raw["rope_scaling"];
    assert_eq!(raw["model_type"], "llama");
    assert_eq!(scaling["rope_type"], "llama3");
    Config {
        hidden_size: raw["hidden_size"].as_u64().unwrap(),
        intermediate_size: raw["intermediate_size"].as_u64().unwrap(),
        num_hidden_layers: raw["num_hidden_layers"].as_u64().unwrap() as u32,
        num_attention_heads: raw["num_attention_heads"].as_u64().unwrap() as u32,
        num_key_value_heads: raw["num_key_value_heads"].as_u64().unwrap() as u32,
        head_dimension: raw["head_dim"].as_u64().unwrap(),
        vocabulary_size: raw["vocab_size"].as_u64().unwrap(),
        rms_norm_eps: raw["rms_norm_eps"].as_f64().unwrap(),
        rope_theta: raw["rope_theta"].as_f64().unwrap(),
        tie_word_embeddings: raw["tie_word_embeddings"].as_bool().unwrap(),
        attention_bias: raw["attention_bias"].as_bool().unwrap(),
        hidden_activation: raw["hidden_act"].as_str().map(str::to_string),
        rope_scaling: Some(RopeScaling::Banded(BandedRopeScaling {
            factor: scaling["factor"].as_f64().unwrap(),
            low_freq_factor: scaling["low_freq_factor"].as_f64().unwrap(),
            high_freq_factor: scaling["high_freq_factor"].as_f64().unwrap(),
            original_max_position_embeddings: scaling["original_max_position_embeddings"]
                .as_u64()
                .unwrap(),
        })),
    }
}

fn segment(phase: Phase, sequence_length: u64, offset: u64, start: u32, end: u32) -> Segment {
    Segment {
        phase,
        sequence_length,
        position_offset: offset,
        start_layer: start,
        end_layer: end,
    }
}

fn check(name: &str, config: &Config, segment: Segment) {
    let mut sink = TextSink::default();
    build_graph(&mut sink, config, segment).unwrap();
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(format!("{name}.txt"));
    if std::env::var_os("UPDATE_GOLDEN").is_some() {
        std::fs::write(&path, &sink.text).unwrap();
        return;
    }
    let expected = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("golden graph {}: {error}", path.display()));
    assert_eq!(
        sink.text, expected,
        "graph '{name}' differs from tests/golden/{name}.txt"
    );
}

#[test]
fn prefill_one_layer_without_bias() {
    check(
        "prefill_tiny_no_bias",
        &tiny(1, false, false),
        segment(Phase::Prefill, 3, 0, 0, 1),
    );
}

#[test]
fn decode_one_layer_without_bias() {
    check(
        "decode_tiny_no_bias",
        &tiny(1, false, false),
        segment(Phase::Decode, 1, 11, 0, 1),
    );
}

#[test]
fn prefill_one_layer_with_attention_bias() {
    check(
        "prefill_tiny_attention_bias",
        &tiny(1, true, false),
        segment(Phase::Prefill, 3, 0, 0, 1),
    );
}

#[test]
fn decode_one_layer_with_attention_bias() {
    check(
        "decode_tiny_attention_bias",
        &tiny(1, true, false),
        segment(Phase::Decode, 1, 11, 0, 1),
    );
}

#[test]
fn prefill_the_whole_two_layer_graph_with_tied_embeddings() {
    check(
        "prefill_tiny_two_layers_tied",
        &tiny(2, false, true),
        segment(Phase::Prefill, 3, 0, 0, 2),
    );
}

#[test]
fn decode_middle_segment() {
    check(
        "decode_tiny_middle_segment",
        &tiny(3, false, true),
        segment(Phase::Decode, 1, 11, 1, 2),
    );
}

#[test]
fn decode_last_segment_aliases_the_embedding() {
    check(
        "decode_tiny_last_segment_tied",
        &tiny(3, false, true),
        segment(Phase::Decode, 1, 11, 2, 3),
    );
}

#[test]
fn llama_3_2_1b_first_two_layers() {
    check(
        "llama_3_2_1b_prefill_layers_0_2",
        &llama_3_2_1b(),
        segment(Phase::Prefill, 5, 0, 0, 2),
    );
}

#[test]
fn llama_3_2_1b_last_two_layers() {
    check(
        "llama_3_2_1b_decode_layers_14_16",
        &llama_3_2_1b(),
        segment(Phase::Decode, 1, 4096, 14, 16),
    );
}
