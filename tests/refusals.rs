//! A checkpoint this Component cannot serve is refused before any call to the
//! graph-builder: the Runtime is left with no half-built graph.

use magnetar_component_Llama::plan::{Config, Phase, RopeScaling, Segment, build_graph};
use magnetar_component_sdk::testing::RecordingSink;

fn config() -> Config {
    Config {
        hidden_size: 8,
        intermediate_size: 12,
        num_hidden_layers: 1,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dimension: 4,
        vocabulary_size: 17,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        tie_word_embeddings: false,
        attention_bias: false,
        hidden_activation: Some("silu".into()),
        rope_scaling: None,
    }
}

fn segment() -> Segment {
    Segment {
        phase: Phase::Prefill,
        sequence_length: 3,
        position_offset: 0,
        start_layer: 0,
        end_layer: 1,
    }
}

fn refused(config: &Config) -> String {
    let mut sink = RecordingSink::new();
    let reason = build_graph(&mut sink, config, segment()).unwrap_err();
    assert!(sink.began.is_empty(), "begin_graph was called");
    assert!(sink.nodes.is_empty(), "a node was added");
    assert!(sink.weights.is_empty(), "a weight edge was asked for");
    reason
}

#[test]
fn an_activation_other_than_silu_is_refused() {
    let mut config = config();
    config.hidden_activation = Some("gelu".into());
    assert_eq!(
        refused(&config),
        "the Llama Component does not support activation \"gelu\""
    );
}

#[test]
fn an_unknown_activation_is_refused() {
    let mut config = config();
    config.hidden_activation = None;
    assert_eq!(
        refused(&config),
        "the Llama Component requires a known checkpoint activation"
    );
}

#[test]
fn yarn_frequency_scaling_is_refused() {
    let mut config = config();
    config.rope_scaling = Some(RopeScaling::Yarn);
    assert_eq!(refused(&config), "the Llama Component does not apply YaRN");
}

#[test]
fn a_supported_checkpoint_builds() {
    let mut sink = RecordingSink::new();
    build_graph(&mut sink, &config(), segment()).unwrap();
    assert_eq!(sink.began.len(), 1);
    assert!(sink.has("lm_head"));
}
