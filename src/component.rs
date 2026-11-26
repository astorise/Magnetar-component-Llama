//! Adapt the host-independent Llama graph builder to Magnetar's WIT capability.

use crate::plan::{self, BandedRopeScaling, Config, Phase, RopeScaling, Segment};
use magnetar_component_api::v1_8::{
    Guest, export,
    magnetar::model_component_graph::{graph_builder as gb, model_config as mc},
};

struct Host;

// The graph-builder adapter is the SDK's, written once for every Component.
magnetar_component_sdk::wit_graph_sink!(Host, gb);

/// The architecture the Runtime bound to the active Model Instance.
fn config() -> Config {
    let a = mc::model_config();
    Config {
        hidden_size: a.hidden_size,
        intermediate_size: a.intermediate_size,
        num_hidden_layers: a.num_hidden_layers,
        num_attention_heads: a.num_attention_heads,
        num_key_value_heads: a.num_key_value_heads,
        head_dimension: a.head_dimension,
        vocabulary_size: a.vocabulary_size,
        rms_norm_eps: a.rms_norm_eps,
        rope_theta: a.rope_theta,
        tie_word_embeddings: a.tie_word_embeddings,
        attention_bias: a.attention_bias,
        hidden_activation: mc::hidden_activation(),
        rope_scaling: mc::rope_scaling().map(|scaling| match scaling {
            mc::RopeFrequencyScaling::Banded(banded) => RopeScaling::Banded(BandedRopeScaling {
                factor: banded.factor,
                low_freq_factor: banded.low_freq_factor,
                high_freq_factor: banded.high_freq_factor,
                original_max_position_embeddings: banded.original_max_position_embeddings,
            }),
            mc::RopeFrequencyScaling::Yarn(_) => RopeScaling::Yarn,
        }),
    }
}

fn build(
    phase: Phase,
    sequence_length: u64,
    position_offset: u64,
    range: Option<(u32, u32)>,
) -> String {
    let config = config();
    let (start_layer, end_layer) = range.unwrap_or((0, config.num_hidden_layers));
    plan::build_graph(
        &mut Host,
        &config,
        Segment {
            phase,
            sequence_length,
            position_offset,
            start_layer,
            end_layer,
        },
    )
    .unwrap_or_else(|reason| panic!("{reason}"))
}

struct Component;

impl Guest for Component {
    fn build_prefill_graph(sequence_length: u64) -> String {
        build(Phase::Prefill, sequence_length, 0, None)
    }

    fn build_decode_graph(cached_token_count: u64) -> String {
        build(Phase::Decode, 1, cached_token_count, None)
    }

    fn build_prefill_graph_segment(
        sequence_length: u64,
        start_layer: u32,
        end_layer: u32,
    ) -> String {
        build(
            Phase::Prefill,
            sequence_length,
            0,
            Some((start_layer, end_layer)),
        )
    }

    fn build_decode_graph_segment(
        cached_token_count: u64,
        start_layer: u32,
        end_layer: u32,
    ) -> String {
        build(
            Phase::Decode,
            1,
            cached_token_count,
            Some((start_layer, end_layer)),
        )
    }
}

export!(Component);
