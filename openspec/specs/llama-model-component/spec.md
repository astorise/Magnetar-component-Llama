# llama-model-component Specification

## Purpose
TBD - created by archiving change add-real-llama-model-component. Update Purpose after archive.
## Requirements
### Requirement: Llama Model Component Baseline

Magnetar SHALL define a Llama-like Model Component baseline for decoder-only inference architecture behavior, implementing the same `model-component-graph-contract` (`magnetar:model-component-graph@1.2.0`) `qwen-model-component` implements.

#### Scenario: Resolve Llama component

- **GIVEN** a Model Artifact declares Llama-compatible architecture metadata
- **WHEN** Model Loading resolves architecture support
- **THEN** Runtime resolves a compatible, registered Llama Model Component

---

### Requirement: Llama Component Is Not Provider

The Llama Model Component SHALL NOT introduce or require a LlamaProvider, select a concrete Provider, or receive a native Runtime handle -- it requests Capabilities (`graph-builder`, `model-config`) only, the same architectural invariant every Model Component observes.

#### Scenario: No Provider selection reachable from the Component

- **GIVEN** the Llama Model Component's WIT world
- **WHEN** its imports are inspected
- **THEN** no Provider- or Device-selecting interface is imported

---

### Requirement: Llama Component Is Configuration-Driven

The Llama Model Component's graph-building logic SHALL derive every architecture dimension (hidden size, layer count, head counts, head dimension, vocabulary size, RMSNorm epsilon, RoPE parameters, tied embeddings, attention bias) from the `model-config` Capability at the start of each `build-prefill-graph`/`build-decode-graph` call, never from a compiled-in constant.

#### Scenario: The same binary serves two differently-shaped configurations

- **GIVEN** the same compiled Llama Component binary
- **WHEN** it builds a graph for two distinct `architecture-config` values (differing in layer count, head counts, or attention bias)
- **THEN** the produced graphs differ accordingly, without recompiling or replacing the Component binary

---

### Requirement: Llama and Qwen Share The Same Generic Decoder Contract

The Llama Model Component's graph-building logic SHALL produce, for an identical `architecture-config`, a graph with the same node count and operator sequence as the Qwen Model Component's own graph for that same configuration -- proving the two independently-compiled Components implement the same generic decoder block the shared contract models, not two incidentally-similar but independently-diverging implementations.

#### Scenario: Byte-identical graphs for an identical configuration

- **GIVEN** the real, independently-compiled Llama and Qwen Model Component binaries, and one `architecture-config` with no QKV attention bias
- **WHEN** each builds a prefill graph against that same configuration
- **THEN** the two graphs have the same node count and the same operator-sequence hash

#### Scenario: A real per-family configuration difference is honored, not hardcoded

- **GIVEN** the real Llama Model Component binary
- **WHEN** it builds a graph against a bias-bearing `architecture-config` (what a real Qwen2 checkpoint declares) instead of a bias-free one (what a real Llama checkpoint declares)
- **THEN** the produced graph differs (additional bias-add nodes), driven entirely by the supplied configuration, not by any hardcoded per-family branch in the Component

