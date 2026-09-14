# magnetar-component-llama

## Purpose

A WebAssembly Component implementing the Llama decoder-only model
architecture family for the [Magnetar](https://github.com/astorise/Magnetar)
local AI Runtime. It supplies prefill/decode graph semantics for
Llama-compatible Model Artifacts through the Runtime's Model Component
contract, without ever selecting a Provider, Device, or receiving native
Runtime handles -- per Magnetar's architectural invariants (Components
request Capabilities, not Providers or Devices).

## Status

**Real, configurable Model Component.** Implements the same
`magnetar:model-component-graph@1.2.0` `model-component-graph-producer`
world [`components/qwen`](https://github.com/astorise/Magnetar-component-Qwen)
implements, with structurally identical graph-building logic: Llama (1/2/3)
and Qwen2/2.5 are both real, well-documented instances of the same
pre-norm, RoPE, grouped-query-attention, SwiGLU-gated decoder block --
Qwen2's own architecture is Llama's with an added QKV bias term, not a
different block shape (see `src/lib.rs`'s own module doc comment for the
full reasoning). Every axis that actually differs between the two
families is already expressed generically by `model-config`'s
`architecture-config` record; there is no separate Llama-specific graph
shape to add beyond what that contract already models. Magnetar's Runtime
still has no interim, Rust-native Llama implementation (unlike Qwen's own
`qwen_model_component.rs` fixture path) -- this WASM Component is Llama's
only graph-production authority.

## Governing contract

Implements `model-component-graph-contract` (`magnetar:model-component-
graph@1.2.0`), the same WIT interface `components/qwen` targets -- see
that Component's README and the main
[Magnetar](https://github.com/astorise/Magnetar) repository's
`openspec/specs/model-component-graph-contract/spec.md` for the full
contract. Llama-specific requirements (real checkpoint compatibility,
tokenizer family, adapter/quantization scope) are tracked in the main
repository's `llama-model-component` capability spec.

## Relationship to magnetar-runtime

This Component would be loaded and validated by the Runtime's Component
Engine (Wasmtime-backed on native targets, browser-hosted on `wasm32`) and
never run with direct hardware, filesystem, or network access. It is
pinned into the main [Magnetar](https://github.com/astorise/Magnetar)
repository as a git submodule at `components/llama`.
