//! Production, configurable Llama Model Component
//! (`magnetar:model-component-graph@1.8.0`).
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
//!
//! # Layout
//!
//! The graph is built by [`plan`], which knows nothing of WebAssembly: it
//! writes to the SDK's `GraphSink` and is tested on the host with golden
//! graphs. The adapter to the contract's `graph-builder` and `model-config`
//! Capabilities is enabled for WebAssembly, or with the `bindings` feature.

pub mod plan;

#[cfg(any(target_arch = "wasm32", feature = "bindings"))]
mod component;
