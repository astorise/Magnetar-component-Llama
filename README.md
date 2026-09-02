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

**Empty template.** This crate is currently a bare `cargo new --lib`
scaffold: no Llama graph logic, no WIT bindings, no Component Model exports
exist here yet. Unlike [`components/qwen`](https://github.com/astorise/Magnetar-component-Qwen),
Magnetar's Runtime has no interim, Rust-native Llama implementation either
-- Llama support does not exist anywhere in the Magnetar workspace today.
This submodule is a placeholder for future work, not yet scheduled against
any OpenSpec change.

## Governing contract

None dedicated yet. Once real work starts here, it will implement the same
`model-component-graph-contract` WIT interface `components/qwen` targets
(see that Component's README), since both are decoder-only architectures
sharing the same Runtime-side graph-production contract -- Llama-specific
requirements (attention variant, rotary embedding parameters, tokenizer
family, etc.) would be scoped in a dedicated `llama-model-component`
capability spec in the main [Magnetar](https://github.com/astorise/Magnetar)
repository's `openspec/specs/` when that work begins.

## Relationship to magnetar-runtime

This Component would be loaded and validated by the Runtime's Component
Engine (Wasmtime-backed on native targets, browser-hosted on `wasm32`) and
never run with direct hardware, filesystem, or network access. It is
pinned into the main [Magnetar](https://github.com/astorise/Magnetar)
repository as a git submodule at `components/llama`.
