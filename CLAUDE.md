# Mini-Pic

This Rust / Burn repo is designed to enable hyper-efficient, high-quality 64x64 image generation using modern diffusion methods.

There are two training pipelines here: `python-train/` (PyTorch, documented as the active one in
its own README) and the native `src/` Burn pipeline (`training::run`, `src/bin/infer.rs`). Check
which one you're actually looking at before diagnosing a training/quality issue - they have
independently drifted (see 2026-09-23 session note below for a case where the Burn side had a
real architecture bug the Python side didn't share).