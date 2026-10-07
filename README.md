# ambient-engine

An optional reference implementation of the inference endpoint that
[ambient](https://github.com/openduo/ambient) talks to.

Ambient defines a protocol and its channels. It works with any OpenAI-compatible
chat-completions endpoint that supports function tools. This repository is one such endpoint: a
patched `llama-server` for the ternary `Ternary-Bonsai-2-27B` model on a single NVIDIA card with
compute capability 8.6 or 8.9 (see Supported GPUs). You do not need it to run ambient. A hosted
API or another server works the same way.

## Contents

| path | what it is |
| --- | --- |
| `UPSTREAM` | upstream repository and the pinned commit |
| `patches/ambient-engine.patch` | one patch against that commit |
| `build.sh` | fetches the commit, applies the patch, builds `llama-server` with CUDA |
| `launch/` | recommended `llama-server` arguments, with a note per flag |
| `LICENSE`, `NOTICE` | licensing, see below |

No upstream source is stored here.

## Upstream

[`PrismML-Eng/llama.cpp`](https://github.com/PrismML-Eng/llama.cpp) at
`7dffb158de30ebb8ef9d64f33c6b0b2d7c1e6313`. That fork adds the `PTQ1_0` and `PQ2_0` formats the
model uses; `ggml-org/llama.cpp` cannot load them.

## What the patch adds

Model-specific CUDA kernels and server changes. Fused and specialized kernels run only when their
conditions hold; otherwise the separate operations run. For `PTQ1_0` weights the mat-vec activation
layout itself changes: on CUDA, the activation quantizer and every `PTQ1_0` mat-vec kernel, the
generic one included, use the planar-transposed layout, so these paths have no unmodified upstream
fallback.

- `PTQ1_0` mat-vec kernels for one to four token columns, on a planar-transposed Q8_1 activation
  layout. Used for plain 2D matmuls whose shared-memory reduction fits. The accumulation order
  does not depend on the column count.
- On Ada (sm_89), fused SwiGLU, FWHT and Q8_1 quantization for folded projections that pass the
  type and shape checks. Other cards and shapes use the separate operations.
- On Ada (sm_89), a gated delta net kernel that updates four state columns per warp, for longer
  batches.
- Recurrent-state snapshots taken inside the prompt decode and used as prompt-cache checkpoints.
  This needs a model whose GGUF architecture is `qwen35` or `qwen35moe`, and speculative decoding
  (for example `--spec-type draft-mtp`) with `--spec-draft-n-max` from 1 to 3; the recurrent
  rollback planes hold the snapshots. Otherwise checkpoints are saved as upstream does.
- A bounded pool of device-resident checkpoints. When it is full, checkpoints use host memory.
- The MTP drafter receives every prompt row after a cache restore. With one active sequence, an
  MTP drafter whose GGUF architecture is `qwen35`, `--spec-draft-n-max 3` and `--spec-draft-p-min`
  at 0 (the default), the three draft steps run as one greedy chain inside one graph. In that chain
  the draft head uses only the base Qwen vocabulary rows, unless a LoRA adapter is loaded or the
  output head is folded. The target model verifies each draft token.
- Chat prompts are tokenized one message at a time when message delimiters start with a special
  token that does not strip whitespace to its left and the tokenizer appends no EOS or SEP token,
  with a bounded cache of unchanged messages.
- OpenAI-format responses skip detokenizing the prompt.

## Supported GPUs

| architecture | cards | status |
| --- | --- | --- |
| sm_89 (Ada) | for example RTX 4090 | supported, tested |
| sm_86 (Ampere) | for example RTX 3090 | builds; not tested on Ampere hardware |
| compute capability 9.0 and newer | | not supported; use an engine built for large-memory GPUs |

The recommended launch arguments also need enough free device memory; see
[Device memory](launch/README.md#device-memory).

## Weights

Download from the publishers. Check the sha256 after download.

| role | repository | file | sha256 |
| --- | --- | --- | --- |
| target | [`prism-ml/Ternary-Bonsai-2-27B-gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) | `Ternary-Bonsai-2-27B-PTQ1_0.gguf` | `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3` |
| drafter | [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` | `50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e` |

## Build

```bash
./build.sh                 # CUDA architectures 86 (Ampere) and 89 (Ada)
```

Requirements: bash, git, CMake with a build tool (Make or Ninja), a CUDA toolkit with `nvcc` that
targets compute capability 8.9, and a C++ compiler. Compiling needs no GPU. `--arch 89` or
`--arch 86` builds for one architecture only; `build.sh` refuses any other architecture.

Tested toolchain (x86-64 Linux, Ubuntu 22.04): CUDA 12.4 (`nvcc` 12.4.131), GCC 11.4.0, CMake 3.29.0
with GNU Make 4.3. Other versions are untested.

Each run creates a new source checkout (`work/llama.cpp`) and a new build directory
(`work/build`) and stops if either exists. To build again, for example after a failed run or with
other options, remove `work/` or pass new `--src` and `--build` paths. `./build.sh --help` lists
the options.

CPU code is built for the build machine's CPU (`GGML_NATIVE`, upstream default), so the binaries
may not run on a CPU with fewer features. Binaries, shared libraries, `LICENSE` and `NOTICE` land
in `work/build/bin`.

## Run

See [`launch/README.md`](launch/README.md).

## Connect a client

`llama-server` serves an OpenAI-compatible API. Point an OpenAI-compatible client at
`http://<host>:<port>/v1/chat/completions` and use the model id from `GET /v1/models` (the
`--alias` value when set). If you start the server with `--api-key`, the client sends the same key
as a Bearer token. To use this server with ambient, follow ambient's documentation:
<https://github.com/openduo/ambient>.

## License

The patch, scripts and documentation in this repository are licensed under FSL-1.1-Apache-2.0
(`LICENSE`). The upstream code the patch modifies remains under its MIT license; see `NOTICE`.

FSL-1.1-Apache-2.0 is a source-available license, not an OSI-approved open-source license. Each
version converts to Apache-2.0 two years after it is made available; see Release below.

## Release

- Published: 2026-10-07
- Apache-2.0 from: 2028-10-07
