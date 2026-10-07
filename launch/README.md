# Launch arguments

| file | target | status |
| --- | --- | --- |
| `sm_89.args` | one card, sm_89 (Ada) or sm_86 (Ampere) | recommended on sm_89; not tested on Ampere hardware |

The same arguments apply to both architectures. Compute capability 9.0 and newer is not supported.

## Device memory

With `sm_89.args` the server requires 9,704 MiB of device memory after load. The bounded device
checkpoint pool (see Prompt checkpoints) can raise this to at most 12,336 MiB, so the card needs at
least 12,336 MiB free for the server. Both figures include the MTP drafter. A larger `--ctx-size`
needs more.

## Run

Run from the repository root after a default `./build.sh`. With a custom `--build DIR`, set `bin` to
`DIR/bin`. The model paths are sed replacement text: escape any `#`, `&` or `\` in them.

```bash
bin=work/build/bin
sed -e 's#<MODEL_GGUF>#/path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf#' \
    -e 's#<DRAFTER_GGUF>#/path/to/mtp-Qwen3.8-27B-Q4_0.gguf#' launch/sm_89.args \
  | grep -v '^#' | tr '\n' '\0' \
  | LD_LIBRARY_PATH="$bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" xargs -0 "$bin/llama-server"
```

`llama-server` loads `libllama`, `libggml*` and `libmtmd` from its own build directory, so put that
directory on `LD_LIBRARY_PATH` (the command above keeps any existing entries after it). The CUDA
runtime libraries must also be resolvable.

Once the model is loaded, `curl http://127.0.0.1:8080/health` returns HTTP 200; while it is still
loading, it returns 503.

## Flags

| flag | what it does | VRAM effect |
| --- | --- | --- |
| `--model <MODEL_GGUF>` | Target model, the `PTQ1_0` GGUF. | Holds all target weights. |
| `--host 127.0.0.1` | Bind to loopback only. Put a proxy in front for remote access. | None. |
| `--port 8080` | Listen port; 8080 is the upstream default. Any free port works; the client URL must match it. | None. |
| `--ctx-size 32768` | Context length of the single slot. It must hold prompt plus output. | Attention KV cache grows linearly with it and is allocated at load. The recurrent state does not depend on it. |
| `--n-gpu-layers all` | Offload every layer to the GPU. | All weights live in VRAM. |
| `--parallel 1` | One server slot. Requests are served one at a time. | One sequence of KV cache and recurrent state. |
| `--jinja` | Render prompts with the chat template stored in the GGUF. Needed for tool calls. | None. |
| `--reasoning off` | Disable thinking in the chat template (`enable_thinking=false`). | None. |
| `--flash-attn on` | Use the Flash Attention kernels. | Smaller attention scratch buffers. |
| `--cache-type-k f16` | K cache precision. | f16: 2 bytes per element. Quantized types are smaller. |
| `--cache-type-v f16` | V cache precision. | As for K. |
| `--cache-prompt` | Reuse the longest common token prefix with the previous request. Hybrid models restore from saved checkpoints, see below. | Checkpoints, see below. |
| `--no-context-shift` | When the context is full, fail the request instead of discarding old tokens. The recurrent state cannot be shifted. | None. |
| `--metrics` | Enable the Prometheus-format metrics endpoint at `/metrics`. Optional. | None. |
| `--no-warmup` | Skip the empty warm-up run at load. One-time setup then happens during the first request. Optional. | Buffers that warm-up would allocate are allocated on the first request instead. |
| `--alias Ternary-Bonsai-2-27B-PTQ1_0` | Model id reported by `/v1/models` and accepted in requests. | None. |
| `--spec-type draft-mtp` | Speculative decoding with a multi-token-prediction drafter. The target model verifies every drafted token. | Adds the drafter, below. |
| `--spec-draft-model <DRAFTER_GGUF>` | The MTP drafter GGUF. | Holds the drafter weights and the drafter's own KV cache. |
| `--spec-draft-n-max 3` | Draft up to three tokens per step. With 3, one active sequence, an MTP drafter whose GGUF architecture is `qwen35` and `--spec-draft-p-min` at 0 (the default), the three draft steps run as one greedy chain inside one graph; otherwise drafting takes the ordinary path. Values from 1 to 3 also enable the in-decode checkpoints below. | Sizes the draft output buffers and the recurrent rollback state for in-decode checkpoints. Device memory above is for 3. |

## Prompt checkpoints

The model mixes attention layers with recurrent (gated delta net) layers. A recurrent state cannot
be truncated, so prefix reuse restores a saved checkpoint of it. Checkpoints are kept on the device
in a pool of at most `device_state_pool_max` snapshots (`src/llama-context.h`). When the pool is
full, later checkpoints use the host memory path. This bounds the device checkpoint pool only;
other allocations (weights, KV cache, compute buffers, the drafter) follow upstream behavior.

## Environment

The model publisher's card mentions `BONSAI_THINKING=0`. This server does not read that variable;
`--reasoning off` controls thinking.
