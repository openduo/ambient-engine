# Launch arguments

| file | target | status |
| --- | --- | --- |
| `sm_89.args` | one card, sm_89 (Ada) or sm_86 (Ampere) | recommended on sm_89; not tested on Ampere hardware |

The same arguments apply to both architectures. Compute capability 9.0 and newer is not supported.

## Device memory

With `sm_89.args` the server requires 9,276 MiB of device memory after load. The bounded device
checkpoint pool and the in-decode snapshot buffers (see Prompt checkpoints) can raise this to at
most 12,170 MiB, so the card needs at least 12,170 MiB free for the server. Both figures include the
MTP drafter. A larger `--ctx-size` or more slots need more; for two slots see Two slots.

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
| `--parallel 1` | One server slot. Requests are served one at a time. For two slots see Two slots. | One sequence of KV cache and recurrent state. |
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
in a pool of at most `--ctx-checkpoints-device N` snapshots (default 16, environment variable
`LLAMA_ARG_CTX_CHECKPOINTS_DEVICE`). The pool belongs to the server's context, so all slots share
it. When the pool is full, later checkpoints use the host memory path. This bounds the device
checkpoint pool only; other allocations (weights, KV cache, compute buffers, the drafter) follow
upstream behavior.

With the recommended arguments each device checkpoint holds about 150 MiB, and the device memory
figures above assume the default of 16. A smaller N lowers the ceiling by that amount per
checkpoint. The cost is time: a checkpoint kept in host memory is copied over PCIe when it is saved
and when it is restored, so prompt reuse from it takes longer than from a device checkpoint. `0`
keeps every checkpoint in host memory.

On CUDA a prompt decode that takes in-decode checkpoints writes them into up to three checkpoint
buffers per slot. These buffers come from free pool entries first; when the pool has none, they are
allocated beyond N and released at the next decode, so at most three per slot exist above the pool.

## Two slots

`sm_89.args` runs one slot, so requests are served one at a time. To serve two requests at once,
change two values in a copy of the file:

| flag | one slot | two slots |
| --- | --- | --- |
| `--parallel` | `1` | `2` |
| `--ctx-size` | `32768` | `65536` |

`--ctx-size` is the total for all slots, so each slot still holds 32,768 tokens. With two slots the
server requires 11,624 MiB of device memory after load and up to 14,978 MiB with the device
checkpoint pool and the snapshot buffers full; both include the MTP drafter. A 24 GB card has room
for this. `--ctx-checkpoints-device` lowers the ceiling (see Prompt checkpoints).

To reduce device memory further, store the target model's attention KV cache as 8-bit values: in
the same copy, set `--cache-type-k q8_0` and `--cache-type-v q8_0`. This roughly halves the KV
cache, the largest allocation that grows with `--ctx-size`. Outputs change slightly compared with
f16, because attention reads quantized keys and values. The drafter's KV cache stays f16
(`--spec-draft-type-k` and `--spec-draft-type-v`, default f16).

## Environment

The model publisher's card mentions `BONSAI_THINKING=0`. This server does not read that variable;
`--reasoning off` controls thinking.
