# native: llama.cpp patch series

Base: upstream commit in `LLAMA_COMMIT`. `build.sh [SRC_DIR]` clones (or reuses) a checkout, applies
`patches/0*.patch` in order, and builds the pipeline tools with CUDA.

| Patch | What | Needed for |
|---|---|---|
| 0001 kld-swap and perplexity-fast | in-place tensor swap/retype + scoring; weight-allocator hook, CUDA graph `node_src_type` | probes, KLD eval |
| 0002 load-mode dio+map | load references larger than VRAM + RAM | reference / capture runs |
| 0003 cuda host buffer chunking | 8 GiB pinned chunks | large host spill |
| 0004 fit underflow fix | `--fit` crash with only full layers left | eval with `--fit on` |
| 0005 cuda reference precision toggles | opt-in f32 paths (`GGML_CUDA_MOE_F32`, `GGML_CUDA_MM_INVARIANT`, ...) | measuring the ggml reference floor |
| 0006 imatrix parallel moe | OpenMP expert accumulation | imatrix stage |
| 0007 debug tensor dumps | `LLAMA_DUMP_TENSORS_DIR`, `LLAMA_NAN_PROBE` | comparing a capture with the ggml graph |
| 0008 convert mxfp4 and writer speed | faster MXFP4 repack / GGUFWriter | building reference GGUFs |
| 0009 cuda gpu quantizer | `ggml_cuda_quantize_weights`, byte-identical to CPU | refine / errtab (via ctypes) |
| 0010 capture library | `libllama-capture`: eval-callback access to every weight matmul's input | `ggufrefine.capture.ggml` |
| 0011 perplexity tokens file | `llama-perplexity --tokens-file`: evaluate / write a KLD base from int32 token ids | eval sets with chat markup (`ggufrefine.corpus`) |
