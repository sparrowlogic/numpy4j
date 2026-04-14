# Optimization Notes — whisper4j (another project)

Lessons learned during performance tuning. Reference before attempting new optimizations.

## ✅ Optimizations That Worked

### F16 GELU Lookup Table (6x speedup on Conv+GELU)
- Pre-compute GELU for all 65536 F16 values at class load time
- `Float.floatToFloat16(x)` → table index → result
- Matches whisper.cpp's `ggml_table_gelu_f16` approach
- Eliminates `Math.tanh()` per element (the dominant cost)
- **Conv1+GELU: 24ms → 4ms**

### Pre-transposed Linear Weights (2x encoder block speedup)
- Transpose weight matrices once at construction time into native memory
- Forward pass uses `matmul(weightT)` instead of `matmulTransB(weight)`
- Eliminates CblasTrans flag overhead and enables better memory access
- Trade-off: model load time increases (232ms → 478ms) — acceptable one-time cost
- **Encoder block: 80ms → 33ms**

### Native Tensor Allocation (eliminates ensureNative copies)
- `Tensor.ofNative()` allocates off-heap from the start
- Conv1d im2col and bias results use native allocation
- Avoids `ensureNative()` heap→off-heap copy in every matmul call
- Critical for BLAS path which requires native MemorySegment

### Bluestein FFT (23x mel spectrogram speedup)
- Direct DFT for n=400 was O(n²) per frame × 3001 frames = dominant bottleneck
- Bluestein's algorithm converts to power-of-2 FFT via chirp-z convolution
- **Mel spectrogram: 4700ms → 200ms**

### Apple Accelerate BLAS via Panama FFM
- `cblas_sgemm` dispatches to AMX coprocessor on Apple Silicon
- ~14x throughput over pure Java for large matrices
- `vDSP` functions for parallel softmax (softmaxRows)
- `vvexpf` for vectorized exp in softmax
- Key: use `CblasRowMajor` with correct `lda`/`ldb`/`ldc` parameters

### GGML Mel Filters (correctness fix that also improved perf)
- Using pre-computed mel filters from GGML file instead of computing our own
- Eliminates mel filter bank computation and ensures exact match with whisper.cpp

## ❌ Optimizations That Did NOT Work

### SIMD matmul-vec for M=1 on Apple Silicon
- **Attempted:** Replace `cblas_sgemm` with Java Vector API dot products for single-token decoder steps
- **Result:** 2x SLOWER (10.8s → 22.6s for physicsworks.wav)
- **Why:** Apple Accelerate's sgemm already uses AMX coprocessor which is faster than NEON SIMD for any matrix size. The BLAS call overhead (~5μs) is real but AMX throughput compensates.
- **Lesson:** On Apple Silicon, ALWAYS use Accelerate BLAS. Don't try to beat AMX with Vector API.
- **Exception:** On non-Apple platforms (no BLAS), the SIMD path IS faster than tiled matmul for M=1.

### SIMD matmul-vec with K threshold (K >= 256)
- **Attempted:** Only use SIMD for large K (Linear layers K=512) but keep BLAS for small K (attention K=64)
- **Result:** Still slower (15.6s vs 10.8s baseline)
- **Why:** The accumulation pattern `out += x[k] * B_row[k]` has poor cache behavior — reads entire output vector N times. For N=2048 (MLP), this thrashes L1 cache.
- **Lesson:** Row-accumulation matmul-vec is cache-hostile for large N. Dot-product-per-output-element is better but requires column access (stride-N), which is also cache-hostile.

### Pre-allocated KV Cache Buffer
- **Attempted:** Pre-allocate (batchHeads, 448, headDim) buffer, append in-place to avoid O(N²) copying
- **Result:** Introduced bugs — `viewRows` still required copying to create contiguous tensors for BLAS. The `.cap` metadata tracking added complexity and broke when caches were shared across decode calls.
- **Why:** BLAS requires contiguous memory per batch element. A pre-allocated buffer with maxLen stride can't be used directly — you need to copy the active portion anyway.
- **Lesson:** KV cache append-and-copy is O(N) per step, O(N²) total. For N≤224 tokens, this is ~50K floats copied per step — negligible vs the matmul compute. Don't optimize what isn't the bottleneck.

### Sigmoid GELU Approximation (x * σ(1.702x))
- **Attempted:** Use `x / (1 + exp(-1.702x))` via vDSP vectorized exp
- **Result:** Produced wrong transcription output
- **Why:** Whisper was trained with tanh GELU `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))`. The sigmoid approximation has ~1% relative error which compounds across 6 encoder + 6 decoder blocks.
- **Lesson:** Activation function must match training exactly. Use F16 lookup table for speed without sacrificing accuracy.

### Accelerate vDSP GELU (x * sigmoid(1.702x) via vvexpf)
- **Attempted:** Vectorized sigmoid GELU using Apple vDSP functions
- **Result:** Same as above — wrong activation function
- **Lesson:** Same as above. The vDSP approach would work IF we used the tanh formula, but `vvtanhf` doesn't exist in Accelerate. The F16 table is simpler and faster.

## 🔶 Partially Effective Optimizations

### matmulTransB (CblasTrans flag)
- Using `sgemmTransB` avoids explicit transpose allocation
- Saves ~0.5ms per encoder block (transpose of 512×512 weight)
- But with pre-transposed weights, this is no longer needed — `matmul` with pre-transposed weight is equivalent and slightly faster
- **Current state:** Linear uses pre-transposed weights + `matmul`. `matmulTransB` still used for QK^T in attention (where K is not pre-transposed).

### Parallel Softmax (vDSP multi-threaded)
- Splits softmax rows across CPU cores using platform threads
- Helps for large attention matrices (1500×1500 in encoder)
- Negligible for decoder (8×1×N softmax — too small to parallelize)

## 📊 Where Time Is Actually Spent (base.en, single token step)

Per decoder block (2.2ms total):
- Cross-attention: 1069μs (49%) — QK^T matmul dominates (8 heads × 64×1500)
- Self-attention: 823μs (38%) — K/V append + QK^T + attn@V
- MLP: 259μs (12%) — two Linear (512→2048, 2048→512) + GELU
- LayerNorm: 11μs (0.5%) — negligible

The 66 BLAS calls per token step contribute ~330μs of overhead (15% of total).
The remaining 85% is actual AMX compute — cannot be optimized in pure Java.

## 🎯 Remaining Optimization Opportunities

### Fused Decoder Block (would close the gap)
Write a single C function that executes an entire decoder block (self-attn + cross-attn + MLP) and call it via Panama FFM. This eliminates 11 BLAS calls per block → 1 call per block. Estimated savings: ~250μs per block × 6 = 1.5ms per token step.

### CoreML/ANE Offload (would exceed whisper.cpp)
The Apple Neural Engine can run the full decoder in ~1ms. Architecture: Java → Unix socket → Swift service → CoreML. This would make the decoder faster than whisper.cpp's CPU path.

### Speculative Decoding
Generate multiple candidate tokens in parallel, verify in one forward pass. Amortizes BLAS overhead across N candidates. Requires architectural changes to the decode loop.

### Quantized Inference (INT8/INT4)
Keep weights in quantized format and use quantized matmul. Reduces memory bandwidth (the real bottleneck for large models). whisper.cpp supports Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 — our GGML loader already dequantizes these but we could keep them quantized.

---

## Session: 2026-04-14 — Decoder Performance for Medium/Large Models

### Scoped Arena for Encoder (success — 68x → 1x overhead eliminated)
- Attempted: Per-layer `Arena.ofShared()` in WhisperEncoder.forward() to batch-free intermediate tensors
- Result: Encoder went from ~41s to ~3s per chunk for large-v3-turbo (32 layers, 1280 state)
- Why: `Arena.ofAuto()` relies on GC cleaners to free native memory. 32 encoder layers allocate ~10GB of intermediates per pass. The cleaner couldn't keep up, causing OS-level virtual memory thrashing. `Arena.ofShared()` frees everything instantly on `.close()`.
- Lesson: For tight loops allocating large native buffers, always use explicit arena lifecycle. `Arena.ofAuto()` is only suitable for long-lived allocations. Note: `Arena.ofConfined()` deadlocks when AccelerateBlas.softmaxRows spawns worker threads — must use `ofShared()`.

### M=1 SIMD Matmul Priority (success — decoder dispatch overhead eliminated)
- Attempted: Reordered matmul/matmulScaled/matmulTransB to check M==1 before AccelerateBlas.isAvailable()
- Result: Decoder per-token matmul calls use inline SIMD instead of BLAS sgemm
- Why: For M=1 (single token decode), BLAS sgemm has ~5μs FFM downcall overhead per call. With ~53 matmul calls per token (turbo), that's ~265μs wasted. The SIMD dot-product path has zero dispatch cost.
- Lesson: BLAS is only faster than SIMD for M>1 (actual matrix-matrix multiply). For matrix-vector (M=1), inline SIMD always wins.

### KV Cache Pre-allocated Buffer (success — medium.en 233s → 46s)
- Attempted: 2x-growth buffer for KV cache append, avoiding Arena.ofAuto() allocation per step
- Result: Decoder time for medium.en dropped from 198s to 26s (7.6x speedup)
- Why: Original appendHeadsCache allocated a new tensor via Arena.ofAuto() on every decode step, copying all existing data plus new data. The Arena allocation overhead dominated — not the O(n²) copy itself, but the native memory allocation/deallocation churn. The buffer approach reuses the backing store and only reallocates on 2x growth.
- Lesson: Arena.ofAuto() allocation is expensive in tight loops. Pre-allocate with growth factor for append-heavy patterns.

### Fused SIMD Attention Kernel (marginal — ~5% decoder improvement)
- Attempted: Combined QK^T + softmax + AV into one SIMD-vectorized pass per head for qLen==1
- Result: medium.en decoder 26s → 23s. Eliminates 5 tensor allocations per attention call.
- Why: The attention compute (headDim=64 dot products) is small relative to the Linear projection compute (state=1024 matmuls). The allocation savings help but the compute savings are modest.
- Lesson: Fused kernels help most when the fused operations dominate total compute. For Whisper, Linear projections (MLP, Q/K/V/out) dominate decoder time, not attention.

### Smart Temperature Fallback (success — base.en 114s → 32s, turbo 626s → 136s)
- Attempted: Early exit from temperature fallback when t=0.0 has OK compression ratio, plus best-result tracking across temperatures
- Result: base.en 3.6x speedup, turbo 4.6x speedup. All models above real-time.
- Why: Greedy decoding produces avgLogprob marginally below -1.0 threshold, triggering all 6 fallback retries. Higher temperatures produce worse results (-9.9 avgLogprob). The original code used the last (worst) result. Now it keeps the best and exits early when higher temps degrade.
- Lesson: Temperature fallback is the single largest performance variable for smaller models. A bad fallback policy can 6x the decode time with zero quality benefit.

### Decoder logits.getRow() (success — eliminated full tensor copy)
- Attempted: Read only the last vocab row from logits tensor instead of copying entire (1, seqLen, 51866) tensor
- Result: Eliminated ~200KB allocation per decode step
- Why: `logits.data()` copied the entire tensor from off-heap to heap. Only the last row (51866 floats) was needed.
- Lesson: Always check if you're copying more data than needed from off-heap tensors.

### LayerNorm Cached Gamma/Beta (success — eliminated 96 copies per encoder pass)
- Attempted: Pre-cache gamma.data() and beta.data() float arrays at LayerNorm construction time
- Result: Eliminated 96 off-heap-to-heap copies per encoder pass (3 layerNorms × 32 layers)
- Why: gamma/beta weights never change but were being copied from off-heap on every forward() call
- Lesson: Cache immutable weight data at construction time, not on every forward pass.

### Linear addInPlace Bias (success — eliminated 192 allocations per encoder pass)
- Attempted: Changed `out.add(bias)` to `out.addInPlace(bias)` in Linear.forward()
- Result: Eliminated one tensor allocation per Linear call
- Why: The matmul output tensor is freshly allocated and owned — safe to modify in-place
- Lesson: Use in-place operations when the input tensor won't be reused.

### Final Performance Summary (physicsworks.wav, 203s audio, Apple Silicon)

| Model | Before | After | Speedup | RTF |
|-------|--------|-------|---------|-----|
| tiny.en (39M) | ~36s | 13s | 2.8x | 15.6x |
| base.en (74M) | ~126s | 32s | 3.9x | 6.3x |
| small.en (244M) | ~45s | 15s | 3.0x | 13.5x |
| medium.en (769M) | 233s | 43s | 5.4x | 4.7x |
| large-v3-turbo (1.5G) | 626s | 136s | 4.6x | 1.5x |
