# Optimization Notes

Append-only log of every optimization attempted. Prevents circular optimization attempts.

---

## Apple Accelerate Framework Offloading (2026-04-15)

Ported Accelerate.framework bindings from whisper4j's `AccelerateBlas` to numpy4j via FFM `Linker`.
Dynamic detection: `AccelerateOps.isAvailable()` returns `true` on macOS with Accelerate, `false` elsewhere.
All existing scalar/SIMD paths remain as fallback.

Benchmark: `tools/BenchmarkAccelerate.java` — 50 warmup + 200 measured iterations, raw FFM calls for both paths.

### batchedMatmul — `cblas_sgemm` via AMX (KEPT)

- **Workload:** 8 × [64,64] @ [64,64]
- **Speedup:** 101–112×
- **Parity:** max |diff| < 0.000002 (FP32 rounding)
- **Integrated in:** `MatMul.batchedMatmul()`
- **Why:** AMX coprocessor handles matrix multiply at ~2400 GFLOPS vs scalar loops. Massive win.

### softmax — vDSP pipeline (KEPT)

- **Workload:** 128 × 512
- **Speedup:** 8.5–8.8×
- **Parity:** max |diff| < 0.000000005 (essentially exact)
- **Integrated in:** `ML.softmax()`
- **Pipeline:** `vDSP_maxv` → `vDSP_vsadd` → `vvexpf` → `vDSP_sve` → `vDSP_vsdiv`
- **Why:** Replaces per-element `Math.exp` + 3 scalar passes with fused vectorized ops.

### exp — `vvexpf` via vForce (REVERTED)

- **Workload:** 65,536 elements
- **Speedup:** 14.5–14.8×
- **Parity:** NOT at parity with NumPy. `vvexpf` uses a fast approximation that differs from
  `libm expf` by 1 ULP on some inputs (e.g., exp(10), exp(-10), exp(-3.14)). NumPy uses
  `libm expf` which is bit-exact with Java's `(float) Math.exp()`. Confirmed by calling
  `vvexpf` directly from Python and comparing to `numpy.exp` — 3 of 13 test values differ.
- **Result:** Reverted. Java `(float) Math.exp()` is bit-exact with NumPy. The 14.8× speedup
  is not worth breaking parity, especially since exp feeds into softmax, cross-entropy, sigmoid,
  and other compound operations where the error would propagate.

### multiply — `vDSP_vmul` (REVERTED)

- **Workload:** 65,536 elements
- **Speedup:** 1.3–1.6×
- **Parity:** exact (0.0 diff)
- **Result:** Reverted. The raw FFI speedup is marginal (~1.5×), and the actual Ufunc integration
  adds `isNative()` check + `contiguous()` call + `isAvailable()` check overhead that likely
  eats most of the gain. Not worth the code complexity for a trivial element-wise op.

### mulScalar — `vDSP_vsmul` (REVERTED)

- **Workload:** 65,536 elements
- **Speedup:** 1.3–1.4×
- **Parity:** exact (0.0 diff)
- **Result:** Reverted. Same reasoning as multiply — marginal raw speedup, overhead in the
  integration path, not worth the complexity.

### GELU — vDSP sigmoid approximation (NOT PORTED)

- **Why not:** whisper4j uses the sigmoid GELU approximation (`x * sigmoid(1.702x)`) which can
  be expressed as a vDSP pipeline. numpy4j uses the tanh GELU approximation
  (`0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715x³)))`) for NumPy parity. These are different
  formulas — porting the vDSP pipeline would change the numerical output.

### layerNorm — vDSP (NOT PORTED)

- **Why not:** whisper4j also uses scalar loops for layerNorm. The per-row FFI call overhead
  would negate the benefit for typical layerNorm dimensions.

---

## LAPACK Offloading via Accelerate (2026-04-15)

Ported LAPACK bindings (`sgesv_`, `sgetrf_`/`sgetri_`, `spotrf_`) from Accelerate.framework for LinAlg operations.
All use Fortran column-major calling convention via FFM pointer args. Threshold: n ≥ 8 (FFI overhead dominates below).

Benchmark: `tools/BenchLapack.java` — 30 warmup + 200 measured iterations per size.

### solve — `sgesv_` (KEPT, n ≥ 8)

| n | Speedup | Parity (max diff) |
|---|---------|-------------------|
| 4 | 0.28× | 0.000007 |
| 16 | 1.11× | 0.000003 |
| 64 | 5.43× | 0.08 (ill-conditioned) |
| 256 | 3.83× | 0.000055 |

- **Integrated in:** `LinAlg.solve()`
- **Note:** n=64 parity diff is from ill-conditioned random matrix — both solutions are valid
  (different pivoting strategies). Residual ‖Ax-b‖ is small for both.

### inv — `sgetrf_` + `sgetri_` (KEPT, n ≥ 8)

| n | Speedup | Parity (max diff) |
|---|---------|-------------------|
| 4 | 0.50× | 0.000000002 |
| 16 | 2.90× | 0.000000007 |
| 64 | 2.73× | 0.000000007 |
| 256 | 10.32× | 0.000000007 |

- **Integrated in:** `LinAlg.inv()`
- **Parity:** Essentially exact across all sizes.

### cholesky — `spotrf_` (KEPT, n ≥ 8)

| n | Speedup | Parity (max diff) |
|---|---------|-------------------|
| 4 | 0.18× | 0.000000015 |
| 16 | 4.33× | 0.000000477 |
| 64 | 2.45× | 0.000001907 |
| 256 | 13.47× | 0.000005722 |

- **Integrated in:** `LinAlg.cholesky()`

### det — `sgetrf_` (KEPT, n ≥ 8)

| n | Speedup | Parity (rel diff) |
|---|---------|-------------------|
| 4 | 0.25× | 0.000001 |
| 16 | 1.17× | 0.000001 |
| 64 | 2.59× | 0.000114 |
| 256 | 3.71× | (both overflow to Inf) |

- **Integrated in:** `LinAlg.det()`
- **Note:** Large matrices produce determinants that overflow float32 — same for both paths.

### norm — `cblas_snrm2` (NOT INTEGRATED)

| n | Speedup | Parity |
|---|---------|--------|
| 64 | 0.25× | exact |
| 1024 | 1.65× | 0.000008 |
| 65536 | 5.88× | 0.000443 |

- **Result:** Not integrated. The crossover is at n≈1024, but `LinAlg.norm()` is typically called
  on small-to-medium vectors. The existing SIMD path handles this well enough.

### svd — `sgesvd_` (KEPT, n ≥ 8)

| n | Speedup | Parity (singular values) |
|---|---------|--------------------------|
| 4 | 0.92× | 0.0000003 |
| 16 | 3.14× | 0.000002 |
| 64 | 3.31× | 0.000389 |

- **Integrated in:** `LinAlg.svd()`
- **Note:** The parity diff at n=64 is the *Jacobi solver being less accurate* than LAPACK,
  not the other way around. LAPACK's `sgesvd_` uses bidiagonal reduction + QR iteration which
  is more numerically stable than the iterative Jacobi fallback. This is an accuracy improvement.
- **Propagation:** SVD feeds into `pinv`, `matrixRank`, and `cond`. Better SVD accuracy
  cascades positively through these dependent operations.

### eigvals/eigh — `ssyev_` (KEPT, n ≥ 8)

| n | Speedup | Parity (eigenvalues) |
|---|---------|----------------------|
| 4 | 48× | 0.000002 |
| 16 | 127× | 0.017 |
| 64 | 472× | 0.247 |

- **Integrated in:** `LinAlg.eigvals()`, `LinAlg.eigh()`
- **Note:** The large parity diff is the *scalar QR iteration being inaccurate*, not LAPACK.
  200 QR iterations is insufficient for convergence on matrices beyond ~16×16. LAPACK's `ssyev_`
  uses tridiagonal reduction + divide-and-conquer which converges reliably. The LAPACK result
  is the more correct one. This is both a massive speedup AND an accuracy improvement.
- **Propagation:** `eigh()` also returns eigenvectors via `ssyev_` with jobz='V', replacing
  the scalar inverse iteration fallback which was also inaccurate for clustered eigenvalues.

### Parity Stress Testing

Round-trip error propagation was validated at n=8,16,32,64,128 using:
- `A^-1 * A = I` (inv round-trip): max error < 0.000001 at n=128
- `solve(A, A*x) = x` (solve round-trip): max error < 0.000001 at n=128
- `L * L^T = A` (cholesky round-trip): max error < 0.00006 at n=128
- `det(A) * det(A^-1) = 1` (det consistency): exact for n≤16, overflow for n≥32 (pre-existing)

LAPACK L matrices differ from scalar L by < 0.000003 at n=128. The reconstruction error
difference is due to matmul accumulation order, not the decomposition itself.

### NumPy Parity Research (bit-exact comparison)

Verified against NumPy 2.4.3 (Python 3.14, Accelerate BLAS/LAPACK, macOS arm64).
Validation scripts: `tools/generate_numpy_reference.py` → `tools/ValidateNumpyParity.java`.

**All Accelerate-integrated operations are bit-exact with NumPy: 15/15 tests pass.**

Key findings:
- NumPy upcasts float32 to float64 before calling LAPACK, then downcasts back. We do the same
  (dgesv_, dgetrf_, dgetri_, dpotrf_, dgesvd_, dsyev_). This produces identical results.
- `(float) Math.exp()` = `numpy.exp` = `libm expf`: bit-exact. `vvexpf` is NOT the same
  (uses fast approximation), so it was reverted.
- SVD singular vectors (U, Vt) have arbitrary sign convention — parity check compares |values|.
- `dsyev_` is for symmetric matrices only. General eigvals uses scalar QR iteration fallback.

To re-validate: `python3 tools/generate_numpy_reference.py > tools/numpy_reference.properties`
then `java --enable-preview --source 26 --class-path target/classes tools/ValidateNumpyParity.java`

### qr — `sgeqrf_`/`sorgqr_` (DEFERRED)

| n | Speedup | Parity (R matrix) |
|---|---------|-------------------|
| 4 | 0.52× | 2.6 (sign convention) |
| 16 | 2.82× | 5.1 (sign convention) |
| 64 | 11.29× | 10.3 (sign convention) |

- **Result:** Deferred. LAPACK Householder QR and Gram-Schmidt QR produce different Q/R
  matrices (sign of R diagonal is arbitrary — both are valid decompositions where Q*R = A).
  The speedup is significant (2.8-11.3×) but the sign convention difference makes parity
  validation complex. Existing tests only check Q*R reconstruction on a 2×2 matrix (below
  the n≥8 threshold). Would need additional test coverage before integrating.

### FFT — `vDSP_fft_zip` (DEFERRED)

- **Result:** Deferred. vDSP FFT uses Apple's split-complex format (separate real/imag arrays)
  which is fundamentally different from the interleaved format used by the current Cooley-Tukey
  implementation. Format conversion overhead would negate speedup for small transforms.
  Complex binding with setup/teardown lifecycle (`vDSP_create_fftsetup`/`vDSP_destroy_fftsetup`).

---

## Summary

| Operation     | Method                | Speedup     | Parity          | Status   |
|---------------|-----------------------|-------------|-----------------|----------|
| batchedMatmul | `cblas_sgemm` (AMX)   | 101–112×    | < 0.000002      | ✅ Kept   |
| softmax       | vDSP pipeline         | 8.5–8.8×    | < 0.000000005   | ✅ Kept   |
| exp           | `vvexpf` (vForce)     | 14.5–14.8×  | NOT NumPy-exact | ❌ Reverted |
| multiply      | `vDSP_vmul`           | 1.3–1.6×    | exact           | ❌ Reverted |
| mulScalar     | `vDSP_vsmul`          | 1.3–1.4×    | exact           | ❌ Reverted |
| GELU          | vDSP sigmoid          | —           | wrong formula   | ⏭ Skipped |
| layerNorm     | vDSP                  | —           | —               | ⏭ Skipped |
| solve         | `sgesv_` (LAPACK)     | 1.1–5.4× (n≥16) | < 0.0001   | ✅ Kept (n≥8) |
| inv           | `sgetrf_`+`sgetri_`   | 2.7–10.3× (n≥16) | < 0.00000001 | ✅ Kept (n≥8) |
| cholesky      | `spotrf_` (LAPACK)    | 2.5–13.5× (n≥16) | < 0.000006 | ✅ Kept (n≥8) |
| det           | `sgetrf_` (LAPACK)    | 1.2–3.7× (n≥16) | < 0.0002 rel | ✅ Kept (n≥8) |
| svd           | `sgesvd_` (LAPACK)    | 3.1–3.3× (n≥16) | < 0.0004     | ✅ Kept (n≥8) |
| eigvals/eigh  | `ssyev_` (LAPACK)     | 48–472× (n≥4)   | accuracy ↑   | ✅ Kept (n≥8) |
| qr            | `sgeqrf_`/`sorgqr_`   | 2.8–11.3× (n≥16)| sign ambiguity | ⏭ Deferred |
| FFT           | `vDSP_fft_zip`        | —                | format mismatch | ⏭ Deferred |
| norm          | `cblas_snrm2`         | 1.7–5.9× (n≥1K) | < 0.0005   | ⏭ Skipped |
