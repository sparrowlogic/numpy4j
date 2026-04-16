package com.sparrowlogic.numpy4j;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.util.logging.Logger;

/**
 * Apple Accelerate framework bindings via FFM for hardware-accelerated linear algebra.
 *
 * <p>Provides three acceleration layers:</p>
 * <ul>
 *   <li><b>BLAS</b> — {@code cblas_sgemm} for matrix multiply (dispatches to AMX on Apple Silicon)</li>
 *   <li><b>vDSP</b> — vectorized softmax pipeline ({@code vDSP_maxv}, {@code vvexpf}, etc.)</li>
 *   <li><b>LAPACK</b> — double-precision routines ({@code dgesv_}, {@code dgetrf_}, etc.)
 *       with float32 cast-back to match NumPy's behavior</li>
 * </ul>
 *
 * <p>All operations fall back gracefully on non-macOS platforms:
 * {@link #isAvailable()} returns {@code false} and callers use scalar paths.</p>
 */
@SuppressWarnings({"checkstyle:ClassDataAbstractionCoupling", "checkstyle:ClassFanOutComplexity"})
final class AccelerateOps {

    private static final Logger LOG = Logger.getLogger(AccelerateOps.class.getName());
    private static final int ROW_MAJOR = 101;
    private static final int NO_TRANS = 111;
    private static final int H_SGEMM = 0;
    private static final int H_VVEXPF = 1;
    private static final int H_MAXV = 2;
    private static final int H_VSADD = 3;
    private static final int H_SVE = 4;
    private static final int H_VSDIV = 5;
    private static final int H_DGESV = 6;
    private static final int H_DGETRF = 7;
    private static final int H_DGETRI = 8;
    private static final int H_DPOTRF = 9;
    private static final int H_DGESVD = 10;
    private static final int H_DSYEV = 11;
    private static final int HANDLE_COUNT = 12;
    private static final MethodHandle[] H;
    private static final boolean AVAILABLE;

    static {
        MethodHandle[] tmp = null;
        try {
            var lib = SymbolLookup.libraryLookup(
                    "/System/Library/Frameworks/Accelerate.framework/Accelerate", Arena.global());
            tmp = initHandles(Linker.nativeLinker(), lib);
            LOG.info("Apple Accelerate loaded (BLAS + vDSP + LAPACK)");
        } catch (final Throwable t) {
            LOG.fine("Accelerate not available: " + t.getMessage());
        }
        H = tmp != null ? tmp : new MethodHandle[HANDLE_COUNT];
        AVAILABLE = tmp != null;
    }

    private AccelerateOps() {
    }

    /**
     * Returns {@code true} if Apple Accelerate is available on this platform.
     *
     * @return whether Accelerate framework was loaded successfully
     */
    static boolean isAvailable() {
        return AVAILABLE;
    }

    // ── Initialization ──

    private static MethodHandle[] initHandles(final Linker l, final SymbolLookup lib) {
        var a = ValueLayout.ADDRESS;
        var i = ValueLayout.JAVA_INT;
        var f = ValueLayout.JAVA_FLOAT;
        var j = ValueLayout.JAVA_LONG;
        return new MethodHandle[]{
            l.downcallHandle(lib.find("cblas_sgemm").orElseThrow(),
                    FunctionDescriptor.ofVoid(i, i, i, i, i, i, f, a, i, a, i, f, a, i)),
            l.downcallHandle(lib.find("vvexpf").orElseThrow(),
                    FunctionDescriptor.ofVoid(a, a, a)),
            l.downcallHandle(lib.find("vDSP_maxv").orElseThrow(),
                    FunctionDescriptor.ofVoid(a, j, a, j)),
            l.downcallHandle(lib.find("vDSP_vsadd").orElseThrow(),
                    FunctionDescriptor.ofVoid(a, j, a, a, j, j)),
            l.downcallHandle(lib.find("vDSP_sve").orElseThrow(),
                    FunctionDescriptor.ofVoid(a, j, a, j)),
            l.downcallHandle(lib.find("vDSP_vsdiv").orElseThrow(),
                    FunctionDescriptor.ofVoid(a, j, a, a, j, j)),
            lapackHandle(l, lib, "dgesv_", 8),
            lapackHandle(l, lib, "dgetrf_", 6),
            lapackHandle(l, lib, "dgetri_", 7),
            lapackHandle(l, lib, "dpotrf_", 5),
            lapackHandle(l, lib, "dgesvd_", 14),
            lapackHandle(l, lib, "dsyev_", 9),
        };
    }

    private static MethodHandle lapackHandle(final Linker l, final SymbolLookup lib,
                                              final String name, final int nArgs) {
        var layouts = new MemoryLayout[nArgs];
        java.util.Arrays.fill(layouts, ValueLayout.ADDRESS);
        return l.downcallHandle(lib.find(name).orElseThrow(), FunctionDescriptor.ofVoid(layouts));
    }

    // ── BLAS ──

    /**
     * Matrix multiply via {@code cblas_sgemm}: C = alpha * A @ B + beta * C (row-major).
     * On Apple Silicon this dispatches to the AMX coprocessor.
     *
     * @param m     rows of A and C
     * @param n     columns of B and C
     * @param k     columns of A / rows of B
     * @param alpha scalar multiplier for A*B
     * @param a     matrix A segment (native, row-major)
     * @param lda   leading dimension of A
     * @param b     matrix B segment (native, row-major)
     * @param ldb   leading dimension of B
     * @param beta  scalar multiplier for C
     * @param c     output matrix C segment (native, row-major)
     * @param ldc   leading dimension of C
     */
    @SuppressWarnings("checkstyle:ParameterNumber")
    static void sgemm(final int m, final int n, final int k, final float alpha,
                      final MemorySegment a, final int lda,
                      final MemorySegment b, final int ldb,
                      final float beta, final MemorySegment c, final int ldc) {
        try {
            H[H_SGEMM].invokeExact(ROW_MAJOR, NO_TRANS, NO_TRANS,
                    m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        } catch (final Throwable t) {
            throw new IllegalStateException("sgemm failed", t);
        }
    }

    // ── vDSP ──

    /**
     * In-place softmax over contiguous rows using the vDSP pipeline:
     * {@code maxv → vsadd(-max) → vvexpf → sve → vsdiv(sum)}.
     *
     * @param seg  native memory segment containing rows × cols floats
     * @param rows number of rows (outer dimension)
     * @param cols number of columns (last-axis length to normalize over)
     */
    static void softmaxRows(final MemorySegment seg, final int rows, final int cols) {
        long rowBytes = (long) cols * Float.BYTES;
        try (var arena = Arena.ofConfined()) {
            var maxBuf = arena.allocate(Float.BYTES, Float.BYTES);
            var negBuf = arena.allocate(Float.BYTES, Float.BYTES);
            var sumBuf = arena.allocate(Float.BYTES, Float.BYTES);
            var cntBuf = arena.allocate(Integer.BYTES, Integer.BYTES);
            cntBuf.set(ValueLayout.JAVA_INT, 0, cols);
            for (int r = 0; r < rows; r++) {
                var row = seg.asSlice((long) r * rowBytes, rowBytes);
                H[H_MAXV].invokeExact(row, 1L, maxBuf, (long) cols);
                negBuf.set(ValueLayout.JAVA_FLOAT, 0, -maxBuf.get(ValueLayout.JAVA_FLOAT, 0));
                H[H_VSADD].invokeExact(row, 1L, negBuf, row, 1L, (long) cols);
                H[H_VVEXPF].invokeExact(row, row, cntBuf);
                H[H_SVE].invokeExact(row, 1L, sumBuf, (long) cols);
                H[H_VSDIV].invokeExact(row, 1L, sumBuf, row, 1L, (long) cols);
            }
        } catch (final Throwable t) {
            throw new IllegalStateException("vDSP softmax failed", t);
        }
    }

    // ── LAPACK (double-precision for NumPy bit-exact parity) ──

    /**
     * Solves Ax = b via LAPACK {@code dgesv_} (LU factorization with partial pivoting).
     * Input is float32, internally upcasted to float64 for NumPy-matching precision.
     *
     * @param n     system size (n × n matrix)
     * @param a     row-major coefficient matrix (overwritten internally)
     * @param b     right-hand side vector; overwritten with solution x on return
     * @param arena confined arena for scratch allocations
     * @return LAPACK info code (0 = success, &gt;0 = singular)
     */
    static int solve(final int n, final float[] a, final float[] b, final Arena arena) {
        var aSeg = f32ToF64ColMajor(arena, a, n, n);
        var bSeg = f32ToF64(arena, b);
        var nSeg = intSeg(arena, n);
        var nrhs = intSeg(arena, 1);
        var ipiv = arena.allocate((long) n * Integer.BYTES, Integer.BYTES);
        var info = intSeg(arena, 0);
        try {
            H[H_DGESV].invokeExact(nSeg, nrhs, aSeg, nSeg, ipiv, bSeg, nSeg, info);
        } catch (final Throwable t) {
            throw new IllegalStateException("dgesv_ failed", t);
        }
        f64ToF32(bSeg, b, n);
        return info.get(ValueLayout.JAVA_INT, 0);
    }

    /**
     * Matrix inverse via LAPACK {@code dgetrf_} (LU) + {@code dgetri_} (inversion).
     *
     * @param n      matrix dimension
     * @param a      row-major input matrix
     * @param result row-major output inverse (n × n)
     * @param arena  confined arena for scratch allocations
     * @return LAPACK info code (0 = success)
     */
    static int inv(final int n, final float[] a, final float[] result, final Arena arena) {
        var aSeg = f32ToF64ColMajor(arena, a, n, n);
        var nSeg = intSeg(arena, n);
        var ipiv = arena.allocate((long) n * Integer.BYTES, Integer.BYTES);
        var info = intSeg(arena, 0);
        int lwork = n * 64;
        var work = arena.allocate((long) lwork * Double.BYTES, Double.BYTES);
        var lwSeg = intSeg(arena, lwork);
        try {
            H[H_DGETRF].invokeExact(nSeg, nSeg, aSeg, nSeg, ipiv, info);
            if (info.get(ValueLayout.JAVA_INT, 0) != 0) {
                return info.get(ValueLayout.JAVA_INT, 0);
            }
            H[H_DGETRI].invokeExact(nSeg, aSeg, nSeg, ipiv, work, lwSeg, info);
        } catch (final Throwable t) {
            throw new IllegalStateException("dgetrf_/dgetri_ failed", t);
        }
        f64ColMajorToF32(aSeg, result, n, n);
        return info.get(ValueLayout.JAVA_INT, 0);
    }

    /**
     * Cholesky decomposition via LAPACK {@code dpotrf_} (lower triangular).
     *
     * @param n      matrix dimension
     * @param a      row-major symmetric positive-definite input
     * @param result row-major output with lower triangle filled
     * @param arena  confined arena for scratch allocations
     * @return LAPACK info code (0 = success, &gt;0 = not positive definite)
     */
    static int cholesky(final int n, final float[] a, final float[] result, final Arena arena) {
        var aSeg = f32ToF64ColMajor(arena, a, n, n);
        var nSeg = intSeg(arena, n);
        var info = intSeg(arena, 0);
        var uplo = arena.allocateFrom("L");
        try {
            H[H_DPOTRF].invokeExact(uplo, nSeg, aSeg, nSeg, info);
        } catch (final Throwable t) {
            throw new IllegalStateException("dpotrf_ failed", t);
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                result[i * n + j] = (float) aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) j * n + i);
            }
        }
        return info.get(ValueLayout.JAVA_INT, 0);
    }

    /**
     * Determinant via LAPACK {@code dgetrf_} (LU factorization).
     * Computes the product of the LU diagonal in double precision, then casts to float.
     *
     * @param n     matrix dimension
     * @param a     row-major input matrix
     * @param arena confined arena for scratch allocations
     * @return the determinant as float32
     */
    static float det(final int n, final float[] a, final Arena arena) {
        var aSeg = f32ToF64ColMajor(arena, a, n, n);
        var nSeg = intSeg(arena, n);
        var ipiv = arena.allocate((long) n * Integer.BYTES, Integer.BYTES);
        var info = intSeg(arena, 0);
        try {
            H[H_DGETRF].invokeExact(nSeg, nSeg, aSeg, nSeg, ipiv, info);
        } catch (final Throwable t) {
            throw new IllegalStateException("dgetrf_ failed", t);
        }
        if (info.get(ValueLayout.JAVA_INT, 0) != 0) {
            return 0f;
        }
        double d = 1.0;
        int swaps = 0;
        for (int i = 0; i < n; i++) {
            d *= aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + i);
            if (ipiv.getAtIndex(ValueLayout.JAVA_INT, i) != i + 1) {
                swaps++;
            }
        }
        return (float) (swaps % 2 == 0 ? d : -d);
    }

    /**
     * Symmetric eigenvalue decomposition via LAPACK {@code dsyev_}.
     * Returns eigenvalues in ascending order. Optionally computes eigenvectors.
     *
     * @param n           matrix dimension
     * @param a           row-major symmetric input matrix
     * @param eigenvalues output array for eigenvalues (length n, ascending)
     * @param vectors     output array for eigenvectors (row-major n × n), or null for eigenvalues only
     * @param arena       confined arena for scratch allocations
     * @return LAPACK info code (0 = success)
     */
    static int eigvals(final int n, final float[] a, final float[] eigenvalues,
                       final float[] vectors, final Arena arena) {
        boolean wantVectors = vectors != null;
        var jobz = arena.allocateFrom(wantVectors ? "V" : "N");
        var uplo = arena.allocateFrom("U");
        var nSeg = intSeg(arena, n);
        var aSeg = f32ToF64ColMajor(arena, a, n, n);
        var wSeg = arena.allocate((long) n * Double.BYTES, Double.BYTES);
        int lwork = n * 64;
        var workSeg = arena.allocate((long) lwork * Double.BYTES, Double.BYTES);
        var lwSeg = intSeg(arena, lwork);
        var info = intSeg(arena, 0);
        try {
            H[H_DSYEV].invokeExact(jobz, uplo, nSeg, aSeg, nSeg, wSeg, workSeg, lwSeg, info);
        } catch (final Throwable t) {
            throw new IllegalStateException("dsyev_ failed", t);
        }
        f64ToF32(wSeg, eigenvalues, n);
        if (wantVectors) {
            f64ColMajorToF32(aSeg, vectors, n, n);
        }
        return info.get(ValueLayout.JAVA_INT, 0);
    }

    /**
     * Singular Value Decomposition via LAPACK {@code dgesvd_} (economy/thin SVD).
     * Returns U (m × min), S (min), and Vt (min × n) where min = min(m, n).
     *
     * @param m     number of rows
     * @param n     number of columns
     * @param a     row-major input matrix (m × n)
     * @param s     output singular values (length min(m,n), descending)
     * @param u     output left singular vectors (row-major, m × min(m,n))
     * @param vt    output right singular vectors transposed (row-major, min(m,n) × n)
     * @param arena confined arena for scratch allocations
     * @return LAPACK info code (0 = success)
     */
    @SuppressWarnings("checkstyle:ParameterNumber")
    static int svd(final int m, final int n, final float[] a, final float[] s,
                   final float[] u, final float[] vt, final Arena arena) {
        int mn = Math.min(m, n);
        var jobu = arena.allocateFrom("S");
        var jobvt = arena.allocateFrom("S");
        var mSeg = intSeg(arena, m);
        var nSeg = intSeg(arena, n);
        var aSeg = f32ToF64ColMajor(arena, a, m, n);
        var sSeg = arena.allocate((long) mn * Double.BYTES, Double.BYTES);
        var uSeg = arena.allocate((long) m * mn * Double.BYTES, Double.BYTES);
        var lduSeg = intSeg(arena, m);
        var vtSeg = arena.allocate((long) mn * n * Double.BYTES, Double.BYTES);
        var ldvtSeg = intSeg(arena, mn);
        int lwork = Math.max(1, 3 * mn + Math.max(m, n)) * 2;
        var workSeg = arena.allocate((long) lwork * Double.BYTES, Double.BYTES);
        var lwSeg = intSeg(arena, lwork);
        var info = intSeg(arena, 0);
        try {
            H[H_DGESVD].invokeExact(jobu, jobvt, mSeg, nSeg, aSeg, mSeg,
                    sSeg, uSeg, lduSeg, vtSeg, ldvtSeg, workSeg, lwSeg, info);
        } catch (final Throwable t) {
            throw new IllegalStateException("dgesvd_ failed", t);
        }
        f64ToF32(sSeg, s, mn);
        f64ColMajorToF32(uSeg, u, m, mn);
        f64ColMajorToF32(vtSeg, vt, mn, n);
        return info.get(ValueLayout.JAVA_INT, 0);
    }

    // ── Segment helpers ──

    /**
     * Copies a heap {@link MemorySegment} to native (off-heap) memory for BLAS compatibility.
     * Returns the original segment if it is already native.
     *
     * @param seg      source segment
     * @param elements number of float elements
     * @param arena    arena for allocation
     * @return native segment suitable for BLAS calls
     */
    static MemorySegment ensureNative(final MemorySegment seg, final long elements,
                                      final Arena arena) {
        if (seg.isNative()) {
            return seg;
        }
        long bytes = elements * Float.BYTES;
        MemorySegment nat = arena.allocate(bytes, Float.BYTES);
        MemorySegment.copy(seg, 0, nat, 0, bytes);
        return nat;
    }

    /** Converts row-major float32 array to column-major float64 segment for LAPACK. */
    private static MemorySegment f32ToF64ColMajor(final Arena arena, final float[] rm,
                                                   final int m, final int n) {
        var seg = arena.allocate((long) m * n * Double.BYTES, Double.BYTES);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                seg.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) j * m + i, rm[i * n + j]);
            }
        }
        return seg;
    }

    /** Converts float32 array to contiguous float64 segment. */
    private static MemorySegment f32ToF64(final Arena arena, final float[] data) {
        var seg = arena.allocate((long) data.length * Double.BYTES, Double.BYTES);
        for (int i = 0; i < data.length; i++) {
            seg.setAtIndex(ValueLayout.JAVA_DOUBLE, i, data[i]);
        }
        return seg;
    }

    /** Reads float64 segment back into float32 array (downcast). */
    private static void f64ToF32(final MemorySegment seg, final float[] out, final int n) {
        for (int i = 0; i < n; i++) {
            out[i] = (float) seg.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
        }
    }

    /** Reads column-major float64 segment into row-major float32 array. */
    private static void f64ColMajorToF32(final MemorySegment cm, final float[] rm,
                                          final int m, final int n) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                rm[i * n + j] = (float) cm.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) j * m + i);
            }
        }
    }

    /** Allocates a single-element int segment for LAPACK pointer arguments. */
    private static MemorySegment intSeg(final Arena arena, final int val) {
        var seg = arena.allocate(Integer.BYTES, Integer.BYTES);
        seg.set(ValueLayout.JAVA_INT, 0, val);
        return seg;
    }
}
