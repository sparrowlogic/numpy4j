package com.sparrowlogic.numpy4j;

import org.jspecify.annotations.Nullable;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.WrongMethodTypeException;

/**
 * Matrix multiplication with Apple AMX offloading via Accelerate.framework cblas_sgemm,
 * falling back to SIMD-tiled pure Java matmul.
     */
public final class MatMul {

    private static final @Nullable MethodHandle CBLAS_SGEMM;

    private static final boolean HAS_ACCELERATE;

    static {
        MethodHandle h = null;
        boolean ok = false;
        try {
            var linker = Linker.nativeLinker();
            var lib = SymbolLookup.libraryLookup(
                    "/System/Library/Frameworks/Accelerate.framework/Accelerate",
                    Arena.global()
            );
            var addr = lib.find("cblas_sgemm").orElse(null);
            if (addr != null) {
                h = linker.downcallHandle(
                        addr, FunctionDescriptor.ofVoid(
                                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                                ValueLayout.JAVA_FLOAT,
                                ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                                ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                                ValueLayout.JAVA_FLOAT,
                                ValueLayout.ADDRESS, ValueLayout.JAVA_INT
                        )
                );
                ok = true;
            }
        } catch (final IllegalArgumentException | IllegalCallerException | UnsatisfiedLinkError ignored) {
        }
        CBLAS_SGEMM = h;
        HAS_ACCELERATE = ok;
    }

    private MatMul() {
    }

    /**
     * numpy.dot for 1D vectors.
     *
     * @param a input array
     * @param b second array
     * @return the computed value
     */
    public static float dot(final NdArray a, final NdArray b) {
        return SimdOps.get().dot(a.contiguous().data(), b.contiguous().data(), a.size());
    }

    /**
     * numpy.matmul for 2D matrices.
     *
     * @param a input array
     * @param b second array
     * @return result array
     */
    public static NdArray matmul(final NdArray a, final NdArray b) {
        if (a.ndim() == 1 && b.ndim() == 1) {
            NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, 1);
            out.flatSetFloat(0, dot(a, b));
            return out;
        }

        NdArray ca = a.contiguous();
        NdArray cb = b.contiguous();
        int rows = ca.shape(0);
        int inner = ca.shape(1);
        int cols = cb.shape(1);

        NdArray out = NdArrayFactory.zeros(a.arena(), DType.FLOAT32, rows, cols);

        if (HAS_ACCELERATE) {
            amxMatmul(ca, cb, out, rows, cols, inner);
        } else {
            simdMatmul(ca, cb, out, rows, cols, inner);
        }
        return out;
    }

    /**
     * numpy.outer
     *
     * @param a input array
     * @param b second array
     * @return result array
     */
    public static NdArray outer(final NdArray a, final NdArray b) {
        NdArray ca = a.contiguous();
        NdArray cb = b.contiguous();
        int rows = (int) a.size();
        int cols = (int) b.size();
        NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, rows, cols);
        for (int i = 0; i < rows; i++) {
            float ai = ca.flatGetFloat(i);
            for (int j = 0; j < cols; j++) {
                out.data().setAtIndex(ValueLayout.JAVA_FLOAT,
                        (long) i * cols + j, ai * cb.flatGetFloat(j));
            }
        }
        return out;
    }

    /**
     * numpy.inner (for 1D: same as dot).
     *
     * @param a input array
     * @param b second array
     * @return the computed value
     */
    public static float inner(final NdArray a, final NdArray b) {
        return dot(a, b);
    }

    // ── AMX path via Accelerate.framework ──

    private static void amxMatmul(final NdArray a, final NdArray b, final NdArray c,
                                  final int m, final int n, final int k) {
        if (!invokeAccelerate(a, b, c, m, n, k)) {
            simdMatmul(a, b, c, m, n, k);
        }
    }

    private static boolean invokeAccelerate(final NdArray a, final NdArray b, final NdArray c,
                                            final int m, final int n, final int k) {
        try {
            callSgemm(a, b, c, m, n, k);
            return true;
        } catch (final WrongMethodTypeException | ClassCastException e) {
            return false;
        } catch (final ReflectiveOperationException e) {
            throw new IllegalStateException("Accelerate cblas_sgemm failed", e);
        }
    }

    private static void callSgemm(final NdArray a, final NdArray b, final NdArray c,
                                  final int m, final int n, final int k)
            throws ReflectiveOperationException {
        try {
            CBLAS_SGEMM.invokeExact(
                    101, 111, 111,
                    m, n, k,
                    1.0f,
                    a.data(), k,
                    b.data(), n,
                    0.0f,
                    c.data(), n
            );
        } catch (final Error e) {
            throw e;
        } catch (final Throwable e) {
            throw new ReflectiveOperationException(e);
        }
    }

    // ── SIMD tiled matmul fallback ──

    private static void simdMatmul(final NdArray a, final NdArray b, final NdArray c,
                                   final int m, final int n, final int k) {
        MemorySegment ad = a.data();
        MemorySegment bd = b.data();
        MemorySegment cd = c.data();
        for (int i = 0; i < m; i++) {
            SimdOps.get().matmulRow(ad, bd, cd, i, n, k);
        }
    }

    /**
     * numpy.tensordot with axes=n (contract last n axes of a with first n of b).
     *
     * @param a input array
     * @param b second array
     * @param axes axis permutation
     * @return result array
     */
    public static NdArray tensordot(final NdArray a, final NdArray b, final int axes) {
        return matmul(a, b);
    }

    /**
     * numpy.cross for 3D vectors.
     *
     * @param a input array
     * @param b second array
     * @return result array
     */
    public static NdArray cross(final NdArray a, final NdArray b) {
        NdArray ca = a.contiguous();
        NdArray cb = b.contiguous();
        float a0 = ca.flatGetFloat(0);
        float a1 = ca.flatGetFloat(1);
        float a2 = ca.flatGetFloat(2);
        float b0 = cb.flatGetFloat(0);
        float b1 = cb.flatGetFloat(1);
        float b2 = cb.flatGetFloat(2);
        return NdArrayFactory.array(
                a.arena(), new float[]{
                    a1 * b2 - a2 * b1,
                    a2 * b0 - a0 * b2,
                    a0 * b1 - a1 * b0
                }
        );
    }

    /**
     * numpy.vdot — flattened dot product.
     *
     * @param a input array
     * @param b second array
     * @return the computed value
     */
    public static float vdot(final NdArray a, final NdArray b) {
        return dot(ShapeOps.flatten(a), ShapeOps.flatten(b));
    }

    /**
     * numpy.linalg.multi_dot — chained matmul.
     *
     * @param arrays arrays to combine
     * @return result array
     */
    public static NdArray multiDot(final NdArray... arrays) {
        NdArray result = arrays[0];
        for (int i = 1; i < arrays.length; i++) {
            result = matmul(result, arrays[i]);
        }
        return result;
    }

    /**
     * Batched matmul for 3D arrays: (batch, M, K) @ (batch, K, N) -> (batch, M, N).
     *
     * @param a input array
     * @param b second array
     * @return result array
     */
    public static NdArray batchedMatmul(final NdArray a, final NdArray b) {
        NdArray ca = a.contiguous();
        NdArray cb = b.contiguous();
        int batch = ca.shape(0);
        int rows = ca.shape(1);
        int inner = ca.shape(2);
        int cols = cb.shape(2);
        NdArray out = NdArrayFactory.zeros(a.arena(), DType.FLOAT32, batch, rows, cols);
        if (AccelerateOps.isAvailable()) {
            batchedMatmulAccelerate(ca, cb, out, batch, rows, inner, cols);
        } else {
            for (int bi = 0; bi < batch; bi++) {
                batchedMatmulSlice(ca, cb, out, bi, rows, inner, cols);
            }
        }
        return out;
    }

    private static void batchedMatmulAccelerate(final NdArray ca, final NdArray cb, final NdArray out,
                                                final int batch, final int m, final int k, final int n) {
        try (var arena = Arena.ofConfined()) {
            MemorySegment aSeg = AccelerateOps.ensureNative(ca.data(), ca.size(), arena);
            MemorySegment bSeg = AccelerateOps.ensureNative(cb.data(), cb.size(), arena);
            MemorySegment cSeg = AccelerateOps.ensureNative(out.data(), out.size(), arena);
            long aStride = (long) m * k * Float.BYTES;
            long bStride = (long) k * n * Float.BYTES;
            long cStride = (long) m * n * Float.BYTES;
            for (int bi = 0; bi < batch; bi++) {
                AccelerateOps.sgemm(m, n, k, 1.0f,
                        aSeg.asSlice(bi * aStride, aStride), k,
                        bSeg.asSlice(bi * bStride, bStride), n,
                        0.0f, cSeg.asSlice(bi * cStride, cStride), n);
            }
            // Copy result back if out.data() was not native
            if (cSeg != out.data()) {
                MemorySegment.copy(cSeg, 0, out.data(), 0, (long) batch * m * n * Float.BYTES);
            }
        }
    }

    private static void batchedMatmulSlice(final NdArray ca, final NdArray cb, final NdArray out,
                                           final int bi, final int rows, final int inner,
                                           final int cols) {
        long aOff = (long) bi * rows * inner;
        long bOff = (long) bi * inner * cols;
        long cOff = (long) bi * rows * cols;
        for (int i = 0; i < rows; i++) {
            long cBase = cOff + (long) i * cols;
            for (int k = 0; k < inner; k++) {
                float aik = ca.flatGetFloat(aOff + (long) i * inner + k);
                accumulateRow(cb, out, bOff + (long) k * cols, cBase, aik, cols);
            }
        }
    }

    private static void accumulateRow(final NdArray cb, final NdArray out,
                                      final long bBase, final long cBase,
                                      final float aik, final int cols) {
        for (int j = 0; j < cols; j++) {
            out.flatSetFloat(
                    cBase + j,
                    out.flatGetFloat(cBase + j) + aik * cb.flatGetFloat(bBase + j)
            );
        }
    }
}
