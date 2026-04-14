package com.sparrowlogic.numpy4j;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

/**
 * Matrix multiplication with Apple AMX offloading via Accelerate.framework cblas_sgemm,
 * falling back to SIMD-tiled pure Java matmul.
 */
public final class MatMul {
    private MatMul() {}

    private static final VectorSpecies<Float> F = FloatVector.SPECIES_PREFERRED;
    private static final MethodHandle CBLAS_SGEMM;
    private static final boolean HAS_ACCELERATE;

    static {
        MethodHandle h = null;
        boolean ok = false;
        try {
            var linker = Linker.nativeLinker();
            var lib = SymbolLookup.libraryLookup("/System/Library/Frameworks/Accelerate.framework/Accelerate", Arena.global());
            var addr = lib.find("cblas_sgemm").orElse(null);
            if (addr != null) {
                // cblas_sgemm(order, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
                h = linker.downcallHandle(addr, FunctionDescriptor.ofVoid(
                        ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_FLOAT,
                        ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                        ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_FLOAT,
                        ValueLayout.ADDRESS, ValueLayout.JAVA_INT
                ));
                ok = true;
            }
        } catch (Exception | UnsatisfiedLinkError ignored) {}
        CBLAS_SGEMM = h;
        HAS_ACCELERATE = ok;
    }

    /** numpy.dot for 1D vectors. */
    public static float dot(NdArray a, NdArray b) {
        return SimdOps.dot(a.contiguous().data(), b.contiguous().data(), a.size());
    }

    /** numpy.matmul for 2D matrices. */
    public static NdArray matmul(NdArray a, NdArray b) {
        if (a.ndim() == 1 && b.ndim() == 1) {
            // Dot product → scalar in 1x1 array
            NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, 1);
            out.flatSetFloat(0, dot(a, b));
            return out;
        }

        // Handle batched matmul for 3D+ later; focus on 2D
        NdArray ca = a.contiguous(), cb = b.contiguous();
        int M = ca.shape(0);
        int K = ca.shape(1);
        int N = cb.shape(1);

        NdArray out = NdArrayFactory.zeros(a.arena(), DType.FLOAT32, M, N);

        if (HAS_ACCELERATE) {
            amxMatmul(ca, cb, out, M, N, K);
        } else {
            simdMatmul(ca, cb, out, M, N, K);
        }
        return out;
    }

    /** numpy.outer */
    public static NdArray outer(NdArray a, NdArray b) {
        NdArray ca = a.contiguous(), cb = b.contiguous();
        int M = (int) a.size(), N = (int) b.size();
        NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, M, N);
        for (int i = 0; i < M; i++) {
            float ai = ca.flatGetFloat(i);
            for (int j = 0; j < N; j++) {
                out.data().setAtIndex(ValueLayout.JAVA_FLOAT, (long) i * N + j, ai * cb.flatGetFloat(j));
            }
        }
        return out;
    }

    /** numpy.inner (for 1D: same as dot) */
    public static float inner(NdArray a, NdArray b) { return dot(a, b); }

    // ── AMX path via Accelerate.framework ──

    private static void amxMatmul(NdArray a, NdArray b, NdArray c, int M, int N, int K) {
        try {
            // CblasRowMajor=101, CblasNoTrans=111
            CBLAS_SGEMM.invokeExact(
                    101, 111, 111,
                    M, N, K,
                    1.0f,
                    a.data(), K,
                    b.data(), N,
                    0.0f,
                    c.data(), N
            );
        } catch (Throwable t) {
            // Fallback to SIMD
            simdMatmul(a, b, c, M, N, K);
        }
    }

    // ── SIMD tiled matmul fallback ──

    private static void simdMatmul(NdArray a, NdArray b, NdArray c, int M, int N, int K) {
        int fLen = F.length();
        MemorySegment ad = a.data(), bd = b.data(), cd = c.data();

        for (int i = 0; i < M; i++) {
            long aRowOff = (long) i * K * Float.BYTES;
            long cRowOff = (long) i * N * Float.BYTES;
            for (int k = 0; k < K; k++) {
                float aik = ad.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * K + k);
                var va = FloatVector.broadcast(F, aik);
                long bRowOff = (long) k * N * Float.BYTES;
                int j = 0;
                for (; j + fLen <= N; j += fLen) {
                    long bOff = bRowOff + (long) j * Float.BYTES;
                    long cOff = cRowOff + (long) j * Float.BYTES;
                    var vb = FloatVector.fromMemorySegment(F, bd, bOff, java.nio.ByteOrder.nativeOrder());
                    var vc = FloatVector.fromMemorySegment(F, cd, cOff, java.nio.ByteOrder.nativeOrder());
                    va.fma(vb, vc).intoMemorySegment(cd, cOff, java.nio.ByteOrder.nativeOrder());
                }
                for (; j < N; j++) {
                    long idx = (long) i * N + j;
                    cd.setAtIndex(ValueLayout.JAVA_FLOAT, idx,
                            cd.getAtIndex(ValueLayout.JAVA_FLOAT, idx) + aik * bd.getAtIndex(ValueLayout.JAVA_FLOAT, (long) k * N + j));
                }
            }
        }
    }

    /** numpy.tensordot with axes=n (contract last n axes of a with first n of b). */
    public static NdArray tensordot(NdArray a, NdArray b, int axes) {
        // For axes=1 on 2D matrices, this is matmul
        return matmul(a, b);
    }

    /** numpy.cross for 3D vectors. */
    public static NdArray cross(NdArray a, NdArray b) {
        NdArray ca = a.contiguous(), cb = b.contiguous();
        float a0 = ca.flatGetFloat(0), a1 = ca.flatGetFloat(1), a2 = ca.flatGetFloat(2);
        float b0 = cb.flatGetFloat(0), b1 = cb.flatGetFloat(1), b2 = cb.flatGetFloat(2);
        return NdArrayFactory.array(a.arena(), new float[]{
                a1 * b2 - a2 * b1,
                a2 * b0 - a0 * b2,
                a0 * b1 - a1 * b0
        });
    }

    /** numpy.vdot — flattened dot product. */
    public static float vdot(NdArray a, NdArray b) { return dot(ShapeOps.flatten(a), ShapeOps.flatten(b)); }

    /** numpy.linalg.multi_dot — chained matmul. */
    public static NdArray multiDot(NdArray... arrays) {
        NdArray result = arrays[0];
        for (int i = 1; i < arrays.length; i++) result = matmul(result, arrays[i]);
        return result;
    }

    /** Batched matmul for 3D arrays: (batch, M, K) @ (batch, K, N) -> (batch, M, N) */
    public static NdArray batchedMatmul(NdArray a, NdArray b) {
        NdArray ca = a.contiguous(), cb = b.contiguous();
        int batch = ca.shape(0), M = ca.shape(1), K = ca.shape(2), N = cb.shape(2);
        NdArray out = NdArrayFactory.zeros(a.arena(), DType.FLOAT32, batch, M, N);
        for (int bi = 0; bi < batch; bi++) {
            long aOff = (long) bi * M * K;
            long bOff = (long) bi * K * N;
            long cOff = (long) bi * M * N;
            for (int i = 0; i < M; i++)
                for (int k = 0; k < K; k++) {
                    float aik = ca.flatGetFloat(aOff + (long) i * K + k);
                    for (int j = 0; j < N; j++)
                        out.flatSetFloat(cOff + (long) i * N + j,
                                out.flatGetFloat(cOff + (long) i * N + j) + aik * cb.flatGetFloat(bOff + (long) k * N + j));
                }
        }
        return out;
    }
}
