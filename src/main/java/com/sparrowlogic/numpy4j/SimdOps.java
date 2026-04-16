package com.sparrowlogic.numpy4j;

import org.jspecify.annotations.Nullable;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.logging.Logger;

/**
 * SIMD-accelerated operations with automatic fallback.
 * Uses the Vector API (jdk.incubator.vector) when available (Java 26+ with --enable-preview),
 * otherwise falls back to scalar loops. Detection happens once at class load time.
 */
final class SimdOps {

    private static final Logger LOG = Logger.getLogger(SimdOps.class.getName());
    private static final SimdOps INSTANCE;
    private static final boolean VECTOR_API_AVAILABLE;

    private final @Nullable VectorOpsDelegate delegate;

    static {
        VectorOpsDelegate d = null;
        boolean avail = false;
        try {
            Class.forName("jdk.incubator.vector.FloatVector");
            d = (VectorOpsDelegate) Class
                    .forName("com.sparrowlogic.numpy4j.VectorSimdOps")
                    .getDeclaredConstructor()
                    .newInstance();
            avail = true;
            LOG.info("Vector API available — SIMD acceleration enabled");
        } catch (final Throwable t) {
            LOG.info("Vector API not available — using scalar fallback: " + t.getMessage());
        }
        VECTOR_API_AVAILABLE = avail;
        INSTANCE = new SimdOps(d);
    }

    private SimdOps(final @Nullable VectorOpsDelegate delegate) {
        this.delegate = delegate;
    }

    static SimdOps get() {
        return INSTANCE;
    }

    static boolean isVectorApiAvailable() {
        return VECTOR_API_AVAILABLE;
    }

    // ── Unary ops ──

    void unaryOp(final MemorySegment src, final MemorySegment dst, final long n,
                 final ScalarUnaryOp op) {
        if (this.delegate != null) {
            this.delegate.unaryOp(src, dst, n, op);
            return;
        }
        for (long i = 0; i < n; i++) {
            dst.setAtIndex(ValueLayout.JAVA_FLOAT, i,
                    op.apply(src.getAtIndex(ValueLayout.JAVA_FLOAT, i)));
        }
    }

    // ── Reductions ──

    float reduceSum(final MemorySegment src, final long n) {
        if (this.delegate != null) {
            return this.delegate.reduceSum(src, n);
        }
        float sum = 0;
        for (long i = 0; i < n; i++) {
            sum += src.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        }
        return sum;
    }

    float reduceMax(final MemorySegment src, final long n) {
        if (this.delegate != null) {
            return this.delegate.reduceMax(src, n);
        }
        float max = Float.NEGATIVE_INFINITY;
        for (long i = 0; i < n; i++) {
            max = Math.max(max, src.getAtIndex(ValueLayout.JAVA_FLOAT, i));
        }
        return max;
    }

    float reduceMin(final MemorySegment src, final long n) {
        if (this.delegate != null) {
            return this.delegate.reduceMin(src, n);
        }
        float min = Float.POSITIVE_INFINITY;
        for (long i = 0; i < n; i++) {
            min = Math.min(min, src.getAtIndex(ValueLayout.JAVA_FLOAT, i));
        }
        return min;
    }

    // ── Dot product ──

    float dot(final MemorySegment a, final MemorySegment b, final long n) {
        if (this.delegate != null) {
            return this.delegate.dot(a, b, n);
        }
        float sum = 0;
        for (long i = 0; i < n; i++) {
            sum += a.getAtIndex(ValueLayout.JAVA_FLOAT, i)
                    * b.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        }
        return sum;
    }

    // ── Matmul row (SIMD-tiled) ──

    @SuppressWarnings("checkstyle:ParameterNumber")
    void matmulRow(final MemorySegment ad, final MemorySegment bd, final MemorySegment cd,
                   final int i, final int n, final int k) {
        if (this.delegate != null) {
            this.delegate.matmulRow(ad, bd, cd, i, n, k);
            return;
        }
        scalarMatmulRow(ad, bd, cd, i, n, k);
    }

    @SuppressWarnings("checkstyle:ParameterNumber")
    static void scalarMatmulRow(final MemorySegment ad, final MemorySegment bd,
                                final MemorySegment cd,
                                final int i, final int n, final int k) {
        for (int ki = 0; ki < k; ki++) {
            float aik = ad.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * k + ki);
            for (int j = 0; j < n; j++) {
                long idx = (long) i * n + j;
                cd.setAtIndex(ValueLayout.JAVA_FLOAT, idx,
                        cd.getAtIndex(ValueLayout.JAVA_FLOAT, idx)
                                + aik * bd.getAtIndex(ValueLayout.JAVA_FLOAT,
                                (long) ki * n + j));
            }
        }
    }

    /** Interface for the Vector API delegate — loaded reflectively. */
    interface VectorOpsDelegate {

        void unaryOp(MemorySegment src, MemorySegment dst, long n, ScalarUnaryOp op);

        float reduceSum(MemorySegment src, long n);

        float reduceMax(MemorySegment src, long n);

        float reduceMin(MemorySegment src, long n);

        float dot(MemorySegment a, MemorySegment b, long n);

        @SuppressWarnings("checkstyle:ParameterNumber")
        void matmulRow(MemorySegment ad, MemorySegment bd, MemorySegment cd,
                       int i, int n, int k);
    }

    @FunctionalInterface
    interface ScalarUnaryOp {
        float apply(float a);
    }
}
