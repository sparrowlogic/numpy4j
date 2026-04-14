package com.sparrowlogic.numpy4j;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * SIMD-accelerated element-wise operations using Java Vector API (preview).
 * All ops work on contiguous FLOAT32 MemorySegments.
 */
public final class SimdOps {
    private SimdOps() {}

    static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int F_LEN = F_SPECIES.length();

    // ── Binary ops ──

    @FunctionalInterface
    public interface BinaryOp {
        FloatVector apply(FloatVector a, FloatVector b);
    }

    public static void binaryOp(MemorySegment a, MemorySegment b, MemorySegment out, long n, BinaryOp op) {
        long i = 0;
        for (; i + F_LEN <= n; i += F_LEN) {
            var va = FloatVector.fromMemorySegment(F_SPECIES, a, i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
            var vb = FloatVector.fromMemorySegment(F_SPECIES, b, i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
            op.apply(va, vb).intoMemorySegment(out, i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
        }
        // Scalar tail
        for (; i < n; i++) {
            float fa = a.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float fb = b.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            // Fallback: use lane 0 of vector op on broadcast
            out.setAtIndex(ValueLayout.JAVA_FLOAT, i, scalarBinaryFallback(fa, fb, op));
        }
    }

    private static float scalarBinaryFallback(float a, float b, BinaryOp op) {
        var va = FloatVector.broadcast(F_SPECIES, a);
        var vb = FloatVector.broadcast(F_SPECIES, b);
        return op.apply(va, vb).lane(0);
    }

    // ── Unary ops ──

    @FunctionalInterface
    public interface UnaryOp {
        FloatVector apply(FloatVector a);
    }

    public static void unaryOp(MemorySegment src, MemorySegment dst, long n, UnaryOp op) {
        long i = 0;
        for (; i + F_LEN <= n; i += F_LEN) {
            var v = FloatVector.fromMemorySegment(F_SPECIES, src, i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
            op.apply(v).intoMemorySegment(dst, i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
        }
        for (; i < n; i++) {
            float f = src.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            dst.setAtIndex(ValueLayout.JAVA_FLOAT, i, scalarUnaryFallback(f, op));
        }
    }

    private static float scalarUnaryFallback(float a, UnaryOp op) {
        return op.apply(FloatVector.broadcast(F_SPECIES, a)).lane(0);
    }

    // ── Reduction ──

    public static float reduceSum(MemorySegment src, long n) {
        FloatVector acc = FloatVector.zero(F_SPECIES);
        long i = 0;
        for (; i + F_LEN <= n; i += F_LEN) {
            var v = FloatVector.fromMemorySegment(F_SPECIES, src, i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
            acc = acc.add(v);
        }
        float sum = acc.reduceLanes(jdk.incubator.vector.VectorOperators.ADD);
        for (; i < n; i++) sum += src.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        return sum;
    }

    public static float reduceMax(MemorySegment src, long n) {
        float max = Float.NEGATIVE_INFINITY;
        long i = 0;
        if (n >= F_LEN) {
            var acc = FloatVector.fromMemorySegment(F_SPECIES, src, 0, java.nio.ByteOrder.nativeOrder());
            i = F_LEN;
            for (; i + F_LEN <= n; i += F_LEN) {
                var v = FloatVector.fromMemorySegment(F_SPECIES, src, i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
                acc = acc.max(v);
            }
            max = acc.reduceLanes(jdk.incubator.vector.VectorOperators.MAX);
        }
        for (; i < n; i++) max = Math.max(max, src.getAtIndex(ValueLayout.JAVA_FLOAT, i));
        return max;
    }

    public static float reduceMin(MemorySegment src, long n) {
        float min = Float.POSITIVE_INFINITY;
        long i = 0;
        if (n >= F_LEN) {
            var acc = FloatVector.fromMemorySegment(F_SPECIES, src, 0, java.nio.ByteOrder.nativeOrder());
            i = F_LEN;
            for (; i + F_LEN <= n; i += F_LEN) {
                var v = FloatVector.fromMemorySegment(F_SPECIES, src, i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
                acc = acc.min(v);
            }
            min = acc.reduceLanes(jdk.incubator.vector.VectorOperators.MIN);
        }
        for (; i < n; i++) min = Math.min(min, src.getAtIndex(ValueLayout.JAVA_FLOAT, i));
        return min;
    }

    /** Dot product of two float segments. */
    public static float dot(MemorySegment a, MemorySegment b, long n) {
        FloatVector acc = FloatVector.zero(F_SPECIES);
        long i = 0;
        for (; i + F_LEN <= n; i += F_LEN) {
            var va = FloatVector.fromMemorySegment(F_SPECIES, a, i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
            var vb = FloatVector.fromMemorySegment(F_SPECIES, b, i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
            acc = va.fma(vb, acc);
        }
        float sum = acc.reduceLanes(jdk.incubator.vector.VectorOperators.ADD);
        for (; i < n; i++) {
            sum += a.getAtIndex(ValueLayout.JAVA_FLOAT, i) * b.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        }
        return sum;
    }
}
