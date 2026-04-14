package com.sparrowlogic.numpy4j;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Vector API (JEP 529) implementation of SIMD operations.
 * Loaded reflectively by {@link SimdOps} — never referenced directly from Java 25 code.
 */
final class VectorSimdOps implements SimdOps.VectorOpsDelegate {

    private static final VectorSpecies<Float> F = FloatVector.SPECIES_PREFERRED;

    @Override
    public void unaryOp(final MemorySegment src, final MemorySegment dst, final long n,
                        final SimdOps.ScalarUnaryOp op) {
        for (long i = 0; i < n; i++) {
            dst.setAtIndex(ValueLayout.JAVA_FLOAT, i,
                    op.apply(src.getAtIndex(ValueLayout.JAVA_FLOAT, i)));
        }
    }

    @Override
    public float reduceSum(final MemorySegment src, final long n) {
        var acc = FloatVector.zero(F);
        long i = 0;
        int fLen = F.length();
        for (; i + fLen <= n; i += fLen) {
            var v = FloatVector.fromMemorySegment(F, src,
                    i * Float.BYTES, ByteOrder.nativeOrder());
            acc = acc.add(v);
        }
        float sum = acc.reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) {
            sum += src.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        }
        return sum;
    }

    @Override
    public float reduceMax(final MemorySegment src, final long n) {
        float max = Float.NEGATIVE_INFINITY;
        long i = 0;
        int fLen = F.length();
        if (n >= fLen) {
            var acc = FloatVector.fromMemorySegment(F, src, 0, ByteOrder.nativeOrder());
            i = fLen;
            for (; i + fLen <= n; i += fLen) {
                var v = FloatVector.fromMemorySegment(F, src,
                        i * Float.BYTES, ByteOrder.nativeOrder());
                acc = acc.max(v);
            }
            max = acc.reduceLanes(VectorOperators.MAX);
        }
        for (; i < n; i++) {
            max = Math.max(max, src.getAtIndex(ValueLayout.JAVA_FLOAT, i));
        }
        return max;
    }

    @Override
    public float reduceMin(final MemorySegment src, final long n) {
        float min = Float.POSITIVE_INFINITY;
        long i = 0;
        int fLen = F.length();
        if (n >= fLen) {
            var acc = FloatVector.fromMemorySegment(F, src, 0, ByteOrder.nativeOrder());
            i = fLen;
            for (; i + fLen <= n; i += fLen) {
                var v = FloatVector.fromMemorySegment(F, src,
                        i * Float.BYTES, ByteOrder.nativeOrder());
                acc = acc.min(v);
            }
            min = acc.reduceLanes(VectorOperators.MIN);
        }
        for (; i < n; i++) {
            min = Math.min(min, src.getAtIndex(ValueLayout.JAVA_FLOAT, i));
        }
        return min;
    }

    @Override
    public float dot(final MemorySegment a, final MemorySegment b, final long n) {
        var acc = FloatVector.zero(F);
        long i = 0;
        int fLen = F.length();
        for (; i + fLen <= n; i += fLen) {
            var va = FloatVector.fromMemorySegment(F, a,
                    i * Float.BYTES, ByteOrder.nativeOrder());
            var vb = FloatVector.fromMemorySegment(F, b,
                    i * Float.BYTES, ByteOrder.nativeOrder());
            acc = va.fma(vb, acc);
        }
        float sum = acc.reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) {
            sum += a.getAtIndex(ValueLayout.JAVA_FLOAT, i)
                    * b.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        }
        return sum;
    }

    @Override
    @SuppressWarnings("checkstyle:ParameterNumber")
    public void matmulRow(final MemorySegment ad, final MemorySegment bd,
                          final MemorySegment cd,
                          final int i, final int n, final int k) {
        int fLen = F.length();
        long cRowOff = (long) i * n * Float.BYTES;
        for (int ki = 0; ki < k; ki++) {
            float aik = ad.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * k + ki);
            var va = FloatVector.broadcast(F, aik);
            long bRowOff = (long) ki * n * Float.BYTES;
            int j = 0;
            for (; j + fLen <= n; j += fLen) {
                long bOff = bRowOff + (long) j * Float.BYTES;
                long cOff = cRowOff + (long) j * Float.BYTES;
                var vb = FloatVector.fromMemorySegment(F, bd, bOff, ByteOrder.nativeOrder());
                var vc = FloatVector.fromMemorySegment(F, cd, cOff, ByteOrder.nativeOrder());
                va.fma(vb, vc).intoMemorySegment(cd, cOff, ByteOrder.nativeOrder());
            }
            for (; j < n; j++) {
                long idx = (long) i * n + j;
                cd.setAtIndex(ValueLayout.JAVA_FLOAT, idx,
                        cd.getAtIndex(ValueLayout.JAVA_FLOAT, idx)
                                + aik * bd.getAtIndex(ValueLayout.JAVA_FLOAT,
                                (long) ki * n + j));
            }
        }
    }
}
