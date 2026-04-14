package com.sparrowlogic.numpy4j;

import jdk.incubator.vector.FloatVector;

import java.lang.foreign.ValueLayout;
import java.util.Arrays;

/**
 * Universal functions (ufuncs) — element-wise operations matching NumPy.
 * Supports broadcasting for binary ops. SIMD-accelerated for contiguous same-shape.
 */
public final class Ufunc {
    private Ufunc() {}

    @FunctionalInterface
    interface ScalarBinaryOp { float apply(float a, float b); }
    @FunctionalInterface
    interface ScalarUnaryOp { float apply(float a); }

    // ── Binary arithmetic ──

    public static NdArray add(NdArray a, NdArray b) { return broadcastBinary(a, b, Float::sum); }
    public static NdArray subtract(NdArray a, NdArray b) { return broadcastBinary(a, b, (x, y) -> x - y); }
    public static NdArray multiply(NdArray a, NdArray b) { return broadcastBinary(a, b, (x, y) -> x * y); }
    public static NdArray divide(NdArray a, NdArray b) { return broadcastBinary(a, b, (x, y) -> x / y); }
    public static NdArray maximum(NdArray a, NdArray b) { return broadcastBinary(a, b, Math::max); }
    public static NdArray minimum(NdArray a, NdArray b) { return broadcastBinary(a, b, Math::min); }
    public static NdArray power(NdArray a, NdArray b) { return broadcastBinary(a, b, (x, y) -> (float) Math.pow(x, y)); }
    public static NdArray mod(NdArray a, NdArray b) { return broadcastBinary(a, b, (x, y) -> x % y); }

    // ── Comparison ops ──

    public static NdArray equal(NdArray a, NdArray b) { return broadcastBinary(a, b, (x, y) -> x == y ? 1f : 0f); }
    public static NdArray notEqual(NdArray a, NdArray b) { return broadcastBinary(a, b, (x, y) -> x != y ? 1f : 0f); }
    public static NdArray greater(NdArray a, NdArray b) { return broadcastBinary(a, b, (x, y) -> x > y ? 1f : 0f); }
    public static NdArray greaterEqual(NdArray a, NdArray b) { return broadcastBinary(a, b, (x, y) -> x >= y ? 1f : 0f); }
    public static NdArray less(NdArray a, NdArray b) { return broadcastBinary(a, b, (x, y) -> x < y ? 1f : 0f); }
    public static NdArray lessEqual(NdArray a, NdArray b) { return broadcastBinary(a, b, (x, y) -> x <= y ? 1f : 0f); }

    // ── Logical ops ──

    public static NdArray logicalAnd(NdArray a, NdArray b) { return broadcastBinary(a, b, (x, y) -> (x != 0 && y != 0) ? 1f : 0f); }
    public static NdArray logicalOr(NdArray a, NdArray b) { return broadcastBinary(a, b, (x, y) -> (x != 0 || y != 0) ? 1f : 0f); }
    public static NdArray logicalXor(NdArray a, NdArray b) { return broadcastBinary(a, b, (x, y) -> ((x != 0) ^ (y != 0)) ? 1f : 0f); }
    public static NdArray logicalNot(NdArray a) { return unaryOpScalar(a, x -> x == 0 ? 1f : 0f); }

    // ── Type-check ops ──

    public static NdArray isnan(NdArray a) { return unaryOpScalar(a, x -> Float.isNaN(x) ? 1f : 0f); }
    public static NdArray isinf(NdArray a) { return unaryOpScalar(a, x -> Float.isInfinite(x) ? 1f : 0f); }
    public static NdArray isfinite(NdArray a) { return unaryOpScalar(a, x -> Float.isFinite(x) ? 1f : 0f); }

    // ── Scalar binary ──

    public static NdArray addScalar(NdArray a, float s) { return unaryOp(a, v -> v.add(s)); }
    public static NdArray mulScalar(NdArray a, float s) { return unaryOp(a, v -> v.mul(s)); }
    public static NdArray divScalar(NdArray a, float s) { return unaryOp(a, v -> v.div(s)); }

    // ── Unary math ──

    public static NdArray neg(NdArray a) { return unaryOp(a, FloatVector::neg); }
    public static NdArray abs(NdArray a) { return unaryOp(a, v -> v.abs()); }
    public static NdArray sqrt(NdArray a) { return unaryOp(a, v -> v.sqrt()); }
    public static NdArray exp(NdArray a) { return unaryOpScalar(a, v -> (float) Math.exp(v)); }
    public static NdArray log(NdArray a) { return unaryOpScalar(a, v -> (float) Math.log(v)); }
    public static NdArray sin(NdArray a) { return unaryOpScalar(a, v -> (float) Math.sin(v)); }
    public static NdArray cos(NdArray a) { return unaryOpScalar(a, v -> (float) Math.cos(v)); }
    public static NdArray tan(NdArray a) { return unaryOpScalar(a, v -> (float) Math.tan(v)); }
    public static NdArray tanh(NdArray a) { return unaryOpScalar(a, v -> (float) Math.tanh(v)); }

    // ── Math extras ──

    public static NdArray square(NdArray a) { return unaryOp(a, v -> v.mul(v)); }
    public static NdArray exp2(NdArray a) { return unaryOpScalar(a, v -> (float) Math.pow(2, v)); }
    public static NdArray floorDivide(NdArray a, NdArray b) { return broadcastBinary(a, b, (x, y) -> (float) Math.floor(x / y)); }
    public static NdArray fmod(NdArray a, NdArray b) { return broadcastBinary(a, b, (x, y) -> x % y); }

    public static NdArray floor(NdArray a) { return unaryOpScalar(a, v -> (float) Math.floor(v)); }
    public static NdArray ceil(NdArray a) { return unaryOpScalar(a, v -> (float) Math.ceil(v)); }
    public static NdArray round(NdArray a) { return unaryOpScalar(a, v -> (float) Math.rint(v)); }
    public static NdArray sign(NdArray a) { return unaryOpScalar(a, v -> (float) Math.signum(v)); }
    public static NdArray sinh(NdArray a) { return unaryOpScalar(a, v -> (float) Math.sinh(v)); }
    public static NdArray cosh(NdArray a) { return unaryOpScalar(a, v -> (float) Math.cosh(v)); }
    public static NdArray arcsin(NdArray a) { return unaryOpScalar(a, v -> (float) Math.asin(v)); }
    public static NdArray arccos(NdArray a) { return unaryOpScalar(a, v -> (float) Math.acos(v)); }
    public static NdArray arctan(NdArray a) { return unaryOpScalar(a, v -> (float) Math.atan(v)); }
    public static NdArray arctan2(NdArray y, NdArray x) { return broadcastBinary(y, x, (a, b) -> (float) Math.atan2(a, b)); }
    public static NdArray log2(NdArray a) { return unaryOpScalar(a, v -> (float) (Math.log(v) / Math.log(2))); }
    public static NdArray log10(NdArray a) { return unaryOpScalar(a, v -> (float) Math.log10(v)); }
    public static NdArray log1p(NdArray a) { return unaryOpScalar(a, v -> (float) Math.log1p(v)); }
    public static NdArray expm1(NdArray a) { return unaryOpScalar(a, v -> (float) Math.expm1(v)); }

    /** GELU (tanh approximation) */
    public static NdArray gelu(NdArray a) {
        return unaryOpScalar(a, x -> {
            double xd = x;
            return (float) (0.5 * xd * (1.0 + Math.tanh(Math.sqrt(2.0 / Math.PI) * (xd + 0.044715 * xd * xd * xd))));
        });
    }

    public static NdArray sigmoid(NdArray a) { return unaryOpScalar(a, x -> (float) (1.0 / (1.0 + Math.exp(-x)))); }
    public static NdArray relu(NdArray a) { return unaryOp(a, v -> v.max(FloatVector.zero(SimdOps.F_SPECIES))); }
    public static NdArray reciprocal(NdArray a) { return unaryOp(a, v -> FloatVector.broadcast(SimdOps.F_SPECIES, 1f).div(v)); }

    public static NdArray clip(NdArray a, float min, float max) {
        return unaryOp(a, v -> v.max(FloatVector.broadcast(SimdOps.F_SPECIES, min))
                                .min(FloatVector.broadcast(SimdOps.F_SPECIES, max)));
    }

    // ── Broadcasting binary dispatch ──

    private static NdArray broadcastBinary(NdArray a, NdArray b, ScalarBinaryOp op) {
        if (Arrays.equals(a.shape(), b.shape())) {
            // Fast path: same shape, use SIMD where possible
            NdArray ca = a.contiguous(), cb = b.contiguous();
            NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, a.shape());
            for (long i = 0; i < a.size(); i++)
                out.flatSetFloat(i, op.apply(ca.flatGetFloat(i), cb.flatGetFloat(i)));
            return out;
        }
        // Broadcasting path
        int[] outShape = ShapeUtils.broadcastShape(a.shape(), b.shape());
        int[] aStrides = ShapeUtils.broadcastStrides(a.shape(), a.strides(), outShape);
        int[] bStrides = ShapeUtils.broadcastStrides(b.shape(), b.strides(), outShape);
        long outSize = ShapeUtils.size(outShape);
        NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, outShape);

        for (long i = 0; i < outSize; i++) {
            int[] idx = ShapeUtils.unravelIndex(i, outShape);
            long aFlat = a.offset();
            for (int d = 0; d < idx.length; d++) aFlat += (long) idx[d] * aStrides[d];
            long bFlat = b.offset();
            for (int d = 0; d < idx.length; d++) bFlat += (long) idx[d] * bStrides[d];
            float va = a.data().getAtIndex(ValueLayout.JAVA_FLOAT, aFlat);
            float vb = b.data().getAtIndex(ValueLayout.JAVA_FLOAT, bFlat);
            out.flatSetFloat(i, op.apply(va, vb));
        }
        return out;
    }

    // ── SIMD unary dispatch ──

    private static NdArray unaryOp(NdArray a, SimdOps.UnaryOp op) {
        NdArray ca = a.contiguous();
        NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, a.shape());
        SimdOps.unaryOp(ca.data(), out.data(), a.size(), op);
        return out;
    }

    private static NdArray unaryOpScalar(NdArray a, ScalarUnaryOp op) {
        NdArray ca = a.contiguous();
        NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, a.shape());
        for (long i = 0; i < a.size(); i++)
            out.flatSetFloat(i, op.apply(ca.flatGetFloat(i)));
        return out;
    }
}
