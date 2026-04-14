package com.sparrowlogic.numpy4j;

import java.lang.foreign.ValueLayout;
import java.util.Arrays;

/**
 * Universal functions (ufuncs) — element-wise operations matching NumPy.
 * Supports broadcasting for binary ops. SIMD-accelerated via SimdOps delegate.
 */
public final class Ufunc {
    private static final double LOG2 = Math.log(2);

    private Ufunc() {
    }

    // ── Binary arithmetic ──

    public static NdArray add(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, Float::sum);
    }

    public static NdArray subtract(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, (x, y) -> x - y);
    }

    public static NdArray multiply(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, (x, y) -> x * y);
    }

    public static NdArray divide(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, (x, y) -> x / y);
    }

    public static NdArray maximum(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, Math::max);
    }

    public static NdArray minimum(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, Math::min);
    }

    public static NdArray power(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, (x, y) -> (float) Math.pow(x, y));
    }

    public static NdArray mod(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, (x, y) -> x % y);
    }

    // ── Comparison ops ──

    public static NdArray equal(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, (x, y) -> x == y ? 1f : 0f);
    }

    public static NdArray notEqual(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, (x, y) -> x != y ? 1f : 0f);
    }

    public static NdArray greater(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, (x, y) -> x > y ? 1f : 0f);
    }

    public static NdArray greaterEqual(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, (x, y) -> x >= y ? 1f : 0f);
    }

    public static NdArray less(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, (x, y) -> x < y ? 1f : 0f);
    }

    public static NdArray lessEqual(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, (x, y) -> x <= y ? 1f : 0f);
    }

    // ── Logical ops ──

    public static NdArray logicalAnd(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, (x, y) -> (x != 0 && y != 0) ? 1f : 0f);
    }

    public static NdArray logicalOr(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, (x, y) -> (x != 0 || y != 0) ? 1f : 0f);
    }

    public static NdArray logicalXor(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, (x, y) -> x != 0 ^ y != 0 ? 1f : 0f);
    }

    public static NdArray logicalNot(final NdArray a) {
        return unaryOp(a, x -> x == 0 ? 1f : 0f);
    }

    // ── Type-check ops ──

    public static NdArray isnan(final NdArray a) {
        return unaryOp(a, x -> Float.isNaN(x) ? 1f : 0f);
    }

    public static NdArray isinf(final NdArray a) {
        return unaryOp(a, x -> Float.isInfinite(x) ? 1f : 0f);
    }

    public static NdArray isfinite(final NdArray a) {
        return unaryOp(a, x -> Float.isFinite(x) ? 1f : 0f);
    }

    // ── Scalar binary ──

    public static NdArray addScalar(final NdArray a, final float s) {
        return unaryOp(a, v -> v + s);
    }

    public static NdArray mulScalar(final NdArray a, final float s) {
        return unaryOp(a, v -> v * s);
    }

    public static NdArray divScalar(final NdArray a, final float s) {
        return unaryOp(a, v -> v / s);
    }

    // ── Unary math ──

    public static NdArray neg(final NdArray a) {
        return unaryOp(a, v -> -v);
    }

    public static NdArray abs(final NdArray a) {
        return unaryOp(a, v -> Math.abs(v));
    }

    public static NdArray sqrt(final NdArray a) {
        return unaryOp(a, v -> (float) Math.sqrt(v));
    }

    public static NdArray exp(final NdArray a) {
        return unaryOp(a, v -> (float) Math.exp(v));
    }

    public static NdArray log(final NdArray a) {
        return unaryOp(a, v -> (float) Math.log(v));
    }

    public static NdArray sin(final NdArray a) {
        return unaryOp(a, v -> (float) Math.sin(v));
    }

    public static NdArray cos(final NdArray a) {
        return unaryOp(a, v -> (float) Math.cos(v));
    }

    public static NdArray tan(final NdArray a) {
        return unaryOp(a, v -> (float) Math.tan(v));
    }

    public static NdArray tanh(final NdArray a) {
        return unaryOp(a, v -> (float) Math.tanh(v));
    }

    // ── Math extras ──

    public static NdArray square(final NdArray a) {
        return unaryOp(a, v -> v * v);
    }

    public static NdArray exp2(final NdArray a) {
        return unaryOp(a, v -> (float) Math.pow(2, v));
    }

    public static NdArray floorDivide(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, (x, y) -> (float) Math.floor(x / y));
    }

    public static NdArray fmod(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, (x, y) -> x % y);
    }

    public static NdArray floor(final NdArray a) {
        return unaryOp(a, v -> (float) Math.floor(v));
    }

    public static NdArray ceil(final NdArray a) {
        return unaryOp(a, v -> (float) Math.ceil(v));
    }

    public static NdArray round(final NdArray a) {
        return unaryOp(a, v -> (float) Math.rint(v));
    }

    public static NdArray sign(final NdArray a) {
        return unaryOp(a, v -> Math.signum(v));
    }

    public static NdArray sinh(final NdArray a) {
        return unaryOp(a, v -> (float) Math.sinh(v));
    }

    public static NdArray cosh(final NdArray a) {
        return unaryOp(a, v -> (float) Math.cosh(v));
    }

    public static NdArray arcsin(final NdArray a) {
        return unaryOp(a, v -> (float) Math.asin(v));
    }

    public static NdArray arccos(final NdArray a) {
        return unaryOp(a, v -> (float) Math.acos(v));
    }

    public static NdArray arctan(final NdArray a) {
        return unaryOp(a, v -> (float) Math.atan(v));
    }

    public static NdArray arctan2(final NdArray y, final NdArray x) {
        return broadcastBinary(y, x, (a, b) -> (float) Math.atan2(a, b));
    }

    public static NdArray log2(final NdArray a) {
        return unaryOp(a, v -> (float) (Math.log(v) / LOG2));
    }

    public static NdArray log10(final NdArray a) {
        return unaryOp(a, v -> (float) Math.log10(v));
    }

    public static NdArray log1p(final NdArray a) {
        return unaryOp(a, v -> (float) Math.log1p(v));
    }

    public static NdArray expm1(final NdArray a) {
        return unaryOp(a, v -> (float) Math.expm1(v));
    }

    /**
     * GELU (tanh approximation).
     */
    public static NdArray gelu(final NdArray a) {
        return unaryOp(
                a, x -> {
                    double xd = x;
                    return (float) (0.5 * xd * (1.0 + Math.tanh(
                            Math.sqrt(2.0 / Math.PI) * (xd + 0.044715 * xd * xd * xd))));
                }
        );
    }

    public static NdArray sigmoid(final NdArray a) {
        return unaryOp(a, x -> (float) (1.0 / (1.0 + Math.exp(-x))));
    }

    public static NdArray relu(final NdArray a) {
        return unaryOp(a, v -> Math.max(0f, v));
    }

    public static NdArray reciprocal(final NdArray a) {
        return unaryOp(a, v -> 1f / v);
    }

    public static NdArray clip(final NdArray a, final float min, final float max) {
        return unaryOp(a, v -> Math.max(min, Math.min(max, v)));
    }

    // ── Broadcasting binary dispatch ──

    private static NdArray broadcastBinary(final NdArray a, final NdArray b,
                                           final ScalarBinaryOp op) {
        if (Arrays.equals(a.shape(), b.shape())) {
            NdArray ca = a.contiguous();
            NdArray cb = b.contiguous();
            NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, a.shape());
            for (long i = 0; i < a.size(); i++) {
                out.flatSetFloat(i, op.apply(ca.flatGetFloat(i), cb.flatGetFloat(i)));
            }
            return out;
        }
        return broadcastBinarySlow(a, b, op);
    }

    private static NdArray broadcastBinarySlow(final NdArray a, final NdArray b,
                                               final ScalarBinaryOp op) {
        int[] outShape = ShapeUtils.broadcastShape(a.shape(), b.shape());
        int[] aStrides = ShapeUtils.broadcastStrides(a.shape(), a.strides(), outShape);
        int[] bStrides = ShapeUtils.broadcastStrides(b.shape(), b.strides(), outShape);
        long outSize = ShapeUtils.size(outShape);
        NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, outShape);

        for (long i = 0; i < outSize; i++) {
            int[] idx = ShapeUtils.unravelIndex(i, outShape);
            long aFlat = a.offset();
            long bFlat = b.offset();
            for (int d = 0; d < idx.length; d++) {
                aFlat += (long) idx[d] * aStrides[d];
                bFlat += (long) idx[d] * bStrides[d];
            }
            float va = a.data().getAtIndex(ValueLayout.JAVA_FLOAT, aFlat);
            float vb = b.data().getAtIndex(ValueLayout.JAVA_FLOAT, bFlat);
            out.flatSetFloat(i, op.apply(va, vb));
        }
        return out;
    }

    // ── Unary dispatch via SimdOps ──

    private static NdArray unaryOp(final NdArray a, final SimdOps.ScalarUnaryOp op) {
        NdArray ca = a.contiguous();
        NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, a.shape());
        SimdOps.get().unaryOp(ca.data(), out.data(), a.size(), op);
        return out;
    }

    @FunctionalInterface
    interface ScalarBinaryOp {
        float apply(float a, float b);
    }
}
