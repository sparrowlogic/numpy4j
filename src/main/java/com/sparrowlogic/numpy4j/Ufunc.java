package com.sparrowlogic.numpy4j;

import java.lang.foreign.ValueLayout;
import java.util.Arrays;

/**
 * Universal functions (ufuncs) — element-wise operations matching NumPy.
 *
 * <p>All binary operations support NumPy-style broadcasting. Unary operations
 * are SIMD-accelerated via {@link SimdOps} when the Vector API is available.</p>
 *
 * <p>All operations produce FLOAT32 output arrays.</p>
 *
 * @see NdArray
 */
public final class Ufunc {
    private static final double LOG2 = Math.log(2);

    private Ufunc() {
    }

    // ── Binary arithmetic ──

    /**
     * Element-wise addition ({@code numpy.add}).
     * @param a first operand
     * @param b second operand
     * @return a + b
     */
    public static NdArray add(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, Float::sum);
    }

    /**
     * Element-wise subtraction ({@code numpy.subtract}).
     * @param a first operand
     * @param b second operand
     * @return a - b
     */
    public static NdArray subtract(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, (x, y) -> x - y);
    }

    /**
     * Element-wise multiplication ({@code numpy.multiply}).
     * @param a first operand
     * @param b second operand
     * @return a * b
     */
    public static NdArray multiply(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, (x, y) -> x * y);
    }

    /**
     * Element-wise true division ({@code numpy.divide}).
     * @param a dividend
     * @param b divisor
     * @return a / b
     */
    public static NdArray divide(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, (x, y) -> x / y);
    }

    /**
     * Element-wise maximum ({@code numpy.maximum}).
     * @param a first operand
     * @param b second operand
     * @return max(a, b)
     */
    public static NdArray maximum(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, Math::max);
    }

    /**
     * Element-wise minimum ({@code numpy.minimum}).
     * @param a first operand
     * @param b second operand
     * @return min(a, b)
     */
    public static NdArray minimum(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, Math::min);
    }

    /**
     * Element-wise exponentiation ({@code numpy.power}).
     * @param a base
     * @param b exponent
     * @return a ** b
     */
    public static NdArray power(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, (x, y) -> (float) Math.pow(x, y));
    }

    /**
     * Element-wise remainder ({@code numpy.mod}).
     * @param a dividend
     * @param b divisor
     * @return a % b
     */
    public static NdArray mod(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, (x, y) -> x % y);
    }

    // ── Comparison ops ──

    /**
     * Element-wise equality ({@code numpy.equal}). Returns 1.0 where equal, 0.0 otherwise.
     * @param a first operand
     * @param b second operand
     * @return boolean mask as FLOAT32
     */
    public static NdArray equal(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, (x, y) -> x == y ? 1f : 0f);
    }

    /**
     * Element-wise inequality ({@code numpy.not_equal}).
     * @param a first operand
     * @param b second operand
     * @return boolean mask as FLOAT32
     */
    public static NdArray notEqual(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, (x, y) -> x != y ? 1f : 0f);
    }

    /**
     * Element-wise greater-than ({@code numpy.greater}).
     * @param a first operand
     * @param b second operand
     * @return boolean mask as FLOAT32
     */
    public static NdArray greater(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, (x, y) -> x > y ? 1f : 0f);
    }

    /**
     * Element-wise greater-or-equal ({@code numpy.greater_equal}).
     * @param a first operand
     * @param b second operand
     * @return boolean mask as FLOAT32
     */
    public static NdArray greaterEqual(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, (x, y) -> x >= y ? 1f : 0f);
    }

    /**
     * Element-wise less-than ({@code numpy.less}).
     * @param a first operand
     * @param b second operand
     * @return boolean mask as FLOAT32
     */
    public static NdArray less(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, (x, y) -> x < y ? 1f : 0f);
    }

    /**
     * Element-wise less-or-equal ({@code numpy.less_equal}).
     * @param a first operand
     * @param b second operand
     * @return boolean mask as FLOAT32
     */
    public static NdArray lessEqual(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, (x, y) -> x <= y ? 1f : 0f);
    }

    // ── Logical ops ──

    /**
     * Element-wise logical AND ({@code numpy.logical_and}). Non-zero is true.
     * @param a first operand
     * @param b second operand
     * @return boolean mask as FLOAT32
     */
    public static NdArray logicalAnd(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, (x, y) -> (x != 0 && y != 0) ? 1f : 0f);
    }

    /**
     * Element-wise logical OR ({@code numpy.logical_or}).
     * @param a first operand
     * @param b second operand
     * @return boolean mask as FLOAT32
     */
    public static NdArray logicalOr(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, (x, y) -> (x != 0 || y != 0) ? 1f : 0f);
    }

    /**
     * Element-wise logical XOR ({@code numpy.logical_xor}).
     * @param a first operand
     * @param b second operand
     * @return boolean mask as FLOAT32
     */
    public static NdArray logicalXor(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, (x, y) -> x != 0 ^ y != 0 ? 1f : 0f);
    }

    /**
     * Element-wise logical NOT ({@code numpy.logical_not}).
     * @param a input array
     * @return boolean mask as FLOAT32
     */
    public static NdArray logicalNot(final NdArray a) {
        return unaryOp(a, x -> x == 0 ? 1f : 0f);
    }

    // ── Type-check ops ──

    /**
     * Element-wise NaN test ({@code numpy.isnan}).
     * @param a input array
     * @return 1.0 where NaN, 0.0 otherwise
     */
    public static NdArray isnan(final NdArray a) {
        return unaryOp(a, x -> Float.isNaN(x) ? 1f : 0f);
    }

    /**
     * Element-wise infinity test ({@code numpy.isinf}).
     * @param a input array
     * @return 1.0 where ±inf, 0.0 otherwise
     */
    public static NdArray isinf(final NdArray a) {
        return unaryOp(a, x -> Float.isInfinite(x) ? 1f : 0f);
    }

    /**
     * Element-wise finiteness test ({@code numpy.isfinite}).
     * @param a input array
     * @return 1.0 where finite, 0.0 otherwise
     */
    public static NdArray isfinite(final NdArray a) {
        return unaryOp(a, x -> Float.isFinite(x) ? 1f : 0f);
    }

    // ── Scalar binary ──

    /**
     * Adds a scalar to every element.
     * @param a input array
     * @param s scalar addend
     * @return a + s
     */
    public static NdArray addScalar(final NdArray a, final float s) {
        return unaryOp(a, v -> v + s);
    }

    /**
     * Multiplies every element by a scalar.
     * @param a input array
     * @param s scalar factor
     * @return a * s
     */
    public static NdArray mulScalar(final NdArray a, final float s) {
        return unaryOp(a, v -> v * s);
    }

    /**
     * Divides every element by a scalar.
     * @param a input array
     * @param s scalar divisor
     * @return a / s
     */
    public static NdArray divScalar(final NdArray a, final float s) {
        return unaryOp(a, v -> v / s);
    }

    // ── Unary math ──

    /**
     * Element-wise negation ({@code numpy.negative}).
     * @param a input array
     * @return -a
     */
    public static NdArray neg(final NdArray a) {
        return unaryOp(a, v -> -v);
    }

    /**
     * Element-wise absolute value ({@code numpy.abs}).
     * @param a input array
     * @return |a|
     */
    public static NdArray abs(final NdArray a) {
        return unaryOp(a, v -> Math.abs(v));
    }

    /**
     * Element-wise square root ({@code numpy.sqrt}).
     * @param a input array
     * @return √a
     */
    public static NdArray sqrt(final NdArray a) {
        return unaryOp(a, v -> (float) Math.sqrt(v));
    }

    /**
     * Element-wise exponential ({@code numpy.exp}).
     * @param a input array
     * @return e^a
     */
    public static NdArray exp(final NdArray a) {
        return unaryOp(a, v -> (float) Math.exp(v));
    }

    /**
     * Element-wise natural logarithm ({@code numpy.log}).
     * @param a input array
     * @return ln(a)
     */
    public static NdArray log(final NdArray a) {
        return unaryOp(a, v -> (float) Math.log(v));
    }

    /**
     * Element-wise sine ({@code numpy.sin}).
     * @param a input array (radians)
     * @return sin(a)
     */
    public static NdArray sin(final NdArray a) {
        return unaryOp(a, v -> (float) Math.sin(v));
    }

    /**
     * Element-wise cosine ({@code numpy.cos}).
     * @param a input array (radians)
     * @return cos(a)
     */
    public static NdArray cos(final NdArray a) {
        return unaryOp(a, v -> (float) Math.cos(v));
    }

    /**
     * Element-wise tangent ({@code numpy.tan}).
     * @param a input array (radians)
     * @return tan(a)
     */
    public static NdArray tan(final NdArray a) {
        return unaryOp(a, v -> (float) Math.tan(v));
    }

    /**
     * Element-wise hyperbolic tangent ({@code numpy.tanh}).
     * @param a input array
     * @return tanh(a)
     */
    public static NdArray tanh(final NdArray a) {
        return unaryOp(a, v -> (float) Math.tanh(v));
    }

    // ── Math extras ──

    /**
     * Element-wise square ({@code numpy.square}).
     * @param a input array
     * @return a²
     */
    public static NdArray square(final NdArray a) {
        return unaryOp(a, v -> v * v);
    }

    /**
     * Element-wise base-2 exponential ({@code numpy.exp2}).
     * @param a input array
     * @return 2^a
     */
    public static NdArray exp2(final NdArray a) {
        return unaryOp(a, v -> (float) Math.pow(2, v));
    }

    /**
     * Element-wise floor division ({@code numpy.floor_divide}).
     * @param a dividend
     * @param b divisor
     * @return floor(a / b)
     */
    public static NdArray floorDivide(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, (x, y) -> (float) Math.floor(x / y));
    }

    /**
     * Element-wise float modulus ({@code numpy.fmod}).
     * @param a dividend
     * @param b divisor
     * @return fmod(a, b)
     */
    public static NdArray fmod(final NdArray a, final NdArray b) {
        return broadcastBinary(a, b, (x, y) -> x % y);
    }

    /**
     * Element-wise floor ({@code numpy.floor}).
     * @param a input array
     * @return floor(a)
     */
    public static NdArray floor(final NdArray a) {
        return unaryOp(a, v -> (float) Math.floor(v));
    }

    /**
     * Element-wise ceiling ({@code numpy.ceil}).
     * @param a input array
     * @return ceil(a)
     */
    public static NdArray ceil(final NdArray a) {
        return unaryOp(a, v -> (float) Math.ceil(v));
    }

    /**
     * Element-wise rounding to nearest even ({@code numpy.round}).
     * @param a input array
     * @return round(a)
     */
    public static NdArray round(final NdArray a) {
        return unaryOp(a, v -> (float) Math.rint(v));
    }

    /**
     * Element-wise sign function ({@code numpy.sign}). Returns -1, 0, or 1.
     * @param a input array
     * @return sign(a)
     */
    public static NdArray sign(final NdArray a) {
        return unaryOp(a, v -> Math.signum(v));
    }

    /**
     * Element-wise hyperbolic sine ({@code numpy.sinh}).
     * @param a input array
     * @return sinh(a)
     */
    public static NdArray sinh(final NdArray a) {
        return unaryOp(a, v -> (float) Math.sinh(v));
    }

    /**
     * Element-wise hyperbolic cosine ({@code numpy.cosh}).
     * @param a input array
     * @return cosh(a)
     */
    public static NdArray cosh(final NdArray a) {
        return unaryOp(a, v -> (float) Math.cosh(v));
    }

    /**
     * Element-wise inverse sine ({@code numpy.arcsin}).
     * @param a input array
     * @return arcsin(a)
     */
    public static NdArray arcsin(final NdArray a) {
        return unaryOp(a, v -> (float) Math.asin(v));
    }

    /**
     * Element-wise inverse cosine ({@code numpy.arccos}).
     * @param a input array
     * @return arccos(a)
     */
    public static NdArray arccos(final NdArray a) {
        return unaryOp(a, v -> (float) Math.acos(v));
    }

    /**
     * Element-wise inverse tangent ({@code numpy.arctan}).
     * @param a input array
     * @return arctan(a)
     */
    public static NdArray arctan(final NdArray a) {
        return unaryOp(a, v -> (float) Math.atan(v));
    }

    /**
     * Element-wise two-argument inverse tangent ({@code numpy.arctan2}).
     * @param y y-coordinates
     * @param x x-coordinates
     * @return arctan2(y, x)
     */
    public static NdArray arctan2(final NdArray y, final NdArray x) {
        return broadcastBinary(y, x, (a, b) -> (float) Math.atan2(a, b));
    }

    /**
     * Element-wise base-2 logarithm ({@code numpy.log2}).
     * @param a input array
     * @return log₂(a)
     */
    public static NdArray log2(final NdArray a) {
        return unaryOp(a, v -> (float) (Math.log(v) / LOG2));
    }

    /**
     * Element-wise base-10 logarithm ({@code numpy.log10}).
     * @param a input array
     * @return log₁₀(a)
     */
    public static NdArray log10(final NdArray a) {
        return unaryOp(a, v -> (float) Math.log10(v));
    }

    /**
     * Element-wise {@code log(1 + x)} ({@code numpy.log1p}). Accurate for small x.
     * @param a input array
     * @return log(1 + a)
     */
    public static NdArray log1p(final NdArray a) {
        return unaryOp(a, v -> (float) Math.log1p(v));
    }

    /**
     * Element-wise {@code exp(x) - 1} ({@code numpy.expm1}). Accurate for small x.
     * @param a input array
     * @return exp(a) - 1
     */
    public static NdArray expm1(final NdArray a) {
        return unaryOp(a, v -> (float) Math.expm1(v));
    }

    /**
     * Element-wise GELU activation using the tanh approximation.
     *
     * <p>Formula: {@code 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))}</p>
     *
     * @param a input array
     * @return GELU-activated array
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

    /**
     * Element-wise sigmoid activation: {@code 1 / (1 + exp(-x))}.
     * @param a input array
     * @return sigmoid(a)
     */
    public static NdArray sigmoid(final NdArray a) {
        return unaryOp(a, x -> (float) (1.0 / (1.0 + Math.exp(-x))));
    }

    /**
     * Element-wise ReLU activation: {@code max(0, x)}.
     * @param a input array
     * @return relu(a)
     */
    public static NdArray relu(final NdArray a) {
        return unaryOp(a, v -> Math.max(0f, v));
    }

    /**
     * Element-wise reciprocal ({@code numpy.reciprocal}): {@code 1 / x}.
     * @param a input array
     * @return 1/a
     */
    public static NdArray reciprocal(final NdArray a) {
        return unaryOp(a, v -> 1f / v);
    }

    /**
     * Element-wise clipping to {@code [min, max]} ({@code numpy.clip}).
     *
     * @param a   input array
     * @param min lower bound
     * @param max upper bound
     * @return clipped array
     */
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
