package com.sparrowlogic.numpy4j;

import java.lang.foreign.ValueLayout;

/**
 * Reduction operations matching NumPy: sum, prod, mean, std, var, min, max,
 * argmin, argmax, cumsum, cumprod, all, any, nonzero.
     */
public final class Reductions {
    private Reductions() {
    }

    // ── Full reductions (no axis) ──

    /**
     * Sum of all elements ({@code numpy.sum}).
     *
     * @param a input array
     * @return the computed value
     */
    public static float sum(final NdArray a) {
        NdArray c = a.contiguous();
        return SimdOps.get().reduceSum(c.data(), a.size());
    }

    /**
     * Sum of all elements ({@code numpy.sum}).
     *
     * @param a input array
     * @param axis axis to reduce
     * @return result array
     */
    public static NdArray sum(final NdArray a, final int axis) {
        return axisReduce(a, axis, false, 0f, Float::sum);
    }

    /**
     * Sum of all elements ({@code numpy.sum}).
     *
     * @param a input array
     * @param axis axis to reduce
     * @param keepdims whether to keep reduced dimensions
     * @return result array
     */
    public static NdArray sum(final NdArray a, final int axis, final boolean keepdims) {
        return axisReduce(a, axis, keepdims, 0f, Float::sum);
    }

    /**
     * Product of all elements ({@code numpy.prod}).
     *
     * @param a input array
     * @return the computed value
     */
    public static float prod(final NdArray a) {
        NdArray c = a.contiguous();
        float p = 1f;
        for (long i = 0; i < a.size(); i++) {
            p *= c.flatGetFloat(i);
        }
        return p;
    }

    /**
     * Arithmetic mean ({@code numpy.mean}).
     *
     * @param a input array
     * @return the computed value
     */
    public static float mean(final NdArray a) {
        return sum(a) / a.size();
    }

    /**
     * Arithmetic mean ({@code numpy.mean}).
     *
     * @param a input array
     * @param axis axis to reduce
     * @return result array
     */
    public static NdArray mean(final NdArray a, final int axis) {
        NdArray s = sum(a, axis);
        return Ufunc.divScalar(s, a.shape(axis));
    }

    /**
     * Variance ({@code numpy.var}).
     *
     * @param a input array
     * @return the computed value
     */
    public static float var(final NdArray a) {
        float m = mean(a);
        NdArray c = a.contiguous();
        float acc = 0;
        for (long i = 0; i < a.size(); i++) {
            float d = c.flatGetFloat(i) - m;
            acc += d * d;
        }
        return acc / a.size();
    }

    /**
     * Standard deviation ({@code numpy.std}).
     *
     * @param a input array
     * @return the computed value
     */
    public static float std(final NdArray a) {
        return (float) Math.sqrt(var(a));
    }

    /**
     * Maximum value ({@code numpy.max}).
     *
     * @param a input array
     * @return the computed value
     */
    public static float max(final NdArray a) {
        return SimdOps.get().reduceMax(a.contiguous().data(), a.size());
    }

    /**
     * Maximum value ({@code numpy.max}).
     *
     * @param a input array
     * @param axis axis to reduce
     * @return result array
     */
    public static NdArray max(final NdArray a, final int axis) {
        return axisReduce(a, axis, false, Float.NEGATIVE_INFINITY, Math::max);
    }

    /**
     * Minimum value ({@code numpy.min}).
     *
     * @param a input array
     * @return the computed value
     */
    public static float min(final NdArray a) {
        return SimdOps.get().reduceMin(a.contiguous().data(), a.size());
    }

    /**
     * Minimum value ({@code numpy.min}).
     *
     * @param a input array
     * @param axis axis to reduce
     * @return result array
     */
    public static NdArray min(final NdArray a, final int axis) {
        return axisReduce(a, axis, false, Float.POSITIVE_INFINITY, Math::min);
    }

    /**
     * Index of maximum value ({@code numpy.argmax}).
     *
     * @param a input array
     * @return the index
     */
    public static long argmax(final NdArray a) {
        NdArray c = a.contiguous();
        float best = Float.NEGATIVE_INFINITY;
        long idx = 0;
        for (long i = 0; i < a.size(); i++) {
            float v = c.flatGetFloat(i);
            if (v > best) {
                best = v;
                idx = i;
            }
        }
        return idx;
    }

    /**
     * Index of maximum value ({@code numpy.argmax}).
     *
     * @param a input array
     * @param axis axis to reduce
     * @return result array
     */
    public static NdArray argmax(final NdArray a, final int axis) {
        return axisArgReduce(a, axis, true);
    }

    /**
     * Index of minimum value ({@code numpy.argmin}).
     *
     * @param a input array
     * @return the index
     */
    public static long argmin(final NdArray a) {
        NdArray c = a.contiguous();
        float best = Float.POSITIVE_INFINITY;
        long idx = 0;
        for (long i = 0; i < a.size(); i++) {
            float v = c.flatGetFloat(i);
            if (v < best) {
                best = v;
                idx = i;
            }
        }
        return idx;
    }

    // ── Cumulative ──

    /**
     * Index of minimum value ({@code numpy.argmin}).
     *
     * @param a input array
     * @param axis axis to reduce
     * @return result array
     */
    public static NdArray argmin(final NdArray a, final int axis) {
        return axisArgReduce(a, axis, false);
    }

    /**
     * Cumulative sum ({@code numpy.cumsum}).
     *
     * @param a input array
     * @return result array
     */
    public static NdArray cumsum(final NdArray a) {
        NdArray c = a.contiguous();
        NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, (int) a.size());
        float acc = 0;
        for (long i = 0; i < a.size(); i++) {
            acc += c.flatGetFloat(i);
            out.flatSetFloat(i, acc);
        }
        return out;
    }

    /**
     * Cumulative product ({@code numpy.cumprod}).
     *
     * @param a input array
     * @return result array
     */
    public static NdArray cumprod(final NdArray a) {
        NdArray c = a.contiguous();
        NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, (int) a.size());
        float acc = 1;
        for (long i = 0; i < a.size(); i++) {
            acc *= c.flatGetFloat(i);
            out.flatSetFloat(i, acc);
        }
        return out;
    }

    // ── Boolean reductions ──

    /**
     * True if all elements are non-zero ({@code numpy.all}).
     *
     * @param a input array
     * @return the result
     */
    public static boolean all(final NdArray a) {
        NdArray c = a.contiguous();
        for (long i = 0; i < a.size(); i++) {
            if (c.flatGetFloat(i) == 0f) {
                return false;
            }
        }
        return true;
    }

    /**
     * True if any element is non-zero ({@code numpy.any}).
     *
     * @param a input array
     * @return the result
     */
    public static boolean any(final NdArray a) {
        NdArray c = a.contiguous();
        for (long i = 0; i < a.size(); i++) {
            if (c.flatGetFloat(i) != 0f) {
                return true;
            }
        }
        return false;
    }

    // ── Internal ──

    private static NdArray axisReduce(
            final NdArray a,
            final int axis,
            final boolean keepdims,
            final float init,
            final ReduceOp op
    ) {
        int ax = axis < 0 ? axis + a.ndim() : axis;
        int[] inShape = a.shape();
        int axisLen = inShape[ax];

        // Output shape: remove axis (or keep as 1)
        int[] outShape;
        if (keepdims) {
            outShape = inShape.clone();
            outShape[ax] = 1;
        } else {
            outShape = new int[inShape.length - 1];
            for (int i = 0, j = 0; i < inShape.length; i++) {
                if (i != ax) {
                    outShape[j++] = inShape[i];
                }
            }
        }

        NdArray out = NdArrayFactory.full(a.arena(), DType.FLOAT32, init, outShape);
        long outSize = ShapeUtils.size(outShape);
        int[] outStrides = ShapeUtils.cStrides(outShape);
        int[] inStrides = a.strides();

        for (long oi = 0; oi < outSize; oi++) {
            int[] outIdx = ShapeUtils.unravelIndex(oi, outShape);
            float acc = init;
            for (int k = 0; k < axisLen; k++) {
                int[] inIdx = buildInputIndex(outIdx, inShape, ax, k, keepdims);
                long flatIn = computeFlatIndex(inIdx, inStrides, a.offset());
                acc = op.apply(acc, a.data().getAtIndex(ValueLayout.JAVA_FLOAT, flatIn));
            }
            out.data().setAtIndex(ValueLayout.JAVA_FLOAT, oi, acc);
        }
        return out;
    }

    private static NdArray axisArgReduce(final NdArray a, final int axis, final boolean isMax) {
        int ax = axis < 0 ? axis + a.ndim() : axis;
        int[] inShape = a.shape();
        int axisLen = inShape[ax];
        int[] outShape = new int[inShape.length - 1];
        for (int i = 0, j = 0; i < inShape.length; i++) {
            if (i != ax) {
                outShape[j++] = inShape[i];
            }
        }

        NdArray out = NdArrayFactory.empty(a.arena(), DType.INT32, outShape);
        long outSize = ShapeUtils.size(outShape);
        int[] inStrides = a.strides();

        for (long oi = 0; oi < outSize; oi++) {
            int[] outIdx = ShapeUtils.unravelIndex(oi, outShape);
            float best = isMax ? Float.NEGATIVE_INFINITY : Float.POSITIVE_INFINITY;
            int bestK = 0;
            for (int k = 0; k < axisLen; k++) {
                int[] inIdx = buildInputIndex(outIdx, inShape, ax, k, false);
                long flatIn = computeFlatIndex(inIdx, inStrides, a.offset());
                float v = a.data().getAtIndex(ValueLayout.JAVA_FLOAT, flatIn);
                if (isMax ? v > best : v < best) {
                    best = v;
                    bestK = k;
                }
            }
            out.data().setAtIndex(ValueLayout.JAVA_INT, oi, bestK);
        }
        return out;
    }

    // ── Nan-aware reductions ──

    private static int[] buildInputIndex(final int[] outIdx, final int[] inShape, final int ax,
                                         final int k, final boolean keepdims) {
        int[] inIdx = new int[inShape.length];
        int oj = 0;
        for (int d = 0; d < inShape.length; d++) {
            if (d == ax) {
                inIdx[d] = k;
            } else {
                inIdx[d] = outIdx[keepdims ? d : oj++];
            }
        }
        return inIdx;
    }

    private static long computeFlatIndex(final int[] inIdx, final int[] strides, final long offset) {
        long flat = offset;
        for (int d = 0; d < inIdx.length; d++) {
            flat += (long) inIdx[d] * strides[d];
        }
        return flat;
    }

    /**
     * Sum ignoring NaN ({@code numpy.nansum}).
     *
     * @param a input array
     * @return the computed value
     */
    public static float nansum(final NdArray a) {
        NdArray c = a.contiguous();
        float sum = 0;
        for (long i = 0; i < a.size(); i++) {
            float v = c.flatGetFloat(i);
            if (!Float.isNaN(v)) {
                sum += v;
            }
        }
        return sum;
    }

    /**
     * Mean ignoring NaN ({@code numpy.nanmean}).
     *
     * @param a input array
     * @return the computed value
     */
    public static float nanmean(final NdArray a) {
        NdArray c = a.contiguous();
        float sum = 0;
        int count = 0;
        for (long i = 0; i < a.size(); i++) {
            float v = c.flatGetFloat(i);
            if (!Float.isNaN(v)) {
                sum += v;
                count++;
            }
        }
        return sum / count;
    }

    /**
     * Standard deviation ignoring NaN ({@code numpy.nanstd}).
     *
     * @param a input array
     * @return the computed value
     */
    public static float nanstd(final NdArray a) {
        return (float) Math.sqrt(nanvar(a));
    }

    /**
     * Variance ignoring NaN ({@code numpy.nanvar}).
     *
     * @param a input array
     * @return the computed value
     */
    public static float nanvar(final NdArray a) {
        float m = nanmean(a);
        NdArray c = a.contiguous();
        float acc = 0;
        int count = 0;
        for (long i = 0; i < a.size(); i++) {
            float v = c.flatGetFloat(i);
            if (!Float.isNaN(v)) {
                float d = v - m;
                acc += d * d;
                count++;
            }
        }
        return acc / count;
    }

    /**
     * Maximum ignoring NaN ({@code numpy.nanmax}).
     *
     * @param a input array
     * @return the computed value
     */
    public static float nanmax(final NdArray a) {
        NdArray c = a.contiguous();
        float max = Float.NEGATIVE_INFINITY;
        for (long i = 0; i < a.size(); i++) {
            float v = c.flatGetFloat(i);
            if (!Float.isNaN(v) && v > max) {
                max = v;
            }
        }
        return max;
    }

    /**
     * Minimum ignoring NaN ({@code numpy.nanmin}).
     *
     * @param a input array
     * @return the computed value
     */
    public static float nanmin(final NdArray a) {
        NdArray c = a.contiguous();
        float min = Float.POSITIVE_INFINITY;
        for (long i = 0; i < a.size(); i++) {
            float v = c.flatGetFloat(i);
            if (!Float.isNaN(v) && v < min) {
                min = v;
            }
        }
        return min;
    }

    /**
     * Index of maximum ignoring NaN ({@code numpy.nanargmax}).
     *
     * @param a input array
     * @return the index
     */
    public static long nanargmax(final NdArray a) {
        NdArray c = a.contiguous();
        float best = Float.NEGATIVE_INFINITY;
        long idx = 0;
        for (long i = 0; i < a.size(); i++) {
            float v = c.flatGetFloat(i);
            if (!Float.isNaN(v) && v > best) {
                best = v;
                idx = i;
            }
        }
        return idx;
    }

    /**
     * Index of minimum ignoring NaN ({@code numpy.nanargmin}).
     *
     * @param a input array
     * @return the index
     */
    public static long nanargmin(final NdArray a) {
        NdArray c = a.contiguous();
        float best = Float.POSITIVE_INFINITY;
        long idx = 0;
        for (long i = 0; i < a.size(); i++) {
            float v = c.flatGetFloat(i);
            if (!Float.isNaN(v) && v < best) {
                best = v;
                idx = i;
            }
        }
        return idx;
    }

    // ── Statistics ──

    /**
     * Median value ({@code numpy.median}).
     *
     * @param a input array
     * @return the computed value
     */
    public static float median(final NdArray a) {
        float[] sorted = a.toFloatArray();
        java.util.Arrays.sort(sorted);
        int n = sorted.length;
        return n % 2 == 0 ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2f : sorted[n / 2];
    }

    /**
     * Percentile value ({@code numpy.percentile}).
     *
     * @param a input array
     * @param q percentile (0-100)
     * @return the computed value
     */
    public static float percentile(final NdArray a, final float q) {
        float[] sorted = a.toFloatArray();
        java.util.Arrays.sort(sorted);
        float idx = q / 100f * (sorted.length - 1);
        int lo = (int) Math.floor(idx);
        int hi = (int) Math.ceil(idx);
        if (lo == hi) {
            return sorted[lo];
        }
        return sorted[lo] + (idx - lo) * (sorted[hi] - sorted[lo]);
    }

    /**
     * Histogram: returns {counts, bin_edges}.
     *
     * @param a input array
     * @param bins bins
     * @return result array
     */
    public static NdArray[] histogram(final NdArray a, final int bins) {
        float[] data = a.toFloatArray();
        float min = Float.MAX_VALUE;
        float max = Float.MIN_VALUE;
        for (final float v : data) {
            if (v < min) {
                min = v;
            }
            if (v > max) {
                max = v;
            }
        }
        float binWidth = (max - min) / bins;

        NdArray counts = NdArrayFactory.zeros(a.arena(), DType.FLOAT32, bins);
        NdArray edges = NdArrayFactory.empty(a.arena(), DType.FLOAT32, bins + 1);
        for (int i = 0; i <= bins; i++) {
            edges.flatSetFloat(i, min + i * binWidth);
        }

        for (final float v : data) {
            int bin = (int) ((v - min) / binWidth);
            if (bin >= bins) {
                bin = bins - 1;
            }
            counts.flatSetFloat(bin, counts.flatGetFloat(bin) + 1);
        }
        return new NdArray[]{counts, edges};
    }

    /**
     * numpy.allclose
     *
     * @param a input array
     * @param b second array
     * @param rtol rtol
     * @param atol atol
     * @return the result
     */
    public static boolean allclose(final NdArray a, final NdArray b, final float rtol, final float atol) {
        NdArray ca = a.contiguous();
        NdArray cb = b.contiguous();
        for (long i = 0; i < a.size(); i++) {
            float va = ca.flatGetFloat(i);
            float vb = cb.flatGetFloat(i);
            if (Math.abs(va - vb) > atol + rtol * Math.abs(vb)) {
                return false;
            }
        }
        return true;
    }

    /**
     * numpy.array_equal
     *
     * @param a input array
     * @param b second array
     * @return the result
     */
    public static boolean arrayEqual(final NdArray a, final NdArray b) {
        if (!java.util.Arrays.equals(a.shape(), b.shape())) {
            return false;
        }
        NdArray ca = a.contiguous();
        NdArray cb = b.contiguous();
        return elementsEqual(ca, cb, a.size());
    }

    private static boolean elementsEqual(final NdArray ca, final NdArray cb, final long size) {
        for (long i = 0; i < size; i++) {
            if (ca.flatGetFloat(i) != cb.flatGetFloat(i)) {
                return false;
            }
        }
        return true;
    }

    /**
     * numpy.count_nonzero
     *
     * @param a input array
     * @return the computed value
     */
    public static int countNonzero(final NdArray a) {
        NdArray c = a.contiguous();
        int count = 0;
        for (long i = 0; i < a.size(); i++) {
            if (c.flatGetFloat(i) != 0f) {
                count++;
            }
        }
        return count;
    }

    /**
     * numpy.quantile (same as percentile but q in [0,1])
     *
     * @param a input array
     * @param q q
     * @return the computed value
     */
    public static float quantile(final NdArray a, final float q) {
        return percentile(a, q * 100f);
    }

    /**
     * numpy.average with optional weights
     *
     * @param a input array
     * @param weights weights
     * @return the computed value
     */
    public static float average(final NdArray a, final NdArray weights) {
        NdArray ca = a.contiguous();
        NdArray cw = weights.contiguous();
        float sumWV = 0;
        float sumW = 0;
        for (long i = 0; i < a.size(); i++) {
            float w = cw.flatGetFloat(i);
            sumWV += ca.flatGetFloat(i) * w;
            sumW += w;
        }
        return sumWV / sumW;
    }

    @FunctionalInterface
    private interface ReduceOp {
        float apply(float acc, float val);
    }
}
