package com.sparrowlogic.numpy4j;

import java.lang.foreign.ValueLayout;

/**
 * Reduction operations matching NumPy: sum, prod, mean, std, var, min, max,
 * argmin, argmax, cumsum, cumprod, all, any, nonzero.
 */
public final class Reductions {
    private Reductions() {}

    // ── Full reductions (no axis) ──

    public static float sum(NdArray a) {
        NdArray c = a.contiguous();
        return SimdOps.reduceSum(c.data(), a.size());
    }

    public static float prod(NdArray a) {
        NdArray c = a.contiguous();
        float p = 1f;
        for (long i = 0; i < a.size(); i++) p *= c.flatGetFloat(i);
        return p;
    }

    public static float mean(NdArray a) { return sum(a) / a.size(); }

    public static float var(NdArray a) {
        float m = mean(a);
        NdArray c = a.contiguous();
        float acc = 0;
        for (long i = 0; i < a.size(); i++) {
            float d = c.flatGetFloat(i) - m;
            acc += d * d;
        }
        return acc / a.size();
    }

    public static float std(NdArray a) { return (float) Math.sqrt(var(a)); }

    public static float max(NdArray a) {
        return SimdOps.reduceMax(a.contiguous().data(), a.size());
    }

    public static float min(NdArray a) {
        return SimdOps.reduceMin(a.contiguous().data(), a.size());
    }

    public static long argmax(NdArray a) {
        NdArray c = a.contiguous();
        float best = Float.NEGATIVE_INFINITY;
        long idx = 0;
        for (long i = 0; i < a.size(); i++) {
            float v = c.flatGetFloat(i);
            if (v > best) { best = v; idx = i; }
        }
        return idx;
    }

    public static long argmin(NdArray a) {
        NdArray c = a.contiguous();
        float best = Float.POSITIVE_INFINITY;
        long idx = 0;
        for (long i = 0; i < a.size(); i++) {
            float v = c.flatGetFloat(i);
            if (v < best) { best = v; idx = i; }
        }
        return idx;
    }

    // ── Cumulative ──

    public static NdArray cumsum(NdArray a) {
        NdArray c = a.contiguous();
        NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, (int) a.size());
        float acc = 0;
        for (long i = 0; i < a.size(); i++) {
            acc += c.flatGetFloat(i);
            out.flatSetFloat(i, acc);
        }
        return out;
    }

    public static NdArray cumprod(NdArray a) {
        NdArray c = a.contiguous();
        NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, (int) a.size());
        float acc = 1;
        for (long i = 0; i < a.size(); i++) {
            acc *= c.flatGetFloat(i);
            out.flatSetFloat(i, acc);
        }
        return out;
    }

    // ── Axis-aware reductions ──

    /** Sum along a specific axis. */
    public static NdArray sum(NdArray a, int axis) {
        return axisReduce(a, axis, false, 0f, Float::sum);
    }

    public static NdArray sum(NdArray a, int axis, boolean keepdims) {
        return axisReduce(a, axis, keepdims, 0f, Float::sum);
    }

    public static NdArray max(NdArray a, int axis) {
        return axisReduce(a, axis, false, Float.NEGATIVE_INFINITY, Math::max);
    }

    public static NdArray min(NdArray a, int axis) {
        return axisReduce(a, axis, false, Float.POSITIVE_INFINITY, Math::min);
    }

    public static NdArray mean(NdArray a, int axis) {
        NdArray s = sum(a, axis);
        return Ufunc.divScalar(s, a.shape(axis));
    }

    /** Argmax along axis — returns INT32 array. */
    public static NdArray argmax(NdArray a, int axis) {
        return axisArgReduce(a, axis, true);
    }

    public static NdArray argmin(NdArray a, int axis) {
        return axisArgReduce(a, axis, false);
    }

    // ── Boolean reductions ──

    public static boolean all(NdArray a) {
        NdArray c = a.contiguous();
        for (long i = 0; i < a.size(); i++) if (c.flatGetFloat(i) == 0f) return false;
        return true;
    }

    public static boolean any(NdArray a) {
        NdArray c = a.contiguous();
        for (long i = 0; i < a.size(); i++) if (c.flatGetFloat(i) != 0f) return true;
        return false;
    }

    // ── Internal ──

    @FunctionalInterface
    private interface ReduceOp { float apply(float acc, float val); }

    private static NdArray axisReduce(NdArray a, int axis, boolean keepdims, float init, ReduceOp op) {
        if (axis < 0) axis += a.ndim();
        int[] inShape = a.shape();
        int axisLen = inShape[axis];

        // Output shape: remove axis (or keep as 1)
        int[] outShape;
        if (keepdims) {
            outShape = inShape.clone();
            outShape[axis] = 1;
        } else {
            outShape = new int[inShape.length - 1];
            for (int i = 0, j = 0; i < inShape.length; i++) {
                if (i != axis) outShape[j++] = inShape[i];
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
                // Build input index
                int[] inIdx = new int[inShape.length];
                int oj = 0;
                for (int d = 0; d < inShape.length; d++) {
                    if (d == axis) {
                        inIdx[d] = k;
                    } else {
                        inIdx[d] = outIdx[keepdims ? d : oj++];
                    }
                }
                long flatIn = a.offset();
                for (int d = 0; d < inIdx.length; d++) flatIn += (long) inIdx[d] * inStrides[d];
                acc = op.apply(acc, a.data().getAtIndex(ValueLayout.JAVA_FLOAT, flatIn));
            }
            out.data().setAtIndex(ValueLayout.JAVA_FLOAT, oi, acc);
        }
        return out;
    }

    private static NdArray axisArgReduce(NdArray a, int axis, boolean isMax) {
        if (axis < 0) axis += a.ndim();
        int[] inShape = a.shape();
        int axisLen = inShape[axis];
        int[] outShape = new int[inShape.length - 1];
        for (int i = 0, j = 0; i < inShape.length; i++) {
            if (i != axis) outShape[j++] = inShape[i];
        }

        NdArray out = NdArrayFactory.empty(a.arena(), DType.INT32, outShape);
        long outSize = ShapeUtils.size(outShape);
        int[] inStrides = a.strides();

        for (long oi = 0; oi < outSize; oi++) {
            int[] outIdx = ShapeUtils.unravelIndex(oi, outShape);
            float best = isMax ? Float.NEGATIVE_INFINITY : Float.POSITIVE_INFINITY;
            int bestK = 0;
            for (int k = 0; k < axisLen; k++) {
                int[] inIdx = new int[inShape.length];
                int oj = 0;
                for (int d = 0; d < inShape.length; d++) {
                    inIdx[d] = (d == axis) ? k : outIdx[oj++];
                }
                long flatIn = a.offset();
                for (int d = 0; d < inIdx.length; d++) flatIn += (long) inIdx[d] * inStrides[d];
                float v = a.data().getAtIndex(ValueLayout.JAVA_FLOAT, flatIn);
                if (isMax ? v > best : v < best) { best = v; bestK = k; }
            }
            out.data().setAtIndex(ValueLayout.JAVA_INT, oi, bestK);
        }
        return out;
    }

    // ── Nan-aware reductions ──

    public static float nansum(NdArray a) {
        NdArray c = a.contiguous();
        float sum = 0;
        for (long i = 0; i < a.size(); i++) { float v = c.flatGetFloat(i); if (!Float.isNaN(v)) sum += v; }
        return sum;
    }

    public static float nanmean(NdArray a) {
        NdArray c = a.contiguous();
        float sum = 0; int count = 0;
        for (long i = 0; i < a.size(); i++) { float v = c.flatGetFloat(i); if (!Float.isNaN(v)) { sum += v; count++; } }
        return sum / count;
    }

    public static float nanstd(NdArray a) { return (float) Math.sqrt(nanvar(a)); }

    public static float nanvar(NdArray a) {
        float m = nanmean(a);
        NdArray c = a.contiguous();
        float acc = 0; int count = 0;
        for (long i = 0; i < a.size(); i++) {
            float v = c.flatGetFloat(i);
            if (!Float.isNaN(v)) { float d = v - m; acc += d * d; count++; }
        }
        return acc / count;
    }

    public static float nanmax(NdArray a) {
        NdArray c = a.contiguous();
        float max = Float.NEGATIVE_INFINITY;
        for (long i = 0; i < a.size(); i++) { float v = c.flatGetFloat(i); if (!Float.isNaN(v) && v > max) max = v; }
        return max;
    }

    public static float nanmin(NdArray a) {
        NdArray c = a.contiguous();
        float min = Float.POSITIVE_INFINITY;
        for (long i = 0; i < a.size(); i++) { float v = c.flatGetFloat(i); if (!Float.isNaN(v) && v < min) min = v; }
        return min;
    }

    public static long nanargmax(NdArray a) {
        NdArray c = a.contiguous();
        float best = Float.NEGATIVE_INFINITY; long idx = 0;
        for (long i = 0; i < a.size(); i++) { float v = c.flatGetFloat(i); if (!Float.isNaN(v) && v > best) { best = v; idx = i; } }
        return idx;
    }

    public static long nanargmin(NdArray a) {
        NdArray c = a.contiguous();
        float best = Float.POSITIVE_INFINITY; long idx = 0;
        for (long i = 0; i < a.size(); i++) { float v = c.flatGetFloat(i); if (!Float.isNaN(v) && v < best) { best = v; idx = i; } }
        return idx;
    }

    // ── Statistics ──

    public static float median(NdArray a) {
        float[] sorted = a.toFloatArray();
        java.util.Arrays.sort(sorted);
        int n = sorted.length;
        return n % 2 == 0 ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2f : sorted[n / 2];
    }

    public static float percentile(NdArray a, float q) {
        float[] sorted = a.toFloatArray();
        java.util.Arrays.sort(sorted);
        float idx = q / 100f * (sorted.length - 1);
        int lo = (int) Math.floor(idx);
        int hi = (int) Math.ceil(idx);
        if (lo == hi) return sorted[lo];
        return sorted[lo] + (idx - lo) * (sorted[hi] - sorted[lo]);
    }

    /** Histogram: returns {counts, bin_edges}. */
    public static NdArray[] histogram(NdArray a, int bins) {
        float[] data = a.toFloatArray();
        float min = Float.MAX_VALUE, max = Float.MIN_VALUE;
        for (float v : data) { if (v < min) min = v; if (v > max) max = v; }
        float binWidth = (max - min) / bins;

        NdArray counts = NdArrayFactory.zeros(a.arena(), DType.FLOAT32, bins);
        NdArray edges = NdArrayFactory.empty(a.arena(), DType.FLOAT32, bins + 1);
        for (int i = 0; i <= bins; i++) edges.flatSetFloat(i, min + i * binWidth);

        for (float v : data) {
            int bin = (int) ((v - min) / binWidth);
            if (bin >= bins) bin = bins - 1;
            counts.flatSetFloat(bin, counts.flatGetFloat(bin) + 1);
        }
        return new NdArray[]{counts, edges};
    }

    /** numpy.allclose */
    public static boolean allclose(NdArray a, NdArray b, float rtol, float atol) {
        NdArray ca = a.contiguous(), cb = b.contiguous();
        for (long i = 0; i < a.size(); i++) {
            float va = ca.flatGetFloat(i), vb = cb.flatGetFloat(i);
            if (Math.abs(va - vb) > atol + rtol * Math.abs(vb)) return false;
        }
        return true;
    }

    /** numpy.array_equal */
    public static boolean arrayEqual(NdArray a, NdArray b) {
        if (!java.util.Arrays.equals(a.shape(), b.shape())) return false;
        NdArray ca = a.contiguous(), cb = b.contiguous();
        for (long i = 0; i < a.size(); i++)
            if (ca.flatGetFloat(i) != cb.flatGetFloat(i)) return false;
        return true;
    }

    /** numpy.count_nonzero */
    public static int countNonzero(NdArray a) {
        NdArray c = a.contiguous();
        int count = 0;
        for (long i = 0; i < a.size(); i++) if (c.flatGetFloat(i) != 0f) count++;
        return count;
    }

    /** numpy.quantile (same as percentile but q in [0,1]) */
    public static float quantile(NdArray a, float q) {
        return percentile(a, q * 100f);
    }

    /** numpy.average with optional weights */
    public static float average(NdArray a, NdArray weights) {
        NdArray ca = a.contiguous(), cw = weights.contiguous();
        float sumWV = 0, sumW = 0;
        for (long i = 0; i < a.size(); i++) {
            float w = cw.flatGetFloat(i);
            sumWV += ca.flatGetFloat(i) * w;
            sumW += w;
        }
        return sumWV / sumW;
    }
}
