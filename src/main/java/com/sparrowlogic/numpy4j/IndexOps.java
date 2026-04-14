package com.sparrowlogic.numpy4j;

import java.lang.foreign.ValueLayout;
import java.util.Arrays;

/**
 * Indexing, slicing, and advanced indexing matching NumPy.
 */
public final class IndexOps {
    private IndexOps() {}

    /** Slice along axis 0: a[start:stop:step] — returns a view when step=1. */
    public static NdArray slice(NdArray a, int start, int stop, int step) {
        return slice(a, 0, start, stop, step);
    }

    /** Slice along a specific axis. */
    public static NdArray slice(NdArray a, int axis, int start, int stop, int step) {
        if (axis < 0) axis += a.ndim();
        int[] shape = a.shape();
        int[] strides = a.strides();
        int axisLen = shape[axis];

        if (start < 0) start += axisLen;
        if (stop < 0) stop += axisLen;
        start = Math.max(0, Math.min(start, axisLen));
        stop = Math.max(0, Math.min(stop, axisLen));

        int newLen = Math.max(0, (stop - start + step - 1) / step);
        int[] newShape = shape.clone();
        newShape[axis] = newLen;
        int[] newStrides = strides.clone();
        newStrides[axis] = strides[axis] * step;
        long newOffset = a.offset() + (long) start * strides[axis];

        return new NdArray(a.data(), a.arena(), a.dtype(), newShape, newStrides, newOffset);
    }

    /** Take elements along axis using integer indices. */
    public static NdArray take(NdArray a, int[] indices, int axis) {
        if (axis < 0) axis += a.ndim();
        int[] inShape = a.shape();
        int[] outShape = inShape.clone();
        outShape[axis] = indices.length;
        NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, outShape);

        long outSize = ShapeUtils.size(outShape);
        int[] outStrides = ShapeUtils.cStrides(outShape);
        int[] inStrides = a.strides();

        for (long oi = 0; oi < outSize; oi++) {
            int[] outIdx = ShapeUtils.unravelIndex(oi, outShape);
            int[] inIdx = outIdx.clone();
            inIdx[axis] = indices[outIdx[axis]];
            if (inIdx[axis] < 0) inIdx[axis] += inShape[axis];

            long flatIn = a.offset();
            for (int d = 0; d < inIdx.length; d++) flatIn += (long) inIdx[d] * inStrides[d];
            out.data().setAtIndex(ValueLayout.JAVA_FLOAT, oi,
                    a.data().getAtIndex(ValueLayout.JAVA_FLOAT, flatIn));
        }
        return out;
    }

    /** Boolean mask indexing — returns 1D array of elements where mask is true. */
    public static NdArray booleanIndex(NdArray a, NdArray mask) {
        NdArray ca = a.contiguous();
        // Count true values
        int count = 0;
        for (long i = 0; i < mask.size(); i++) {
            if (mask.data().getAtIndex(ValueLayout.JAVA_FLOAT, i) != 0f) count++;
        }
        NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, count);
        int j = 0;
        for (long i = 0; i < mask.size(); i++) {
            if (mask.data().getAtIndex(ValueLayout.JAVA_FLOAT, i) != 0f) {
                out.flatSetFloat(j++, ca.flatGetFloat(i));
            }
        }
        return out;
    }

    /** numpy.where(condition, x, y) — element-wise ternary. */
    public static NdArray where(NdArray condition, NdArray x, NdArray y) {
        NdArray out = NdArrayFactory.empty(x.arena(), DType.FLOAT32, x.shape());
        NdArray cc = condition.contiguous(), cx = x.contiguous(), cy = y.contiguous();
        for (long i = 0; i < x.size(); i++) {
            out.flatSetFloat(i, cc.flatGetFloat(i) != 0f ? cx.flatGetFloat(i) : cy.flatGetFloat(i));
        }
        return out;
    }

    /** Sort along last axis (returns new array). */
    public static NdArray sort(NdArray a) {
        NdArray out = a.contiguous();
        // Copy data
        NdArray result = NdArrayFactory.empty(a.arena(), DType.FLOAT32, a.shape());
        result.data().copyFrom(out.data().asSlice(0, a.size() * Float.BYTES));

        if (a.ndim() == 1) {
            float[] arr = result.toFloatArray();
            Arrays.sort(arr);
            java.lang.foreign.MemorySegment.copy(arr, 0, result.data(), ValueLayout.JAVA_FLOAT, 0, arr.length);
        } else {
            // Sort along last axis
            int lastDim = a.shape(a.ndim() - 1);
            long nSlices = a.size() / lastDim;
            float[] buf = new float[lastDim];
            for (long s = 0; s < nSlices; s++) {
                long base = s * lastDim;
                for (int i = 0; i < lastDim; i++) buf[i] = result.flatGetFloat(base + i);
                Arrays.sort(buf);
                for (int i = 0; i < lastDim; i++) result.flatSetFloat(base + i, buf[i]);
            }
        }
        return result;
    }

    /** Argsort along last axis. Returns INT32 indices. */
    public static NdArray argsort(NdArray a) {
        NdArray c = a.contiguous();
        int lastDim = a.shape(a.ndim() - 1);
        long nSlices = a.size() / lastDim;
        NdArray out = NdArrayFactory.empty(a.arena(), DType.INT32, a.shape());

        Integer[] indices = new Integer[lastDim];
        float[] buf = new float[lastDim];
        for (long s = 0; s < nSlices; s++) {
            long base = s * lastDim;
            for (int i = 0; i < lastDim; i++) { buf[i] = c.flatGetFloat(base + i); indices[i] = i; }
            Arrays.sort(indices, (x, y) -> Float.compare(buf[x], buf[y]));
            for (int i = 0; i < lastDim; i++) {
                out.data().setAtIndex(ValueLayout.JAVA_INT, base + i, indices[i]);
            }
        }
        return out;
    }

    /** Searchsorted — binary search in sorted 1D array. */
    public static int searchsorted(NdArray a, float value) {
        NdArray c = a.contiguous();
        int lo = 0, hi = (int) a.size();
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (c.flatGetFloat(mid) < value) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    /** numpy.nonzero — returns array of indices where elements are nonzero. */
    public static NdArray nonzero(NdArray a) {
        NdArray c = a.contiguous();
        int count = 0;
        for (long i = 0; i < a.size(); i++) if (c.flatGetFloat(i) != 0f) count++;
        NdArray out = NdArrayFactory.empty(a.arena(), DType.INT32, count);
        int j = 0;
        for (long i = 0; i < a.size(); i++)
            if (c.flatGetFloat(i) != 0f) out.data().setAtIndex(java.lang.foreign.ValueLayout.JAVA_INT, j++, (int) i);
        return out;
    }

    /** numpy.argwhere — same as nonzero for 1D. */
    public static NdArray argwhere(NdArray a) { return nonzero(a); }

    /** numpy.partition — partially sort so element at kth position is in sorted position. */
    public static NdArray partition(NdArray a, int kth) {
        float[] data = a.toFloatArray();
        // Quickselect-style: just sort (correct but not optimal)
        float[] sorted = data.clone();
        Arrays.sort(sorted);
        float pivot = sorted[kth];
        float[] result = new float[data.length];
        int lo = 0, hi = data.length - 1;
        // Place elements < pivot first, then pivot, then rest
        java.util.List<Float> less = new java.util.ArrayList<>(), eq = new java.util.ArrayList<>(), greater = new java.util.ArrayList<>();
        for (float v : data) {
            if (v < pivot) less.add(v);
            else if (v == pivot) eq.add(v);
            else greater.add(v);
        }
        int idx = 0;
        for (float v : less) result[idx++] = v;
        for (float v : eq) result[idx++] = v;
        for (float v : greater) result[idx++] = v;
        return NdArrayFactory.array(a.arena(), result);
    }

    /** numpy.argpartition — indices that would partition. */
    public static NdArray argpartition(NdArray a, int kth) {
        float[] data = a.toFloatArray();
        Integer[] indices = new Integer[data.length];
        for (int i = 0; i < data.length; i++) indices[i] = i;
        float[] sorted = data.clone();
        Arrays.sort(sorted);
        float pivot = sorted[kth];
        // Sort indices by partition order
        Arrays.sort(indices, (x, y) -> {
            boolean xLess = data[x] < pivot, yLess = data[y] < pivot;
            if (xLess != yLess) return xLess ? -1 : 1;
            return 0;
        });
        NdArray out = NdArrayFactory.empty(a.arena(), DType.INT32, data.length);
        for (int i = 0; i < data.length; i++)
            out.data().setAtIndex(java.lang.foreign.ValueLayout.JAVA_INT, i, indices[i]);
        return out;
    }

    /** Fancy indexing — select elements by integer index array (1D). */
    public static NdArray fancyIndex(NdArray a, int[] indices) {
        NdArray c = a.contiguous();
        NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, indices.length);
        for (int i = 0; i < indices.length; i++)
            out.flatSetFloat(i, c.flatGetFloat(indices[i]));
        return out;
    }

    /** numpy.put — set values at flat indices. */
    public static void put(NdArray a, int[] indices, float[] values) {
        for (int i = 0; i < indices.length; i++)
            a.flatSetFloat(indices[i], values[i % values.length]);
    }
}
