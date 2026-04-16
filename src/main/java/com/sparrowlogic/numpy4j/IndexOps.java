package com.sparrowlogic.numpy4j;

import java.lang.foreign.ValueLayout;
import java.util.Arrays;

/**
 * Indexing, slicing, and advanced indexing matching NumPy.
 */
public final class IndexOps {
    private IndexOps() {
    }

    /**
     * Slices along axis 0: {@code a[start:stop:step]} — returns a view when step=1.
     *
     * @param a     input array
     * @param start start index (inclusive, supports negative)
     * @param stop  stop index (exclusive, supports negative)
     * @param step  step size
     * @return sliced view
     */
    public static NdArray slice(final NdArray a, final int start, final int stop, final int step) {
        return slice(a, 0, start, stop, step);
    }

    /**
     * Slices along a specific axis.
     *
     * @param a     input array
     * @param axis  axis to slice along
     * @param start start index (inclusive, supports negative)
     * @param stop  stop index (exclusive, supports negative)
     * @param step  step size
     * @return sliced view
     */
    public static NdArray slice(final NdArray a, final int axis, final int start, final int stop, final int step) {
        int ax = axis < 0 ? axis + a.ndim() : axis;
        int[] shape = a.shape();
        int axisLen = shape[ax];

        int s0 = start < 0 ? start + axisLen : start;
        int s1 = stop < 0 ? stop + axisLen : stop;
        s0 = Math.max(0, Math.min(s0, axisLen));
        s1 = Math.max(0, Math.min(s1, axisLen));

        int newLen = Math.max(0, (s1 - s0 + step - 1) / step);
        int[] strides = a.strides();
        int[] newShape = shape.clone();
        newShape[ax] = newLen;
        int[] newStrides = strides.clone();
        newStrides[ax] = strides[ax] * step;
        long newOffset = a.offset() + (long) s0 * strides[ax];

        return new NdArray(a.data(), a.arena(), a.dtype(), newShape, newStrides, newOffset);
    }

    /**
     * Takes elements along an axis using integer indices ({@code numpy.take}).
     *
     * @param a       input array
     * @param indices integer indices to select
     * @param axis    axis to take along
     * @return gathered array
     */
    public static NdArray take(final NdArray a, final int[] indices, final int axis) {
        int ax = axis < 0 ? axis + a.ndim() : axis;
        int[] inShape = a.shape();
        int[] outShape = inShape.clone();
        outShape[ax] = indices.length;
        NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, outShape);

        long outSize = ShapeUtils.size(outShape);
        int[] outStrides = ShapeUtils.cStrides(outShape);
        int[] inStrides = a.strides();

        for (long oi = 0; oi < outSize; oi++) {
            int[] outIdx = ShapeUtils.unravelIndex(oi, outShape);
            int[] inIdx = outIdx.clone();
            inIdx[ax] = indices[outIdx[ax]];
            if (inIdx[ax] < 0) {
                inIdx[ax] += inShape[ax];
            }

            long flatIn = a.offset();
            for (int d = 0; d < inIdx.length; d++) {
                flatIn += (long) inIdx[d] * inStrides[d];
            }
            out.data().setAtIndex(
                    ValueLayout.JAVA_FLOAT, oi,
                    a.data().getAtIndex(ValueLayout.JAVA_FLOAT, flatIn)
            );
        }
        return out;
    }

    /**
     * Boolean mask indexing — returns 1-D array of elements where mask is non-zero.
     *
     * @param a    input array
     * @param mask boolean mask (non-zero = true)
     * @return 1-D array of selected elements
     */
    public static NdArray booleanIndex(final NdArray a, final NdArray mask) {
        NdArray ca = a.contiguous();
        // Count true values
        int count = 0;
        for (long i = 0; i < mask.size(); i++) {
            if (mask.data().getAtIndex(ValueLayout.JAVA_FLOAT, i) != 0f) {
                count++;
            }
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

    /**
     * Element-wise ternary selection ({@code numpy.where}).
     *
     * @param condition boolean mask (non-zero = true)
     * @param x         values where condition is true
     * @param y         values where condition is false
     * @return selected array
     */
    public static NdArray where(final NdArray condition, final NdArray x, final NdArray y) {
        NdArray out = NdArrayFactory.empty(x.arena(), DType.FLOAT32, x.shape());
        NdArray cc = condition.contiguous();
        NdArray cx = x.contiguous();
        NdArray cy = y.contiguous();
        for (long i = 0; i < x.size(); i++) {
            out.flatSetFloat(i, cc.flatGetFloat(i) != 0f ? cx.flatGetFloat(i) : cy.flatGetFloat(i));
        }
        return out;
    }

    /**
     * Sorts along the last axis ({@code numpy.sort}). Returns a new sorted array.
     *
     * @param a input array
     * @return sorted copy
     */
    public static NdArray sort(final NdArray a) {
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
                for (int i = 0; i < lastDim; i++) {
                    buf[i] = result.flatGetFloat(base + i);
                }
                Arrays.sort(buf);
                for (int i = 0; i < lastDim; i++) {
                    result.flatSetFloat(base + i, buf[i]);
                }
            }
        }
        return result;
    }

    /**
     * Returns indices that would sort the array along the last axis ({@code numpy.argsort}).
     *
     * @param a input array
     * @return INT32 array of sort indices
     */
    public static NdArray argsort(final NdArray a) {
        NdArray c = a.contiguous();
        int lastDim = a.shape(a.ndim() - 1);
        long nSlices = a.size() / lastDim;
        NdArray out = NdArrayFactory.empty(a.arena(), DType.INT32, a.shape());

        Integer[] indices = new Integer[lastDim];
        float[] buf = new float[lastDim];
        for (long s = 0; s < nSlices; s++) {
            long base = s * lastDim;
            for (int i = 0; i < lastDim; i++) {
                buf[i] = c.flatGetFloat(base + i);
                indices[i] = i;
            }
            Arrays.sort(indices, (x, y) -> Float.compare(buf[x], buf[y]));
            for (int i = 0; i < lastDim; i++) {
                out.data().setAtIndex(ValueLayout.JAVA_INT, base + i, indices[i]);
            }
        }
        return out;
    }

    /**
     * Binary search in a sorted 1-D array ({@code numpy.searchsorted}).
     *
     * @param a     sorted input array
     * @param value value to search for
     * @return insertion index
     */
    public static int searchsorted(final NdArray a, final float value) {
        NdArray c = a.contiguous();
        int lo = 0;
        int hi = (int) a.size();
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (c.flatGetFloat(mid) < value) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    /**
     * Returns flat indices of non-zero elements ({@code numpy.nonzero}).
     *
     * @param a input array
     * @return INT32 array of non-zero indices
     */
    public static NdArray nonzero(final NdArray a) {
        NdArray c = a.contiguous();
        int count = 0;
        for (long i = 0; i < a.size(); i++) {
            if (c.flatGetFloat(i) != 0f) {
                count++;
            }
        }
        NdArray out = NdArrayFactory.empty(a.arena(), DType.INT32, count);
        int j = 0;
        for (long i = 0; i < a.size(); i++) {
            if (c.flatGetFloat(i) != 0f) {
                out.data().setAtIndex(java.lang.foreign.ValueLayout.JAVA_INT, j++, (int) i);
            }
        }
        return out;
    }

    /**
     * Returns indices of non-zero elements ({@code numpy.argwhere}). Same as {@link #nonzero(NdArray)} for 1-D.
     *
     * @param a input array
     * @return INT32 array of non-zero indices
     */
    public static NdArray argwhere(final NdArray a) {
        return nonzero(a);
    }

    /**
     * Partially sorts so the kth element is in its sorted position ({@code numpy.partition}).
     *
     * @param a   input array
     * @param kth partition index
     * @return partitioned array
     */
    public static NdArray partition(final NdArray a, final int kth) {
        float[] data = a.toFloatArray();
        // Quickselect-style: just sort (correct but not optimal)
        float[] sorted = data.clone();
        Arrays.sort(sorted);
        float pivot = sorted[kth];
        float[] result = new float[data.length];
        int lo = 0;
        int hi = data.length - 1;
        // Place elements < pivot first, then pivot, then rest
        java.util.List<Float> less = new java.util.ArrayList<>();
        java.util.List<Float> eq = new java.util.ArrayList<>();
        java.util.List<Float> greater = new java.util.ArrayList<>();
        for (final float v : data) {
            if (v < pivot) {
                less.add(v);
            } else if (v == pivot) {
                eq.add(v);
            } else {
                greater.add(v);
            }
        }
        int idx = 0;
        for (final float v : less) {
            result[idx++] = v;
        }
        for (final float v : eq) {
            result[idx++] = v;
        }
        for (final float v : greater) {
            result[idx++] = v;
        }
        return NdArrayFactory.array(a.arena(), result);
    }

    /**
     * Returns indices that would partition the array ({@code numpy.argpartition}).
     *
     * @param a   input array
     * @param kth partition index
     * @return INT32 array of partition indices
     */
    public static NdArray argpartition(final NdArray a, final int kth) {
        float[] data = a.toFloatArray();
        Integer[] indices = new Integer[data.length];
        for (int i = 0; i < data.length; i++) {
            indices[i] = i;
        }
        float[] sorted = data.clone();
        Arrays.sort(sorted);
        float pivot = sorted[kth];
        // Sort indices by partition order
        Arrays.sort(
                indices, (x, y) -> {
                    boolean xLess = data[x] < pivot;
                    boolean yLess = data[y] < pivot;
                    if (xLess != yLess) {
                        return xLess ? -1 : 1;
                    }
                    return 0;
                }
        );
        NdArray out = NdArrayFactory.empty(a.arena(), DType.INT32, data.length);
        for (int i = 0; i < data.length; i++) {
            out.data().setAtIndex(java.lang.foreign.ValueLayout.JAVA_INT, i, indices[i]);
        }
        return out;
    }

    /**
     * Fancy indexing — selects elements by integer index array from a 1-D array.
     *
     * @param a       input array
     * @param indices integer indices to select
     * @return 1-D array of selected elements
     */
    public static NdArray fancyIndex(final NdArray a, final int[] indices) {
        NdArray c = a.contiguous();
        NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, indices.length);
        for (int i = 0; i < indices.length; i++) {
            out.flatSetFloat(i, c.flatGetFloat(indices[i]));
        }
        return out;
    }

    /**
     * Sets values at flat indices in-place ({@code numpy.put}).
     *
     * @param a       target array (modified in-place)
     * @param indices flat indices to set
     * @param values  values to write (cycled if shorter than indices)
     */
    public static void put(final NdArray a, final int[] indices, final float[] values) {
        for (int i = 0; i < indices.length; i++) {
            a.flatSetFloat(indices[i], values[i % values.length]);
        }
    }
}
