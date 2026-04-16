package com.sparrowlogic.numpy4j;

import java.util.Arrays;

/**
 * Shape, stride, and broadcast utilities matching NumPy's C-contiguous (row-major) layout.
 *
 * <p>All stride values are in <em>elements</em>, not bytes. This matches the convention
 * used by {@link NdArray} internally.</p>
 *
 * @see NdArray
 */
public final class ShapeUtils {
    private ShapeUtils() {
    }

    /**
     * Returns the total number of elements for a given shape.
     *
     * @param shape dimension sizes
     * @return product of all dimensions
     */
    public static long size(final int[] shape) {
        long s = 1;
        for (final int d : shape) {
            s *= d;
        }
        return s;
    }

    /**
     * Computes C-contiguous (row-major) strides in elements for the given shape.
     *
     * <p>For shape {@code [2, 3, 4]}, returns {@code [12, 4, 1]}.</p>
     *
     * @param shape dimension sizes
     * @return strides array (same length as {@code shape})
     */
    public static int[] cStrides(final int[] shape) {
        int[] strides = new int[shape.length];
        if (shape.length > 0) {
            strides[shape.length - 1] = 1;
            for (int i = shape.length - 2; i >= 0; i--) {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        return strides;
    }

    /**
     * Converts a multi-dimensional index to a flat element offset using the given strides.
     *
     * @param indices multi-dimensional index (one per axis)
     * @param strides element strides (one per axis)
     * @return flat element offset
     */
    public static long flatIndex(final int[] indices, final int[] strides) {
        long idx = 0;
        for (int i = 0; i < indices.length; i++) {
            idx += (long) indices[i] * strides[i];
        }
        return idx;
    }

    /**
     * Converts a flat element index back to multi-dimensional indices.
     *
     * <p>Inverse of {@link #flatIndex(int[], int[])} for C-contiguous strides.</p>
     *
     * @param flatIdx flat element index
     * @param shape   dimension sizes
     * @return multi-dimensional indices (one per axis)
     */
    public static int[] unravelIndex(final long flatIdx, final int[] shape) {
        int[] indices = new int[shape.length];
        long remaining = flatIdx;
        for (int i = shape.length - 1; i >= 0; i--) {
            indices[i] = (int) (remaining % shape[i]);
            remaining /= shape[i];
        }
        return indices;
    }

    /**
     * Computes the broadcast-compatible output shape for two input shapes per NumPy rules.
     *
     * <p>Shapes are aligned from the trailing dimension. Dimensions of size 1 are
     * stretched to match the other operand.</p>
     *
     * @param a first shape
     * @param b second shape
     * @return broadcast output shape
     * @throws IllegalArgumentException if shapes are not broadcast-compatible
     */
    public static int[] broadcastShape(final int[] a, final int[] b) {
        int ndim = Math.max(a.length, b.length);
        int[] result = new int[ndim];
        for (int i = 0; i < ndim; i++) {
            int da = i < ndim - a.length ? 1 : a[i - (ndim - a.length)];
            int db = i < ndim - b.length ? 1 : b[i - (ndim - b.length)];
            if (da != db && da != 1 && db != 1) {
                throw new IllegalArgumentException(
                        "Cannot broadcast shapes " + Arrays.toString(a) + " and " + Arrays.toString(b));
            }
            result[i] = Math.max(da, db);
        }
        return result;
    }

    /**
     * Computes broadcast strides for an array being broadcast to a target shape.
     *
     * <p>Dimensions of size 1 in the source get stride 0 (repeated reads).
     * Leading dimensions not present in the source also get stride 0.</p>
     *
     * @param shape       original array shape
     * @param strides     original array strides
     * @param targetShape broadcast target shape
     * @return broadcast strides (same length as {@code targetShape})
     */
    public static int[] broadcastStrides(final int[] shape, final int[] strides, final int[] targetShape) {
        int[] result = new int[targetShape.length];
        int offset = targetShape.length - shape.length;
        for (int i = 0; i < targetShape.length; i++) {
            if (i < offset || shape[i - offset] == 1) {
                result[i] = 0;
            } else {
                result[i] = strides[i - offset];
            }
        }
        return result;
    }
}
