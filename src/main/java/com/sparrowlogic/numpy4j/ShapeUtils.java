package com.sparrowlogic.numpy4j;

import java.util.Arrays;

/**
 * Shape and stride utilities matching NumPy's C-contiguous (row-major) layout.
 */
public final class ShapeUtils {
    private ShapeUtils() {
    }

    /**
     * Total number of elements for a given shape.
     */
    public static long size(final int[] shape) {
        long s = 1;
        for (final int d : shape) {
            s *= d;
        }
        return s;
    }

    /**
     * C-contiguous strides in elements (not bytes).
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
     * Convert multi-dimensional index to flat offset using strides.
     */
    public static long flatIndex(final int[] indices, final int[] strides) {
        long idx = 0;
        for (int i = 0; i < indices.length; i++) {
            idx += (long) indices[i] * strides[i];
        }
        return idx;
    }

    /**
     * Convert flat index to multi-dimensional indices.
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
     * Compute broadcast shape for two shapes per NumPy rules.
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
     * Compute broadcast strides (0 for broadcast dimensions).
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
