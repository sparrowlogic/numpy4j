package com.sparrowlogic.numpy4j;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Factory methods for creating {@link NdArray} instances.
 *
 * <p>Mirrors NumPy array creation routines: {@code numpy.zeros}, {@code numpy.ones},
 * {@code numpy.arange}, {@code numpy.linspace}, {@code numpy.eye}, etc.</p>
 *
 * <p>All methods require an {@link Arena} that owns the allocated off-heap memory.</p>
 *
 * @see NdArray
 * @see DType
 */
public final class NdArrayFactory {
    private NdArrayFactory() {
    }

    /**
     * Allocates an uninitialized array ({@code numpy.empty}).
     *
     * @param arena memory arena for allocation
     * @param dtype element data type
     * @param shape dimension sizes
     * @return a new uninitialized array
     */
    public static NdArray empty(final Arena arena, final DType dtype, final int... shape) {
        long size = ShapeUtils.size(shape);
        MemorySegment seg = arena.allocate(dtype.totalBytes(size), dtype.bytes());
        return new NdArray(seg, arena, dtype, shape, ShapeUtils.cStrides(shape), 0);
    }

    /**
     * Creates an array filled with zeros ({@code numpy.zeros}).
     *
     * @param arena memory arena for allocation
     * @param dtype element data type
     * @param shape dimension sizes
     * @return a new zero-filled array
     */
    public static NdArray zeros(final Arena arena, final DType dtype, final int... shape) {
        NdArray a = empty(arena, dtype, shape);
        a.data().fill((byte) 0);
        return a;
    }

    /**
     * Creates an array filled with ones ({@code numpy.ones}).
     *
     * @param arena memory arena for allocation
     * @param dtype element data type
     * @param shape dimension sizes
     * @return a new array filled with 1.0
     */
    public static NdArray ones(final Arena arena, final DType dtype, final int... shape) {
        return full(arena, dtype, 1.0, shape);
    }

    /**
     * Creates an array filled with a scalar value ({@code numpy.full}).
     *
     * @param arena memory arena for allocation
     * @param dtype element data type
     * @param value fill value (cast to the target dtype)
     * @param shape dimension sizes
     * @return a new array filled with {@code value}
     */
    public static NdArray full(final Arena arena, final DType dtype, final double value, final int... shape) {
        NdArray a = empty(arena, dtype, shape);
        fillArray(a, dtype, value);
        return a;
    }

    private static void fillArray(final NdArray a, final DType dtype, final double value) {
        long size = a.size();
        switch (dtype) {
            case FLOAT32 -> fillFloat32(a, size, (float) value);
            case FLOAT64 -> fillFloat64(a, size, value);
            case INT32 -> fillInt32(a, size, (int) value);
            case INT64 -> fillInt64(a, size, (long) value);
            case BOOL -> fillBool(a, size, value != 0 ? (byte) 1 : (byte) 0);
            default -> throw new UnsupportedOperationException("Unsupported dtype: " + dtype);
        }
    }

    private static void fillFloat32(final NdArray a, final long size, final float value) {
        for (long i = 0; i < size; i++) {
            a.flatSetFloat(i, value);
        }
    }

    private static void fillFloat64(final NdArray a, final long size, final double value) {
        for (long i = 0; i < size; i++) {
            a.flatSetDouble(i, value);
        }
    }

    private static void fillInt32(final NdArray a, final long size, final int value) {
        for (long i = 0; i < size; i++) {
            a.data().setAtIndex(ValueLayout.JAVA_INT, i, value);
        }
    }

    private static void fillInt64(final NdArray a, final long size, final long value) {
        for (long i = 0; i < size; i++) {
            a.data().setAtIndex(ValueLayout.JAVA_LONG, i, value);
        }
    }

    private static void fillBool(final NdArray a, final long size, final byte value) {
        for (long i = 0; i < size; i++) {
            a.data().set(ValueLayout.JAVA_BYTE, i, value);
        }
    }

    /**
     * Creates a FLOAT32 array from a {@code float[]} ({@code numpy.array}).
     *
     * @param arena memory arena for allocation
     * @param data  source data
     * @param shape dimension sizes (defaults to 1-D if omitted)
     * @return a new FLOAT32 array
     */
    public static NdArray array(final Arena arena, final float[] data, final int... shape) {
        int[] resolvedShape = shape.length == 0 ? new int[]{data.length} : shape;
        NdArray a = empty(arena, DType.FLOAT32, resolvedShape);
        MemorySegment.copy(data, 0, a.data(), ValueLayout.JAVA_FLOAT, 0, data.length);
        return a;
    }

    /**
     * Creates a FLOAT64 array from a {@code double[]} ({@code numpy.array}).
     *
     * @param arena memory arena for allocation
     * @param data  source data
     * @param shape dimension sizes (defaults to 1-D if omitted)
     * @return a new FLOAT64 array
     */
    public static NdArray array(final Arena arena, final double[] data, final int... shape) {
        int[] resolvedShape = shape.length == 0 ? new int[]{data.length} : shape;
        NdArray a = empty(arena, DType.FLOAT64, resolvedShape);
        MemorySegment.copy(data, 0, a.data(), ValueLayout.JAVA_DOUBLE, 0, data.length);
        return a;
    }

    /**
     * Creates an INT32 array from an {@code int[]} ({@code numpy.array}).
     *
     * @param arena memory arena for allocation
     * @param data  source data
     * @param shape dimension sizes (defaults to 1-D if omitted)
     * @return a new INT32 array
     */
    public static NdArray array(final Arena arena, final int[] data, final int... shape) {
        int[] resolvedShape = shape.length == 0 ? new int[]{data.length} : shape;
        NdArray a = empty(arena, DType.INT32, resolvedShape);
        MemorySegment.copy(data, 0, a.data(), ValueLayout.JAVA_INT, 0, data.length);
        return a;
    }

    /**
     * Creates a FLOAT32 range array ({@code numpy.arange}).
     *
     * @param arena memory arena for allocation
     * @param start start value (inclusive)
     * @param stop  stop value (exclusive)
     * @param step  step between values
     * @return a new 1-D FLOAT32 array
     */
    public static NdArray arange(final Arena arena, final float start, final float stop, final float step) {
        int n = Math.max(0, (int) Math.ceil((stop - start) / step));
        NdArray a = empty(arena, DType.FLOAT32, n);
        for (int i = 0; i < n; i++) {
            a.flatSetFloat(i, start + i * step);
        }
        return a;
    }

    /**
     * Creates a FLOAT32 range array from 0 to {@code n} ({@code numpy.arange(n)}).
     *
     * @param arena memory arena for allocation
     * @param n     number of elements
     * @return a new 1-D FLOAT32 array {@code [0, 1, ..., n-1]}
     */
    public static NdArray arange(final Arena arena, final int n) {
        return arange(arena, 0f, (float) n, 1f);
    }

    /**
     * Creates a FLOAT64 array of evenly spaced values ({@code numpy.linspace}).
     *
     * @param arena memory arena for allocation
     * @param start start value (inclusive)
     * @param stop  stop value (inclusive)
     * @param num   number of samples
     * @return a new 1-D FLOAT64 array
     */
    public static NdArray linspace(final Arena arena, final double start, final double stop, final int num) {
        NdArray a = empty(arena, DType.FLOAT64, num);
        double step = num > 1 ? (stop - start) / (num - 1) : 0;
        for (int i = 0; i < num; i++) {
            a.flatSetDouble(i, start + i * step);
        }
        return a;
    }

    /**
     * Creates a FLOAT32 identity matrix ({@code numpy.eye}).
     *
     * @param arena memory arena for allocation
     * @param n     matrix size (n × n)
     * @return a new identity matrix
     */
    public static NdArray eye(final Arena arena, final int n) {
        return eye(arena, n, DType.FLOAT32);
    }

    /**
     * Creates an identity matrix with the specified dtype ({@code numpy.eye}).
     *
     * @param arena memory arena for allocation
     * @param n     matrix size (n × n)
     * @param dtype element data type
     * @return a new identity matrix
     */
    public static NdArray eye(final Arena arena, final int n, final DType dtype) {
        NdArray a = zeros(arena, dtype, n, n);
        for (int i = 0; i < n; i++) {
            switch (dtype) {
                case FLOAT32 -> a.setFloat(1f, i, i);
                case FLOAT64 -> a.setDouble(1.0, i, i);
                case INT32 -> a.setInt(1, i, i);
                case INT64 -> a.setLong(1L, i, i);
                default -> {
                }
            }
        }
        return a;
    }

    /**
     * Creates a zero-filled array with the same shape and dtype as the input ({@code numpy.zeros_like}).
     *
     * @param a template array
     * @return a new zero-filled array
     */
    public static NdArray zerosLike(final NdArray a) {
        return zeros(a.arena(), a.dtype(), a.shape());
    }

    /**
     * Creates a ones-filled array with the same shape and dtype as the input ({@code numpy.ones_like}).
     *
     * @param a template array
     * @return a new ones-filled array
     */
    public static NdArray onesLike(final NdArray a) {
        return ones(a.arena(), a.dtype(), a.shape());
    }

    /**
     * Creates a scalar-filled array with the same shape and dtype as the input ({@code numpy.full_like}).
     *
     * @param a     template array
     * @param value fill value
     * @return a new filled array
     */
    public static NdArray fullLike(final NdArray a, final double value) {
        return full(a.arena(), a.dtype(), value, a.shape());
    }

    /**
     * Creates an uninitialized array with the same shape and dtype as the input ({@code numpy.empty_like}).
     *
     * @param a template array
     * @return a new uninitialized array
     */
    public static NdArray emptyLike(final NdArray a) {
        return empty(a.arena(), a.dtype(), a.shape());
    }

    /**
     * Creates a FLOAT32 array of values spaced evenly on a log scale ({@code numpy.logspace}).
     *
     * @param arena memory arena for allocation
     * @param start exponent of the start value ({@code 10^start})
     * @param stop  exponent of the stop value ({@code 10^stop})
     * @param num   number of samples
     * @return a new 1-D FLOAT32 array
     */
    public static NdArray logspace(final Arena arena, final double start, final double stop, final int num) {
        NdArray a = empty(arena, DType.FLOAT32, num);
        double step = num > 1 ? (stop - start) / (num - 1) : 0;
        for (int i = 0; i < num; i++) {
            a.flatSetFloat(i, (float) Math.pow(10, start + i * step));
        }
        return a;
    }

    /**
     * Creates a FLOAT32 array of values spaced evenly on a geometric scale ({@code numpy.geomspace}).
     *
     * @param arena memory arena for allocation
     * @param start start value (must be positive)
     * @param stop  stop value (must be positive)
     * @param num   number of samples
     * @return a new 1-D FLOAT32 array
     */
    public static NdArray geomspace(final Arena arena, final double start, final double stop, final int num) {
        NdArray a = empty(arena, DType.FLOAT32, num);
        double logStart = Math.log(start);
        double logStop = Math.log(stop);
        double step = num > 1 ? (logStop - logStart) / (num - 1) : 0;
        for (int i = 0; i < num; i++) {
            a.flatSetFloat(i, (float) Math.exp(logStart + i * step));
        }
        return a;
    }

    /**
     * Creates a FLOAT32 array by evaluating a function at each index ({@code numpy.fromfunction}).
     *
     * @param arena memory arena for allocation
     * @param fn    function mapping multi-dimensional indices to a float value
     * @param shape dimension sizes
     * @return a new FLOAT32 array
     */
    public static NdArray fromfunction(final Arena arena, final IndexFunction fn, final int... shape) {
        NdArray a = empty(arena, DType.FLOAT32, shape);
        long size = ShapeUtils.size(shape);
        for (long i = 0; i < size; i++) {
            int[] idx = ShapeUtils.unravelIndex(i, shape);
            a.flatSetFloat(i, fn.apply(idx));
        }
        return a;
    }

    /**
     * Functional interface for index-based array construction.
     *
     * @see #fromfunction(Arena, IndexFunction, int...)
     */
    @FunctionalInterface
    public interface IndexFunction {

        /**
         * Computes the element value for the given multi-dimensional indices.
         *
         * @param indices array indices (one per dimension)
         * @return the computed float value
         */
        float apply(int[] indices);
    }
}
