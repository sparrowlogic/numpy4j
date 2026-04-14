package com.sparrowlogic.numpy4j;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Factory methods for creating NdArrays. Mirrors numpy.zeros, numpy.ones, numpy.arange, etc.
 */
public final class NdArrayFactory {
    private NdArrayFactory() {
    }

    /**
     * Allocate uninitialized array.
     */
    public static NdArray empty(final Arena arena, final DType dtype, final int... shape) {
        long size = ShapeUtils.size(shape);
        MemorySegment seg = arena.allocate(dtype.totalBytes(size), dtype.bytes());
        return new NdArray(seg, arena, dtype, shape, ShapeUtils.cStrides(shape), 0);
    }

    /**
     * All zeros.
     */
    public static NdArray zeros(final Arena arena, final DType dtype, final int... shape) {
        NdArray a = empty(arena, dtype, shape);
        a.data().fill((byte) 0);
        return a;
    }

    /**
     * All ones.
     */
    public static NdArray ones(final Arena arena, final DType dtype, final int... shape) {
        return full(arena, dtype, 1.0, shape);
    }

    /**
     * Fill with scalar value.
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
     * Create from float array.
     */
    public static NdArray array(final Arena arena, final float[] data, final int... shape) {
        int[] resolvedShape = shape.length == 0 ? new int[]{data.length} : shape;
        NdArray a = empty(arena, DType.FLOAT32, resolvedShape);
        MemorySegment.copy(data, 0, a.data(), ValueLayout.JAVA_FLOAT, 0, data.length);
        return a;
    }

    /**
     * Create from double array.
     */
    public static NdArray array(final Arena arena, final double[] data, final int... shape) {
        int[] resolvedShape = shape.length == 0 ? new int[]{data.length} : shape;
        NdArray a = empty(arena, DType.FLOAT64, resolvedShape);
        MemorySegment.copy(data, 0, a.data(), ValueLayout.JAVA_DOUBLE, 0, data.length);
        return a;
    }

    /**
     * Create from int array.
     */
    public static NdArray array(final Arena arena, final int[] data, final int... shape) {
        int[] resolvedShape = shape.length == 0 ? new int[]{data.length} : shape;
        NdArray a = empty(arena, DType.INT32, resolvedShape);
        MemorySegment.copy(data, 0, a.data(), ValueLayout.JAVA_INT, 0, data.length);
        return a;
    }

    /**
     * numpy.arange — float32
     */
    public static NdArray arange(final Arena arena, final float start, final float stop, final float step) {
        int n = Math.max(0, (int) Math.ceil((stop - start) / step));
        NdArray a = empty(arena, DType.FLOAT32, n);
        for (int i = 0; i < n; i++) {
            a.flatSetFloat(i, start + i * step);
        }
        return a;
    }

    public static NdArray arange(final Arena arena, final int n) {
        return arange(arena, 0f, (float) n, 1f);
    }

    /**
     * numpy.linspace — float64
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
     * numpy.eye — identity matrix, float32
     */
    public static NdArray eye(final Arena arena, final int n) {
        return eye(arena, n, DType.FLOAT32);
    }

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
     * numpy.zeros_like
     */
    public static NdArray zerosLike(final NdArray a) {
        return zeros(a.arena(), a.dtype(), a.shape());
    }

    /**
     * numpy.ones_like
     */
    public static NdArray onesLike(final NdArray a) {
        return ones(a.arena(), a.dtype(), a.shape());
    }

    /**
     * numpy.full_like
     */
    public static NdArray fullLike(final NdArray a, final double value) {
        return full(a.arena(), a.dtype(), value, a.shape());
    }

    /**
     * numpy.empty_like
     */
    public static NdArray emptyLike(final NdArray a) {
        return empty(a.arena(), a.dtype(), a.shape());
    }

    /**
     * numpy.logspace
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
     * numpy.geomspace
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
     * numpy.fromfunction
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

    @FunctionalInterface
    public interface IndexFunction {
        float apply(int[] indices);
    }
}
