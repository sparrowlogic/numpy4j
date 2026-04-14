package com.sparrowlogic.numpy4j;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Factory methods for creating NdArrays. Mirrors numpy.zeros, numpy.ones, numpy.arange, etc.
 */
public final class NdArrayFactory {
    private NdArrayFactory() {}

    /** Allocate uninitialized array. */
    public static NdArray empty(Arena arena, DType dtype, int... shape) {
        long size = ShapeUtils.size(shape);
        MemorySegment seg = arena.allocate(dtype.totalBytes(size), dtype.bytes());
        return new NdArray(seg, arena, dtype, shape, ShapeUtils.cStrides(shape), 0);
    }

    /** All zeros. */
    public static NdArray zeros(Arena arena, DType dtype, int... shape) {
        NdArray a = empty(arena, dtype, shape);
        a.data().fill((byte) 0);
        return a;
    }

    /** All ones. */
    public static NdArray ones(Arena arena, DType dtype, int... shape) {
        return full(arena, dtype, 1.0, shape);
    }

    /** Fill with scalar value. */
    public static NdArray full(Arena arena, DType dtype, double value, int... shape) {
        NdArray a = empty(arena, dtype, shape);
        long size = a.size();
        switch (dtype) {
            case FLOAT32 -> { for (long i = 0; i < size; i++) a.flatSetFloat(i, (float) value); }
            case FLOAT64 -> { for (long i = 0; i < size; i++) a.flatSetDouble(i, value); }
            case INT32 -> {
                int v = (int) value;
                for (long i = 0; i < size; i++) a.data().setAtIndex(ValueLayout.JAVA_INT, i, v);
            }
            case INT64 -> {
                long v = (long) value;
                for (long i = 0; i < size; i++) a.data().setAtIndex(ValueLayout.JAVA_LONG, i, v);
            }
            case BOOL -> {
                byte v = value != 0 ? (byte) 1 : (byte) 0;
                for (long i = 0; i < size; i++) a.data().set(ValueLayout.JAVA_BYTE, i, v);
            }
        }
        return a;
    }

    /** Create from float array. */
    public static NdArray array(Arena arena, float[] data, int... shape) {
        if (shape.length == 0) shape = new int[]{data.length};
        NdArray a = empty(arena, DType.FLOAT32, shape);
        MemorySegment.copy(data, 0, a.data(), ValueLayout.JAVA_FLOAT, 0, data.length);
        return a;
    }

    /** Create from double array. */
    public static NdArray array(Arena arena, double[] data, int... shape) {
        if (shape.length == 0) shape = new int[]{data.length};
        NdArray a = empty(arena, DType.FLOAT64, shape);
        MemorySegment.copy(data, 0, a.data(), ValueLayout.JAVA_DOUBLE, 0, data.length);
        return a;
    }

    /** Create from int array. */
    public static NdArray array(Arena arena, int[] data, int... shape) {
        if (shape.length == 0) shape = new int[]{data.length};
        NdArray a = empty(arena, DType.INT32, shape);
        MemorySegment.copy(data, 0, a.data(), ValueLayout.JAVA_INT, 0, data.length);
        return a;
    }

    /** numpy.arange — float32 */
    public static NdArray arange(Arena arena, float start, float stop, float step) {
        int n = Math.max(0, (int) Math.ceil((stop - start) / step));
        NdArray a = empty(arena, DType.FLOAT32, n);
        for (int i = 0; i < n; i++) a.flatSetFloat(i, start + i * step);
        return a;
    }

    public static NdArray arange(Arena arena, int n) {
        return arange(arena, 0f, (float) n, 1f);
    }

    /** numpy.linspace — float64 */
    public static NdArray linspace(Arena arena, double start, double stop, int num) {
        NdArray a = empty(arena, DType.FLOAT64, num);
        double step = num > 1 ? (stop - start) / (num - 1) : 0;
        for (int i = 0; i < num; i++) a.flatSetDouble(i, start + i * step);
        return a;
    }

    /** numpy.eye — identity matrix, float32 */
    public static NdArray eye(Arena arena, int n) {
        return eye(arena, n, DType.FLOAT32);
    }

    public static NdArray eye(Arena arena, int n, DType dtype) {
        NdArray a = zeros(arena, dtype, n, n);
        for (int i = 0; i < n; i++) {
            switch (dtype) {
                case FLOAT32 -> a.setFloat(1f, i, i);
                case FLOAT64 -> a.setDouble(1.0, i, i);
                case INT32 -> a.setInt(1, i, i);
                case INT64 -> a.setLong(1L, i, i);
                default -> {}
            }
        }
        return a;
    }

    /** numpy.zeros_like */
    public static NdArray zerosLike(NdArray a) { return zeros(a.arena(), a.dtype(), a.shape()); }

    /** numpy.ones_like */
    public static NdArray onesLike(NdArray a) { return ones(a.arena(), a.dtype(), a.shape()); }

    /** numpy.full_like */
    public static NdArray fullLike(NdArray a, double value) { return full(a.arena(), a.dtype(), value, a.shape()); }

    /** numpy.empty_like */
    public static NdArray emptyLike(NdArray a) { return empty(a.arena(), a.dtype(), a.shape()); }

    /** numpy.logspace */
    public static NdArray logspace(Arena arena, double start, double stop, int num) {
        NdArray a = empty(arena, DType.FLOAT32, num);
        double step = num > 1 ? (stop - start) / (num - 1) : 0;
        for (int i = 0; i < num; i++) a.flatSetFloat(i, (float) Math.pow(10, start + i * step));
        return a;
    }

    /** numpy.geomspace */
    public static NdArray geomspace(Arena arena, double start, double stop, int num) {
        NdArray a = empty(arena, DType.FLOAT32, num);
        double logStart = Math.log(start), logStop = Math.log(stop);
        double step = num > 1 ? (logStop - logStart) / (num - 1) : 0;
        for (int i = 0; i < num; i++) a.flatSetFloat(i, (float) Math.exp(logStart + i * step));
        return a;
    }

    /** numpy.fromfunction */
    @FunctionalInterface
    public interface IndexFunction { float apply(int[] indices); }

    public static NdArray fromfunction(Arena arena, IndexFunction fn, int... shape) {
        NdArray a = empty(arena, DType.FLOAT32, shape);
        long size = ShapeUtils.size(shape);
        for (long i = 0; i < size; i++) {
            int[] idx = ShapeUtils.unravelIndex(i, shape);
            a.flatSetFloat(i, fn.apply(idx));
        }
        return a;
    }
}
