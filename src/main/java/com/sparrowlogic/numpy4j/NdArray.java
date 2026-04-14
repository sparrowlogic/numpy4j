package com.sparrowlogic.numpy4j;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;

/**
 * N-dimensional array backed by off-heap memory via FFM Arena.
 * Mirrors numpy.ndarray with C-contiguous (row-major) layout.
 */
public final class NdArray {
    private final MemorySegment data;
    private final Arena arena;
    private final DType dtype;
    private final int[] shape;
    private final int[] strides; // element strides (not byte strides)
    private final long offset;   // element offset into data
    private final long size;     // total elements

    /** Internal constructor — use factory methods. */
    NdArray(MemorySegment data, Arena arena, DType dtype, int[] shape, int[] strides, long offset) {
        this.data = data;
        this.arena = arena;
        this.dtype = dtype;
        this.shape = shape.clone();
        this.strides = strides.clone();
        this.offset = offset;
        this.size = ShapeUtils.size(shape);
    }

    // ── Accessors ──

    public DType dtype() { return dtype; }
    public int[] shape() { return shape.clone(); }
    public int[] strides() { return strides.clone(); }
    public int ndim() { return shape.length; }
    public long size() { return size; }
    public MemorySegment data() { return data; }
    public long offset() { return offset; }
    public Arena arena() { return arena; }

    /** Shape along a specific axis. */
    public int shape(int axis) {
        if (axis < 0) axis += shape.length;
        return shape[axis];
    }

    // ── Scalar element access ──

    public float getFloat(long... indices) {
        return data.getAtIndex(ValueLayout.JAVA_FLOAT, flatIndex(indices));
    }

    public double getDouble(long... indices) {
        return data.getAtIndex(ValueLayout.JAVA_DOUBLE, flatIndex(indices));
    }

    public int getInt(long... indices) {
        return data.getAtIndex(ValueLayout.JAVA_INT, flatIndex(indices));
    }

    public long getLong(long... indices) {
        return data.getAtIndex(ValueLayout.JAVA_LONG, flatIndex(indices));
    }

    public void setFloat(float value, long... indices) {
        data.setAtIndex(ValueLayout.JAVA_FLOAT, flatIndex(indices), value);
    }

    public void setDouble(double value, long... indices) {
        data.setAtIndex(ValueLayout.JAVA_DOUBLE, flatIndex(indices), value);
    }

    public void setInt(int value, long... indices) {
        data.setAtIndex(ValueLayout.JAVA_INT, flatIndex(indices), value);
    }

    public void setLong(long value, long... indices) {
        data.setAtIndex(ValueLayout.JAVA_LONG, flatIndex(indices), value);
    }

    /** Flat element access by linear index. */
    public float flatGetFloat(long i) {
        return data.getAtIndex(ValueLayout.JAVA_FLOAT, offset + i);
    }

    public void flatSetFloat(long i, float v) {
        data.setAtIndex(ValueLayout.JAVA_FLOAT, offset + i, v);
    }

    public double flatGetDouble(long i) {
        return data.getAtIndex(ValueLayout.JAVA_DOUBLE, offset + i);
    }

    public void flatSetDouble(long i, double v) {
        data.setAtIndex(ValueLayout.JAVA_DOUBLE, offset + i, v);
    }

    private long flatIndex(long[] indices) {
        long idx = offset;
        for (int i = 0; i < indices.length; i++) {
            long ix = indices[i] < 0 ? indices[i] + shape[i] : indices[i];
            idx += ix * strides[i];
        }
        return idx;
    }

    /** Whether this array is C-contiguous. */
    public boolean isContiguous() {
        return Arrays.equals(strides, ShapeUtils.cStrides(shape));
    }

    /** Return a contiguous copy if not already contiguous. */
    public NdArray contiguous() {
        if (isContiguous() && offset == 0) return this;
        NdArray out = NdArrayFactory.empty(arena, dtype, shape);
        for (long i = 0; i < size; i++) {
            int[] idx = ShapeUtils.unravelIndex(i, shape);
            long srcFlat = offset;
            for (int d = 0; d < idx.length; d++) srcFlat += (long) idx[d] * strides[d];
            long dstFlat = i;
            switch (dtype) {
                case FLOAT32 -> out.data.setAtIndex(ValueLayout.JAVA_FLOAT, dstFlat,
                        data.getAtIndex(ValueLayout.JAVA_FLOAT, srcFlat));
                case FLOAT64 -> out.data.setAtIndex(ValueLayout.JAVA_DOUBLE, dstFlat,
                        data.getAtIndex(ValueLayout.JAVA_DOUBLE, srcFlat));
                case INT32 -> out.data.setAtIndex(ValueLayout.JAVA_INT, dstFlat,
                        data.getAtIndex(ValueLayout.JAVA_INT, srcFlat));
                case INT64 -> out.data.setAtIndex(ValueLayout.JAVA_LONG, dstFlat,
                        data.getAtIndex(ValueLayout.JAVA_LONG, srcFlat));
                case BOOL -> out.data.set(ValueLayout.JAVA_BYTE, dstFlat,
                        data.get(ValueLayout.JAVA_BYTE, srcFlat));
            }
        }
        return out;
    }

    /** Copy data to a float array (for FLOAT32 arrays). */
    public float[] toFloatArray() {
        NdArray c = contiguous();
        float[] out = new float[(int) size];
        MemorySegment.copy(c.data, ValueLayout.JAVA_FLOAT, 0, out, 0, out.length);
        return out;
    }

    /** Copy data to a double array (for FLOAT64 arrays). */
    public double[] toDoubleArray() {
        NdArray c = contiguous();
        double[] out = new double[(int) size];
        MemorySegment.copy(c.data, ValueLayout.JAVA_DOUBLE, 0, out, 0, out.length);
        return out;
    }

    /** Copy data to an int array (for INT32 arrays). */
    public int[] toIntArray() {
        NdArray c = contiguous();
        int[] out = new int[(int) size];
        MemorySegment.copy(c.data, ValueLayout.JAVA_INT, 0, out, 0, out.length);
        return out;
    }

    @Override
    public String toString() {
        return "NdArray(shape=" + Arrays.toString(shape) + ", dtype=" + dtype + ")";
    }

    // ── Proxy methods for numpy-style chaining ──
    // These delegate to the static utility classes so users can write a.reshape(3,4) instead of ShapeOps.reshape(a, 3, 4)

    /** numpy ndarray.reshape() */
    public NdArray reshape(int... newShape) { return ShapeOps.reshape(this, newShape); }

    /** numpy ndarray.T / ndarray.transpose() */
    public NdArray transpose() { return ShapeOps.transpose(this); }

    /** numpy ndarray.transpose(axes) */
    public NdArray transpose(int... axes) { return ShapeOps.transpose(this, axes); }

    /** numpy ndarray.flatten() */
    public NdArray flatten() { return ShapeOps.flatten(this); }

    /** numpy ndarray.ravel() */
    public NdArray ravel() { return ShapeOps.ravel(this); }

    /** numpy ndarray.squeeze() */
    public NdArray squeeze() { return ShapeOps.squeeze(this); }

    /** numpy ndarray.sum() */
    public float sum() { return Reductions.sum(this); }

    /** numpy ndarray.sum(axis) */
    public NdArray sum(int axis) { return Reductions.sum(this, axis); }

    /** numpy ndarray.mean() */
    public float mean() { return Reductions.mean(this); }

    /** numpy ndarray.std() */
    public float std() { return Reductions.std(this); }

    /** numpy ndarray.var() */
    public float var() { return Reductions.var(this); }

    /** numpy ndarray.min() */
    public float min() { return Reductions.min(this); }

    /** numpy ndarray.max() */
    public float max() { return Reductions.max(this); }

    /** numpy ndarray.argmin() */
    public long argmin() { return Reductions.argmin(this); }

    /** numpy ndarray.argmax() */
    public long argmax() { return Reductions.argmax(this); }

    /** numpy ndarray.clip(min, max) */
    public NdArray clip(float min, float max) { return Ufunc.clip(this, min, max); }

    /** numpy ndarray.sort() — returns sorted copy */
    public NdArray sort() { return IndexOps.sort(this); }

    /** numpy ndarray.argsort() */
    public NdArray argsort() { return IndexOps.argsort(this); }

    /** numpy ndarray.dot(other) */
    public float dot(NdArray other) { return MatMul.dot(this, other); }

    /** numpy ndarray @ other (matmul) */
    public NdArray matmul(NdArray other) { return MatMul.matmul(this, other); }

    /** numpy ndarray.copy() — always returns a new independent copy */
    public NdArray copy() {
        NdArray out = NdArrayFactory.empty(arena, dtype, shape);
        out.data.copyFrom(contiguous().data.asSlice(0, dtype.totalBytes(size)));
        return out;
    }

    /** numpy ndarray.astype — returns FLOAT32 copy (dtype conversion placeholder) */
    public NdArray astype(DType target) {
        if (target == dtype) return contiguous();
        // For now, only FLOAT32 storage is fully supported
        return contiguous();
    }
}
