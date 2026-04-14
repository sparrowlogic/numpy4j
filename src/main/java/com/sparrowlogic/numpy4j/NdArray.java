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

    private final int[] strides;

    private final long offset;

    private final long size;

    /**
     * Internal constructor — use factory methods.
     */
    NdArray(
            final MemorySegment data,
            final Arena arena,
            final DType dtype,
            final int[] shape,
            final int[] strides,
            final long offset
    ) {
        this.data = data;
        this.arena = arena;
        this.dtype = dtype;
        this.shape = shape.clone();
        this.strides = strides.clone();
        this.offset = offset;
        this.size = ShapeUtils.size(shape);
    }

    // ── Accessors ──

    public DType dtype() {
        return this.dtype;
    }

    public int[] shape() {
        return this.shape.clone();
    }

    /**
     * Shape along a specific axis.
     */
    public int shape(final int axis) {
        int ax = axis < 0 ? axis + this.shape.length : axis;
        return this.shape[ax];
    }

    public int[] strides() {
        return this.strides.clone();
    }

    public int ndim() {
        return this.shape.length;
    }

    public long size() {
        return this.size;
    }

    public MemorySegment data() {
        return this.data;
    }

    public long offset() {
        return this.offset;
    }

    public Arena arena() {
        return this.arena;
    }

    // ── Scalar element access ──

    public float getFloat(final long... indices) {
        return this.data.getAtIndex(ValueLayout.JAVA_FLOAT, this.flatIndex(indices));
    }

    public double getDouble(final long... indices) {
        return this.data.getAtIndex(ValueLayout.JAVA_DOUBLE, this.flatIndex(indices));
    }

    public int getInt(final long... indices) {
        return this.data.getAtIndex(ValueLayout.JAVA_INT, this.flatIndex(indices));
    }

    public long getLong(final long... indices) {
        return this.data.getAtIndex(ValueLayout.JAVA_LONG, this.flatIndex(indices));
    }

    public void setFloat(final float value, final long... indices) {
        this.data.setAtIndex(ValueLayout.JAVA_FLOAT, this.flatIndex(indices), value);
    }

    public void setDouble(final double value, final long... indices) {
        this.data.setAtIndex(ValueLayout.JAVA_DOUBLE, this.flatIndex(indices), value);
    }

    public void setInt(final int value, final long... indices) {
        this.data.setAtIndex(ValueLayout.JAVA_INT, this.flatIndex(indices), value);
    }

    public void setLong(final long value, final long... indices) {
        this.data.setAtIndex(ValueLayout.JAVA_LONG, this.flatIndex(indices), value);
    }

    /**
     * Flat element access by linear index.
     */
    public float flatGetFloat(final long i) {
        return this.data.getAtIndex(ValueLayout.JAVA_FLOAT, this.offset + i);
    }

    public void flatSetFloat(final long i, final float v) {
        this.data.setAtIndex(ValueLayout.JAVA_FLOAT, this.offset + i, v);
    }

    public double flatGetDouble(final long i) {
        return this.data.getAtIndex(ValueLayout.JAVA_DOUBLE, this.offset + i);
    }

    public void flatSetDouble(final long i, final double v) {
        this.data.setAtIndex(ValueLayout.JAVA_DOUBLE, this.offset + i, v);
    }

    private long flatIndex(final long[] indices) {
        long idx = this.offset;
        for (int i = 0; i < indices.length; i++) {
            long ix = indices[i] < 0 ? indices[i] + this.shape[i] : indices[i];
            idx += ix * this.strides[i];
        }
        return idx;
    }

    /**
     * Whether this array is C-contiguous.
     */
    public boolean isContiguous() {
        return Arrays.equals(this.strides, ShapeUtils.cStrides(this.shape));
    }

    /**
     * Return a contiguous copy if not already contiguous.
     */
    public NdArray contiguous() {
        if (this.isContiguous() && this.offset == 0) {
            return this;
        }
        NdArray out = NdArrayFactory.empty(this.arena, this.dtype, this.shape);
        for (long i = 0; i < this.size; i++) {
            int[] idx = ShapeUtils.unravelIndex(i, this.shape);
            long srcFlat = this.offset;
            for (int d = 0; d < idx.length; d++) {
                srcFlat += (long) idx[d] * this.strides[d];
            }
            long dstFlat = i;
            switch (this.dtype) {
                case FLOAT32 -> out.data.setAtIndex(
                        ValueLayout.JAVA_FLOAT, dstFlat,
                        this.data.getAtIndex(ValueLayout.JAVA_FLOAT, srcFlat)
                );
                case FLOAT64 -> out.data.setAtIndex(
                        ValueLayout.JAVA_DOUBLE, dstFlat,
                        this.data.getAtIndex(ValueLayout.JAVA_DOUBLE, srcFlat)
                );
                case INT32 -> out.data.setAtIndex(
                        ValueLayout.JAVA_INT, dstFlat,
                        this.data.getAtIndex(ValueLayout.JAVA_INT, srcFlat)
                );
                case INT64 -> out.data.setAtIndex(
                        ValueLayout.JAVA_LONG, dstFlat,
                        this.data.getAtIndex(ValueLayout.JAVA_LONG, srcFlat)
                );
                case BOOL -> out.data.set(
                        ValueLayout.JAVA_BYTE, dstFlat,
                        this.data.get(ValueLayout.JAVA_BYTE, srcFlat)
                );
                default -> throw new UnsupportedOperationException("Unsupported dtype: " + this.dtype);
            }
        }
        return out;
    }

    /**
     * Copy data to a float array (for FLOAT32 arrays).
     */
    public float[] toFloatArray() {
        NdArray c = this.contiguous();
        float[] out = new float[(int) this.size];
        MemorySegment.copy(c.data, ValueLayout.JAVA_FLOAT, 0, out, 0, out.length);
        return out;
    }

    /**
     * Copy data to a double array (for FLOAT64 arrays).
     */
    public double[] toDoubleArray() {
        NdArray c = this.contiguous();
        double[] out = new double[(int) this.size];
        MemorySegment.copy(c.data, ValueLayout.JAVA_DOUBLE, 0, out, 0, out.length);
        return out;
    }

    /**
     * Copy data to an int array (for INT32 arrays).
     */
    public int[] toIntArray() {
        NdArray c = this.contiguous();
        int[] out = new int[(int) this.size];
        MemorySegment.copy(c.data, ValueLayout.JAVA_INT, 0, out, 0, out.length);
        return out;
    }

    @Override
    public String toString() {
        return "NdArray(shape=" + Arrays.toString(this.shape) + ", dtype=" + this.dtype + ")";
    }

    // ── Proxy methods for numpy-style chaining ──
    // These delegate to the static utility classes so users can write a.reshape(3,4) instead of ShapeOps.reshape(a, 3, 4)

    /**
     * numpy ndarray.reshape()
     */
    public NdArray reshape(final int... newShape) {
        return ShapeOps.reshape(this, newShape);
    }

    /**
     * numpy ndarray.T / ndarray.transpose()
     */
    public NdArray transpose() {
        return ShapeOps.transpose(this);
    }

    /**
     * numpy ndarray.transpose(axes)
     */
    public NdArray transpose(final int... axes) {
        return ShapeOps.transpose(this, axes);
    }

    /**
     * numpy ndarray.flatten()
     */
    public NdArray flatten() {
        return ShapeOps.flatten(this);
    }

    /**
     * numpy ndarray.ravel()
     */
    public NdArray ravel() {
        return ShapeOps.ravel(this);
    }

    /**
     * numpy ndarray.squeeze()
     */
    public NdArray squeeze() {
        return ShapeOps.squeeze(this);
    }

    /**
     * numpy ndarray.sum()
     */
    public float sum() {
        return Reductions.sum(this);
    }

    /**
     * numpy ndarray.sum(axis)
     */
    public NdArray sum(final int axis) {
        return Reductions.sum(this, axis);
    }

    /**
     * numpy ndarray.mean()
     */
    public float mean() {
        return Reductions.mean(this);
    }

    /**
     * numpy ndarray.std()
     */
    public float std() {
        return Reductions.std(this);
    }

    /**
     * numpy ndarray.var()
     */
    public float var() {
        return Reductions.var(this);
    }

    /**
     * numpy ndarray.min()
     */
    public float min() {
        return Reductions.min(this);
    }

    /**
     * numpy ndarray.max()
     */
    public float max() {
        return Reductions.max(this);
    }

    /**
     * numpy ndarray.argmin()
     */
    public long argmin() {
        return Reductions.argmin(this);
    }

    /**
     * numpy ndarray.argmax()
     */
    public long argmax() {
        return Reductions.argmax(this);
    }

    /**
     * numpy ndarray.clip(min, max)
     */
    public NdArray clip(final float min, final float max) {
        return Ufunc.clip(this, min, max);
    }

    /**
     * numpy ndarray.sort() — returns sorted copy
     */
    public NdArray sort() {
        return IndexOps.sort(this);
    }

    /**
     * numpy ndarray.argsort()
     */
    public NdArray argsort() {
        return IndexOps.argsort(this);
    }

    /**
     * numpy ndarray.dot(other)
     */
    public float dot(final NdArray other) {
        return MatMul.dot(this, other);
    }

    /**
     * numpy ndarray @ other (matmul)
     */
    public NdArray matmul(final NdArray other) {
        return MatMul.matmul(this, other);
    }

    /**
     * numpy ndarray.copy() — always returns a new independent copy
     */
    public NdArray copy() {
        NdArray out = NdArrayFactory.empty(this.arena, this.dtype, this.shape);
        out.data.copyFrom(this.contiguous().data.asSlice(0, this.dtype.totalBytes(this.size)));
        return out;
    }

    /**
     * numpy ndarray.astype — returns FLOAT32 copy (dtype conversion placeholder)
     */
    public NdArray astype(final DType target) {
        if (target == this.dtype) {
            return this.contiguous();
        }
        // For now, only FLOAT32 storage is fully supported
        return this.contiguous();
    }
}
