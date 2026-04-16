package com.sparrowlogic.numpy4j;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;

/**
 * N-dimensional array backed by off-heap memory via the FFM {@link Arena}.
 *
 * <p>Mirrors {@code numpy.ndarray} with C-contiguous (row-major) layout.
 * Memory is managed by the owning {@link Arena} — when the arena is closed,
 * all arrays allocated from it become invalid.</p>
 *
 * <p>Create instances via {@link NdArrayFactory} factory methods.</p>
 *
 * @see NdArrayFactory
 * @see DType
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
     * Internal constructor — use {@link NdArrayFactory} methods instead.
     *
     * @param data    backing off-heap memory segment
     * @param arena   owning arena for lifetime management
     * @param dtype   element data type
     * @param shape   dimension sizes
     * @param strides element strides per dimension
     * @param offset  flat element offset into {@code data}
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

    /**
     * Returns the data type of this array's elements.
     *
     * @return the dtype
     */
    public DType dtype() {
        return this.dtype;
    }

    /**
     * Returns a copy of this array's shape (dimension sizes).
     *
     * @return shape array (defensive copy)
     */
    public int[] shape() {
        return this.shape.clone();
    }

    /**
     * Returns the size of a specific axis, supporting negative indexing.
     *
     * @param axis axis index (negative counts from the end)
     * @return dimension size along the given axis
     */
    public int shape(final int axis) {
        int ax = axis < 0 ? axis + this.shape.length : axis;
        return this.shape[ax];
    }

    /**
     * Returns a copy of this array's strides (in elements, not bytes).
     *
     * @return strides array (defensive copy)
     */
    public int[] strides() {
        return this.strides.clone();
    }

    /**
     * Returns the number of dimensions (axes).
     *
     * @return number of dimensions
     */
    public int ndim() {
        return this.shape.length;
    }

    /**
     * Returns the total number of elements in this array.
     *
     * @return total element count (product of all dimensions)
     */
    public long size() {
        return this.size;
    }

    /**
     * Returns the backing off-heap {@link MemorySegment}.
     *
     * @return the raw memory segment
     */
    public MemorySegment data() {
        return this.data;
    }

    /**
     * Returns the flat element offset into the backing segment.
     *
     * @return element offset (0 for contiguous arrays created by factories)
     */
    public long offset() {
        return this.offset;
    }

    /**
     * Returns the {@link Arena} that owns this array's memory.
     *
     * @return the owning arena
     */
    public Arena arena() {
        return this.arena;
    }

    // ── Scalar element access ──

    /**
     * Reads a {@code float} element at the given multi-dimensional indices.
     * Supports negative indexing per axis.
     *
     * @param indices one index per dimension (negative counts from end)
     * @return the float value at the specified position
     */
    public float getFloat(final long... indices) {
        return this.data.getAtIndex(ValueLayout.JAVA_FLOAT, this.flatIndex(indices));
    }

    /**
     * Reads a {@code double} element at the given multi-dimensional indices.
     *
     * @param indices one index per dimension (negative counts from end)
     * @return the double value at the specified position
     */
    public double getDouble(final long... indices) {
        return this.data.getAtIndex(ValueLayout.JAVA_DOUBLE, this.flatIndex(indices));
    }

    /**
     * Reads an {@code int} element at the given multi-dimensional indices.
     *
     * @param indices one index per dimension (negative counts from end)
     * @return the int value at the specified position
     */
    public int getInt(final long... indices) {
        return this.data.getAtIndex(ValueLayout.JAVA_INT, this.flatIndex(indices));
    }

    /**
     * Reads a {@code long} element at the given multi-dimensional indices.
     *
     * @param indices one index per dimension (negative counts from end)
     * @return the long value at the specified position
     */
    public long getLong(final long... indices) {
        return this.data.getAtIndex(ValueLayout.JAVA_LONG, this.flatIndex(indices));
    }

    /**
     * Writes a {@code float} value at the given multi-dimensional indices.
     *
     * @param value   the value to write
     * @param indices one index per dimension (negative counts from end)
     */
    public void setFloat(final float value, final long... indices) {
        this.data.setAtIndex(ValueLayout.JAVA_FLOAT, this.flatIndex(indices), value);
    }

    /**
     * Writes a {@code double} value at the given multi-dimensional indices.
     *
     * @param value   the value to write
     * @param indices one index per dimension (negative counts from end)
     */
    public void setDouble(final double value, final long... indices) {
        this.data.setAtIndex(ValueLayout.JAVA_DOUBLE, this.flatIndex(indices), value);
    }

    /**
     * Writes an {@code int} value at the given multi-dimensional indices.
     *
     * @param value   the value to write
     * @param indices one index per dimension (negative counts from end)
     */
    public void setInt(final int value, final long... indices) {
        this.data.setAtIndex(ValueLayout.JAVA_INT, this.flatIndex(indices), value);
    }

    /**
     * Writes a {@code long} value at the given multi-dimensional indices.
     *
     * @param value   the value to write
     * @param indices one index per dimension (negative counts from end)
     */
    public void setLong(final long value, final long... indices) {
        this.data.setAtIndex(ValueLayout.JAVA_LONG, this.flatIndex(indices), value);
    }

    /**
     * Reads a {@code float} element by flat (linear) index.
     *
     * @param i zero-based flat index
     * @return the float value
     */
    public float flatGetFloat(final long i) {
        return this.data.getAtIndex(ValueLayout.JAVA_FLOAT, this.offset + i);
    }

    /**
     * Writes a {@code float} value by flat (linear) index.
     *
     * @param i zero-based flat index
     * @param v the value to write
     */
    public void flatSetFloat(final long i, final float v) {
        this.data.setAtIndex(ValueLayout.JAVA_FLOAT, this.offset + i, v);
    }

    /**
     * Reads a {@code double} element by flat (linear) index.
     *
     * @param i zero-based flat index
     * @return the double value
     */
    public double flatGetDouble(final long i) {
        return this.data.getAtIndex(ValueLayout.JAVA_DOUBLE, this.offset + i);
    }

    /**
     * Writes a {@code double} value by flat (linear) index.
     *
     * @param i zero-based flat index
     * @param v the value to write
     */
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
     * Returns {@code true} if this array is C-contiguous (row-major, no gaps or strides).
     *
     * @return whether the array is contiguous
     */
    public boolean isContiguous() {
        return Arrays.equals(this.strides, ShapeUtils.cStrides(this.shape));
    }

    /**
     * Returns a C-contiguous copy of this array, or {@code this} if already contiguous.
     *
     * @return a contiguous array with the same shape and data
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
     * Copies this array's data into a new {@code float[]}. Requires {@link DType#FLOAT32}.
     *
     * @return a new float array containing all elements in row-major order
     */
    public float[] toFloatArray() {
        NdArray c = this.contiguous();
        float[] out = new float[(int) this.size];
        MemorySegment.copy(c.data, ValueLayout.JAVA_FLOAT, 0, out, 0, out.length);
        return out;
    }

    /**
     * Copies this array's data into a new {@code double[]}. Requires {@link DType#FLOAT64}.
     *
     * @return a new double array containing all elements in row-major order
     */
    public double[] toDoubleArray() {
        NdArray c = this.contiguous();
        double[] out = new double[(int) this.size];
        MemorySegment.copy(c.data, ValueLayout.JAVA_DOUBLE, 0, out, 0, out.length);
        return out;
    }

    /**
     * Copies this array's data into a new {@code int[]}. Requires {@link DType#INT32}.
     *
     * @return a new int array containing all elements in row-major order
     */
    public int[] toIntArray() {
        NdArray c = this.contiguous();
        int[] out = new int[(int) this.size];
        MemorySegment.copy(c.data, ValueLayout.JAVA_INT, 0, out, 0, out.length);
        return out;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return "NdArray(shape=" + Arrays.toString(this.shape) + ", dtype=" + this.dtype + ")";
    }

    // ── Proxy methods for numpy-style chaining ──

    /**
     * Returns a reshaped view of this array. Equivalent to {@link ShapeOps#reshape(NdArray, int...)}.
     *
     * @param newShape target shape ({@code -1} infers one dimension)
     * @return reshaped array
     * @see ShapeOps#reshape(NdArray, int...)
     */
    public NdArray reshape(final int... newShape) {
        return ShapeOps.reshape(this, newShape);
    }

    /**
     * Returns a transposed view (reversed axes). Equivalent to {@code numpy.ndarray.T}.
     *
     * @return transposed view
     * @see ShapeOps#transpose(NdArray)
     */
    public NdArray transpose() {
        return ShapeOps.transpose(this);
    }

    /**
     * Returns a transposed view with the given axis permutation.
     *
     * @param axes axis order
     * @return transposed view
     * @see ShapeOps#transpose(NdArray, int...)
     */
    public NdArray transpose(final int... axes) {
        return ShapeOps.transpose(this, axes);
    }

    /**
     * Returns a 1-D contiguous copy. Equivalent to {@code numpy.ndarray.flatten()}.
     *
     * @return flattened array
     * @see ShapeOps#flatten(NdArray)
     */
    public NdArray flatten() {
        return ShapeOps.flatten(this);
    }

    /**
     * Returns a 1-D contiguous copy. Equivalent to {@code numpy.ndarray.ravel()}.
     *
     * @return raveled array
     * @see ShapeOps#ravel(NdArray)
     */
    public NdArray ravel() {
        return ShapeOps.ravel(this);
    }

    /**
     * Removes all size-1 dimensions. Equivalent to {@code numpy.ndarray.squeeze()}.
     *
     * @return squeezed view
     * @see ShapeOps#squeeze(NdArray)
     */
    public NdArray squeeze() {
        return ShapeOps.squeeze(this);
    }

    /**
     * Returns the sum of all elements.
     *
     * @return scalar sum
     * @see Reductions#sum(NdArray)
     */
    public float sum() {
        return Reductions.sum(this);
    }

    /**
     * Returns the sum along the given axis.
     *
     * @param axis axis to reduce
     * @return reduced array
     * @see Reductions#sum(NdArray, int)
     */
    public NdArray sum(final int axis) {
        return Reductions.sum(this, axis);
    }

    /**
     * Returns the arithmetic mean of all elements.
     *
     * @return scalar mean
     * @see Reductions#mean(NdArray)
     */
    public float mean() {
        return Reductions.mean(this);
    }

    /**
     * Returns the standard deviation of all elements.
     *
     * @return scalar standard deviation
     * @see Reductions#std(NdArray)
     */
    public float std() {
        return Reductions.std(this);
    }

    /**
     * Returns the variance of all elements.
     *
     * @return scalar variance
     * @see Reductions#var(NdArray)
     */
    public float var() {
        return Reductions.var(this);
    }

    /**
     * Returns the minimum element value.
     *
     * @return scalar minimum
     * @see Reductions#min(NdArray)
     */
    public float min() {
        return Reductions.min(this);
    }

    /**
     * Returns the maximum element value.
     *
     * @return scalar maximum
     * @see Reductions#max(NdArray)
     */
    public float max() {
        return Reductions.max(this);
    }

    /**
     * Returns the flat index of the minimum element.
     *
     * @return index of the minimum value
     * @see Reductions#argmin(NdArray)
     */
    public long argmin() {
        return Reductions.argmin(this);
    }

    /**
     * Returns the flat index of the maximum element.
     *
     * @return index of the maximum value
     * @see Reductions#argmax(NdArray)
     */
    public long argmax() {
        return Reductions.argmax(this);
    }

    /**
     * Clips all elements to {@code [min, max]}.
     *
     * @param min lower bound
     * @param max upper bound
     * @return clipped array
     * @see Ufunc#clip(NdArray, float, float)
     */
    public NdArray clip(final float min, final float max) {
        return Ufunc.clip(this, min, max);
    }

    /**
     * Returns a sorted copy of this array.
     *
     * @return sorted array
     * @see IndexOps#sort(NdArray)
     */
    public NdArray sort() {
        return IndexOps.sort(this);
    }

    /**
     * Returns the indices that would sort this array.
     *
     * @return INT32 array of sort indices
     * @see IndexOps#argsort(NdArray)
     */
    public NdArray argsort() {
        return IndexOps.argsort(this);
    }

    /**
     * Computes the dot product with another 1-D array.
     *
     * @param other the other array
     * @return scalar dot product
     * @see MatMul#dot(NdArray, NdArray)
     */
    public float dot(final NdArray other) {
        return MatMul.dot(this, other);
    }

    /**
     * Matrix-multiplies this array with another. Equivalent to the {@code @} operator in NumPy.
     *
     * @param other the right-hand operand
     * @return result of matrix multiplication
     * @see MatMul#matmul(NdArray, NdArray)
     */
    public NdArray matmul(final NdArray other) {
        return MatMul.matmul(this, other);
    }

    /**
     * Returns an independent deep copy of this array.
     *
     * @return a new array with copied data
     */
    public NdArray copy() {
        NdArray out = NdArrayFactory.empty(this.arena, this.dtype, this.shape);
        out.data.copyFrom(this.contiguous().data.asSlice(0, this.dtype.totalBytes(this.size)));
        return out;
    }

    /**
     * Returns a copy cast to the target dtype.
     *
     * <p>Currently returns a contiguous copy; full dtype conversion is a placeholder.</p>
     *
     * @param target the desired dtype
     * @return array with the target dtype
     */
    public NdArray astype(final DType target) {
        if (target == this.dtype) {
            return this.contiguous();
        }
        return this.contiguous();
    }
}
