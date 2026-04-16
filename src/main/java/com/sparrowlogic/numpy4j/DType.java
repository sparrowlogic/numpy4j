package com.sparrowlogic.numpy4j;

import java.lang.foreign.ValueLayout;

/**
 * NumPy-compatible data types backed by FFM {@link ValueLayout}s for off-heap access.
 *
 * <p>Each constant maps to a fixed-size primitive layout used by {@link NdArray}
 * for element storage in {@link java.lang.foreign.MemorySegment} regions.</p>
 *
 * @see NdArray
 * @see NdArrayFactory
 */
public enum DType {

    /** 32-bit IEEE 754 float ({@code numpy.float32}). */
    FLOAT32(Float.BYTES, ValueLayout.JAVA_FLOAT),

    /** 64-bit IEEE 754 double ({@code numpy.float64}). */
    FLOAT64(Double.BYTES, ValueLayout.JAVA_DOUBLE),

    /** 32-bit signed integer ({@code numpy.int32}). */
    INT32(Integer.BYTES, ValueLayout.JAVA_INT),

    /** 64-bit signed integer ({@code numpy.int64}). */
    INT64(Long.BYTES, ValueLayout.JAVA_LONG),

    /** 8-bit boolean ({@code numpy.bool_}). Non-zero is {@code true}. */
    BOOL(Byte.BYTES, ValueLayout.JAVA_BYTE);

    private final int bytes;

    private final ValueLayout layout;

    DType(final int bytes, final ValueLayout layout) {
        this.bytes = bytes;
        this.layout = layout;
    }

    /**
     * Returns the size of a single element in bytes.
     *
     * @return element size in bytes
     */
    public int bytes() {
        return this.bytes;
    }

    /**
     * Returns the FFM {@link ValueLayout} used for off-heap element access.
     *
     * @return the value layout for this dtype
     */
    public ValueLayout layout() {
        return this.layout;
    }

    /**
     * Returns the total byte count required to store {@code n} elements.
     *
     * @param n number of elements
     * @return total bytes ({@code n * bytes()})
     */
    public long totalBytes(final long n) {
        return n * this.bytes;
    }
}
