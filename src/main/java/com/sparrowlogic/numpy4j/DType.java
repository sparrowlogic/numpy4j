package com.sparrowlogic.numpy4j;

import java.lang.foreign.ValueLayout;

/**
 * NumPy-compatible data types backed by FFM ValueLayouts for off-heap access.
 */
public enum DType {
    FLOAT32(Float.BYTES, ValueLayout.JAVA_FLOAT),
    FLOAT64(Double.BYTES, ValueLayout.JAVA_DOUBLE),
    INT32(Integer.BYTES, ValueLayout.JAVA_INT),
    INT64(Long.BYTES, ValueLayout.JAVA_LONG),
    BOOL(Byte.BYTES, ValueLayout.JAVA_BYTE);

    private final int bytes;

    private final ValueLayout layout;

    DType(final int bytes, final ValueLayout layout) {
        this.bytes = bytes;
        this.layout = layout;
    }

    public int bytes() {
        return this.bytes;
    }

    public ValueLayout layout() {
        return this.layout;
    }

    /**
     * Total bytes for n elements.
     */
    public long totalBytes(final long n) {
        return n * this.bytes;
    }
}
