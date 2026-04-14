package com.sparrowlogic.numpy4j;

import org.junit.jupiter.api.Test;
import java.lang.foreign.Arena;
import static org.junit.jupiter.api.Assertions.*;

/** Tests for {@link NdArray} — storage, accessors, contiguity, proxy methods. */
public class NdArrayTest {
    private static final float T = 0.001f;

    @Test void shapeAndSize() { try (var a = Arena.ofConfined()) { NdArray x = NdArrayFactory.zeros(a, DType.FLOAT32, 2, 3); assertArrayEquals(new int[]{2,3}, x.shape()); assertEquals(2, x.ndim()); assertEquals(6, x.size()); assertEquals(3, x.shape(1)); } }
    @Test void getSetFloat() { try (var a = Arena.ofConfined()) { NdArray x = NdArrayFactory.zeros(a, DType.FLOAT32, 2, 3); x.setFloat(42f, 1, 2); assertEquals(42f, x.getFloat(1, 2), T); } }
    @Test void flatGetSetFloat() { try (var a = Arena.ofConfined()) { NdArray x = NdArrayFactory.array(a, new float[]{10,20,30}); assertEquals(10f, x.flatGetFloat(0), T); x.flatSetFloat(1, 99f); assertEquals(99f, x.flatGetFloat(1), T); } }
    @Test void toFloatArray() { try (var a = Arena.ofConfined()) { assertArrayEquals(new float[]{1,2,3}, NdArrayFactory.array(a, new float[]{1,2,3}).toFloatArray(), T); } }
    @Test void toIntArray() { try (var a = Arena.ofConfined()) { assertArrayEquals(new int[]{10,20,30}, NdArrayFactory.array(a, new int[]{10,20,30}).toIntArray()); } }
    @Test void isContiguous() { try (var a = Arena.ofConfined()) { NdArray x = NdArrayFactory.arange(a, 6); assertTrue(x.isContiguous()); assertFalse(ShapeOps.transpose(ShapeOps.reshape(x, 2, 3)).isContiguous()); } }
    @Test void negativeIndexing() { try (var a = Arena.ofConfined()) { assertEquals(30f, NdArrayFactory.array(a, new float[]{10,20,30}).getFloat(-1), T); } }

    // ── Proxy method tests ──
    @Test void proxyReshape() { try (var a = Arena.ofConfined()) { assertArrayEquals(new int[]{2,3}, NdArrayFactory.arange(a, 6).reshape(2, 3).shape()); } }
    @Test void proxyTranspose() { try (var a = Arena.ofConfined()) { assertArrayEquals(new int[]{3,2}, NdArrayFactory.arange(a, 6).reshape(2, 3).transpose().shape()); } }
    @Test void proxyFlatten() { try (var a = Arena.ofConfined()) { assertEquals(6, NdArrayFactory.arange(a, 6).reshape(2, 3).flatten().size()); } }
    @Test void proxySqueeze() { try (var a = Arena.ofConfined()) { assertArrayEquals(new int[]{3}, NdArrayFactory.arange(a, 3).reshape(1, 3, 1).squeeze().shape()); } }
    @Test void proxySum() { try (var a = Arena.ofConfined()) { assertEquals(6f, NdArrayFactory.array(a, new float[]{1,2,3}).sum(), T); } }
    @Test void proxyMean() { try (var a = Arena.ofConfined()) { assertEquals(2f, NdArrayFactory.array(a, new float[]{1,2,3}).mean(), T); } }
    @Test void proxyMinMax() { try (var a = Arena.ofConfined()) { NdArray x = NdArrayFactory.array(a, new float[]{3,1,2}); assertEquals(1f, x.min(), T); assertEquals(3f, x.max(), T); } }
    @Test void proxyArgminArgmax() { try (var a = Arena.ofConfined()) { NdArray x = NdArrayFactory.array(a, new float[]{3,1,2}); assertEquals(1, x.argmin()); assertEquals(0, x.argmax()); } }
    @Test void proxyClip() { try (var a = Arena.ofConfined()) { float[] r = NdArrayFactory.array(a, new float[]{1,2,3,4}).clip(1.5f, 3.5f).toFloatArray(); assertEquals(1.5f, r[0], T); assertEquals(3.5f, r[3], T); } }
    @Test void proxySort() { try (var a = Arena.ofConfined()) { assertEquals(1f, NdArrayFactory.array(a, new float[]{3,1,2}).sort().flatGetFloat(0), T); } }
    @Test void proxyDot() { try (var a = Arena.ofConfined()) { assertEquals(32f, NdArrayFactory.array(a, new float[]{1,2,3}).dot(NdArrayFactory.array(a, new float[]{4,5,6})), T); } }
    @Test void proxyMatmul() { try (var a = Arena.ofConfined()) { NdArray r = NdArrayFactory.array(a, new float[]{1,2,3,4}, 2, 2).matmul(NdArrayFactory.array(a, new float[]{5,6,7,8}, 2, 2)); assertEquals(19f, r.flatGetFloat(0), T); } }
    @Test void proxyCopy() { try (var a = Arena.ofConfined()) { NdArray x = NdArrayFactory.array(a, new float[]{1,2,3}); NdArray c = x.copy(); c.flatSetFloat(0, 99f); assertEquals(1f, x.flatGetFloat(0), T); } }

    // ── Missing proxy methods ──
    @Test void proxyRavel() { try (var a = Arena.ofConfined()) { assertEquals(6, NdArrayFactory.arange(a, 6).reshape(2, 3).ravel().size()); } }
    @Test void proxyStd() { try (var a = Arena.ofConfined()) { assertEquals(0.816f, NdArrayFactory.array(a, new float[]{1,2,3}).std(), 0.01f); } }
    @Test void proxyVar() { try (var a = Arena.ofConfined()) { assertEquals(0.667f, NdArrayFactory.array(a, new float[]{1,2,3}).var(), 0.01f); } }
    @Test void proxySumAxis() { try (var a = Arena.ofConfined()) { assertArrayEquals(new float[]{5,7,9}, NdArrayFactory.array(a, new float[]{1,2,3,4,5,6}, 2, 3).sum(0).toFloatArray(), T); } }
    @Test void proxyArgsort() { try (var a = Arena.ofConfined()) { assertEquals(1, NdArrayFactory.array(a, new float[]{3,1,2}).argsort().toIntArray()[0]); } }
    @Test void proxyAstype() { try (var a = Arena.ofConfined()) { NdArray x = NdArrayFactory.array(a, new float[]{1,2,3}); assertNotNull(x.astype(DType.FLOAT32)); assertNotNull(x.astype(DType.FLOAT64)); } }
    @Test void proxyTransposeAxes() { try (var a = Arena.ofConfined()) { assertArrayEquals(new int[]{3,2}, NdArrayFactory.arange(a, 6).reshape(2, 3).transpose(1, 0).shape()); } }
    @Test void toStringTest() { try (var a = Arena.ofConfined()) { assertTrue(NdArrayFactory.arange(a, 3).toString().contains("shape=")); } }

    // ── contiguous() for non-FLOAT32 dtypes ──
    @Test void contiguousFloat64() {
        try (var a = Arena.ofConfined()) {
            NdArray x = NdArrayFactory.linspace(a, 0, 1, 4); // FLOAT64
            NdArray t = ShapeOps.transpose(ShapeOps.reshape(x, 2, 2));
            NdArray c = t.contiguous();
            assertTrue(c.isContiguous());
        }
    }
    @Test void contiguousInt32() {
        try (var a = Arena.ofConfined()) {
            NdArray x = NdArrayFactory.array(a, new int[]{1,2,3,4}, 2, 2);
            NdArray t = ShapeOps.transpose(x);
            NdArray c = t.contiguous();
            assertTrue(c.isContiguous());
            assertEquals(1, c.getInt(0, 0));
            assertEquals(3, c.getInt(0, 1));
        }
    }
    @Test void copyNonContiguous() {
        try (var a = Arena.ofConfined()) {
            NdArray x = ShapeOps.transpose(NdArrayFactory.array(a, new float[]{1,2,3,4}, 2, 2));
            NdArray c = x.copy();
            assertTrue(c.isContiguous());
            assertEquals(1f, c.flatGetFloat(0), T);
            assertEquals(3f, c.flatGetFloat(1), T);
        }
    }
    @Test void getSetDouble() { try (var a = Arena.ofConfined()) { NdArray x = NdArrayFactory.full(a, DType.FLOAT64, 0, 2, 2); x.setDouble(3.14, 0, 1); assertEquals(3.14, x.getDouble(0, 1), 0.001); } }
    @Test void getSetLong() { try (var a = Arena.ofConfined()) { NdArray x = NdArrayFactory.full(a, DType.INT64, 0, 2); x.setLong(42L, 1); assertEquals(42L, x.getLong(1)); } }
    @Test void toDoubleArray() { try (var a = Arena.ofConfined()) { double[] r = NdArrayFactory.linspace(a, 0, 1, 3).toDoubleArray(); assertEquals(0.0, r[0], 0.001); assertEquals(1.0, r[2], 0.001); } }
}
