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
}
