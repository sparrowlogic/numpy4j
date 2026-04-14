package com.sparrowlogic.numpy4j;

import org.junit.jupiter.api.Test;
import java.lang.foreign.Arena;
import static org.junit.jupiter.api.Assertions.*;

/** Tests for {@link IndexOps}. */
public class IndexOpsTest {
    private static final float T = 0.001f;
    private static void assertArr(String l, NdArray a, float[] e) { float[] ac = a.toFloatArray(); for (int i = 0; i < e.length; i++) assertEquals(e[i], ac[i], T, l+"["+i+"]"); }

    @Test void slice() { try (var a = Arena.ofConfined()) { NdArray s = IndexOps.slice(NdArrayFactory.array(a, new float[]{0,1,2,3,4,5}), 1, 4, 1).contiguous(); assertEquals(3, s.size()); assertArr("slice", s, new float[]{1,2,3}); } }
    @Test void sliceWithStep() { try (var a = Arena.ofConfined()) { NdArray s = IndexOps.slice(NdArrayFactory.array(a, new float[]{0,1,2,3,4,5}), 0, 6, 2).contiguous(); assertArr("step", s, new float[]{0,2,4}); } }
    @Test void take() { try (var a = Arena.ofConfined()) { NdArray r = IndexOps.take(NdArrayFactory.array(a, new float[]{10,20,30,40,50}), new int[]{0,2,4}, 0); assertArr("take", r, new float[]{10,30,50}); } }
    @Test void booleanIndex() { try (var a = Arena.ofConfined()) { NdArray r = IndexOps.booleanIndex(NdArrayFactory.array(a, new float[]{10,20,30,40}), NdArrayFactory.array(a, new float[]{1,0,1,0})); assertArr("bool", r, new float[]{10,30}); } }
    @Test void where() { try (var a = Arena.ofConfined()) { NdArray r = IndexOps.where(NdArrayFactory.array(a, new float[]{1,0,1}), NdArrayFactory.array(a, new float[]{10,20,30}), NdArrayFactory.array(a, new float[]{-1,-2,-3})); assertArr("where", r, new float[]{10,-2,30}); } }
    @Test void sort() { try (var a = Arena.ofConfined()) { assertArr("sort", IndexOps.sort(NdArrayFactory.array(a, new float[]{3,1,4,1,5,9})), new float[]{1,1,3,4,5,9}); } }
    @Test void argsort() { try (var a = Arena.ofConfined()) { NdArray r = IndexOps.argsort(NdArrayFactory.array(a, new float[]{30,10,20})); int[] idx = r.toIntArray(); assertEquals(1, idx[0]); assertEquals(2, idx[1]); assertEquals(0, idx[2]); } }
    @Test void searchsorted() { try (var a = Arena.ofConfined()) { assertEquals(2, IndexOps.searchsorted(NdArrayFactory.array(a, new float[]{1,3,5,7}), 4f)); } }
    @Test void nonzero() { try (var a = Arena.ofConfined()) { NdArray r = IndexOps.nonzero(NdArrayFactory.array(a, new float[]{0,1,0,3,0,5})); assertArrayEquals(new int[]{1,3,5}, r.toIntArray()); } }
    @Test void argwhere() { try (var a = Arena.ofConfined()) { assertEquals(3, IndexOps.argwhere(NdArrayFactory.array(a, new float[]{0,1,0,3,0,5})).size()); } }
    @Test void partition() { try (var a = Arena.ofConfined()) { NdArray r = IndexOps.partition(NdArrayFactory.array(a, new float[]{3,1,4,1,5,9,2,6}), 3); assertEquals(8, r.size()); } }
    @Test void put() { try (var a = Arena.ofConfined()) { NdArray x = NdArrayFactory.array(a, new float[]{1,2,3,4}); IndexOps.put(x, new int[]{1,3}, new float[]{99,88}); assertEquals(99f, x.flatGetFloat(1), T); assertEquals(88f, x.flatGetFloat(3), T); } }
}
