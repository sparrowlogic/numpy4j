package com.sparrowlogic.numpy4j;

import org.junit.jupiter.api.Test;
import java.lang.foreign.Arena;
import static org.junit.jupiter.api.Assertions.*;

/** Tests for {@link Einsum} — matmul, trace, transpose, element-wise dot via einsum notation. */
public class EinsumTest {
    private static final float T = 0.001f;
    private static void assertArr(String l, NdArray a, float[] e) { float[] ac = a.toFloatArray(); for (int i = 0; i < e.length; i++) assertEquals(e[i], ac[i], T, l+"["+i+"]"); }

    @Test void matmul() { try (var a = Arena.ofConfined()) { assertArr("ij,jk->ik", Einsum.einsum("ij,jk->ik", NdArrayFactory.array(a, new float[]{1,2,3,4}, 2, 2), NdArrayFactory.array(a, new float[]{5,6,7,8}, 2, 2)), new float[]{19,22,43,50}); } }
    @Test void trace() { try (var a = Arena.ofConfined()) { assertEquals(5f, Einsum.einsum("ii->", NdArrayFactory.array(a, new float[]{1,2,3,4}, 2, 2)).flatGetFloat(0), T); } }
    @Test void transpose() { try (var a = Arena.ofConfined()) { assertArr("ij->ji", Einsum.einsum("ij->ji", NdArrayFactory.array(a, new float[]{1,2,3,4}, 2, 2)), new float[]{1,3,2,4}); } }
}
