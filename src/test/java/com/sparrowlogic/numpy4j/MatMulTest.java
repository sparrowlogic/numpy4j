package com.sparrowlogic.numpy4j;

import org.junit.jupiter.api.Test;
import java.lang.foreign.Arena;
import static org.junit.jupiter.api.Assertions.*;

public class MatMulTest {
    private static final float T = 0.001f;
    private static void assertArr(String l, NdArray a, float[] e) { float[] ac = a.toFloatArray(); for (int i = 0; i < e.length; i++) assertEquals(e[i], ac[i], T, l+"["+i+"]"); }

    @Test void dot1D() { try (var a = Arena.ofConfined()) { assertEquals(32f, MatMul.dot(NdArrayFactory.array(a, new float[]{1,2,3}), NdArrayFactory.array(a, new float[]{4,5,6})), T); } }
    @Test void matmul2x2() { try (var a = Arena.ofConfined()) { assertArr("mm", MatMul.matmul(NdArrayFactory.array(a, new float[]{1,2,3,4}, 2, 2), NdArrayFactory.array(a, new float[]{5,6,7,8}, 2, 2)), new float[]{19,22,43,50}); } }
    @Test void outer() { try (var a = Arena.ofConfined()) { assertArr("outer", MatMul.outer(NdArrayFactory.array(a, new float[]{1,2}), NdArrayFactory.array(a, new float[]{3,4,5})), new float[]{3,4,5,6,8,10}); } }
    @Test void inner() { try (var a = Arena.ofConfined()) { assertEquals(32f, MatMul.inner(NdArrayFactory.array(a, new float[]{1,2,3}), NdArrayFactory.array(a, new float[]{4,5,6})), T); } }
    @Test void tensordot() { try (var a = Arena.ofConfined()) { assertArr("td", MatMul.tensordot(NdArrayFactory.array(a, new float[]{1,2,3,4}, 2, 2), NdArrayFactory.array(a, new float[]{5,6,7,8}, 2, 2), 1), new float[]{19,22,43,50}); } }
    @Test void cross() { try (var a = Arena.ofConfined()) { assertArr("cross", MatMul.cross(NdArrayFactory.array(a, new float[]{1,2,3}), NdArrayFactory.array(a, new float[]{4,5,6})), new float[]{-3,6,-3}); } }
    @Test void batchedMatmul() { try (var a = Arena.ofConfined()) { NdArray C = MatMul.batchedMatmul(NdArrayFactory.array(a, new float[]{1,2,3,4,5,6,7,8,9,10,11,12}, 2, 2, 3), NdArrayFactory.array(a, new float[]{1,2,3,4,5,6,7,8,9,10,11,12}, 2, 3, 2)); assertEquals(22f, C.flatGetFloat(0), T); assertEquals(28f, C.flatGetFloat(1), T); } }
    @Test void vdot() { try (var a = Arena.ofConfined()) { assertEquals(32f, MatMul.vdot(NdArrayFactory.array(a, new float[]{1,2,3}), NdArrayFactory.array(a, new float[]{4,5,6})), T); } }
    @Test void multiDot() { try (var a = Arena.ofConfined()) { assertArr("md", MatMul.multiDot(NdArrayFactory.array(a, new float[]{1,2,3,4}, 2, 2), NdArrayFactory.array(a, new float[]{5,6,7,8}, 2, 2)), new float[]{19,22,43,50}); } }
}
