package com.sparrowlogic.numpy4j;

import org.junit.jupiter.api.Test;
import java.lang.foreign.Arena;
import static org.junit.jupiter.api.Assertions.*;

/** Tests for {@link LinAlg}. */
public class LinAlgTest {
    private static final float T = 0.001f;
    private static final float TM = 0.05f;
    private static void assertArr(String l, NdArray a, float[] e, float t) { float[] ac = a.toFloatArray(); for (int i = 0; i < e.length; i++) assertEquals(e[i], ac[i], t, l+"["+i+"]"); }

    @Test void norm() { try (var a = Arena.ofConfined()) { assertEquals(5f, LinAlg.norm(NdArrayFactory.array(a, new float[]{3,4})), T); } }
    @Test void normP1() { try (var a = Arena.ofConfined()) { assertEquals(7f, LinAlg.norm(NdArrayFactory.array(a, new float[]{3,4}), 1f), T); } }
    @Test void det() { try (var a = Arena.ofConfined()) { assertEquals(10f, LinAlg.det(NdArrayFactory.array(a, new float[]{4,7,2,6}, 2, 2)), T); } }
    @Test void inv() { try (var a = Arena.ofConfined()) { assertArr("inv", LinAlg.inv(NdArrayFactory.array(a, new float[]{4,7,2,6}, 2, 2)), new float[]{0.6f,-0.7f,-0.2f,0.4f}, T); } }
    @Test void solve() { try (var a = Arena.ofConfined()) { assertArr("solve", LinAlg.solve(NdArrayFactory.array(a, new float[]{4,7,2,6}, 2, 2), NdArrayFactory.array(a, new float[]{1,2})), new float[]{-0.8f,0.6f}, T); } }
    @Test void cholesky() { try (var a = Arena.ofConfined()) { assertArr("chol", LinAlg.cholesky(NdArrayFactory.array(a, new float[]{4,2,2,3}, 2, 2)), new float[]{2,0,1,1.4142135f}, T); } }
    @Test void qrReconstruct() { try (var a = Arena.ofConfined()) { NdArray[] qr = LinAlg.qr(NdArrayFactory.array(a, new float[]{1,2,3,4}, 2, 2)); assertArr("qr", MatMul.matmul(qr[0], qr[1]), new float[]{1,2,3,4}, TM); } }
    @Test void matrixPower() { try (var a = Arena.ofConfined()) { assertArr("A^0", LinAlg.matrixPower(NdArrayFactory.array(a, new float[]{1,2,3,4}, 2, 2), 0), new float[]{1,0,0,1}, T); } }
    @Test void svd() { try (var a = Arena.ofConfined()) { NdArray[] usv = LinAlg.svd(NdArrayFactory.array(a, new float[]{1,2,3,4}, 2, 2)); float[] s = usv[1].toFloatArray(); assertEquals(5.465f, s[0], TM); NdArray Sm = NdArrayFactory.zeros(a, DType.FLOAT32, 2, 2); Sm.setFloat(s[0],0,0); Sm.setFloat(s[1],1,1); assertArr("svd", MatMul.matmul(MatMul.matmul(usv[0], Sm), usv[2]), new float[]{1,2,3,4}, TM); } }
    @Test void eigvals() { try (var a = Arena.ofConfined()) { float[] v = LinAlg.eigvals(NdArrayFactory.array(a, new float[]{1,2,3,4}, 2, 2)).toFloatArray(); java.util.Arrays.sort(v); assertEquals(-0.372f, v[0], TM); assertEquals(5.372f, v[1], TM); } }
    @Test void pinv() { try (var a = Arena.ofConfined()) { NdArray M = NdArrayFactory.array(a, new float[]{1,2,3,4,5,6}, 3, 2); assertArr("pinv", MatMul.matmul(MatMul.matmul(M, LinAlg.pinv(M)), M), new float[]{1,2,3,4,5,6}, TM); } }
    @Test void matrixRank() { try (var a = Arena.ofConfined()) { assertEquals(2, LinAlg.matrixRank(NdArrayFactory.array(a, new float[]{1,2,3,4}, 2, 2))); } }
    @Test void slogdet() { try (var a = Arena.ofConfined()) { float[] r = LinAlg.slogdet(NdArrayFactory.array(a, new float[]{1,2,3,4}, 2, 2)); assertEquals(-1f, r[0], T); assertEquals(0.6931472f, r[1], T); } }
    @Test void lstsq() { try (var a = Arena.ofConfined()) { float[] x = LinAlg.lstsq(NdArrayFactory.array(a, new float[]{1,1,1,2,1,3}, 3, 2), NdArrayFactory.array(a, new float[]{1,2,3})).toFloatArray(); assertEquals(0f, x[0], TM); assertEquals(1f, x[1], TM); } }
    @Test void eigh() { try (var a = Arena.ofConfined()) { NdArray[] r = LinAlg.eigh(NdArrayFactory.array(a, new float[]{4,2,2,3}, 2, 2)); float[] w = r[0].toFloatArray(); assertEquals(1.438f, w[0], TM); assertEquals(5.562f, w[1], TM); } }
    @Test void cond() { try (var a = Arena.ofConfined()) { float c = LinAlg.cond(NdArrayFactory.array(a, new float[]{1,2,3,4}, 2, 2)); assertEquals(14.933f, c, 0.5f); } }
}
