package com.sparrowlogic.numpy4j;

import org.junit.jupiter.api.Test;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import static org.junit.jupiter.api.Assertions.*;

public class MLTest {
    private static final float T = 0.001f;
    private static void assertArr(String l, NdArray a, float[] e) { float[] ac = a.toFloatArray(); for (int i = 0; i < e.length; i++) assertEquals(e[i], ac[i], T, l+"["+i+"]"); }

    @Test void softmax() { try (var a = Arena.ofConfined()) { assertArr("sm", ML.softmax(NdArrayFactory.array(a, new float[]{2,1,0.1f}, 1, 3)), new float[]{0.6590012f,0.2424330f,0.0985659f}); } }
    @Test void logSoftmax() { try (var a = Arena.ofConfined()) { assertArr("lsm", ML.logSoftmax(NdArrayFactory.array(a, new float[]{2,1,0.1f}, 1, 3)), new float[]{-0.4170299f,-1.4170300f,-2.3170300f}); } }
    @Test void greedySearch() { try (var a = Arena.ofConfined()) { assertEquals(0, ML.greedySearch(NdArrayFactory.array(a, new float[]{2,1,0.1f}, 1, 3)).data().getAtIndex(ValueLayout.JAVA_INT, 0)); } }
    @Test void topK() { try (var a = Arena.ofConfined()) { float[] r = ML.topK(NdArrayFactory.array(a, new float[]{2,1,0.1f}, 1, 3), 2).toFloatArray(); assertEquals(2f, r[0], T); assertEquals(Float.NEGATIVE_INFINITY, r[2]); } }
    @Test void layerNorm() { try (var a = Arena.ofConfined()) { assertArr("ln", ML.layerNorm(NdArrayFactory.array(a, new float[]{1,2,3}, 1, 3), NdArrayFactory.array(a, new float[]{1,1,1}), NdArrayFactory.array(a, new float[]{0,0,0}), 1e-5f), new float[]{-1.2247356f,0f,1.2247356f}); } }
    @Test void beamSearch() { try (var a = Arena.ofConfined()) { int[][] b = ML.beamSearch(NdArrayFactory.array(a, new float[]{0.1f,2,1,0.5f}), 2); assertEquals(1, b[0][0]); } }
    @Test void temperatureScale() { try (var a = Arena.ofConfined()) { assertArr("temp", ML.temperatureScale(NdArrayFactory.array(a, new float[]{2,4,6}), 2f), new float[]{1,2,3}); } }
    @Test void crossEntropy() { try (var a = Arena.ofConfined()) { assertEquals(0.4170299f, ML.crossEntropy(NdArrayFactory.array(a, new float[]{2,1,0.1f}, 1, 3), NdArrayFactory.array(a, new float[]{1,0,0}, 1, 3)), 0.01f); } }
    @Test void conv1d() { try (var a = Arena.ofConfined()) { NdArray r = ML.conv1d(NdArrayFactory.array(a, new float[]{1,2,3,4,5}, 1, 1, 5), NdArrayFactory.array(a, new float[]{1,0,-1}, 1, 1, 3), null, 1, 0); assertEquals(-2f, r.flatGetFloat(0), T); } }
    @Test void topP() { try (var a = Arena.ofConfined()) { NdArray r = ML.topP(NdArrayFactory.array(a, new float[]{2,1,0.1f}, 1, 3), 0.9f); assertTrue(r.flatGetFloat(0) > Float.NEGATIVE_INFINITY); } }
}
