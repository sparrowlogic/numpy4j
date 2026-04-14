package com.sparrowlogic.numpy4j;

import org.junit.jupiter.api.Test;
import java.lang.foreign.Arena;
import static org.junit.jupiter.api.Assertions.*;

/** Tests for {@link SignalOps} — convolve, correlate, interp. */
public class SignalOpsTest {
    private static final float T = 0.001f;
    private static void assertArr(String l, NdArray a, float[] e) { float[] ac = a.toFloatArray(); for (int i = 0; i < e.length; i++) assertEquals(e[i], ac[i], T, l+"["+i+"]"); }

    @Test void convolve() { try (var a = Arena.ofConfined()) { assertArr("conv", SignalOps.convolve(NdArrayFactory.array(a, new float[]{1,2,3}), NdArrayFactory.array(a, new float[]{0,1,0.5f})), new float[]{0,1,2.5f,4,1.5f}); } }
    @Test void correlate() { try (var a = Arena.ofConfined()) { assertArr("corr", SignalOps.correlate(NdArrayFactory.array(a, new float[]{1,2,3}), NdArrayFactory.array(a, new float[]{0,1,0.5f})), new float[]{0.5f,2,3.5f,3,0}); } }
    @Test void interp() { assertEquals(25f, SignalOps.interp(2.5f, new float[]{1,2,3}, new float[]{10,20,30}), T); }
    @Test void interpEdge() { assertEquals(10f, SignalOps.interp(0f, new float[]{1,2,3}, new float[]{10,20,30}), T); }
}
