package com.sparrowlogic.numpy4j;

import org.junit.jupiter.api.Test;
import java.lang.foreign.Arena;
import static org.junit.jupiter.api.Assertions.*;

public class ReductionsTest {
    private static final float T = 0.001f;
    private static void assertArr(String l, NdArray a, float[] e) { float[] ac = a.toFloatArray(); for (int i = 0; i < e.length; i++) assertEquals(e[i], ac[i], T, l+"["+i+"]"); }

    @Test void sum() { try (var a = Arena.ofConfined()) { assertEquals(21f, Reductions.sum(NdArrayFactory.array(a, new float[]{1,2,3,4,5,6})), T); } }
    @Test void prod() { try (var a = Arena.ofConfined()) { assertEquals(720f, Reductions.prod(NdArrayFactory.array(a, new float[]{1,2,3,4,5,6})), T); } }
    @Test void mean() { try (var a = Arena.ofConfined()) { assertEquals(3.5f, Reductions.mean(NdArrayFactory.array(a, new float[]{1,2,3,4,5,6})), T); } }
    @Test void var_() { try (var a = Arena.ofConfined()) { assertEquals(2.917f, Reductions.var(NdArrayFactory.array(a, new float[]{1,2,3,4,5,6})), 0.01f); } }
    @Test void std_() { try (var a = Arena.ofConfined()) { assertEquals(1.708f, Reductions.std(NdArrayFactory.array(a, new float[]{1,2,3,4,5,6})), 0.01f); } }
    @Test void maxMin() { try (var a = Arena.ofConfined()) { NdArray x = NdArrayFactory.array(a, new float[]{1,2,3,4,5,6}); assertEquals(6f, Reductions.max(x), T); assertEquals(1f, Reductions.min(x), T); } }
    @Test void argmaxArgmin() { try (var a = Arena.ofConfined()) { NdArray x = NdArrayFactory.array(a, new float[]{1,2,3,4,5,6}); assertEquals(5, Reductions.argmax(x)); assertEquals(0, Reductions.argmin(x)); } }
    @Test void cumsum() { try (var a = Arena.ofConfined()) { assertArr("cs", Reductions.cumsum(NdArrayFactory.array(a, new float[]{1,2,3,4})), new float[]{1,3,6,10}); } }
    @Test void cumprod() { try (var a = Arena.ofConfined()) { assertArr("cp", Reductions.cumprod(NdArrayFactory.array(a, new float[]{1,2,3,4})), new float[]{1,2,6,24}); } }
    @Test void sumAxis() { try (var a = Arena.ofConfined()) { NdArray m = NdArrayFactory.array(a, new float[]{1,2,3,4,5,6}, 2, 3); assertArr("a0", Reductions.sum(m, 0), new float[]{5,7,9}); assertArr("a1", Reductions.sum(m, 1), new float[]{6,15}); } }
    @Test void allAny() { try (var a = Arena.ofConfined()) { assertTrue(Reductions.all(NdArrayFactory.array(a, new float[]{1,2,3}))); assertFalse(Reductions.all(NdArrayFactory.array(a, new float[]{1,0,3}))); assertTrue(Reductions.any(NdArrayFactory.array(a, new float[]{0,0,1}))); } }
    @Test void nansum() { try (var a = Arena.ofConfined()) { assertEquals(9f, Reductions.nansum(NdArrayFactory.array(a, new float[]{1,Float.NaN,3,Float.NaN,5})), T); } }
    @Test void nanmean() { try (var a = Arena.ofConfined()) { assertEquals(3f, Reductions.nanmean(NdArrayFactory.array(a, new float[]{1,Float.NaN,3,Float.NaN,5})), T); } }
    @Test void nanstd() { try (var a = Arena.ofConfined()) { assertEquals(1.633f, Reductions.nanstd(NdArrayFactory.array(a, new float[]{1,Float.NaN,3,Float.NaN,5})), 0.01f); } }
    @Test void nanvar() { try (var a = Arena.ofConfined()) { assertEquals(2.667f, Reductions.nanvar(NdArrayFactory.array(a, new float[]{1,Float.NaN,3,Float.NaN,5})), 0.01f); } }
    @Test void nanmax() { try (var a = Arena.ofConfined()) { assertEquals(5f, Reductions.nanmax(NdArrayFactory.array(a, new float[]{1,Float.NaN,3,Float.NaN,5})), T); } }
    @Test void nanmin() { try (var a = Arena.ofConfined()) { assertEquals(1f, Reductions.nanmin(NdArrayFactory.array(a, new float[]{1,Float.NaN,3,Float.NaN,5})), T); } }
    @Test void nanargmax() { try (var a = Arena.ofConfined()) { assertEquals(4, Reductions.nanargmax(NdArrayFactory.array(a, new float[]{1,Float.NaN,3,Float.NaN,5}))); } }
    @Test void nanargmin() { try (var a = Arena.ofConfined()) { assertEquals(0, Reductions.nanargmin(NdArrayFactory.array(a, new float[]{1,Float.NaN,3,Float.NaN,5}))); } }
    @Test void median() { try (var a = Arena.ofConfined()) { assertEquals(5.5f, Reductions.median(NdArrayFactory.array(a, new float[]{1,2,3,4,5,6,7,8,9,10})), T); } }
    @Test void percentile() { try (var a = Arena.ofConfined()) { assertEquals(3.25f, Reductions.percentile(NdArrayFactory.array(a, new float[]{1,2,3,4,5,6,7,8,9,10}), 25), T); } }
    @Test void histogram() { try (var a = Arena.ofConfined()) { NdArray[] r = Reductions.histogram(NdArrayFactory.array(a, new float[]{1,2,1,3,2,1}), 3); assertArr("c", r[0], new float[]{3,2,1}); } }
    @Test void allclose() { try (var a = Arena.ofConfined()) { assertTrue(Reductions.allclose(NdArrayFactory.array(a, new float[]{1,2}), NdArrayFactory.array(a, new float[]{1,2}), 1e-5f, 1e-8f)); } }
    @Test void arrayEqual() { try (var a = Arena.ofConfined()) { assertTrue(Reductions.arrayEqual(NdArrayFactory.array(a, new float[]{1,2,3}), NdArrayFactory.array(a, new float[]{1,2,3}))); } }
    @Test void countNonzero() { try (var a = Arena.ofConfined()) { assertEquals(3, Reductions.countNonzero(NdArrayFactory.array(a, new float[]{0,1,0,3,0,5}))); } }
    @Test void quantile() { try (var a = Arena.ofConfined()) { assertEquals(3.25f, Reductions.quantile(NdArrayFactory.array(a, new float[]{1,2,3,4,5,6,7,8,9,10}), 0.25f), T); } }
    @Test void average() { try (var a = Arena.ofConfined()) { assertEquals(2f, Reductions.average(NdArrayFactory.array(a, new float[]{1,2,3,4}), NdArrayFactory.array(a, new float[]{4,3,2,1})), T); } }

    // ── Axis-aware reductions coverage ──
    @Test void sumAxisKeepdims() { try (var a = Arena.ofConfined()) { NdArray r = Reductions.sum(NdArrayFactory.array(a, new float[]{1,2,3,4,5,6}, 2, 3), 0, true); assertArrayEquals(new int[]{1, 3}, r.shape()); } }
    @Test void meanAxis() { try (var a = Arena.ofConfined()) { assertArr("ma", Reductions.mean(NdArrayFactory.array(a, new float[]{1,2,3,4,5,6}, 2, 3), 0), new float[]{2.5f, 3.5f, 4.5f}); } }
    @Test void maxAxis() { try (var a = Arena.ofConfined()) { assertArr("mxa", Reductions.max(NdArrayFactory.array(a, new float[]{1,2,3,4,5,6}, 2, 3), 0), new float[]{4,5,6}); } }
    @Test void minAxis() { try (var a = Arena.ofConfined()) { assertArr("mna", Reductions.min(NdArrayFactory.array(a, new float[]{1,2,3,4,5,6}, 2, 3), 0), new float[]{1,2,3}); } }
    @Test void argmaxAxis() { try (var a = Arena.ofConfined()) { NdArray r = Reductions.argmax(NdArrayFactory.array(a, new float[]{1,2,3,4,5,6}, 2, 3), 0); assertEquals(3, r.size()); } }
    @Test void argminAxis() { try (var a = Arena.ofConfined()) { NdArray r = Reductions.argmin(NdArrayFactory.array(a, new float[]{1,2,3,4,5,6}, 2, 3), 0); assertEquals(3, r.size()); } }
}
