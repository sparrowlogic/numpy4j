package com.sparrowlogic.numpy4j;

import org.junit.jupiter.api.Test;
import java.lang.foreign.Arena;
import static org.junit.jupiter.api.Assertions.*;

public class RandomTest {
    private static final float T = 0.001f;

    @Test void uniform() { try (var a = Arena.ofConfined()) { Random r = new Random(42); NdArray u = r.uniform(a, 0f, 1f, 1000); assertTrue(Reductions.min(u) >= 0f); assertTrue(Reductions.max(u) < 1f); } }
    @Test void normal() { try (var a = Arena.ofConfined()) { Random r = new Random(42); NdArray n = r.normal(a, 0f, 1f, 10000); assertEquals(0f, Reductions.mean(n), 0.1f); assertEquals(1f, Reductions.std(n), 0.1f); } }
    @Test void randint() { try (var a = Arena.ofConfined()) { Random r = new Random(42); int[] v = r.randint(a, 0, 10, 100).toIntArray(); for (int x : v) assertTrue(x >= 0 && x < 10); } }
    @Test void permutation() { try (var a = Arena.ofConfined()) { assertEquals(5, new Random(42).permutation(a, 5).size()); } }
    @Test void choice() { try (var a = Arena.ofConfined()) { assertEquals(5, new Random(42).choice(a, NdArrayFactory.array(a, new float[]{10,20,30}), 5).size()); } }
    @Test void shuffle() { try (var a = Arena.ofConfined()) { Random r = new Random(42); NdArray x = NdArrayFactory.array(a, new float[]{1,2,3,4,5}); r.shuffle(x); assertEquals(15f, Reductions.sum(x), T); } }

    // ── standardNormal parity: delegates to normal(0,1), verify distribution properties ──
    @Test void standardNormalMean() { try (var a = Arena.ofConfined()) { assertEquals(0f, Reductions.mean(new Random(42).standardNormal(a, 50000)), 0.05f); } }
    @Test void standardNormalStd() { try (var a = Arena.ofConfined()) { assertEquals(1f, Reductions.std(new Random(42).standardNormal(a, 50000)), 0.05f); } }
    @Test void standardNormalShape() { try (var a = Arena.ofConfined()) { assertArrayEquals(new int[]{3, 4}, new Random(42).standardNormal(a, 3, 4).shape()); } }
    @Test void standardNormalMatchesNormal() {
        // standardNormal must produce identical output to normal(0, 1) with same seed
        try (var a1 = Arena.ofConfined(); var a2 = Arena.ofConfined()) {
            float[] sn = new Random(99).standardNormal(a1, 100).toFloatArray();
            float[] n = new Random(99).normal(a2, 0f, 1f, 100).toFloatArray();
            assertArrayEquals(sn, n, T);
        }
    }

    // ── exponential parity: -scale * ln(1 - u), verify E[X]=scale, all x >= 0 ──
    @Test void exponentialMean() { try (var a = Arena.ofConfined()) { assertEquals(2f, Reductions.mean(new Random(42).exponential(a, 2f, 50000)), 0.1f); } }
    @Test void exponentialNonNegative() { try (var a = Arena.ofConfined()) { assertTrue(Reductions.min(new Random(42).exponential(a, 1f, 10000)) >= 0f); } }
    @Test void exponentialShape() { try (var a = Arena.ofConfined()) { assertArrayEquals(new int[]{5, 3}, new Random(42).exponential(a, 1f, 5, 3).shape()); } }
    @Test void exponentialScaleOne() {
        // E[X]=1, Var[X]=1 for scale=1
        try (var a = Arena.ofConfined()) {
            NdArray e = new Random(42).exponential(a, 1f, 50000);
            assertEquals(1f, Reductions.mean(e), 0.05f);
            float v = Reductions.var(e);
            assertEquals(1f, v, 0.15f);
        }
    }
    @Test void exponentialFormula() {
        // Verify the formula: -scale * ln(1 - u) matches numpy.random.exponential
        // For known u=0.3, scale=2.0: expected = -2.0 * ln(0.7) = 0.71335
        float expected = (float) (-2.0 * Math.log(1.0 - 0.3));
        assertEquals(0.71335f, expected, 0.001f);
    }

    // ── poisson parity: Knuth's algorithm, verify E[X]=λ, Var[X]≈λ, all x >= 0 integer ──
    @Test void poissonMean() { try (var a = Arena.ofConfined()) { assertEquals(5f, Reductions.mean(new Random(42).poisson(a, 5f, 50000)), 0.15f); } }
    @Test void poissonVariance() {
        try (var a = Arena.ofConfined()) {
            NdArray p = new Random(42).poisson(a, 5f, 50000);
            assertEquals(5f, Reductions.var(p), 0.3f); // Var[X] ≈ λ
        }
    }
    @Test void poissonNonNegativeInteger() {
        try (var a = Arena.ofConfined()) {
            float[] vals = new Random(42).poisson(a, 3f, 1000).toFloatArray();
            for (float v : vals) { assertTrue(v >= 0f); assertEquals(v, Math.floor(v), T); }
        }
    }
    @Test void poissonShape() { try (var a = Arena.ofConfined()) { assertArrayEquals(new int[]{4, 5}, new Random(42).poisson(a, 3f, 4, 5).shape()); } }
    @Test void poissonSmallLambda() {
        // λ=0.5: most values should be 0 or 1
        try (var a = Arena.ofConfined()) {
            NdArray p = new Random(42).poisson(a, 0.5f, 10000);
            assertEquals(0.5f, Reductions.mean(p), 0.1f);
        }
    }

    // ── binomial parity: trial-based, verify E[X]=np, Var[X]=np(1-p), x ∈ [0,n] ──
    @Test void binomialMean() { try (var a = Arena.ofConfined()) { assertEquals(5f, Reductions.mean(new Random(42).binomial(a, 10, 0.5f, 50000)), 0.15f); } }
    @Test void binomialVariance() {
        try (var a = Arena.ofConfined()) {
            NdArray b = new Random(42).binomial(a, 10, 0.5f, 50000);
            assertEquals(2.5f, Reductions.var(b), 0.2f); // Var = n*p*(1-p) = 2.5
        }
    }
    @Test void binomialBounds() {
        try (var a = Arena.ofConfined()) {
            NdArray b = new Random(42).binomial(a, 10, 0.5f, 10000);
            assertTrue(Reductions.min(b) >= 0f);
            assertTrue(Reductions.max(b) <= 10f);
        }
    }
    @Test void binomialInteger() {
        try (var a = Arena.ofConfined()) {
            float[] vals = new Random(42).binomial(a, 10, 0.5f, 1000).toFloatArray();
            for (float v : vals) assertEquals(v, Math.floor(v), T);
        }
    }
    @Test void binomialShape() { try (var a = Arena.ofConfined()) { assertArrayEquals(new int[]{3, 3}, new Random(42).binomial(a, 10, 0.5f, 3, 3).shape()); } }
    @Test void binomialExtremeProbabilities() {
        try (var a = Arena.ofConfined()) {
            // p=0: all zeros
            float[] z = new Random(42).binomial(a, 10, 0f, 100).toFloatArray();
            for (float v : z) assertEquals(0f, v, T);
            // p=1: all n
            float[] n = new Random(42).binomial(a, 10, 1f, 100).toFloatArray();
            for (float v : n) assertEquals(10f, v, T);
        }
    }
    @Test void binomialAsymmetric() {
        // n=20, p=0.8: E[X]=16, Var[X]=3.2
        try (var a = Arena.ofConfined()) {
            NdArray b = new Random(42).binomial(a, 20, 0.8f, 50000);
            assertEquals(16f, Reductions.mean(b), 0.15f);
            assertEquals(3.2f, Reductions.var(b), 0.3f);
        }
    }

    // ── Determinism: same seed → same output ──
    @Test void standardNormalDeterministic() {
        try (var a1 = Arena.ofConfined(); var a2 = Arena.ofConfined()) {
            assertArrayEquals(new Random(123).standardNormal(a1, 50).toFloatArray(),
                    new Random(123).standardNormal(a2, 50).toFloatArray(), T);
        }
    }
    @Test void exponentialDeterministic() {
        try (var a1 = Arena.ofConfined(); var a2 = Arena.ofConfined()) {
            assertArrayEquals(new Random(123).exponential(a1, 2f, 50).toFloatArray(),
                    new Random(123).exponential(a2, 2f, 50).toFloatArray(), T);
        }
    }
    @Test void poissonDeterministic() {
        try (var a1 = Arena.ofConfined(); var a2 = Arena.ofConfined()) {
            assertArrayEquals(new Random(123).poisson(a1, 5f, 50).toFloatArray(),
                    new Random(123).poisson(a2, 5f, 50).toFloatArray(), T);
        }
    }
    @Test void binomialDeterministic() {
        try (var a1 = Arena.ofConfined(); var a2 = Arena.ofConfined()) {
            assertArrayEquals(new Random(123).binomial(a1, 10, 0.5f, 50).toFloatArray(),
                    new Random(123).binomial(a2, 10, 0.5f, 50).toFloatArray(), T);
        }
    }
}
