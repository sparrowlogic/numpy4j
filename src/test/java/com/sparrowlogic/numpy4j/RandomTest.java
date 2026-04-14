package com.sparrowlogic.numpy4j;

import org.junit.jupiter.api.Test;
import java.lang.foreign.Arena;
import static org.junit.jupiter.api.Assertions.*;

public class RandomTest {
    @Test void uniform() { try (var a = Arena.ofConfined()) { Random r = new Random(42); NdArray u = r.uniform(a, 0f, 1f, 1000); assertTrue(Reductions.min(u) >= 0f); assertTrue(Reductions.max(u) < 1f); } }
    @Test void normal() { try (var a = Arena.ofConfined()) { Random r = new Random(42); NdArray n = r.normal(a, 0f, 1f, 10000); assertEquals(0f, Reductions.mean(n), 0.1f); assertEquals(1f, Reductions.std(n), 0.1f); } }
    @Test void randint() { try (var a = Arena.ofConfined()) { Random r = new Random(42); int[] v = r.randint(a, 0, 10, 100).toIntArray(); for (int x : v) assertTrue(x >= 0 && x < 10); } }
    @Test void permutation() { try (var a = Arena.ofConfined()) { assertEquals(5, new Random(42).permutation(a, 5).size()); } }
    @Test void choice() { try (var a = Arena.ofConfined()) { assertEquals(5, new Random(42).choice(a, NdArrayFactory.array(a, new float[]{10,20,30}), 5).size()); } }
    @Test void shuffle() { try (var a = Arena.ofConfined()) { Random r = new Random(42); NdArray x = NdArrayFactory.array(a, new float[]{1,2,3,4,5}); r.shuffle(x); assertEquals(15f, Reductions.sum(x), 0.001f); } }
    @Test void standardNormal() { try (var a = Arena.ofConfined()) { assertEquals(0f, Reductions.mean(new Random(42).standardNormal(a, 1000)), 0.2f); } }
    @Test void exponential() { try (var a = Arena.ofConfined()) { float m = Reductions.mean(new Random(42).exponential(a, 1f, 5000)); assertTrue(m > 0.8f && m < 1.2f); } }
    @Test void poissonDist() { try (var a = Arena.ofConfined()) { float m = Reductions.mean(new Random(42).poisson(a, 5f, 5000)); assertTrue(m > 4f && m < 6f); } }
    @Test void binomialDist() { try (var a = Arena.ofConfined()) { float m = Reductions.mean(new Random(42).binomial(a, 10, 0.5f, 5000)); assertTrue(m > 4f && m < 6f); } }
}
