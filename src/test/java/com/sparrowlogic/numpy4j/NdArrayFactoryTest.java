package com.sparrowlogic.numpy4j;

import org.junit.jupiter.api.Test;
import java.lang.foreign.Arena;
import static org.junit.jupiter.api.Assertions.*;

public class NdArrayFactoryTest {
    private static final float T = 0.001f;

    @Test void zeros() { try (var a = Arena.ofConfined()) { assertEquals(0f, NdArrayFactory.zeros(a, DType.FLOAT32, 2, 3).flatGetFloat(0), T); } }
    @Test void ones() { try (var a = Arena.ofConfined()) { assertEquals(1f, NdArrayFactory.ones(a, DType.FLOAT32, 3).flatGetFloat(2), T); } }
    @Test void full() { try (var a = Arena.ofConfined()) { assertEquals(7.5f, NdArrayFactory.full(a, DType.FLOAT32, 7.5, 4).flatGetFloat(3), T); } }
    @Test void arange() { try (var a = Arena.ofConfined()) { assertArrayEquals(new float[]{0,1,2,3,4,5}, NdArrayFactory.arange(a, 6).toFloatArray(), T); } }
    @Test void arangeStep() { try (var a = Arena.ofConfined()) { assertArrayEquals(new float[]{0,2.5f,5,7.5f}, NdArrayFactory.arange(a, 0f, 10f, 2.5f).toFloatArray(), T); } }
    @Test void linspace() { try (var a = Arena.ofConfined()) { double[] r = NdArrayFactory.linspace(a, 0, 1, 5).toDoubleArray(); assertEquals(0.0, r[0], T); assertEquals(0.25, r[1], T); assertEquals(1.0, r[4], T); } }
    @Test void eye() { try (var a = Arena.ofConfined()) { assertArrayEquals(new float[]{1,0,0,0,1,0,0,0,1}, NdArrayFactory.eye(a, 3).toFloatArray(), T); } }
    @Test void arrayFloat() { try (var a = Arena.ofConfined()) { assertEquals(3f, NdArrayFactory.array(a, new float[]{1,2,3,4}, 2, 2).getFloat(1, 0), T); } }
    @Test void arrayInt() { try (var a = Arena.ofConfined()) { assertEquals(20, NdArrayFactory.array(a, new int[]{10,20,30}).getInt(1)); } }
    @Test void zerosLike() { try (var a = Arena.ofConfined()) { assertArrayEquals(new float[]{0,0,0}, NdArrayFactory.zerosLike(NdArrayFactory.array(a, new float[]{1,2,3})).toFloatArray(), T); } }
    @Test void onesLike() { try (var a = Arena.ofConfined()) { assertArrayEquals(new float[]{1,1,1}, NdArrayFactory.onesLike(NdArrayFactory.array(a, new float[]{1,2,3})).toFloatArray(), T); } }
    @Test void fullLike() { try (var a = Arena.ofConfined()) { assertArrayEquals(new float[]{7,7,7}, NdArrayFactory.fullLike(NdArrayFactory.array(a, new float[]{1,2,3}), 7).toFloatArray(), T); } }
    @Test void logspace() { try (var a = Arena.ofConfined()) { NdArray r = NdArrayFactory.logspace(a, 0, 2, 5); assertEquals(1f, r.flatGetFloat(0), T); assertEquals(100f, r.flatGetFloat(4), 0.1f); } }
    @Test void geomspace() { try (var a = Arena.ofConfined()) { NdArray r = NdArrayFactory.geomspace(a, 1, 1000, 4); assertEquals(1f, r.flatGetFloat(0), T); assertEquals(1000f, r.flatGetFloat(3), 0.1f); } }
    @Test void fromfunction() { try (var a = Arena.ofConfined()) { NdArray r = NdArrayFactory.fromfunction(a, idx -> idx[0] + idx[1], 2, 3); assertEquals(0f, r.flatGetFloat(0), T); assertEquals(1f, r.flatGetFloat(1), T); assertEquals(2f, r.flatGetFloat(4), T); } }
}
