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
    @Test void fromfunction() { try (var a = Arena.ofConfined()) { assertArrayEquals(new float[]{0,1,2,3,1,2,3,4,2,3,4,5}, NdArrayFactory.fromfunction(a, idx -> idx[0] + idx[1], 3, 4).toFloatArray(), T); } }
    @Test void fromfunctionMul() { try (var a = Arena.ofConfined()) { assertArrayEquals(new float[]{0,1,2,3,10,11,12,13,20,21,22,23}, NdArrayFactory.fromfunction(a, idx -> idx[0] * 10 + idx[1], 3, 4).toFloatArray(), T); } }
    @Test void fromfunctionSquare() { try (var a = Arena.ofConfined()) { assertArrayEquals(new float[]{0,1,4,9,16}, NdArrayFactory.fromfunction(a, idx -> idx[0] * idx[0], 5).toFloatArray(), T); } }
    @Test void fromfunction3d() { try (var a = Arena.ofConfined()) { assertArrayEquals(new float[]{0,1,1,2,2,3,1,2,2,3,3,4}, NdArrayFactory.fromfunction(a, idx -> idx[0] + idx[1] + idx[2], 2, 3, 2).toFloatArray(), T); } }
    @Test void fromfunctionShape() { try (var a = Arena.ofConfined()) { assertArrayEquals(new int[]{3, 4}, NdArrayFactory.fromfunction(a, idx -> idx[0] + idx[1], 3, 4).shape()); } }

    // ── dtype branch coverage ──
    @Test void fullFloat64() { try (var a = Arena.ofConfined()) { NdArray r = NdArrayFactory.full(a, DType.FLOAT64, 3.14, 3); assertEquals(3.14, r.flatGetDouble(0), 0.001); } }
    @Test void fullInt32() { try (var a = Arena.ofConfined()) { NdArray r = NdArrayFactory.full(a, DType.INT32, 42, 3); assertEquals(42, r.getInt(0)); } }
    @Test void fullInt64() { try (var a = Arena.ofConfined()) { NdArray r = NdArrayFactory.full(a, DType.INT64, 99, 3); assertEquals(99L, r.getLong(0)); } }
    @Test void fullBool() { try (var a = Arena.ofConfined()) { NdArray r = NdArrayFactory.full(a, DType.BOOL, 1, 3); assertEquals((byte) 1, r.data().get(java.lang.foreign.ValueLayout.JAVA_BYTE, 0)); } }
    @Test void eyeFloat64() { try (var a = Arena.ofConfined()) { NdArray r = NdArrayFactory.eye(a, 2, DType.FLOAT64); assertEquals(1.0, r.getDouble(0, 0), 0.001); assertEquals(0.0, r.getDouble(0, 1), 0.001); } }
    @Test void eyeInt32() { try (var a = Arena.ofConfined()) { NdArray r = NdArrayFactory.eye(a, 2, DType.INT32); assertEquals(1, r.getInt(0, 0)); assertEquals(0, r.getInt(0, 1)); } }
    @Test void eyeInt64() { try (var a = Arena.ofConfined()) { NdArray r = NdArrayFactory.eye(a, 2, DType.INT64); assertEquals(1L, r.getLong(0, 0)); assertEquals(0L, r.getLong(0, 1)); } }
    @Test void emptyLike() { try (var a = Arena.ofConfined()) { NdArray r = NdArrayFactory.emptyLike(NdArrayFactory.array(a, new float[]{1,2,3})); assertEquals(3, r.size()); } }
    @Test void arrayDouble() { try (var a = Arena.ofConfined()) { NdArray r = NdArrayFactory.array(a, new double[]{1.0, 2.0, 3.0}); assertEquals(2.0, r.flatGetDouble(1), 0.001); } }
}
