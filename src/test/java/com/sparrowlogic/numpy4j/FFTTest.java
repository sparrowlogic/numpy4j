package com.sparrowlogic.numpy4j;

import org.junit.jupiter.api.Test;
import java.lang.foreign.Arena;
import static org.junit.jupiter.api.Assertions.*;

public class FFTTest {
    private static final float T = 0.001f;

    @Test void fft() { float[] r = FFT.fft(new float[]{1,0,2,0,3,0,4,0}); assertEquals(10f, r[0], T); assertEquals(-2f, r[2], T); assertEquals(2f, r[3], T); }
    @Test void ifftRoundtrip() { float[] in = {1,0,2,0,3,0,4,0}; float[] r = FFT.ifft(FFT.fft(in)); for (int i = 0; i < in.length; i++) assertEquals(in[i], r[i], T); }
    @Test void rfft() { float[] r = FFT.rfft(new float[]{1,2,3,4}); assertEquals(10f, r[0], T); assertEquals(-2f, r[2], T); }
    @Test void irfft() { float[] c = FFT.rfft(new float[]{1,2,3,4}); float[] r = FFT.irfft(c, 4); assertEquals(1f, r[0], T); assertEquals(4f, r[3], T); }
    @Test void fftfreq() { double[] f = FFT.fftfreq(4, 1.0); assertEquals(0.0, f[0], T); assertEquals(0.25, f[1], T); assertEquals(-0.5, f[2], T); }
    @Test void rfftfreq() { double[] f = FFT.rfftfreq(4, 1.0); assertEquals(0.0, f[0], T); assertEquals(0.5, f[2], T); }
    @Test void fftshift1D() { float[] r = FFT.fftshift(new float[]{0,1,2,3,4}); assertEquals(2f, r[0], T); assertEquals(0f, r[3], T); }

    @Test void fft2() {
        try (var arena = Arena.ofConfined()) {
            // [[1,2],[3,4]] -> fft2 real: [10,-2,-4,0], imag: [0,0,0,0]
            NdArray m = NdArrayFactory.array(arena, new float[]{1,2,3,4}, 2, 2);
            NdArray result = FFT.fft2(m);
            // result shape: (2,2,2) — last dim is [re,im]
            assertArrayEquals(new int[]{2, 2, 2}, result.shape());
            assertEquals(10f, result.flatGetFloat(0), T, "fft2 re[0,0]");
            assertEquals(0f, result.flatGetFloat(1), T, "fft2 im[0,0]");
            assertEquals(-2f, result.flatGetFloat(2), T, "fft2 re[0,1]");
            assertEquals(-4f, result.flatGetFloat(4), T, "fft2 re[1,0]");
            assertEquals(0f, result.flatGetFloat(6), T, "fft2 re[1,1]");
        }
    }

    @Test void ifft2Roundtrip() {
        try (var arena = Arena.ofConfined()) {
            NdArray m = NdArrayFactory.array(arena, new float[]{1,2,3,4}, 2, 2);
            NdArray fwd = FFT.fft2(m);
            NdArray back = FFT.ifft2(fwd);
            // back should be (2,2) real values matching original
            assertEquals(1f, back.flatGetFloat(0), T);
            assertEquals(2f, back.flatGetFloat(1), T);
            assertEquals(3f, back.flatGetFloat(2), T);
            assertEquals(4f, back.flatGetFloat(3), T);
        }
    }

    @Test void fftn() {
        try (var arena = Arena.ofConfined()) {
            NdArray m = NdArrayFactory.array(arena, new float[]{1,2,3,4}, 2, 2);
            NdArray result = FFT.fftn(m);
            assertEquals(10f, result.flatGetFloat(0), T, "fftn re[0,0]");
            assertEquals(-2f, result.flatGetFloat(2), T, "fftn re[0,1]");
        }
    }

    @Test void fancyIndex() {
        try (var arena = Arena.ofConfined()) {
            NdArray a = NdArrayFactory.array(arena, new float[]{10,20,30,40,50});
            NdArray r = IndexOps.fancyIndex(a, new int[]{0, 2, 4});
            assertEquals(3, r.size());
            assertEquals(10f, r.flatGetFloat(0), T);
            assertEquals(30f, r.flatGetFloat(1), T);
            assertEquals(50f, r.flatGetFloat(2), T);
        }
    }

    // ── Bluestein (non-power-of-2) coverage ──
    @Test void fftNonPow2Roundtrip() {
        // n=3 (non-power-of-2) triggers Bluestein path; verify roundtrip
        float[] in = {1,0, 2,0, 3,0};
        float[] fwd = FFT.fft(in);
        float[] back = FFT.ifft(fwd);
        for (int i = 0; i < in.length; i++) assertEquals(in[i], back[i], 0.01f, "roundtrip[" + i + "]");
    }

    @Test void fftNonPow2Size5Roundtrip() {
        float[] in = {5,0, 3,0, 1,0, 4,0, 2,0};
        float[] fwd = FFT.fft(in);
        float[] back = FFT.ifft(fwd);
        for (int i = 0; i < in.length; i++) assertEquals(in[i], back[i], 0.01f, "roundtrip5[" + i + "]");
    }

    @Test void fftNonPow2Size7Roundtrip() {
        float[] in = {1,0, 0,1, 1,0, 0,1, 1,0, 0,1, 1,0};
        float[] fwd = FFT.fft(in);
        float[] back = FFT.ifft(fwd);
        for (int i = 0; i < in.length; i++) assertEquals(in[i], back[i], 0.01f, "roundtrip7[" + i + "]");
    }

    @Test void fftnOn1D() {
        try (var arena = Arena.ofConfined()) {
            NdArray a = NdArrayFactory.array(arena, new float[]{1,2,3,4});
            NdArray r = FFT.fftn(a);
            assertEquals(10f, r.flatGetFloat(0), T);
        }
    }

    @Test void fftEmpty() {
        float[] r = FFT.fft(new float[0]);
        assertEquals(0, r.length);
    }

    @Test void irfftRoundtrip() {
        float[] orig = {1,2,3,4,5,6,7,8};
        float[] c = FFT.rfft(orig);
        float[] back = FFT.irfft(c, 8);
        for (int i = 0; i < orig.length; i++) assertEquals(orig[i], back[i], T);
    }

    @Test void fftNdArrayReal() {
        try (var arena = Arena.ofConfined()) {
            NdArray a = NdArrayFactory.array(arena, new float[]{1,2,3,4});
            NdArray r = FFT.fft(a);
            assertEquals(10f, r.flatGetFloat(0), T);
        }
    }

    @Test void fftNdArrayComplex() {
        try (var arena = Arena.ofConfined()) {
            NdArray a = NdArrayFactory.array(arena, new float[]{1,0,2,0,3,0,4,0}, 4, 2);
            NdArray r = FFT.fft(a);
            assertEquals(10f, r.flatGetFloat(0), T);
        }
    }
}
