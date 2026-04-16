package com.sparrowlogic.numpy4j;

/**
 * FFT module matching numpy.fft.
 * Cooley-Tukey for power-of-2, Bluestein for arbitrary lengths.
 */
public final class FFT {
    private FFT() {
    }

    /**
     * Complex FFT ({@code numpy.fft.fft}).
     *
     * @param input interleaved complex array [re0, im0, re1, im1, ...]
     * @return transformed interleaved complex array
     */
    public static float[] fft(final float[] input) {
        int n = input.length / 2;
        if (n == 0) {
            return new float[0];
        }
        return (n & (n - 1)) == 0 ? cooleyTukey(input, false) : bluestein(input, false);
    }

    /**
     * Complex FFT on an {@link NdArray} ({@code numpy.fft.fft}).
     *
     * <p>1-D real input is zero-padded to complex. 2-D input with last dim=2 is treated as interleaved complex.</p>
     *
     * @param a input array (real 1-D or interleaved complex)
     * @return complex result as shape (n, 2)
     */
    public static NdArray fft(final NdArray a) {
        float[] data = a.toFloatArray();
        float[] complex;
        if (a.ndim() == 1) {
            complex = new float[data.length * 2];
            for (int i = 0; i < data.length; i++) {
                complex[2 * i] = data[i];
            }
        } else {
            complex = data;
        }
        float[] result = fft(complex);
        return NdArrayFactory.array(a.arena(), result, result.length / 2, 2);
    }

    /**
     * Inverse complex FFT ({@code numpy.fft.ifft}).
     *
     * @param input interleaved complex array
     * @return inverse-transformed interleaved complex array
     */
    public static float[] ifft(final float[] input) {
        int n = input.length / 2;
        float[] result;
        if ((n & (n - 1)) == 0) {
            result = cooleyTukey(input, true);
        } else {
            result = bluestein(input, true);
        }
        for (int i = 0; i < result.length; i++) {
            result[i] /= n;
        }
        return result;
    }

    /**
     * Real-input FFT ({@code numpy.fft.rfft}).
     *
     * @param realInput real-valued input array
     * @return interleaved complex output of length (n/2+1) pairs
     */
    public static float[] rfft(final float[] realInput) {
        int n = realInput.length;
        float[] complex = new float[n * 2];
        for (int i = 0; i < n; i++) {
            complex[2 * i] = realInput[i];
        }
        float[] full = fft(complex);
        int outLen = n / 2 + 1;
        float[] result = new float[outLen * 2];
        System.arraycopy(full, 0, result, 0, result.length);
        return result;
    }

    /**
     * Inverse real FFT ({@code numpy.fft.irfft}).
     *
     * @param complexInput interleaved complex input from {@link #rfft(float[])}
     * @param n            output length
     * @return real-valued output array of length n
     */
    public static float[] irfft(final float[] complexInput, final int n) {
        float[] full = new float[n * 2];
        int halfLen = complexInput.length / 2;
        System.arraycopy(complexInput, 0, full, 0, complexInput.length);
        // Hermitian symmetry
        for (int i = halfLen; i < n; i++) {
            full[2 * i] = full[2 * (n - i)];
            full[2 * i + 1] = -full[2 * (n - i) + 1];
        }
        float[] result = ifft(full);
        float[] real = new float[n];
        for (int i = 0; i < n; i++) {
            real[i] = result[2 * i];
        }
        return real;
    }

    /**
     * DFT sample frequencies ({@code numpy.fft.fftfreq}).
     *
     * @param n window length
     * @param d sample spacing
     * @return frequency bin centers
     */
    public static double[] fftfreq(final int n, final double d) {
        double[] freq = new double[n];
        double val = 1.0 / (n * d);
        int half = (n + 1) / 2;
        for (int i = 0; i < half; i++) {
            freq[i] = i * val;
        }
        for (int i = half; i < n; i++) {
            freq[i] = (i - n) * val;
        }
        return freq;
    }

    /**
     * DFT sample frequencies for {@link #rfft(float[])} output ({@code numpy.fft.rfftfreq}).
     *
     * @param n window length
     * @param d sample spacing
     * @return frequency bin centers (length n/2+1)
     */
    public static double[] rfftfreq(final int n, final double d) {
        int len = n / 2 + 1;
        double[] freq = new double[len];
        double val = 1.0 / (n * d);
        for (int i = 0; i < len; i++) {
            freq[i] = i * val;
        }
        return freq;
    }

    /**
     * Shifts zero-frequency component to center ({@code numpy.fft.fftshift}).
     *
     * @param input frequency-domain array
     * @return shifted array
     */
    public static float[] fftshift(final float[] input) {
        int n = input.length;
        int half = n / 2;
        float[] out = new float[n];
        System.arraycopy(input, half, out, 0, n - half);
        System.arraycopy(input, 0, out, n - half, half);
        return out;
    }

    // ── NdArray wrappers ──

    /**
     * 2-D FFT ({@code numpy.fft.fft2}).
     *
     * @param a real input array of shape (rows, cols)
     * @return complex output of shape (rows, cols, 2)
     */
    public static NdArray fft2(final NdArray a) {
        int rows = a.shape(0);
        int cols = a.shape(1);
        NdArray c = a.contiguous();
        // Allocate complex output: (rows, cols, 2) for [re, im]
        float[][] reOut = new float[rows][cols];
        float[][] imOut = new float[rows][cols];

        // FFT along each row
        for (int r = 0; r < rows; r++) {
            float[] rowComplex = new float[cols * 2];
            for (int j = 0; j < cols; j++) {
                rowComplex[2 * j] = c.flatGetFloat((long) r * cols + j);
            }
            float[] rowResult = fft(rowComplex);
            for (int j = 0; j < cols; j++) {
                reOut[r][j] = rowResult[2 * j];
                imOut[r][j] = rowResult[2 * j + 1];
            }
        }

        // FFT along each column
        float[][] reResult = new float[rows][cols];
        float[][] imResult = new float[rows][cols];
        for (int j = 0; j < cols; j++) {
            float[] colComplex = new float[rows * 2];
            for (int r = 0; r < rows; r++) {
                colComplex[2 * r] = reOut[r][j];
                colComplex[2 * r + 1] = imOut[r][j];
            }
            float[] colResult = fft(colComplex);
            for (int r = 0; r < rows; r++) {
                reResult[r][j] = colResult[2 * r];
                imResult[r][j] = colResult[2 * r + 1];
            }
        }

        // Pack into (rows, cols, 2)
        float[] out = new float[rows * cols * 2];
        for (int r = 0; r < rows; r++) {
            for (int j = 0; j < cols; j++) {
                out[(r * cols + j) * 2] = reResult[r][j];
                out[(r * cols + j) * 2 + 1] = imResult[r][j];
            }
        }
        return NdArrayFactory.array(a.arena(), out, rows, cols, 2);
    }

    /**
     * 2-D inverse FFT ({@code numpy.fft.ifft2}).
     *
     * @param a complex input of shape (rows, cols, 2)
     * @return real output of shape (rows, cols)
     */
    public static NdArray ifft2(final NdArray a) {
        int rows = a.shape(0);
        int cols = a.shape(1);
        NdArray c = a.contiguous();

        // IFFT along each row
        float[][] reOut = new float[rows][cols];
        float[][] imOut = new float[rows][cols];
        for (int r = 0; r < rows; r++) {
            float[] rowComplex = new float[cols * 2];
            for (int j = 0; j < cols; j++) {
                rowComplex[2 * j] = c.flatGetFloat((long) ((long) r * cols + j) * 2);
                rowComplex[2 * j + 1] = c.flatGetFloat((long) ((long) r * cols + j) * 2 + 1);
            }
            float[] rowResult = ifft(rowComplex);
            for (int j = 0; j < cols; j++) {
                reOut[r][j] = rowResult[2 * j];
                imOut[r][j] = rowResult[2 * j + 1];
            }
        }

        // IFFT along each column
        float[] out = new float[rows * cols];
        for (int j = 0; j < cols; j++) {
            float[] colComplex = new float[rows * 2];
            for (int r = 0; r < rows; r++) {
                colComplex[2 * r] = reOut[r][j];
                colComplex[2 * r + 1] = imOut[r][j];
            }
            float[] colResult = ifft(colComplex);
            for (int r = 0; r < rows; r++) {
                out[r * cols + j] = colResult[2 * r];
            }
        }
        return NdArrayFactory.array(a.arena(), out, rows, cols);
    }

    /**
     * N-dimensional FFT ({@code numpy.fft.fftn}). Delegates to {@link #fft2(NdArray)} for 2-D input.
     *
     * @param a input array
     * @return complex FFT result
     */
    public static NdArray fftn(final NdArray a) {
        if (a.ndim() == 2) {
            return fft2(a);
        }
        if (a.ndim() == 1) {
            return fft(a);
        }
        throw new UnsupportedOperationException("fftn only supports 1D and 2D currently");
    }

    // ── Cooley-Tukey radix-2 ──

    private static float[] cooleyTukey(final float[] input, final boolean inverse) {
        int n = input.length / 2;
        float[] out = input.clone();

        bitReversalPermutation(out, n);

        butterflyPass(out, n, inverse);
        return out;
    }

    // ── Bluestein (chirp-z) for non-power-of-2 ──

    private static float[] bluestein(final float[] input, final boolean inverse) {
        int n = input.length / 2;
        // next power of 2 >= 2n-1
        int m = Integer.highestOneBit(2 * n - 1) << 1;

        double sign = inverse ? 1.0 : -1.0;

        // Chirp sequence
        float[] chirp = new float[2 * m];
        float[] a = new float[2 * m];
        float[] b = new float[2 * m];

        setupChirp(input, a, b, n, m, sign);

        float[] conv = fftConvolve(a, b, m);

        // Multiply by chirp
        float[] result = new float[2 * n];
        for (int k = 0; k < n; k++) {
            double angle = sign * Math.PI * k * k / n;
            float cRe = (float) Math.cos(angle);
            float cIm = (float) Math.sin(angle);
            float rRe = conv[2 * k];
            float rIm = conv[2 * k + 1];
            result[2 * k] = rRe * cRe + rIm * cIm;
            result[2 * k + 1] = rIm * cRe - rRe * cIm;
        }
        return result;
    }

    private static void bitReversalPermutation(final float[] out, final int n) {
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            while ((j & bit) != 0) {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if (i < j) {
                float tr = out[2 * i];
                out[2 * i] = out[2 * j];
                out[2 * j] = tr;
                tr = out[2 * i + 1];
                out[2 * i + 1] = out[2 * j + 1];
                out[2 * j + 1] = tr;
            }
        }
    }

    private static void butterflyPass(final float[] out, final int n, final boolean inverse) {
        double sign = inverse ? 1.0 : -1.0;
        for (int len = 2; len <= n; len <<= 1) {
            double angle = sign * 2.0 * Math.PI / len;
            double wRe = Math.cos(angle);
            double wIm = Math.sin(angle);
            for (int i = 0; i < n; i += len) {
                butterflyBlock(out, i, len, wRe, wIm);
            }
        }
    }

    private static void butterflyBlock(final float[] out, final int start, final int len,
                                       final double wRe, final double wIm) {
        double curRe = 1.0;
        double curIm = 0.0;
        for (int j = 0; j < len / 2; j++) {
            int u = 2 * (start + j);
            int v = 2 * (start + j + len / 2);
            double tRe = curRe * out[v] - curIm * out[v + 1];
            double tIm = curRe * out[v + 1] + curIm * out[v];
            out[v] = (float) (out[u] - tRe);
            out[v + 1] = (float) (out[u + 1] - tIm);
            out[u] = (float) (out[u] + tRe);
            out[u + 1] = (float) (out[u + 1] + tIm);
            double newRe = curRe * wRe - curIm * wIm;
            curIm = curRe * wIm + curIm * wRe;
            curRe = newRe;
        }
    }

    private static void setupChirp(final float[] input, final float[] a, final float[] b,
                                   final int n, final int m, final double sign) {
        for (int k = 0; k < n; k++) {
            double angle = sign * Math.PI * k * k / n;
            float cRe = (float) Math.cos(angle);
            float cIm = (float) Math.sin(angle);
            float xRe = input[2 * k];
            float xIm = input[2 * k + 1];
            a[2 * k] = xRe * cRe + xIm * cIm;
            a[2 * k + 1] = xIm * cRe - xRe * cIm;
            b[2 * k] = cRe;
            b[2 * k + 1] = -cIm;
        }
        for (int k = 1; k < n; k++) {
            b[2 * (m - k)] = b[2 * k];
            b[2 * (m - k) + 1] = b[2 * k + 1];
        }
    }

    private static float[] fftConvolve(final float[] a, final float[] b, final int m) {
        float[] fa = cooleyTukey(a, false);
        float[] fb = cooleyTukey(b, false);
        float[] fc = new float[2 * m];
        for (int i = 0; i < m; i++) {
            float aRe = fa[2 * i];
            float aIm = fa[2 * i + 1];
            float bRe = fb[2 * i];
            float bIm = fb[2 * i + 1];
            fc[2 * i] = aRe * bRe - aIm * bIm;
            fc[2 * i + 1] = aRe * bIm + aIm * bRe;
        }
        float[] conv = cooleyTukey(fc, true);
        for (int i = 0; i < conv.length; i++) {
            conv[i] /= m;
        }
        return conv;
    }
}
