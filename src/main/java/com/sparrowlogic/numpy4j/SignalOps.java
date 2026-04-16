package com.sparrowlogic.numpy4j;

/**
 * Signal processing ops: convolve, correlate, interp.
     */
public final class SignalOps {
    private SignalOps() {
    }

    /**
     * 1D convolution (full mode).
     *
     * @param a input array
     * @param b second array
     * @return result array
     */
    public static NdArray convolve(final NdArray a, final NdArray b) {
        float[] fa = a.toFloatArray();
        float[] fb = b.toFloatArray();
        int na = fa.length;
        int nb = fb.length;
        int nc = na + nb - 1;
        float[] out = new float[nc];
        for (int i = 0; i < na; i++) {
            for (int j = 0; j < nb; j++) {
                out[i + j] += fa[i] * fb[j];
            }
        }
        return NdArrayFactory.array(a.arena(), out);
    }

    /**
     * 1D cross-correlation (full mode).
     *
     * @param a input array
     * @param b second array
     * @return result array
     */
    public static NdArray correlate(final NdArray a, final NdArray b) {
        float[] fa = a.toFloatArray();
        float[] fb = b.toFloatArray();
        int na = fa.length;
        int nb = fb.length;
        int nc = na + nb - 1;
        float[] out = new float[nc];
        for (int i = 0; i < nc; i++) {
            float sum = 0;
            for (int j = 0; j < nb; j++) {
                int ai = i - (nb - 1) + j;
                if (ai >= 0 && ai < na) {
                    sum += fa[ai] * fb[j];
                }
            }
            out[i] = sum;
        }
        return NdArrayFactory.array(a.arena(), out);
    }

    /**
     * 1D linear interpolation.
     *
     * @param x x-coordinate vector
     * @param xp xp
     * @param fp fp
     * @return the computed value
     */
    public static float interp(final float x, final float[] xp, final float[] fp) {
        if (x <= xp[0]) {
            return fp[0];
        }
        return x >= xp[xp.length - 1] ? fp[fp.length - 1] : interpolateSegment(x, xp, fp);
    }

    private static float interpolateSegment(final float x, final float[] xp, final float[] fp) {
        for (int i = 0; i < xp.length - 1; i++) {
            if (x >= xp[i] && x <= xp[i + 1]) {
                float t = (x - xp[i]) / (xp[i + 1] - xp[i]);
                return fp[i] + t * (fp[i + 1] - fp[i]);
            }
        }
        return fp[fp.length - 1];
    }
}
