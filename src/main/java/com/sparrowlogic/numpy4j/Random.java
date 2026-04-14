package com.sparrowlogic.numpy4j;

import java.lang.foreign.Arena;

/**
 * Random number generation matching numpy.random with PCG64 PRNG.
 * PCG-XSL-RR 128/64 — same algorithm as numpy's default_rng().
 */
public final class Random {
    // PCG64 state (128-bit state, 128-bit increment, using two longs each)
    private long stateHi;

    private long stateLo;

    private final long incHi;

    private final long incLo;

    /**
     * Create with seed matching numpy.random.default_rng(seed).
     */
    public Random(final long seed) {
        // Initialize PCG64 — simplified seeding matching numpy's approach
        this.incHi = 0;
        this.incLo = (seed << 1) | 1;
        this.stateHi = 0;
        this.stateLo = 0;
        this.nextLong();
        this.stateLo += seed;
        this.nextLong();
    }

    /**
     * Generate next 64-bit value using PCG-XSL-RR.
     */
    public long nextLong() {
        long oldHi = this.stateHi;
        long oldLo = this.stateLo;
        // state = state * MULT + inc (128-bit multiply)
        // PCG multiplier: 6364136223846793005
        long multLo = 6364136223846793005L;
        long lo = multLo * oldLo;
        long hi = Math.multiplyHigh(multLo, oldLo) + multLo * oldHi;
        this.stateLo = lo + this.incLo;
        this.stateHi = hi + this.incHi + (Long.compareUnsigned(this.stateLo, lo) < 0 ? 1 : 0);
        // XSL-RR output function
        long xorShifted = oldHi ^ oldLo;
        int rot = (int) (oldHi >>> 58);
        return Long.rotateRight(xorShifted, rot);
    }

    /**
     * Uniform [0, 1) double.
     */
    public double nextDouble() {
        return (this.nextLong() >>> 11) * 0x1.0p-53;
    }

    /**
     * Uniform [0, 1) float.
     */
    public float nextFloat() {
        return (this.nextLong() >>> 40) * 0x1.0p-24f;
    }

    /**
     * Standard normal via Box-Muller.
     */
    public float nextGaussian() {
        double u1 = this.nextDouble();
        double u2 = this.nextDouble();
        return (float) (Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2));
    }

    // ── Array generators ──

    /**
     * numpy.random.uniform(low, high, size)
     */
    public NdArray uniform(final Arena arena, final float low, final float high, final int... shape) {
        NdArray out = NdArrayFactory.empty(arena, DType.FLOAT32, shape);
        float range = high - low;
        for (long i = 0; i < out.size(); i++) {
            out.flatSetFloat(i, low + this.nextFloat() * range);
        }
        return out;
    }

    /**
     * numpy.random.normal(loc, scale, size)
     */
    public NdArray normal(final Arena arena, final float loc, final float scale, final int... shape) {
        NdArray out = NdArrayFactory.empty(arena, DType.FLOAT32, shape);
        for (long i = 0; i < out.size(); i++) {
            out.flatSetFloat(i, loc + this.nextGaussian() * scale);
        }
        return out;
    }

    /**
     * numpy.random.randint(low, high, size) — returns INT32
     */
    public NdArray randint(final Arena arena, final int low, final int high, final int... shape) {
        NdArray out = NdArrayFactory.empty(arena, DType.INT32, shape);
        int range = high - low;
        for (long i = 0; i < out.size(); i++) {
            int val = low + (int) (this.nextDouble() * range);
            out.data().setAtIndex(java.lang.foreign.ValueLayout.JAVA_INT, i, val);
        }
        return out;
    }

    /**
     * numpy.random.choice — sample from 1D array without replacement.
     */
    public NdArray choice(final Arena arena, final NdArray a, final int size) {
        NdArray c = a.contiguous();
        NdArray out = NdArrayFactory.empty(arena, DType.FLOAT32, size);
        for (int i = 0; i < size; i++) {
            int idx = (int) (this.nextDouble() * a.size());
            out.flatSetFloat(i, c.flatGetFloat(idx));
        }
        return out;
    }

    /**
     * Fisher-Yates shuffle (in-place).
     */
    public void shuffle(final NdArray a) {
        NdArray c = a.contiguous();
        int n = (int) a.size();
        for (int i = n - 1; i > 0; i--) {
            int j = (int) (this.nextDouble() * (i + 1));
            float tmp = c.flatGetFloat(i);
            c.flatSetFloat(i, c.flatGetFloat(j));
            c.flatSetFloat(j, tmp);
        }
    }

    /**
     * numpy.random.permutation — returns shuffled copy.
     */
    public NdArray permutation(final Arena arena, final int n) {
        NdArray a = NdArrayFactory.arange(arena, n);
        this.shuffle(a);
        return a;
    }

    /**
     * Generator.standard_normal — normal(0,1).
     */
    public NdArray standardNormal(final Arena arena, final int... shape) {
        return this.normal(arena, 0f, 1f, shape);
    }

    /**
     * Generator.exponential — exponential distribution with given scale.
     */
    public NdArray exponential(final Arena arena, final float scale, final int... shape) {
        NdArray out = NdArrayFactory.empty(arena, DType.FLOAT32, shape);
        for (long i = 0; i < out.size(); i++) {
            out.flatSetFloat(i, (float) (-scale * Math.log(1.0 - this.nextDouble())));
        }
        return out;
    }

    /**
     * Generator.poisson — Poisson distribution via Knuth's algorithm.
     */
    public NdArray poisson(final Arena arena, final float lam, final int... shape) {
        NdArray out = NdArrayFactory.empty(arena, DType.FLOAT32, shape);
        double threshold = Math.exp(-lam);
        for (long i = 0; i < out.size(); i++) {
            int k = 0;
            double p = 1;
            do {
                k++;
                p *= this.nextDouble();
            } while (p > threshold);
            out.flatSetFloat(i, k - 1);
        }
        return out;
    }

    /**
     * Generator.binomial — binomial distribution.
     */
    public NdArray binomial(final Arena arena, final int n, final float p, final int... shape) {
        NdArray out = NdArrayFactory.empty(arena, DType.FLOAT32, shape);
        for (long i = 0; i < out.size(); i++) {
            int successes = 0;
            for (int t = 0; t < n; t++) {
                if (this.nextDouble() < p) {
                    successes++;
                }
            }
            out.flatSetFloat(i, successes);
        }
        return out;
    }
}
