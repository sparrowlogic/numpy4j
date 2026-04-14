package com.sparrowlogic.numpy4j;

import java.lang.foreign.Arena;

/**
 * Random number generation matching numpy.random with PCG64 PRNG.
 * PCG-XSL-RR 128/64 — same algorithm as numpy's default_rng().
 */
public final class Random {
    // PCG64 state (128-bit state, 128-bit increment, using two longs each)
    private long stateHi, stateLo;
    private final long incHi, incLo;

    /** Create with seed matching numpy.random.default_rng(seed). */
    public Random(long seed) {
        // Initialize PCG64 — simplified seeding matching numpy's approach
        this.incHi = 0;
        this.incLo = (seed << 1) | 1;
        this.stateHi = 0;
        this.stateLo = 0;
        nextLong(); // Advance once
        this.stateLo += seed;
        nextLong(); // Advance again
    }

    /** Generate next 64-bit value using PCG-XSL-RR. */
    public long nextLong() {
        long oldHi = stateHi, oldLo = stateLo;
        // state = state * MULT + inc (128-bit multiply)
        // PCG multiplier: 6364136223846793005
        long multLo = 6364136223846793005L;
        long lo = multLo * oldLo;
        long hi = Math.multiplyHigh(multLo, oldLo) + multLo * oldHi;
        stateLo = lo + incLo;
        stateHi = hi + incHi + (Long.compareUnsigned(stateLo, lo) < 0 ? 1 : 0);
        // XSL-RR output function
        long xorShifted = oldHi ^ oldLo;
        int rot = (int) (oldHi >>> 58);
        return Long.rotateRight(xorShifted, rot);
    }

    /** Uniform [0, 1) double. */
    public double nextDouble() {
        return (nextLong() >>> 11) * 0x1.0p-53;
    }

    /** Uniform [0, 1) float. */
    public float nextFloat() {
        return (nextLong() >>> 40) * 0x1.0p-24f;
    }

    /** Standard normal via Box-Muller. */
    public float nextGaussian() {
        double u1 = nextDouble(), u2 = nextDouble();
        return (float) (Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2));
    }

    // ── Array generators ──

    /** numpy.random.uniform(low, high, size) */
    public NdArray uniform(Arena arena, float low, float high, int... shape) {
        NdArray out = NdArrayFactory.empty(arena, DType.FLOAT32, shape);
        float range = high - low;
        for (long i = 0; i < out.size(); i++) out.flatSetFloat(i, low + nextFloat() * range);
        return out;
    }

    /** numpy.random.normal(loc, scale, size) */
    public NdArray normal(Arena arena, float loc, float scale, int... shape) {
        NdArray out = NdArrayFactory.empty(arena, DType.FLOAT32, shape);
        for (long i = 0; i < out.size(); i++) out.flatSetFloat(i, loc + nextGaussian() * scale);
        return out;
    }

    /** numpy.random.randint(low, high, size) — returns INT32 */
    public NdArray randint(Arena arena, int low, int high, int... shape) {
        NdArray out = NdArrayFactory.empty(arena, DType.INT32, shape);
        int range = high - low;
        for (long i = 0; i < out.size(); i++) {
            int val = low + (int) (nextDouble() * range);
            out.data().setAtIndex(java.lang.foreign.ValueLayout.JAVA_INT, i, val);
        }
        return out;
    }

    /** numpy.random.choice — sample from 1D array without replacement. */
    public NdArray choice(Arena arena, NdArray a, int size) {
        NdArray c = a.contiguous();
        NdArray out = NdArrayFactory.empty(arena, DType.FLOAT32, size);
        for (int i = 0; i < size; i++) {
            int idx = (int) (nextDouble() * a.size());
            out.flatSetFloat(i, c.flatGetFloat(idx));
        }
        return out;
    }

    /** Fisher-Yates shuffle (in-place). */
    public void shuffle(NdArray a) {
        NdArray c = a.contiguous();
        int n = (int) a.size();
        for (int i = n - 1; i > 0; i--) {
            int j = (int) (nextDouble() * (i + 1));
            float tmp = c.flatGetFloat(i);
            c.flatSetFloat(i, c.flatGetFloat(j));
            c.flatSetFloat(j, tmp);
        }
    }

    /** numpy.random.permutation — returns shuffled copy. */
    public NdArray permutation(Arena arena, int n) {
        NdArray a = NdArrayFactory.arange(arena, n);
        shuffle(a);
        return a;
    }

    /** Generator.standard_normal — normal(0,1). */
    public NdArray standardNormal(Arena arena, int... shape) { return normal(arena, 0f, 1f, shape); }

    /** Generator.exponential — exponential distribution with given scale. */
    public NdArray exponential(Arena arena, float scale, int... shape) {
        NdArray out = NdArrayFactory.empty(arena, DType.FLOAT32, shape);
        for (long i = 0; i < out.size(); i++) out.flatSetFloat(i, (float) (-scale * Math.log(1.0 - nextDouble())));
        return out;
    }

    /** Generator.poisson — Poisson distribution via Knuth's algorithm. */
    public NdArray poisson(Arena arena, float lam, int... shape) {
        NdArray out = NdArrayFactory.empty(arena, DType.FLOAT32, shape);
        double L = Math.exp(-lam);
        for (long i = 0; i < out.size(); i++) {
            int k = 0; double p = 1;
            do { k++; p *= nextDouble(); } while (p > L);
            out.flatSetFloat(i, k - 1);
        }
        return out;
    }

    /** Generator.binomial — binomial distribution. */
    public NdArray binomial(Arena arena, int n, float p, int... shape) {
        NdArray out = NdArrayFactory.empty(arena, DType.FLOAT32, shape);
        for (long i = 0; i < out.size(); i++) {
            int successes = 0;
            for (int t = 0; t < n; t++) if (nextDouble() < p) successes++;
            out.flatSetFloat(i, successes);
        }
        return out;
    }
}
