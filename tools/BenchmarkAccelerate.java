///usr/bin/env java --enable-preview --source 26 "$0" "$@"; exit $?
// Run: java --enable-preview --source 26 tools/BenchmarkAccelerate.java
//
// Benchmarks each Accelerate-optimized op against its scalar fallback.
// Reports: parity (max abs diff), speedup ratio, and wall-clock times.

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.util.Arrays;

/**
 * Self-contained benchmark — no project dependencies.
 * Reimplements both scalar and Accelerate paths inline to measure the delta.
 */
public class BenchmarkAccelerate {

    static final ValueLayout.OfFloat F = ValueLayout.JAVA_FLOAT;
    static final int WARMUP = 50;
    static final int ITERS = 200;

    // ── Accelerate handles ──
    static MethodHandle SGEMM, VVEXPF, MAXV, VSADD, SVE, VSDIV, SVDIV, VSMUL, VMUL;
    static boolean HAS_ACCEL;

    static {
        try {
            var lib = SymbolLookup.libraryLookup(
                    "/System/Library/Frameworks/Accelerate.framework/Accelerate", Arena.global());
            var l = Linker.nativeLinker();
            var A = ValueLayout.ADDRESS;
            var I = ValueLayout.JAVA_INT;
            var FL = ValueLayout.JAVA_FLOAT;
            var J = ValueLayout.JAVA_LONG;
            SGEMM = l.downcallHandle(lib.find("cblas_sgemm").orElseThrow(),
                    FunctionDescriptor.ofVoid(I, I, I, I, I, I, FL, A, I, A, I, FL, A, I));
            VVEXPF = l.downcallHandle(lib.find("vvexpf").orElseThrow(),
                    FunctionDescriptor.ofVoid(A, A, A));
            MAXV = l.downcallHandle(lib.find("vDSP_maxv").orElseThrow(),
                    FunctionDescriptor.ofVoid(A, J, A, J));
            VSADD = l.downcallHandle(lib.find("vDSP_vsadd").orElseThrow(),
                    FunctionDescriptor.ofVoid(A, J, A, A, J, J));
            SVE = l.downcallHandle(lib.find("vDSP_sve").orElseThrow(),
                    FunctionDescriptor.ofVoid(A, J, A, J));
            VSDIV = l.downcallHandle(lib.find("vDSP_vsdiv").orElseThrow(),
                    FunctionDescriptor.ofVoid(A, J, A, A, J, J));
            SVDIV = l.downcallHandle(lib.find("vDSP_svdiv").orElseThrow(),
                    FunctionDescriptor.ofVoid(A, A, J, A, J, J));
            VSMUL = l.downcallHandle(lib.find("vDSP_vsmul").orElseThrow(),
                    FunctionDescriptor.ofVoid(A, J, A, A, J, J));
            VMUL = l.downcallHandle(lib.find("vDSP_vmul").orElseThrow(),
                    FunctionDescriptor.ofVoid(A, J, A, J, A, J, J));
            HAS_ACCEL = true;
        } catch (Throwable t) {
            System.err.println("Accelerate not available: " + t.getMessage());
            HAS_ACCEL = false;
        }
    }

    public static void main(String[] args) throws Throwable {
        if (!HAS_ACCEL) {
            System.out.println("Accelerate not available — nothing to benchmark.");
            return;
        }
        System.out.println("=== Accelerate vs Scalar Benchmark ===\n");

        benchSgemm();
        benchSoftmax();
        benchExp();
        benchVmul();
        benchVsmul();
    }

    // ── sgemm: batched matmul ──

    static void benchSgemm() throws Throwable {
        int batch = 8, m = 64, k = 64, n = 64;
        System.out.printf("## batchedMatmul (%d x [%d,%d]@[%d,%d])%n", batch, m, k, k, n);

        try (var arena = Arena.ofConfined()) {
            var a = allocRandom(arena, batch * m * k);
            var b = allocRandom(arena, batch * k * n);
            var cAccel = arena.allocate((long) batch * m * n * 4, 4);
            var cScalar = arena.allocate((long) batch * m * n * 4, 4);

            // Warmup + run: Accelerate
            for (int w = 0; w < WARMUP; w++) sgemmBatched(a, b, cAccel, batch, m, n, k);
            long t0 = System.nanoTime();
            for (int i = 0; i < ITERS; i++) sgemmBatched(a, b, cAccel, batch, m, n, k);
            long accelNs = System.nanoTime() - t0;

            // Warmup + run: Scalar
            for (int w = 0; w < WARMUP; w++) scalarMatmulBatched(a, b, cScalar, batch, m, n, k);
            t0 = System.nanoTime();
            for (int i = 0; i < ITERS; i++) scalarMatmulBatched(a, b, cScalar, batch, m, n, k);
            long scalarNs = System.nanoTime() - t0;

            printResult(cAccel, cScalar, batch * m * n, accelNs, scalarNs);
        }
    }

    static void sgemmBatched(MemorySegment a, MemorySegment b, MemorySegment c,
                             int batch, int m, int n, int k) throws Throwable {
        long as = (long) m * k * 4, bs = (long) k * n * 4, cs = (long) m * n * 4;
        for (int bi = 0; bi < batch; bi++) {
            SGEMM.invokeExact(101, 111, 111, m, n, k, 1.0f,
                    a.asSlice(bi * as, as), k, b.asSlice(bi * bs, bs), n,
                    0.0f, c.asSlice(bi * cs, cs), n);
        }
    }

    static void scalarMatmulBatched(MemorySegment a, MemorySegment b, MemorySegment c,
                                    int batch, int m, int n, int k) {
        for (int bi = 0; bi < batch; bi++) {
            long aBase = (long) bi * m * k;
            long bBase = (long) bi * k * n;
            long cBase = (long) bi * m * n;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    float sum = 0;
                    for (int ki = 0; ki < k; ki++) {
                        sum += a.getAtIndex(F, aBase + (long) i * k + ki)
                                * b.getAtIndex(F, bBase + (long) ki * n + j);
                    }
                    c.setAtIndex(F, cBase + (long) i * n + j, sum);
                }
            }
        }
    }

    // ── softmax ──

    static void benchSoftmax() throws Throwable {
        int rows = 128, cols = 512;
        System.out.printf("%n## softmax (%d x %d)%n", rows, cols);

        try (var arena = Arena.ofConfined()) {
            var src = allocRandom(arena, rows * cols);
            var accel = arena.allocate((long) rows * cols * 4, 4);
            var scalar = arena.allocate((long) rows * cols * 4, 4);

            // Warmup + run: Accelerate
            for (int w = 0; w < WARMUP; w++) {
                MemorySegment.copy(src, 0, accel, 0, (long) rows * cols * 4);
                accelSoftmax(accel, rows, cols);
            }
            long t0 = System.nanoTime();
            for (int i = 0; i < ITERS; i++) {
                MemorySegment.copy(src, 0, accel, 0, (long) rows * cols * 4);
                accelSoftmax(accel, rows, cols);
            }
            long accelNs = System.nanoTime() - t0;

            // Warmup + run: Scalar
            for (int w = 0; w < WARMUP; w++) {
                MemorySegment.copy(src, 0, scalar, 0, (long) rows * cols * 4);
                scalarSoftmax(scalar, rows, cols);
            }
            t0 = System.nanoTime();
            for (int i = 0; i < ITERS; i++) {
                MemorySegment.copy(src, 0, scalar, 0, (long) rows * cols * 4);
                scalarSoftmax(scalar, rows, cols);
            }
            long scalarNs = System.nanoTime() - t0;

            printResult(accel, scalar, rows * cols, accelNs, scalarNs);
        }
    }

    static void accelSoftmax(MemorySegment seg, int rows, int cols) throws Throwable {
        long rowBytes = (long) cols * 4;
        try (var ar = Arena.ofConfined()) {
            var maxB = ar.allocate(4, 4);
            var negB = ar.allocate(4, 4);
            var sumB = ar.allocate(4, 4);
            var cntB = ar.allocate(4, 4);
            cntB.set(ValueLayout.JAVA_INT, 0, cols);
            for (int r = 0; r < rows; r++) {
                var row = seg.asSlice((long) r * rowBytes, rowBytes);
                MAXV.invokeExact(row, 1L, maxB, (long) cols);
                negB.set(F, 0, -maxB.get(F, 0));
                VSADD.invokeExact(row, 1L, negB, row, 1L, (long) cols);
                VVEXPF.invokeExact(row, row, cntB);
                SVE.invokeExact(row, 1L, sumB, (long) cols);
                VSDIV.invokeExact(row, 1L, sumB, row, 1L, (long) cols);
            }
        }
    }

    static void scalarSoftmax(MemorySegment seg, int rows, int cols) {
        for (int r = 0; r < rows; r++) {
            long base = (long) r * cols;
            float max = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < cols; j++) max = Math.max(max, seg.getAtIndex(F, base + j));
            float sum = 0;
            for (int j = 0; j < cols; j++) {
                float e = (float) Math.exp(seg.getAtIndex(F, base + j) - max);
                seg.setAtIndex(F, base + j, e);
                sum += e;
            }
            float inv = 1.0f / sum;
            for (int j = 0; j < cols; j++) seg.setAtIndex(F, base + j, seg.getAtIndex(F, base + j) * inv);
        }
    }

    // ── exp (vvexpf) ──

    static void benchExp() throws Throwable {
        int n = 65536;
        System.out.printf("%n## exp (%d elements)%n", n);

        try (var arena = Arena.ofConfined()) {
            var src = allocRandom(arena, n);
            // clamp to [-10, 10] to avoid inf
            for (int i = 0; i < n; i++) src.setAtIndex(F, i, src.getAtIndex(F, i) * 10f);
            var accel = arena.allocate((long) n * 4, 4);
            var scalar = arena.allocate((long) n * 4, 4);

            // Warmup + run: Accelerate
            var cnt = arena.allocate(4, 4);
            cnt.set(ValueLayout.JAVA_INT, 0, n);
            for (int w = 0; w < WARMUP; w++) VVEXPF.invokeExact(accel, src, cnt);
            long t0 = System.nanoTime();
            for (int i = 0; i < ITERS; i++) VVEXPF.invokeExact(accel, src, cnt);
            long accelNs = System.nanoTime() - t0;

            // Warmup + run: Scalar
            for (int w = 0; w < WARMUP; w++) scalarExp(src, scalar, n);
            t0 = System.nanoTime();
            for (int i = 0; i < ITERS; i++) scalarExp(src, scalar, n);
            long scalarNs = System.nanoTime() - t0;

            printResult(accel, scalar, n, accelNs, scalarNs);
        }
    }

    static void scalarExp(MemorySegment src, MemorySegment dst, int n) {
        for (int i = 0; i < n; i++) dst.setAtIndex(F, i, (float) Math.exp(src.getAtIndex(F, i)));
    }

    // ── vmul (element-wise multiply) ──

    static void benchVmul() throws Throwable {
        int n = 65536;
        System.out.printf("%n## multiply (%d elements)%n", n);

        try (var arena = Arena.ofConfined()) {
            var a = allocRandom(arena, n);
            var b = allocRandom(arena, n);
            var accel = arena.allocate((long) n * 4, 4);
            var scalar = arena.allocate((long) n * 4, 4);

            for (int w = 0; w < WARMUP; w++) VMUL.invokeExact(a, 1L, b, 1L, accel, 1L, (long) n);
            long t0 = System.nanoTime();
            for (int i = 0; i < ITERS; i++) VMUL.invokeExact(a, 1L, b, 1L, accel, 1L, (long) n);
            long accelNs = System.nanoTime() - t0;

            for (int w = 0; w < WARMUP; w++) scalarMul(a, b, scalar, n);
            t0 = System.nanoTime();
            for (int i = 0; i < ITERS; i++) scalarMul(a, b, scalar, n);
            long scalarNs = System.nanoTime() - t0;

            printResult(accel, scalar, n, accelNs, scalarNs);
        }
    }

    static void scalarMul(MemorySegment a, MemorySegment b, MemorySegment dst, int n) {
        for (int i = 0; i < n; i++) dst.setAtIndex(F, i, a.getAtIndex(F, i) * b.getAtIndex(F, i));
    }

    // ── vsmul (scalar multiply) ──

    static void benchVsmul() throws Throwable {
        int n = 65536;
        System.out.printf("%n## mulScalar (%d elements)%n", n);

        try (var arena = Arena.ofConfined()) {
            var src = allocRandom(arena, n);
            var accel = arena.allocate((long) n * 4, 4);
            var scalar = arena.allocate((long) n * 4, 4);
            var sBuf = arena.allocate(4, 4);
            sBuf.set(F, 0, 2.5f);

            for (int w = 0; w < WARMUP; w++) VSMUL.invokeExact(src, 1L, sBuf, accel, 1L, (long) n);
            long t0 = System.nanoTime();
            for (int i = 0; i < ITERS; i++) VSMUL.invokeExact(src, 1L, sBuf, accel, 1L, (long) n);
            long accelNs = System.nanoTime() - t0;

            for (int w = 0; w < WARMUP; w++) scalarMulS(src, scalar, 2.5f, n);
            t0 = System.nanoTime();
            for (int i = 0; i < ITERS; i++) scalarMulS(src, scalar, 2.5f, n);
            long scalarNs = System.nanoTime() - t0;

            printResult(accel, scalar, n, accelNs, scalarNs);
        }
    }

    static void scalarMulS(MemorySegment src, MemorySegment dst, float s, int n) {
        for (int i = 0; i < n; i++) dst.setAtIndex(F, i, src.getAtIndex(F, i) * s);
    }

    // ── Helpers ──

    static MemorySegment allocRandom(Arena arena, int n) {
        var seg = arena.allocate((long) n * 4, 4);
        var rng = new java.util.Random(42);
        for (int i = 0; i < n; i++) seg.setAtIndex(F, i, rng.nextFloat() * 2 - 1);
        return seg;
    }

    static void printResult(MemorySegment accel, MemorySegment scalar, int n,
                            long accelNs, long scalarNs) {
        float maxDiff = 0;
        for (int i = 0; i < n; i++) {
            float diff = Math.abs(accel.getAtIndex(F, i) - scalar.getAtIndex(F, i));
            if (diff > maxDiff) maxDiff = diff;
        }
        double accelMs = accelNs / 1_000_000.0;
        double scalarMs = scalarNs / 1_000_000.0;
        double speedup = (double) scalarNs / accelNs;
        System.out.printf("  Parity:     max |diff| = %.9f%n", maxDiff);
        System.out.printf("  Accelerate: %.3f ms (%d iters)%n", accelMs, ITERS);
        System.out.printf("  Scalar:     %.3f ms (%d iters)%n", scalarMs, ITERS);
        System.out.printf("  Speedup:    %.2fx%n", speedup);
        System.out.printf("  Per-iter:   %.3f µs (accel) vs %.3f µs (scalar)%n",
                accelMs / ITERS * 1000, scalarMs / ITERS * 1000);
    }
}
