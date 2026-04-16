///usr/bin/env java --enable-preview --source 26 "$0" "$@"; exit $?
// Benchmarks LAPACK/BLAS ops from Accelerate vs current pure-Java LinAlg.
// Run: java --enable-preview --source 26 tools/BenchLapack.java [op]
// ops: solve, inv, cholesky, svd, eigvals, qr, det, norm
// no arg = run all

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.util.Random;

public class BenchLapack {
    static final ValueLayout.OfFloat F = ValueLayout.JAVA_FLOAT;
    static final ValueLayout.OfInt I = ValueLayout.JAVA_INT;
    static final int WARMUP = 30;
    static final int ITERS = 200;
    static SymbolLookup LIB;
    static Linker LINKER;

    public static void main(String[] args) throws Throwable {
        LIB = SymbolLookup.libraryLookup(
            "/System/Library/Frameworks/Accelerate.framework/Accelerate", Arena.global());
        LINKER = Linker.nativeLinker();

        String op = args.length > 0 ? args[0] : "all";
        if (op.equals("all") || op.equals("solve"))    benchSolve();
        if (op.equals("all") || op.equals("inv"))       benchInv();
        if (op.equals("all") || op.equals("cholesky"))  benchCholesky();
        if (op.equals("all") || op.equals("svd"))       benchSvd();
        if (op.equals("all") || op.equals("eigvals"))   benchEigvals();
        if (op.equals("all") || op.equals("qr"))        benchQr();
        if (op.equals("all") || op.equals("det"))       benchDet();
        if (op.equals("all") || op.equals("norm"))      benchNorm();
    }

    // ── solve: sgesv_ ──
    static void benchSolve() throws Throwable {
        // sgesv_(n, nrhs, a, lda, ipiv, b, ldb, info) = 8 pointer args
        var h = lapack("sgesv_", 8);
        for (int n : new int[]{4, 16, 64, 256}) {
            System.out.printf("%n## solve (n=%d)%n", n);
            var rng = new Random(42);
            float[] a = randMatrix(rng, n, n);
            float[] b = randVec(rng, n);

            try (var arena = Arena.ofConfined()) {
                // LAPACK path (column-major)
                var aSeg = allocFloats(arena, toColMajor(a, n, n));
                var bSeg = allocFloats(arena, b.clone());
                var nSeg = allocInt(arena, n);
                var nhrs = allocInt(arena, 1);
                var ipiv = arena.allocate((long)n * Integer.BYTES, Integer.BYTES);
                var info = allocInt(arena, 0);
                var ldSeg = allocInt(arena, n);

                // warmup
                for (int w = 0; w < WARMUP; w++) {
                    copyFloats(aSeg, toColMajor(a, n, n));
                    copyFloats(bSeg, b);
                    h.invokeExact(nSeg, nhrs, aSeg, ldSeg, ipiv, bSeg, ldSeg, info);
                }
                // measure
                long t0 = System.nanoTime();
                for (int i = 0; i < ITERS; i++) {
                    copyFloats(aSeg, toColMajor(a, n, n));
                    copyFloats(bSeg, b);
                    h.invokeExact(nSeg, nhrs, aSeg, ldSeg, ipiv, bSeg, ldSeg, info);
                }
                long accelNs = System.nanoTime() - t0;
                float[] accelResult = readFloats(bSeg, n);

                // Scalar path
                float[] scalarResult = null;
                long t1 = System.nanoTime();
                // warmup
                for (int w = 0; w < WARMUP; w++) scalarResult = scalarSolve(a, b, n);
                long t2 = System.nanoTime();
                for (int i = 0; i < ITERS; i++) scalarResult = scalarSolve(a, b, n);
                long scalarNs = System.nanoTime() - t2;

                printResult(accelResult, scalarResult, accelNs, scalarNs);
            }
        }
    }

    static float[] scalarSolve(float[] a, float[] b, int n) {
        float[][] aug = new float[n][n + 1];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) aug[i][j] = a[i * n + j];
            aug[i][n] = b[i];
        }
        for (int col = 0; col < n; col++) {
            int maxR = col;
            for (int r = col+1; r < n; r++) if (Math.abs(aug[r][col]) > Math.abs(aug[maxR][col])) maxR = r;
            var tmp = aug[col]; aug[col] = aug[maxR]; aug[maxR] = tmp;
            for (int r = col+1; r < n; r++) {
                float f = aug[r][col] / aug[col][col];
                for (int j = col; j <= n; j++) aug[r][j] -= f * aug[col][j];
            }
        }
        float[] x = new float[n];
        for (int i = n-1; i >= 0; i--) {
            x[i] = aug[i][n];
            for (int j = i+1; j < n; j++) x[i] -= aug[i][j] * x[j];
            x[i] /= aug[i][i];
        }
        return x;
    }

    // ── inv: sgetrf_ + sgetri_ ──
    static void benchInv() throws Throwable {
        var hrf = lapack("sgetrf_", 6);
        var hri = lapack("sgetri_", 7);
        for (int n : new int[]{4, 16, 64, 256}) {
            System.out.printf("%n## inv (n=%d)%n", n);
            var rng = new Random(42);
            float[] a = randInvertible(rng, n);
            int lwork = n * 64;

            try (var arena = Arena.ofConfined()) {
                var aSeg = allocFloats(arena, toColMajor(a, n, n));
                var nSeg = allocInt(arena, n);
                var ipiv = arena.allocate((long)n * Integer.BYTES, Integer.BYTES);
                var info = allocInt(arena, 0);
                var work = arena.allocate((long)lwork * Float.BYTES, Float.BYTES);
                var lwSeg = allocInt(arena, lwork);

                for (int w = 0; w < WARMUP; w++) {
                    copyFloats(aSeg, toColMajor(a, n, n));
                    hrf.invokeExact(nSeg, nSeg, aSeg, nSeg, ipiv, info);
                    hri.invokeExact(nSeg, aSeg, nSeg, ipiv, work, lwSeg, info);
                }
                long t0 = System.nanoTime();
                for (int i = 0; i < ITERS; i++) {
                    copyFloats(aSeg, toColMajor(a, n, n));
                    hrf.invokeExact(nSeg, nSeg, aSeg, nSeg, ipiv, info);
                    hri.invokeExact(nSeg, aSeg, nSeg, ipiv, work, lwSeg, info);
                }
                long accelNs = System.nanoTime() - t0;
                float[] accelResult = fromColMajor(readFloats(aSeg, n*n), n, n);

                float[] scalarResult = null;
                for (int w = 0; w < WARMUP; w++) scalarResult = scalarInv(a, n);
                long t1 = System.nanoTime();
                for (int i = 0; i < ITERS; i++) scalarResult = scalarInv(a, n);
                long scalarNs = System.nanoTime() - t1;

                printResult(accelResult, scalarResult, accelNs, scalarNs);
            }
        }
    }

    static float[] scalarInv(float[] a, int n) {
        float[][] aug = new float[n][2*n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) aug[i][j] = a[i*n+j];
            aug[i][n+i] = 1f;
        }
        for (int col = 0; col < n; col++) {
            int maxR = col;
            for (int r = col+1; r < n; r++) if (Math.abs(aug[r][col]) > Math.abs(aug[maxR][col])) maxR = r;
            var tmp = aug[col]; aug[col] = aug[maxR]; aug[maxR] = tmp;
            float piv = aug[col][col];
            for (int j = 0; j < 2*n; j++) aug[col][j] /= piv;
            for (int r = 0; r < n; r++) if (r != col) {
                float f = aug[r][col];
                for (int j = 0; j < 2*n; j++) aug[r][j] -= f * aug[col][j];
            }
        }
        float[] res = new float[n*n];
        for (int i = 0; i < n; i++) System.arraycopy(aug[i], n, res, i*n, n);
        return res;
    }

    // ── cholesky: spotrf_ ──
    static void benchCholesky() throws Throwable {
        var h = lapack("spotrf_", 5);
        for (int n : new int[]{4, 16, 64, 256}) {
            System.out.printf("%n## cholesky (n=%d)%n", n);
            float[] a = randSPD(new Random(42), n);

            try (var arena = Arena.ofConfined()) {
                var uplo = arena.allocateFrom("L");
                var aSeg = allocFloats(arena, toColMajor(a, n, n));
                var nSeg = allocInt(arena, n);
                var info = allocInt(arena, 0);

                for (int w = 0; w < WARMUP; w++) {
                    copyFloats(aSeg, toColMajor(a, n, n));
                    h.invokeExact(uplo, nSeg, aSeg, nSeg, info);
                }
                long t0 = System.nanoTime();
                for (int i = 0; i < ITERS; i++) {
                    copyFloats(aSeg, toColMajor(a, n, n));
                    h.invokeExact(uplo, nSeg, aSeg, nSeg, info);
                }
                long accelNs = System.nanoTime() - t0;
                // Extract lower triangle
                float[] cm = readFloats(aSeg, n*n);
                float[] accelResult = extractLowerColMajor(cm, n);

                float[] scalarResult = null;
                for (int w = 0; w < WARMUP; w++) scalarResult = scalarCholesky(a, n);
                long t1 = System.nanoTime();
                for (int i = 0; i < ITERS; i++) scalarResult = scalarCholesky(a, n);
                long scalarNs = System.nanoTime() - t1;

                printResult(accelResult, scalarResult, accelNs, scalarNs);
            }
        }
    }

    static float[] scalarCholesky(float[] a, int n) {
        float[][] L = new float[n][n];
        for (int i = 0; i < n; i++) for (int j = 0; j <= i; j++) {
            float sum = 0;
            for (int k = 0; k < j; k++) sum += L[i][k] * L[j][k];
            L[i][j] = (i == j) ? (float)Math.sqrt(a[i*n+j] - sum) : (a[i*n+j] - sum) / L[j][j];
        }
        float[] res = new float[n*n];
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) res[i*n+j] = L[i][j];
        return res;
    }

    static float[] extractLowerColMajor(float[] cm, int n) {
        float[] res = new float[n*n];
        for (int i = 0; i < n; i++) for (int j = 0; j <= i; j++)
            res[i*n+j] = cm[j*n+i]; // col-major to row-major lower
        return res;
    }

    // ── det: sgetrf_ ──
    static void benchDet() throws Throwable {
        var hrf = lapack("sgetrf_", 6);
        for (int n : new int[]{4, 16, 64, 256}) {
            System.out.printf("%n## det (n=%d)%n", n);
            var rng = new Random(42);
            float[] a = randMatrix(rng, n, n);

            try (var arena = Arena.ofConfined()) {
                var aSeg = allocFloats(arena, toColMajor(a, n, n));
                var nSeg = allocInt(arena, n);
                var ipiv = arena.allocate((long)n * Integer.BYTES, Integer.BYTES);
                var info = allocInt(arena, 0);

                float accelDet = 0;
                for (int w = 0; w < WARMUP; w++) {
                    copyFloats(aSeg, toColMajor(a, n, n));
                    hrf.invokeExact(nSeg, nSeg, aSeg, nSeg, ipiv, info);
                }
                long t0 = System.nanoTime();
                for (int i = 0; i < ITERS; i++) {
                    copyFloats(aSeg, toColMajor(a, n, n));
                    hrf.invokeExact(nSeg, nSeg, aSeg, nSeg, ipiv, info);
                }
                long accelNs = System.nanoTime() - t0;
                // compute det from LU diagonal
                accelDet = 1f;
                float[] lu = readFloats(aSeg, n*n);
                int swaps = 0;
                for (int i = 0; i < n; i++) {
                    accelDet *= lu[i*n+i]; // col-major diagonal: [i + i*n]
                    if (ipiv.getAtIndex(I, i) != i+1) swaps++;
                }
                if (swaps % 2 != 0) accelDet = -accelDet;

                float scalarDet = 0;
                for (int w = 0; w < WARMUP; w++) scalarDet = scalarDet(a, n);
                long t1 = System.nanoTime();
                for (int i = 0; i < ITERS; i++) scalarDet = scalarDet(a, n);
                long scalarNs = System.nanoTime() - t1;

                printResultScalar(accelDet, scalarDet, accelNs, scalarNs);
            }
        }
    }

    static float scalarDet(float[] a, int n) {
        float[][] lu = new float[n][n];
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) lu[i][j] = a[i*n+j];
        float det = 1f;
        for (int col = 0; col < n; col++) {
            int maxR = col;
            for (int r = col+1; r < n; r++) if (Math.abs(lu[r][col]) > Math.abs(lu[maxR][col])) maxR = r;
            if (maxR != col) { var t = lu[col]; lu[col] = lu[maxR]; lu[maxR] = t; det = -det; }
            if (Math.abs(lu[col][col]) < 1e-12f) return 0f;
            det *= lu[col][col];
            for (int r = col+1; r < n; r++) {
                float f = lu[r][col] / lu[col][col];
                for (int j = col+1; j < n; j++) lu[r][j] -= f * lu[col][j];
            }
        }
        return det;
    }

    // ── norm: cblas_snrm2 ──
    static void benchNorm() throws Throwable {
        var h = LINKER.downcallHandle(LIB.find("cblas_snrm2").orElseThrow(),
            FunctionDescriptor.of(ValueLayout.JAVA_FLOAT, I, ValueLayout.ADDRESS, I));
        for (int n : new int[]{64, 1024, 65536}) {
            System.out.printf("%n## norm (n=%d)%n", n);
            float[] v = randVec(new Random(42), n);

            try (var arena = Arena.ofConfined()) {
                var seg = allocFloats(arena, v);
                float accelVal = 0;
                for (int w = 0; w < WARMUP; w++) accelVal = (float) h.invokeExact(n, seg, 1);
                long t0 = System.nanoTime();
                for (int i = 0; i < ITERS; i++) accelVal = (float) h.invokeExact(n, seg, 1);
                long accelNs = System.nanoTime() - t0;

                float scalarVal = 0;
                for (int w = 0; w < WARMUP; w++) scalarVal = scalarNorm(v);
                long t1 = System.nanoTime();
                for (int i = 0; i < ITERS; i++) scalarVal = scalarNorm(v);
                long scalarNs = System.nanoTime() - t1;

                printResultScalar(accelVal, scalarVal, accelNs, scalarNs);
            }
        }
    }

    static float scalarNorm(float[] v) {
        float sum = 0;
        for (float x : v) sum += x * x;
        return (float) Math.sqrt(sum);
    }

    // ── Stubs for ops we'll fill in next ──
    static void benchSvd() throws Throwable {
        var h = LINKER.downcallHandle(LIB.find("sgesvd_").orElseThrow(),
            FunctionDescriptor.ofVoid(
                // jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        for (int n : new int[]{4, 16, 64}) {
            System.out.printf("%n## svd (m=%d, n=%d)%n", n, n);
            var rng = new Random(42);
            float[] a = randMatrix(rng, n, n);

            try (var arena = Arena.ofConfined()) {
                // LAPACK SVD
                var jobu = arena.allocateFrom("S");
                var jobvt = arena.allocateFrom("S");
                var mSeg = allocInt(arena, n);
                var nSeg = allocInt(arena, n);
                var aSeg = allocFloats(arena, toColMajor(a, n, n));
                var ldaSeg = allocInt(arena, n);
                var sSeg = arena.allocate((long) n * Float.BYTES, Float.BYTES);
                var uSeg = arena.allocate((long) n * n * Float.BYTES, Float.BYTES);
                var lduSeg = allocInt(arena, n);
                var vtSeg = arena.allocate((long) n * n * Float.BYTES, Float.BYTES);
                var ldvtSeg = allocInt(arena, n);
                int lwork = n * 64;
                var workSeg = arena.allocate((long) lwork * Float.BYTES, Float.BYTES);
                var lwSeg = allocInt(arena, lwork);
                var info = allocInt(arena, 0);

                // warmup
                for (int w = 0; w < WARMUP; w++) {
                    copyFloats(aSeg, toColMajor(a, n, n));
                    h.invokeExact(jobu, jobvt, mSeg, nSeg, aSeg, ldaSeg, sSeg, uSeg, lduSeg, vtSeg, ldvtSeg, workSeg, lwSeg, info);
                }
                long t0 = System.nanoTime();
                for (int i = 0; i < ITERS; i++) {
                    copyFloats(aSeg, toColMajor(a, n, n));
                    h.invokeExact(jobu, jobvt, mSeg, nSeg, aSeg, ldaSeg, sSeg, uSeg, lduSeg, vtSeg, ldvtSeg, workSeg, lwSeg, info);
                }
                long accelNs = System.nanoTime() - t0;
                float[] accelS = readFloats(sSeg, n);

                // Scalar SVD (Jacobi)
                float[] scalarS = null;
                for (int w = 0; w < WARMUP; w++) scalarS = scalarSvdSingularValues(a, n);
                long t1 = System.nanoTime();
                for (int i = 0; i < ITERS; i++) scalarS = scalarSvdSingularValues(a, n);
                long scalarNs = System.nanoTime() - t1;

                // Compare singular values (sorted descending)
                java.util.Arrays.sort(accelS);
                java.util.Arrays.sort(scalarS);
                // reverse both
                for (int i = 0; i < n / 2; i++) {
                    float tmp = accelS[i]; accelS[i] = accelS[n-1-i]; accelS[n-1-i] = tmp;
                    tmp = scalarS[i]; scalarS[i] = scalarS[n-1-i]; scalarS[n-1-i] = tmp;
                }
                printResult(accelS, scalarS, accelNs, scalarNs);
            }
        }
    }

    static float[] scalarSvdSingularValues(float[] a, int n) {
        // Compute A^T*A
        float[][] atA = new float[n][n];
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int k = 0; k < n; k++) sum += a[k * n + i] * a[k * n + j];
            atA[i][j] = sum;
        }
        // Jacobi eigenvalue
        float[][] v = new float[n][n];
        for (int i = 0; i < n; i++) v[i][i] = 1f;
        for (int iter = 0; iter < 100; iter++) {
            float off = 0;
            for (int i = 0; i < n; i++) for (int j = i+1; j < n; j++) off += atA[i][j] * atA[i][j];
            if (off < 1e-12f) break;
            for (int p = 0; p < n; p++) for (int q = p+1; q < n; q++) {
                if (Math.abs(atA[p][q]) < 1e-12f) continue;
                float tau = (atA[q][q] - atA[p][p]) / (2 * atA[p][q]);
                float t = (float)(Math.signum(tau) / (Math.abs(tau) + Math.sqrt(1 + tau*tau)));
                float c = (float)(1.0 / Math.sqrt(1 + t*t));
                float s = t * c;
                // rotate
                for (int i = 0; i < n; i++) {
                    float sp = atA[i][p], sq = atA[i][q];
                    atA[i][p] = c*sp - s*sq; atA[i][q] = s*sp + c*sq;
                }
                for (int i = 0; i < n; i++) {
                    float sp = atA[p][i], sq = atA[q][i];
                    atA[p][i] = c*sp - s*sq; atA[q][i] = s*sp + c*sq;
                }
            }
        }
        float[] singVals = new float[n];
        for (int i = 0; i < n; i++) singVals[i] = (float) Math.sqrt(Math.max(0, atA[i][i]));
        java.util.Arrays.sort(singVals);
        for (int i = 0; i < n/2; i++) { float t = singVals[i]; singVals[i] = singVals[n-1-i]; singVals[n-1-i] = t; }
        return singVals;
    }

    static void benchEigvals() throws Throwable {
        // ssyev_(jobz, uplo, n, a, lda, w, work, lwork, info) = 9 ptr args
        var h = lapack("ssyev_", 9);
        for (int n : new int[]{4, 16, 64}) {
            System.out.printf("%n## eigvals (n=%d, symmetric)%n", n);
            var rng = new Random(42);
            float[] a = randSymmetric(rng, n);

            try (var arena = Arena.ofConfined()) {
                var jobz = arena.allocateFrom("V"); // compute eigenvalues + vectors
                var uplo = arena.allocateFrom("U");
                var nSeg = allocInt(arena, n);
                var aSeg = allocFloats(arena, toColMajor(a, n, n));
                var ldaSeg = allocInt(arena, n);
                var wSeg = arena.allocate((long) n * Float.BYTES, Float.BYTES);
                int lwork = n * 64;
                var workSeg = arena.allocate((long) lwork * Float.BYTES, Float.BYTES);
                var lwSeg = allocInt(arena, lwork);
                var info = allocInt(arena, 0);

                for (int w = 0; w < WARMUP; w++) {
                    copyFloats(aSeg, toColMajor(a, n, n));
                    h.invokeExact(jobz, uplo, nSeg, aSeg, ldaSeg, wSeg, workSeg, lwSeg, info);
                }
                long t0 = System.nanoTime();
                for (int i = 0; i < ITERS; i++) {
                    copyFloats(aSeg, toColMajor(a, n, n));
                    h.invokeExact(jobz, uplo, nSeg, aSeg, ldaSeg, wSeg, workSeg, lwSeg, info);
                }
                long accelNs = System.nanoTime() - t0;
                float[] accelEig = readFloats(wSeg, n);
                java.util.Arrays.sort(accelEig);

                float[] scalarEig = null;
                for (int w = 0; w < WARMUP; w++) scalarEig = scalarEigvals(a, n);
                long t1 = System.nanoTime();
                for (int i = 0; i < ITERS; i++) scalarEig = scalarEigvals(a, n);
                long scalarNs = System.nanoTime() - t1;
                java.util.Arrays.sort(scalarEig);

                printResult(accelEig, scalarEig, accelNs, scalarNs);
            }
        }
    }

    static float[] randSymmetric(Random rng, int n) {
        float[] a = new float[n * n];
        for (int i = 0; i < n; i++) for (int j = i; j < n; j++) {
            float v = rng.nextFloat() * 2 - 1;
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
        for (int i = 0; i < n; i++) a[i * n + i] += n; // diag dominance
        return a;
    }

    static float[] scalarEigvals(float[] a, int n) {
        // QR iteration (same as LinAlg.eigvals fallback)
        float[][] h = new float[n][n];
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) h[i][j] = a[i * n + j];
        for (int iter = 0; iter < 200; iter++) {
            // QR decomposition
            float[][] q = new float[n][n];
            float[][] r = new float[n][n];
            for (int i = 0; i < n; i++) q[i][i] = 1f;
            for (int j = 0; j < n; j++) {
                float[] col = new float[n];
                for (int i = 0; i < n; i++) {
                    col[i] = h[i][j];
                    for (int k = 0; k < j; k++) col[i] -= q[i][k] * r[k][j];
                }
                float norm = 0;
                for (float x : col) norm += x * x;
                norm = (float) Math.sqrt(norm);
                r[j][j] = norm;
                if (norm > 1e-12f) for (int i = 0; i < n; i++) q[i][j] = col[i] / norm;
                for (int k = j + 1; k < n; k++) {
                    float dot = 0;
                    for (int i = 0; i < n; i++) dot += q[i][j] * h[i][k];
                    r[j][k] = dot;
                }
            }
            // h = R * Q
            float[][] next = new float[n][n];
            for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
                float sum = 0;
                for (int k = 0; k < n; k++) sum += r[i][k] * q[k][j];
                next[i][j] = sum;
            }
            h = next;
        }
        float[] eig = new float[n];
        for (int i = 0; i < n; i++) eig[i] = h[i][i];
        return eig;
    }

    static void benchQr() throws Throwable {
        // sgeqrf_(m, n, a, lda, tau, work, lwork, info) = 8 ptr args
        // sorgqr_(m, n, k, a, lda, tau, work, lwork, info) = 9 ptr args
        var hqrf = lapack("sgeqrf_", 8);
        var horgqr = lapack("sorgqr_", 9);
        for (int n : new int[]{4, 16, 64}) {
            System.out.printf("%n## qr (n=%d)%n", n);
            var rng = new Random(42);
            float[] a = randMatrix(rng, n, n);

            try (var arena = Arena.ofConfined()) {
                var mSeg = allocInt(arena, n);
                var nSeg = allocInt(arena, n);
                var aSeg = allocFloats(arena, toColMajor(a, n, n));
                var ldaSeg = allocInt(arena, n);
                var tau = arena.allocate((long) n * Float.BYTES, Float.BYTES);
                int lwork = n * 64;
                var workSeg = arena.allocate((long) lwork * Float.BYTES, Float.BYTES);
                var lwSeg = allocInt(arena, lwork);
                var info = allocInt(arena, 0);

                // Extract R from upper triangle, then Q via sorgqr
                for (int w = 0; w < WARMUP; w++) {
                    copyFloats(aSeg, toColMajor(a, n, n));
                    hqrf.invokeExact(mSeg, nSeg, aSeg, ldaSeg, tau, workSeg, lwSeg, info);
                    horgqr.invokeExact(mSeg, nSeg, nSeg, aSeg, ldaSeg, tau, workSeg, lwSeg, info);
                }
                long t0 = System.nanoTime();
                for (int i = 0; i < ITERS; i++) {
                    copyFloats(aSeg, toColMajor(a, n, n));
                    hqrf.invokeExact(mSeg, nSeg, aSeg, ldaSeg, tau, workSeg, lwSeg, info);
                }
                long accelNs = System.nanoTime() - t0;
                // Read R from upper triangle of factored A
                float[] factored = readFloats(aSeg, n * n);
                float[] accelR = new float[n * n];
                for (int i = 0; i < n; i++) for (int j = i; j < n; j++)
                    accelR[i * n + j] = factored[j * n + i]; // col-major upper → row-major

                // Scalar QR (Gram-Schmidt)
                float[] scalarR = null;
                for (int w = 0; w < WARMUP; w++) scalarR = scalarQrR(a, n);
                long t1 = System.nanoTime();
                for (int i = 0; i < ITERS; i++) scalarR = scalarQrR(a, n);
                long scalarNs = System.nanoTime() - t1;

                printResult(accelR, scalarR, accelNs, scalarNs);
            }
        }
    }

    static float[] scalarQrR(float[] a, int n) {
        float[][] mat = new float[n][n];
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) mat[i][j] = a[i * n + j];
        float[][] q = new float[n][n];
        float[][] r = new float[n][n];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) q[i][j] = mat[i][j];
            for (int k = 0; k < j; k++) {
                float dot = 0;
                for (int i = 0; i < n; i++) dot += q[i][k] * mat[i][j];
                r[k][j] = dot;
                for (int i = 0; i < n; i++) q[i][j] -= dot * q[i][k];
            }
            float norm = 0;
            for (int i = 0; i < n; i++) norm += q[i][j] * q[i][j];
            norm = (float) Math.sqrt(norm);
            r[j][j] = norm;
            if (norm > 1e-12f) for (int i = 0; i < n; i++) q[i][j] /= norm;
        }
        float[] rFlat = new float[n * n];
        for (int i = 0; i < n; i++) System.arraycopy(r[i], 0, rFlat, i * n, n);
        return rFlat;
    }

    // ── Helpers ──
    static MethodHandle lapack(String name, int nArgs) {
        var addr = LIB.find(name).orElseThrow();
        var ptrLayouts = new MemoryLayout[nArgs];
        for (int i = 0; i < nArgs; i++) ptrLayouts[i] = ValueLayout.ADDRESS;
        return LINKER.downcallHandle(addr, FunctionDescriptor.ofVoid(ptrLayouts));
    }

    static float[] randMatrix(Random rng, int m, int n) {
        float[] a = new float[m * n];
        for (int i = 0; i < a.length; i++) a[i] = rng.nextFloat() * 2 - 1;
        return a;
    }

    static float[] randVec(Random rng, int n) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) v[i] = rng.nextFloat() * 2 - 1;
        return v;
    }

    static float[] randInvertible(Random rng, int n) {
        float[] a = randMatrix(rng, n, n);
        for (int i = 0; i < n; i++) a[i * n + i] += n; // diag dominance
        return a;
    }

    static float[] randSPD(Random rng, int n) {
        float[] a = randMatrix(rng, n, n);
        float[] spd = new float[n * n];
        // A^T A + nI
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int k = 0; k < n; k++) sum += a[k * n + i] * a[k * n + j];
            spd[i * n + j] = sum + (i == j ? n : 0);
        }
        return spd;
    }

    static float[] toColMajor(float[] rowMajor, int m, int n) {
        float[] cm = new float[m * n];
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) cm[j * m + i] = rowMajor[i * n + j];
        return cm;
    }

    static float[] fromColMajor(float[] cm, int m, int n) {
        float[] rm = new float[m * n];
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) rm[i * n + j] = cm[j * m + i];
        return rm;
    }

    static MemorySegment allocFloats(Arena arena, float[] data) {
        var seg = arena.allocate((long) data.length * Float.BYTES, Float.BYTES);
        MemorySegment.copy(data, 0, seg, F, 0, data.length);
        return seg;
    }

    static void copyFloats(MemorySegment seg, float[] data) {
        MemorySegment.copy(data, 0, seg, F, 0, data.length);
    }

    static float[] readFloats(MemorySegment seg, int n) {
        float[] out = new float[n];
        MemorySegment.copy(seg, F, 0, out, 0, n);
        return out;
    }

    static MemorySegment allocInt(Arena arena, int val) {
        var seg = arena.allocate(Integer.BYTES, Integer.BYTES);
        seg.set(I, 0, val);
        return seg;
    }

    static void printResult(float[] accel, float[] scalar, long accelNs, long scalarNs) {
        float maxDiff = 0;
        for (int i = 0; i < accel.length; i++) {
            float d = Math.abs(accel[i] - scalar[i]);
            if (d > maxDiff) maxDiff = d;
        }
        double accelUs = accelNs / 1000.0 / ITERS;
        double scalarUs = scalarNs / 1000.0 / ITERS;
        double speedup = (double) scalarNs / accelNs;
        System.out.printf("  Parity:  max|diff| = %.9f%n", maxDiff);
        System.out.printf("  Accel:   %.1f µs/iter   Scalar: %.1f µs/iter%n", accelUs, scalarUs);
        System.out.printf("  Speedup: %.2fx%n", speedup);
    }

    static void printResultScalar(float accel, float scalar, long accelNs, long scalarNs) {
        float diff = Math.abs(accel - scalar);
        float relDiff = Math.abs(scalar) > 1e-10f ? diff / Math.abs(scalar) : diff;
        double accelUs = accelNs / 1000.0 / ITERS;
        double scalarUs = scalarNs / 1000.0 / ITERS;
        double speedup = (double) scalarNs / accelNs;
        System.out.printf("  Values:  accel=%.6f  scalar=%.6f  |diff|=%.9f  rel=%.6f%n", accel, scalar, diff, relDiff);
        System.out.printf("  Accel:   %.1f µs/iter   Scalar: %.1f µs/iter%n", accelUs, scalarUs);
        System.out.printf("  Speedup: %.2fx%n", speedup);
    }
}
