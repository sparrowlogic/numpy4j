package com.sparrowlogic.numpy4j;

import java.lang.foreign.Arena;

/**
 * numpy.linalg equivalent. Uses Accelerate.framework LAPACK via FFM on macOS,
 * with pure Java fallbacks for cross-platform.
 */
public final class LinAlg {

    private static final int LAPACK_THRESHOLD = 2;
    private static final String LAPACK_INFO_SUFFIX = ")";

    private LinAlg() {
    }

    // ── Norms ──

    /**
     * Frobenius / L2 norm ({@code numpy.linalg.norm}).
     *
     * @param a input vector or matrix
     * @return the L2 (Frobenius) norm
     */
    public static float norm(final NdArray a) {
        NdArray c = a.contiguous();
        float sum = 0;
        for (long i = 0; i < a.size(); i++) {
            float v = c.flatGetFloat(i);
            sum += v * v;
        }
        return (float) Math.sqrt(sum);
    }

    /**
     * Vector p-norm ({@code numpy.linalg.norm} with ord).
     *
     * @param a   input vector
     * @param ord norm order (1, 2, inf, -inf, 0)
     * @return the p-norm
     */
    public static float norm(final NdArray a, final float ord) {
        if (Float.isInfinite(ord)) {
            return ord > 0 ? Reductions.max(Ufunc.abs(a)) : Reductions.min(Ufunc.abs(a));
        }
        return computeNorm(a, ord);
    }

    private static float computeNorm(final NdArray a, final float ord) {
        NdArray c = a.contiguous();
        if (ord == 0) {
            int count = 0;
            for (long i = 0; i < a.size(); i++) {
                if (c.flatGetFloat(i) != 0f) {
                    count++;
                }
            }
            return count;
        }
        double sum = 0;
        for (long i = 0; i < a.size(); i++) {
            sum += Math.pow(Math.abs(c.flatGetFloat(i)), ord);
        }
        return (float) Math.pow(sum, 1.0 / ord);
    }

    // ── Matrix operations (pure Java) ──

    /**
     * Matrix inverse via Gauss-Jordan elimination ({@code numpy.linalg.inv}).
     *
     * @param a square matrix
     * @return the inverse matrix
     * @throws ArithmeticException if the matrix is singular
     */
    public static NdArray inv(final NdArray a) {
        int n = a.shape(0);
        if (AccelerateOps.isAvailable() && n >= LAPACK_THRESHOLD) {
            return invLapack(a, n);
        }
        return invScalar(a, n);
    }

    private static NdArray invLapack(final NdArray a, final int n) {
        NdArray c = a.contiguous();
        float[] aData = new float[n * n];
        for (int i = 0; i < n * n; i++) {
            aData[i] = c.flatGetFloat(i);
        }
        float[] result = new float[n * n];
        try (var arena = Arena.ofConfined()) {
            int info = AccelerateOps.inv(n, aData, result, arena);
            if (info != 0) {
                throw new ArithmeticException("Singular matrix (LAPACK info=" + info + LAPACK_INFO_SUFFIX);
            }
        }
        return NdArrayFactory.array(a.arena(), result, n, n);
    }

    private static NdArray invScalar(final NdArray a, final int n) {
        float[][] aug = new float[n][2 * n];
        NdArray c = a.contiguous();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                aug[i][j] = c.flatGetFloat((long) i * n + j);
            }
            aug[i][n + i] = 1f;
        }
        for (int col = 0; col < n; col++) {
            int maxRow = findPivotRow(aug, col, n);
            float[] tmp = aug[col];
            aug[col] = aug[maxRow];
            aug[maxRow] = tmp;
            float pivot = aug[col][col];
            if (Math.abs(pivot) < 1e-12f) {
                throw new ArithmeticException("Singular matrix");
            }
            scaleRow(aug[col], pivot, 2 * n);
            eliminateColumn(aug, col, n, 2 * n);
        }
        float[] result = new float[n * n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(aug[i], n, result, i * n, n);
        }
        return NdArrayFactory.array(a.arena(), result, n, n);
    }

    private static int findPivotRow(final float[][] mat, final int col, final int n) {
        int maxRow = col;
        for (int row = col + 1; row < n; row++) {
            if (Math.abs(mat[row][col]) > Math.abs(mat[maxRow][col])) {
                maxRow = row;
            }
        }
        return maxRow;
    }

    private static void scaleRow(final float[] row, final float pivot, final int len) {
        for (int j = 0; j < len; j++) {
            row[j] /= pivot;
        }
    }

    private static void eliminateColumn(final float[][] aug, final int col, final int n, final int width) {
        for (int row = 0; row < n; row++) {
            if (row != col) {
                float factor = aug[row][col];
                for (int j = 0; j < width; j++) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }

    /**
     * Determinant via LU decomposition ({@code numpy.linalg.det}).
     *
     * @param a square matrix
     * @return the determinant
     */
    public static float det(final NdArray a) {
        int n = a.shape(0);
        if (AccelerateOps.isAvailable() && n >= LAPACK_THRESHOLD) {
            return detLapack(a, n);
        }
        return detScalar(a, n);
    }

    private static float detLapack(final NdArray a, final int n) {
        NdArray c = a.contiguous();
        float[] aData = new float[n * n];
        for (int i = 0; i < n * n; i++) {
            aData[i] = c.flatGetFloat(i);
        }
        try (var arena = Arena.ofConfined()) {
            return AccelerateOps.det(n, aData, arena);
        }
    }

    private static float detScalar(final NdArray a, final int n) {
        float[][] lu = toMatrix(a);
        float det = 1f;
        for (int col = 0; col < n; col++) {
            int maxRow = findPivotRow(lu, col, n);
            if (maxRow != col) {
                float[] t = lu[col];
                lu[col] = lu[maxRow];
                lu[maxRow] = t;
                det = -det;
            }
            if (Math.abs(lu[col][col]) < 1e-12f) {
                return 0f;
            }
            det *= lu[col][col];
            eliminateLU(lu, col, n);
        }
        return det;
    }

    private static void eliminateLU(final float[][] lu, final int col, final int n) {
        for (int row = col + 1; row < n; row++) {
            float factor = lu[row][col] / lu[col][col];
            for (int j = col + 1; j < n; j++) {
                lu[row][j] -= factor * lu[col][j];
            }
        }
    }

    /**
     * Solves the linear system Ax = b ({@code numpy.linalg.solve}).
     *
     * @param a coefficient matrix (n × n)
     * @param b right-hand side vector (n)
     * @return solution vector x
     */
    public static NdArray solve(final NdArray a, final NdArray b) {
        int n = a.shape(0);
        if (AccelerateOps.isAvailable() && n >= LAPACK_THRESHOLD) {
            return solveLapack(a, b, n);
        }
        return solveScalar(a, b, n);
    }

    private static NdArray solveLapack(final NdArray a, final NdArray b, final int n) {
        NdArray ca = a.contiguous();
        NdArray cb = b.contiguous();
        float[] aData = new float[n * n];
        float[] bData = new float[n];
        for (int i = 0; i < n * n; i++) {
            aData[i] = ca.flatGetFloat(i);
        }
        for (int i = 0; i < n; i++) {
            bData[i] = cb.flatGetFloat(i);
        }
        try (var arena = Arena.ofConfined()) {
            int info = AccelerateOps.solve(n, aData, bData, arena);
            if (info != 0) {
                throw new ArithmeticException("Singular matrix (LAPACK info=" + info + LAPACK_INFO_SUFFIX);
            }
        }
        return NdArrayFactory.array(a.arena(), bData);
    }

    private static NdArray solveScalar(final NdArray a, final NdArray b, final int n) {
        float[][] aug = new float[n][n + 1];
        NdArray ca = a.contiguous();
        NdArray cb = b.contiguous();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                aug[i][j] = ca.flatGetFloat((long) i * n + j);
            }
            aug[i][n] = cb.flatGetFloat(i);
        }
        forwardElimination(aug, n);
        float[] x = backSubstitution(aug, n);
        return NdArrayFactory.array(a.arena(), x);
    }

    private static void forwardElimination(final float[][] aug, final int n) {
        for (int col = 0; col < n; col++) {
            int maxRow = findPivotRow(aug, col, n);
            float[] t = aug[col];
            aug[col] = aug[maxRow];
            aug[maxRow] = t;
            for (int row = col + 1; row < n; row++) {
                eliminateRow(aug, row, col, n);
            }
        }
    }

    private static void eliminateRow(final float[][] aug, final int row, final int col, final int n) {
        float f = aug[row][col] / aug[col][col];
        for (int j = col; j <= n; j++) {
            aug[row][j] -= f * aug[col][j];
        }
    }

    private static float[] backSubstitution(final float[][] aug, final int n) {
        float[] x = new float[n];
        for (int i = n - 1; i >= 0; i--) {
            x[i] = aug[i][n];
            for (int j = i + 1; j < n; j++) {
                x[i] -= aug[i][j] * x[j];
            }
            x[i] /= aug[i][i];
        }
        return x;
    }

    /**
     * Cholesky decomposition ({@code numpy.linalg.cholesky}).
     *
     * @param a symmetric positive-definite matrix
     * @return lower triangular L where A = L @ L^T
     */
    public static NdArray cholesky(final NdArray a) {
        int n = a.shape(0);
        if (AccelerateOps.isAvailable() && n >= LAPACK_THRESHOLD) {
            return choleskyLapack(a, n);
        }
        return choleskyScalar(a, n);
    }

    private static NdArray choleskyLapack(final NdArray a, final int n) {
        NdArray c = a.contiguous();
        float[] aData = new float[n * n];
        for (int i = 0; i < n * n; i++) {
            aData[i] = c.flatGetFloat(i);
        }
        float[] result = new float[n * n];
        try (var arena = Arena.ofConfined()) {
            int info = AccelerateOps.cholesky(n, aData, result, arena);
            if (info != 0) {
                throw new ArithmeticException("Not positive definite (LAPACK info=" + info + LAPACK_INFO_SUFFIX);
            }
        }
        return NdArrayFactory.array(a.arena(), result, n, n);
    }

    private static NdArray choleskyScalar(final NdArray a, final int n) {
        float[][] lower = new float[n][n];
        NdArray c = a.contiguous();
        for (int i = 0; i < n; i++) {
            choleskyRow(lower, c, i, n);
        }
        return fromMatrix(a.arena(), lower, n, n);
    }

    private static void choleskyRow(final float[][] lower, final NdArray c, final int i, final int n) {
        for (int j = 0; j <= i; j++) {
            float sum = 0;
            for (int k = 0; k < j; k++) {
                sum += lower[i][k] * lower[j][k];
            }
            if (i == j) {
                lower[i][j] = (float) Math.sqrt(c.flatGetFloat((long) i * n + j) - sum);
            } else {
                lower[i][j] = (c.flatGetFloat((long) i * n + j) - sum) / lower[j][j];
            }
        }
    }

    /**
     * QR decomposition via Gram-Schmidt ({@code numpy.linalg.qr}).
     *
     * @param a input matrix (m × n)
     * @return array {Q, R} where Q is orthogonal and R is upper triangular
     */
    public static NdArray[] qr(final NdArray a) {
        int m = a.shape(0);
        int n = a.shape(1);
        float[][] mat = toMatrix(a);
        float[][] qMat = new float[m][n];
        float[][] rMat = new float[n][n];

        for (int j = 0; j < n; j++) {
            qrColumn(mat, qMat, rMat, j, m);
        }
        return new NdArray[]{fromMatrix(a.arena(), qMat, m, n), fromMatrix(a.arena(), rMat, n, n)};
    }

    private static void qrColumn(final float[][] mat, final float[][] qMat, final float[][] rMat,
                                  final int j, final int m) {
        for (int i = 0; i < m; i++) {
            qMat[i][j] = mat[i][j];
        }
        for (int k = 0; k < j; k++) {
            float dot = dotColumn(qMat, mat, k, j, m);
            rMat[k][j] = dot;
            for (int i = 0; i < m; i++) {
                qMat[i][j] -= dot * qMat[i][k];
            }
        }
        float norm = columnNorm(qMat, j, m);
        rMat[j][j] = norm;
        if (norm > 1e-12f) {
            for (int i = 0; i < m; i++) {
                qMat[i][j] /= norm;
            }
        }
    }

    private static float dotColumn(final float[][] a, final float[][] b, final int colA, final int colB,
                                   final int rows) {
        float dot = 0;
        for (int i = 0; i < rows; i++) {
            dot += a[i][colA] * b[i][colB];
        }
        return dot;
    }

    private static float columnNorm(final float[][] mat, final int col, final int rows) {
        float norm = 0;
        for (int i = 0; i < rows; i++) {
            norm += mat[i][col] * mat[i][col];
        }
        return (float) Math.sqrt(norm);
    }

    /**
     * Matrix power: A^n ({@code numpy.linalg.matrix_power}).
     *
     * @param a square matrix
     * @param n exponent (0 returns identity, negative uses inverse)
     * @return A raised to the nth power
     */
    public static NdArray matrixPower(final NdArray a, final int n) {
        if (n == 0) {
            return NdArrayFactory.eye(a.arena(), a.shape(0));
        }
        NdArray base = n < 0 ? inv(a) : a;
        int exp = n < 0 ? -n : n;
        NdArray result = NdArrayFactory.eye(a.arena(), a.shape(0));
        while (exp > 0) {
            if ((exp & 1) == 1) {
                result = MatMul.matmul(result, base);
            }
            base = MatMul.matmul(base, base);
            exp >>= 1;
        }
        return result;
    }

    // ── Helpers ──

    private static boolean isSymmetric(final NdArray a, final int n) {
        NdArray c = a.contiguous();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (c.flatGetFloat((long) i * n + j) != c.flatGetFloat((long) j * n + i)) {
                    return false;
                }
            }
        }
        return true;
    }

    private static float[][] toMatrix(final NdArray a) {
        NdArray c = a.contiguous();
        int m = a.shape(0);
        int n = a.shape(1);
        float[][] mat = new float[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                mat[i][j] = c.flatGetFloat((long) i * n + j);
            }
        }
        return mat;
    }

    private static NdArray fromMatrix(final Arena arena, final float[][] mat, final int m, final int n) {
        float[] flat = new float[m * n];
        for (int i = 0; i < m; i++) {
            System.arraycopy(mat[i], 0, flat, i * n, n);
        }
        return NdArrayFactory.array(arena, flat, m, n);
    }

    private static float[][] matMul2d(final float[][] a, final float[][] b, final int m, final int n, final int k) {
        float[][] result = new float[m][n];
        for (int i = 0; i < m; i++) {
            matMulRow(a, b, result, i, n, k);
        }
        return result;
    }

    private static void matMulRow(final float[][] a, final float[][] b, final float[][] result,
                                  final int i, final int n, final int k) {
        for (int j = 0; j < n; j++) {
            for (int ki = 0; ki < k; ki++) {
                result[i][j] += a[i][ki] * b[ki][j];
            }
        }
    }

    private static float[][] computeAtA(final float[][] mat, final int m, final int n) {
        float[][] atA = new float[n][n];
        for (int i = 0; i < n; i++) {
            computeAtARow(mat, atA, i, m, n);
        }
        return atA;
    }

    private static void computeAtARow(final float[][] mat, final float[][] atA, final int i,
                                      final int m, final int n) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < m; k++) {
                atA[i][j] += mat[k][i] * mat[k][j];
            }
        }
    }

    /**
     * Singular Value Decomposition ({@code numpy.linalg.svd}).
     *
     * @param a input matrix (m × n)
     * @return array {U, S, Vt} where A ≈ U @ diag(S) @ Vt
     */
    public static NdArray[] svd(final NdArray a) {
        int m = a.shape(0);
        int n = a.shape(1);
        if (AccelerateOps.isAvailable() && Math.min(m, n) >= LAPACK_THRESHOLD) {
            return svdLapack(a, m, n);
        }
        return svdScalar(a, m, n);
    }

    private static NdArray[] svdLapack(final NdArray a, final int m, final int n) {
        NdArray c = a.contiguous();
        float[] aData = new float[m * n];
        for (int i = 0; i < m * n; i++) {
            aData[i] = c.flatGetFloat(i);
        }
        int mn = Math.min(m, n);
        float[] s = new float[mn];
        float[] u = new float[m * mn];
        float[] vt = new float[mn * n];
        try (var arena = Arena.ofConfined()) {
            int info = AccelerateOps.svd(m, n, aData, s, u, vt, arena);
            if (info != 0) {
                throw new ArithmeticException("SVD failed (LAPACK info=" + info + LAPACK_INFO_SUFFIX);
            }
        }
        return new NdArray[]{
                NdArrayFactory.array(a.arena(), u, m, mn),
                NdArrayFactory.array(a.arena(), s),
                NdArrayFactory.array(a.arena(), vt, mn, n)
        };
    }

    private static NdArray[] svdScalar(final NdArray a, final int m, final int n) {
        float[][] mat = toMatrix(a);
        float[][] atA = computeAtA(mat, m, n);

        float[][] vMat = new float[n][n];
        for (int i = 0; i < n; i++) {
            vMat[i][i] = 1f;
        }
        float[][] sMat = new float[n][n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(atA[i], 0, sMat[i], 0, n);
        }

        jacobiSweep(sMat, vMat, n);

        float[] singVals = new float[n];
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            singVals[i] = (float) Math.sqrt(Math.max(0, sMat[i][i]));
            order[i] = i;
        }
        java.util.Arrays.sort(order, (x, y) -> Float.compare(singVals[y], singVals[x]));

        float[] sortedS = new float[n];
        float[][] sortedV = new float[n][n];
        for (int i = 0; i < n; i++) {
            sortedS[i] = singVals[order[i]];
            copyColumn(vMat, sortedV, order[i], i, n);
        }

        float[][] uMat = computeU(mat, sortedV, sortedS, m, n);

        return new NdArray[]{
                fromMatrix(a.arena(), uMat, m, n),
                NdArrayFactory.array(a.arena(), sortedS),
                fromMatrix(a.arena(), transposeMatrix(sortedV, n, n), n, n)
        };
    }

    private static void jacobiSweep(final float[][] sMat, final float[][] vMat, final int n) {
        for (int iter = 0; iter < 100; iter++) {
            float offDiag = offDiagonalSum(sMat, n);
            if (offDiag < 1e-12f) {
                break;
            }
            for (int p = 0; p < n; p++) {
                jacobiSweepRow(sMat, vMat, p, n);
            }
        }
    }

    private static void jacobiSweepRow(final float[][] sMat, final float[][] vMat, final int p, final int n) {
        for (int q = p + 1; q < n; q++) {
            if (Math.abs(sMat[p][q]) < 1e-12f) {
                continue;
            }
            float tau = (sMat[q][q] - sMat[p][p]) / (2 * sMat[p][q]);
            float t = (float) (Math.signum(tau) / (Math.abs(tau) + Math.sqrt(1 + tau * tau)));
            float c = (float) (1.0 / Math.sqrt(1 + t * t));
            float s = t * c;
            jacobiRotate(sMat, vMat, p, q, c, s, n);
        }
    }

    private static float offDiagonalSum(final float[][] mat, final int n) {
        float sum = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                sum += mat[i][j] * mat[i][j];
            }
        }
        return sum;
    }

    private static void copyColumn(final float[][] src, final float[][] dst, final int srcCol,
                                   final int dstCol, final int rows) {
        for (int j = 0; j < rows; j++) {
            dst[j][dstCol] = src[j][srcCol];
        }
    }

    private static float[][] computeU(final float[][] mat, final float[][] sortedV, final float[] sortedS,
                                      final int m, final int n) {
        float[][] uMat = new float[m][n];
        for (int i = 0; i < m; i++) {
            computeURow(mat, sortedV, sortedS, uMat, i, n);
        }
        return uMat;
    }

    private static void computeURow(final float[][] mat, final float[][] sortedV, final float[] sortedS,
                                    final float[][] uMat, final int i, final int n) {
        for (int j = 0; j < n; j++) {
            if (sortedS[j] > 1e-10f) {
                float sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += mat[i][k] * sortedV[k][j];
                }
                uMat[i][j] = sum / sortedS[j];
            }
        }
    }

    private static void jacobiRotate(
            final float[][] sMat,
            final float[][] vMat,
            final int p,
            final int q,
            final float c,
            final float s,
            final int n
    ) {
        for (int i = 0; i < n; i++) {
            float sp = sMat[i][p];
            float sq = sMat[i][q];
            sMat[i][p] = c * sp - s * sq;
            sMat[i][q] = s * sp + c * sq;
        }
        for (int i = 0; i < n; i++) {
            float sp = sMat[p][i];
            float sq = sMat[q][i];
            sMat[p][i] = c * sp - s * sq;
            sMat[q][i] = s * sp + c * sq;
        }
        for (int i = 0; i < n; i++) {
            float vp = vMat[i][p];
            float vq = vMat[i][q];
            vMat[i][p] = c * vp - s * vq;
            vMat[i][q] = s * vp + c * vq;
        }
    }

    private static float[][] transposeMatrix(final float[][] m, final int rows, final int cols) {
        float[][] t = new float[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                t[j][i] = m[i][j];
            }
        }
        return t;
    }

    /**
     * Eigenvalues of a square matrix ({@code numpy.linalg.eigvals}).
     *
     * @param a square matrix
     * @return 1-D array of eigenvalues (real parts only)
     */
    public static NdArray eigvals(final NdArray a) {
        int n = a.shape(0);
        if (AccelerateOps.isAvailable() && isSymmetric(a, n)) {
            return eigvalsLapack(a, n);
        }
        return eigvalsScalar(a, n);
    }

    private static NdArray eigvalsLapack(final NdArray a, final int n) {
        NdArray c = a.contiguous();
        float[] aData = new float[n * n];
        for (int i = 0; i < n * n; i++) {
            aData[i] = c.flatGetFloat(i);
        }
        float[] eigenvalues = new float[n];
        try (var arena = Arena.ofConfined()) {
            AccelerateOps.eigvals(n, aData, eigenvalues, null, arena);
        }
        return NdArrayFactory.array(a.arena(), eigenvalues);
    }

    private static NdArray eigvalsScalar(final NdArray a, final int n) {
        float[][] hMat = toMatrix(a);
        for (int iter = 0; iter < 200; iter++) {
            hMat = qrIterationStep(hMat, n);
        }
        float[] eigenvalues = new float[n];
        for (int i = 0; i < n; i++) {
            eigenvalues[i] = hMat[i][i];
        }
        return NdArrayFactory.array(a.arena(), eigenvalues);
    }

    private static float[][] qrIterationStep(final float[][] hMat, final int n) {
        float[][] qMat = new float[n][n];
        float[][] rMat = new float[n][n];
        for (int i = 0; i < n; i++) {
            qMat[i][i] = 1f;
        }
        for (int j = 0; j < n; j++) {
            float[] col = buildQRColumn(hMat, qMat, rMat, j, n);
            float norm = vectorNorm(col);
            rMat[j][j] = norm;
            if (norm > 1e-12f) {
                for (int i = 0; i < n; i++) {
                    qMat[i][j] = col[i] / norm;
                }
            }
            computeRemainingR(qMat, hMat, rMat, j, n);
        }
        return matMul2d(rMat, qMat, n, n, n);
    }

    private static float[] buildQRColumn(final float[][] hMat, final float[][] qMat, final float[][] rMat,
                                         final int j, final int n) {
        float[] col = new float[n];
        for (int i = 0; i < n; i++) {
            col[i] = hMat[i][j];
            for (int k = 0; k < j; k++) {
                col[i] -= qMat[i][k] * rMat[k][j];
            }
        }
        return col;
    }

    private static float vectorNorm(final float[] v) {
        float norm = 0;
        for (final float x : v) {
            norm += x * x;
        }
        return (float) Math.sqrt(norm);
    }

    private static void computeRemainingR(final float[][] qMat, final float[][] hMat, final float[][] rMat,
                                          final int j, final int n) {
        for (int k = j + 1; k < n; k++) {
            float dot = 0;
            for (int i = 0; i < n; i++) {
                dot += qMat[i][j] * hMat[i][k];
            }
            rMat[j][k] = dot;
        }
    }

    /**
     * Moore-Penrose pseudoinverse via SVD ({@code numpy.linalg.pinv}).
     *
     * @param a input matrix
     * @return the pseudoinverse
     */
    public static NdArray pinv(final NdArray a) {
        NdArray[] usv = svd(a);
        NdArray uArr = usv[0];
        NdArray sArr = usv[1];
        NdArray vt = usv[2];
        int m = a.shape(0);
        int n = a.shape(1);
        float[] s = sArr.toFloatArray();
        NdArray sinv = NdArrayFactory.zeros(a.arena(), DType.FLOAT32, n, n);
        int k = Math.min(m, n);
        for (int i = 0; i < k; i++) {
            if (s[i] > 1e-10f) {
                sinv.setFloat(1f / s[i], i, i);
            }
        }
        NdArray vArr = ShapeOps.transpose(vt);
        NdArray ut = ShapeOps.transpose(uArr);
        return MatMul.matmul(MatMul.matmul(vArr, sinv), ut);
    }

    /**
     * Matrix rank via SVD ({@code numpy.linalg.matrix_rank}).
     *
     * @param a input matrix
     * @return the numerical rank
     */
    public static int matrixRank(final NdArray a) {
        NdArray[] usv = svd(a);
        float[] s = usv[1].toFloatArray();
        // float eps
        float tol = s[0] * Math.max(a.shape(0), a.shape(1)) * 1.1920929e-7f;
        int rank = 0;
        for (final float v : s) {
            if (v > tol) {
                rank++;
            }
        }
        return rank;
    }

    /**
     * Sign and log of absolute determinant ({@code numpy.linalg.slogdet}).
     *
     * @param a square matrix
     * @return float array {sign, logabsdet}
     */
    public static float[] slogdet(final NdArray a) {
        float d = det(a);
        float sign = Math.signum(d);
        float logabsdet = (float) Math.log(Math.abs(d));
        return new float[]{sign, logabsdet};
    }

    /**
     * Eigenvalues and eigenvectors of a symmetric matrix ({@code numpy.linalg.eigh}).
     *
     * @param a symmetric matrix
     * @return array {eigenvalues (sorted ascending), eigenvectors (columns)}
     */
    public static NdArray[] eigh(final NdArray a) {
        int n = a.shape(0);
        if (AccelerateOps.isAvailable()) {
            return eighLapack(a, n);
        }
        return eighScalar(a, n);
    }

    private static NdArray[] eighLapack(final NdArray a, final int n) {
        NdArray c = a.contiguous();
        float[] aData = new float[n * n];
        for (int i = 0; i < n * n; i++) {
            aData[i] = c.flatGetFloat(i);
        }
        float[] eigenvalues = new float[n];
        float[] vectors = new float[n * n];
        try (var arena = Arena.ofConfined()) {
            AccelerateOps.eigvals(n, aData, eigenvalues, vectors, arena);
        }
        return new NdArray[]{
                NdArrayFactory.array(a.arena(), eigenvalues),
                NdArrayFactory.array(a.arena(), vectors, n, n)
        };
    }

    private static NdArray[] eighScalar(final NdArray a, final int n) {
        NdArray eigenvalues = eigvals(a);
        float[] vals = eigenvalues.toFloatArray();
        java.util.Arrays.sort(vals);
        NdArray sortedVals = NdArrayFactory.array(a.arena(), vals);
        float[][] vecs = new float[n][n];
        for (int k = 0; k < n; k++) {
            float[] v = computeEigenvector(a, vals[k], k, n);
            for (int i = 0; i < n; i++) {
                vecs[i][k] = v[i];
            }
        }
        return new NdArray[]{sortedVals, fromMatrix(a.arena(), vecs, n, n)};
    }

    private static float[] computeEigenvector(final NdArray a, final float lambda, final int startIdx,
                                              final int n) {
        float[][] shifted = toMatrix(a);
        for (int i = 0; i < n; i++) {
            shifted[i][i] -= lambda - 1e-6f;
        }
        float[] v = new float[n];
        v[startIdx] = 1f;
        for (int iter = 0; iter < 50; iter++) {
            v = matVecMul(shifted, v, n);
            float norm = vectorNorm(v);
            if (norm > 1e-10f) {
                for (int i = 0; i < n; i++) {
                    v[i] /= norm;
                }
            }
        }
        return v;
    }

    private static float[] matVecMul(final float[][] mat, final float[] vec, final int n) {
        float[] result = new float[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                result[i] += mat[i][j] * vec[j];
            }
        }
        return result;
    }

    /**
     * Condition number: ratio of largest to smallest singular value ({@code numpy.linalg.cond}).
     *
     * @param a input matrix
     * @return the condition number (infinity if singular)
     */
    public static float cond(final NdArray a) {
        NdArray[] usv = svd(a);
        float[] s = usv[1].toFloatArray();
        float sMax = s[0];
        float sMin = s[s.length - 1];
        return sMin > 1e-10f ? sMax / sMin : Float.POSITIVE_INFINITY;
    }

    /**
     * Least-squares solution via normal equations ({@code numpy.linalg.lstsq}).
     *
     * @param a coefficient matrix (m × n)
     * @param b right-hand side vector (m)
     * @return least-squares solution x
     */
    public static NdArray lstsq(final NdArray a, final NdArray b) {
        NdArray at = ShapeOps.transpose(a);
        NdArray atA = MatMul.matmul(at, a);
        NdArray atb = MatMul.matmul(at, ShapeOps.reshape(b, (int) b.size(), 1));
        NdArray x = solve(atA, ShapeOps.flatten(atb));
        return x;
    }
}
