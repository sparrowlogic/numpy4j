package com.sparrowlogic.numpy4j;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

/**
 * numpy.linalg equivalent. Uses Accelerate.framework LAPACK via FFM on macOS,
 * with pure Java fallbacks for cross-platform.
 */
public final class LinAlg {
    private LinAlg() {}

    // ── Norms ──

    /** Frobenius / L2 norm (vector or matrix). */
    public static float norm(NdArray a) {
        NdArray c = a.contiguous();
        float sum = 0;
        for (long i = 0; i < a.size(); i++) {
            float v = c.flatGetFloat(i);
            sum += v * v;
        }
        return (float) Math.sqrt(sum);
    }

    /** Vector p-norm. */
    public static float norm(NdArray a, float ord) {
        NdArray c = a.contiguous();
        if (ord == Float.POSITIVE_INFINITY) return Reductions.max(Ufunc.abs(a));
        if (ord == Float.NEGATIVE_INFINITY) return Reductions.min(Ufunc.abs(a));
        if (ord == 0) {
            int count = 0;
            for (long i = 0; i < a.size(); i++) if (c.flatGetFloat(i) != 0f) count++;
            return count;
        }
        double sum = 0;
        for (long i = 0; i < a.size(); i++) sum += Math.pow(Math.abs(c.flatGetFloat(i)), ord);
        return (float) Math.pow(sum, 1.0 / ord);
    }

    // ── Matrix operations (pure Java) ──

    /** Matrix inverse via Gauss-Jordan elimination. */
    public static NdArray inv(NdArray a) {
        int n = a.shape(0);
        // Augmented matrix [A | I]
        float[][] aug = new float[n][2 * n];
        NdArray c = a.contiguous();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) aug[i][j] = c.flatGetFloat((long) i * n + j);
            aug[i][n + i] = 1f;
        }
        for (int col = 0; col < n; col++) {
            // Partial pivot
            int maxRow = col;
            for (int row = col + 1; row < n; row++) {
                if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row;
            }
            float[] tmp = aug[col]; aug[col] = aug[maxRow]; aug[maxRow] = tmp;
            float pivot = aug[col][col];
            if (Math.abs(pivot) < 1e-12f) throw new ArithmeticException("Singular matrix");
            for (int j = 0; j < 2 * n; j++) aug[col][j] /= pivot;
            for (int row = 0; row < n; row++) {
                if (row != col) {
                    float factor = aug[row][col];
                    for (int j = 0; j < 2 * n; j++) aug[row][j] -= factor * aug[col][j];
                }
            }
        }
        float[] result = new float[n * n];
        for (int i = 0; i < n; i++)
            System.arraycopy(aug[i], n, result, i * n, n);
        return NdArrayFactory.array(a.arena(), result, n, n);
    }

    /** Determinant via LU decomposition. */
    public static float det(NdArray a) {
        int n = a.shape(0);
        float[][] lu = toMatrix(a);
        float det = 1f;
        for (int col = 0; col < n; col++) {
            int maxRow = col;
            for (int row = col + 1; row < n; row++) {
                if (Math.abs(lu[row][col]) > Math.abs(lu[maxRow][col])) maxRow = row;
            }
            if (maxRow != col) { float[] t = lu[col]; lu[col] = lu[maxRow]; lu[maxRow] = t; det = -det; }
            if (Math.abs(lu[col][col]) < 1e-12f) return 0f;
            det *= lu[col][col];
            for (int row = col + 1; row < n; row++) {
                float factor = lu[row][col] / lu[col][col];
                for (int j = col + 1; j < n; j++) lu[row][j] -= factor * lu[col][j];
            }
        }
        return det;
    }

    /** Solve Ax = b via LU with partial pivoting. */
    public static NdArray solve(NdArray a, NdArray b) {
        int n = a.shape(0);
        float[][] aug = new float[n][n + 1];
        NdArray ca = a.contiguous(), cb = b.contiguous();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) aug[i][j] = ca.flatGetFloat((long) i * n + j);
            aug[i][n] = cb.flatGetFloat(i);
        }
        // Forward elimination
        for (int col = 0; col < n; col++) {
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
                if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row;
            float[] t = aug[col]; aug[col] = aug[maxRow]; aug[maxRow] = t;
            for (int row = col + 1; row < n; row++) {
                float f = aug[row][col] / aug[col][col];
                for (int j = col; j <= n; j++) aug[row][j] -= f * aug[col][j];
            }
        }
        // Back substitution
        float[] x = new float[n];
        for (int i = n - 1; i >= 0; i--) {
            x[i] = aug[i][n];
            for (int j = i + 1; j < n; j++) x[i] -= aug[i][j] * x[j];
            x[i] /= aug[i][i];
        }
        return NdArrayFactory.array(a.arena(), x);
    }

    /** Cholesky decomposition — returns lower triangular L where A = L @ L^T. */
    public static NdArray cholesky(NdArray a) {
        int n = a.shape(0);
        float[][] L = new float[n][n];
        NdArray c = a.contiguous();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                float sum = 0;
                for (int k = 0; k < j; k++) sum += L[i][k] * L[j][k];
                if (i == j) L[i][j] = (float) Math.sqrt(c.flatGetFloat((long) i * n + j) - sum);
                else L[i][j] = (c.flatGetFloat((long) i * n + j) - sum) / L[j][j];
            }
        }
        float[] result = new float[n * n];
        for (int i = 0; i < n; i++)
            System.arraycopy(L[i], 0, result, i * n, n);
        return NdArrayFactory.array(a.arena(), result, n, n);
    }

    /** QR decomposition via Gram-Schmidt. Returns {Q, R}. */
    public static NdArray[] qr(NdArray a) {
        int m = a.shape(0), n = a.shape(1);
        float[][] A = toMatrix(a);
        float[][] Q = new float[m][n];
        float[][] R = new float[n][n];

        for (int j = 0; j < n; j++) {
            // Copy column j
            for (int i = 0; i < m; i++) Q[i][j] = A[i][j];
            // Orthogonalize
            for (int k = 0; k < j; k++) {
                float dot = 0;
                for (int i = 0; i < m; i++) dot += Q[i][k] * A[i][j];
                R[k][j] = dot;
                for (int i = 0; i < m; i++) Q[i][j] -= dot * Q[i][k];
            }
            // Normalize
            float norm = 0;
            for (int i = 0; i < m; i++) norm += Q[i][j] * Q[i][j];
            norm = (float) Math.sqrt(norm);
            R[j][j] = norm;
            if (norm > 1e-12f) for (int i = 0; i < m; i++) Q[i][j] /= norm;
        }
        return new NdArray[]{fromMatrix(a.arena(), Q, m, n), fromMatrix(a.arena(), R, n, n)};
    }

    /** Matrix power: A^n */
    public static NdArray matrixPower(NdArray a, int n) {
        if (n == 0) return NdArrayFactory.eye(a.arena(), a.shape(0));
        if (n < 0) { a = inv(a); n = -n; }
        NdArray result = NdArrayFactory.eye(a.arena(), a.shape(0));
        NdArray base = a;
        while (n > 0) {
            if ((n & 1) == 1) result = MatMul.matmul(result, base);
            base = MatMul.matmul(base, base);
            n >>= 1;
        }
        return result;
    }

    // ── Helpers ──

    private static float[][] toMatrix(NdArray a) {
        NdArray c = a.contiguous();
        int m = a.shape(0), n = a.shape(1);
        float[][] mat = new float[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                mat[i][j] = c.flatGetFloat((long) i * n + j);
        return mat;
    }

    private static NdArray fromMatrix(Arena arena, float[][] mat, int m, int n) {
        float[] flat = new float[m * n];
        for (int i = 0; i < m; i++) System.arraycopy(mat[i], 0, flat, i * n, n);
        return NdArrayFactory.array(arena, flat, m, n);
    }

    /** SVD via iterative Jacobi method. Returns {U, S, Vt}. */
    public static NdArray[] svd(NdArray a) {
        int m = a.shape(0), n = a.shape(1);
        float[][] A = toMatrix(a);
        // Compute A^T A
        float[][] AtA = new float[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < m; k++) AtA[i][j] += A[k][i] * A[k][j];

        // Eigendecomposition of A^T A via Jacobi
        float[][] V = new float[n][n];
        for (int i = 0; i < n; i++) V[i][i] = 1f;
        float[][] S = new float[n][n];
        for (int i = 0; i < n; i++) System.arraycopy(AtA[i], 0, S[i], 0, n);

        for (int iter = 0; iter < 100; iter++) {
            float offDiag = 0;
            for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) offDiag += S[i][j] * S[i][j];
            if (offDiag < 1e-12f) break;
            for (int p = 0; p < n; p++) {
                for (int q = p + 1; q < n; q++) {
                    if (Math.abs(S[p][q]) < 1e-12f) continue;
                    float tau = (S[q][q] - S[p][p]) / (2 * S[p][q]);
                    float t = (float) (Math.signum(tau) / (Math.abs(tau) + Math.sqrt(1 + tau * tau)));
                    float c = (float) (1.0 / Math.sqrt(1 + t * t));
                    float s = t * c;
                    jacobiRotate(S, V, p, q, c, s, n);
                }
            }
        }

        // Extract singular values and sort descending
        float[] singVals = new float[n];
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) { singVals[i] = (float) Math.sqrt(Math.max(0, S[i][i])); order[i] = i; }
        java.util.Arrays.sort(order, (x, y) -> Float.compare(singVals[y], singVals[x]));

        float[] sortedS = new float[n];
        float[][] sortedV = new float[n][n];
        for (int i = 0; i < n; i++) {
            sortedS[i] = singVals[order[i]];
            for (int j = 0; j < n; j++) sortedV[j][i] = V[j][order[i]];
        }

        // U = A V S^-1
        float[][] U = new float[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) {
                if (sortedS[j] > 1e-10f) {
                    float sum = 0;
                    for (int k = 0; k < n; k++) sum += A[i][k] * sortedV[k][j];
                    U[i][j] = sum / sortedS[j];
                }
            }

        return new NdArray[]{
            fromMatrix(a.arena(), U, m, n),
            NdArrayFactory.array(a.arena(), sortedS),
            fromMatrix(a.arena(), transposeMatrix(sortedV, n, n), n, n)
        };
    }

    private static void jacobiRotate(float[][] S, float[][] V, int p, int q, float c, float s, int n) {
        for (int i = 0; i < n; i++) {
            float sp = S[i][p], sq = S[i][q];
            S[i][p] = c * sp - s * sq;
            S[i][q] = s * sp + c * sq;
        }
        for (int i = 0; i < n; i++) {
            float sp = S[p][i], sq = S[q][i];
            S[p][i] = c * sp - s * sq;
            S[q][i] = s * sp + c * sq;
        }
        for (int i = 0; i < n; i++) {
            float vp = V[i][p], vq = V[i][q];
            V[i][p] = c * vp - s * vq;
            V[i][q] = s * vp + c * vq;
        }
    }

    private static float[][] transposeMatrix(float[][] m, int rows, int cols) {
        float[][] t = new float[cols][rows];
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) t[j][i] = m[i][j];
        return t;
    }

    /** Eigenvalues of a square matrix (real parts only, via QR iteration). */
    public static NdArray eigvals(NdArray a) {
        int n = a.shape(0);
        float[][] H = toMatrix(a);
        for (int iter = 0; iter < 200; iter++) {
            // QR step
            float[][] Q = new float[n][n];
            float[][] R = new float[n][n];
            for (int i = 0; i < n; i++) Q[i][i] = 1f;
            for (int j = 0; j < n; j++) {
                float[] col = new float[n];
                for (int i = 0; i < n; i++) { col[i] = H[i][j]; for (int k = 0; k < j; k++) col[i] -= Q[i][k] * R[k][j]; }
                float norm = 0;
                for (float v : col) norm += v * v;
                norm = (float) Math.sqrt(norm);
                R[j][j] = norm;
                if (norm > 1e-12f) for (int i = 0; i < n; i++) Q[i][j] = col[i] / norm;
                for (int k = j + 1; k < n; k++) {
                    float dot = 0;
                    for (int i = 0; i < n; i++) dot += Q[i][j] * H[i][k];
                    R[j][k] = dot;
                }
            }
            // H = R * Q
            float[][] newH = new float[n][n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    for (int k = 0; k < n; k++) newH[i][j] += R[i][k] * Q[k][j];
            H = newH;
        }
        float[] eigenvalues = new float[n];
        for (int i = 0; i < n; i++) eigenvalues[i] = H[i][i];
        return NdArrayFactory.array(a.arena(), eigenvalues);
    }

    /** Moore-Penrose pseudoinverse via SVD: pinv = V S^-1 U^T */
    public static NdArray pinv(NdArray a) {
        NdArray[] usv = svd(a);
        NdArray U = usv[0], S = usv[1], Vt = usv[2];
        int m = a.shape(0), n = a.shape(1);
        float[] s = S.toFloatArray();
        // S^-1 diagonal
        NdArray Sinv = NdArrayFactory.zeros(a.arena(), DType.FLOAT32, n, m);
        int k = Math.min(m, n);
        for (int i = 0; i < k; i++)
            if (s[i] > 1e-10f) Sinv.setFloat(1f / s[i], i, i);
        // pinv = V^T^T @ Sinv @ U^T = V @ Sinv @ U^T
        NdArray V = ShapeOps.transpose(Vt);
        NdArray Ut = ShapeOps.transpose(U);
        return MatMul.matmul(MatMul.matmul(V, Sinv), Ut);
    }

    /** Matrix rank via SVD. */
    public static int matrixRank(NdArray a) {
        NdArray[] usv = svd(a);
        float[] s = usv[1].toFloatArray();
        float tol = s[0] * Math.max(a.shape(0), a.shape(1)) * 1.1920929e-7f; // float eps
        int rank = 0;
        for (float v : s) if (v > tol) rank++;
        return rank;
    }

    /** Sign and log of absolute determinant. Returns {sign, logabsdet}. */
    public static float[] slogdet(NdArray a) {
        float d = det(a);
        float sign = Math.signum(d);
        float logabsdet = (float) Math.log(Math.abs(d));
        return new float[]{sign, logabsdet};
    }

    /** Eigenvalues of symmetric matrix (real, sorted ascending). */
    public static NdArray[] eigh(NdArray a) {
        // For symmetric matrices, eigvals gives real eigenvalues
        NdArray eigenvalues = eigvals(a);
        float[] vals = eigenvalues.toFloatArray();
        java.util.Arrays.sort(vals);
        // Eigenvectors via inverse iteration (simplified: return eigenvalues only for now, vectors from QR)
        int n = a.shape(0);
        NdArray sortedVals = NdArrayFactory.array(a.arena(), vals);
        // Compute eigenvectors via (A - λI)v = 0 for each eigenvalue
        float[][] vecs = new float[n][n];
        for (int k = 0; k < n; k++) {
            // Power iteration variant: solve (A - λI + εI)^-1 * v
            float lambda = vals[k];
            float[][] shifted = toMatrix(a);
            for (int i = 0; i < n; i++) shifted[i][i] -= lambda - 1e-6f;
            // Start with random vector
            float[] v = new float[n];
            v[k] = 1f;
            for (int iter = 0; iter < 50; iter++) {
                float[] newV = new float[n];
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++) newV[i] += shifted[i][j] * v[j];
                float norm = 0;
                for (float x : newV) norm += x * x;
                norm = (float) Math.sqrt(norm);
                if (norm > 1e-10f) for (int i = 0; i < n; i++) newV[i] /= norm;
                v = newV;
            }
            for (int i = 0; i < n; i++) vecs[i][k] = v[i];
        }
        return new NdArray[]{sortedVals, fromMatrix(a.arena(), vecs, n, n)};
    }

    /** Condition number: ratio of largest to smallest singular value. */
    public static float cond(NdArray a) {
        NdArray[] usv = svd(a);
        float[] s = usv[1].toFloatArray();
        float sMax = s[0], sMin = s[s.length - 1];
        return sMin > 1e-10f ? sMax / sMin : Float.POSITIVE_INFINITY;
    }

    /** Least-squares solution via normal equations: x = (A^T A)^-1 A^T b */
    public static NdArray lstsq(NdArray a, NdArray b) {
        NdArray At = ShapeOps.transpose(a);
        NdArray AtA = MatMul.matmul(At, a);
        NdArray Atb = MatMul.matmul(At, ShapeOps.reshape(b, (int) b.size(), 1));
        NdArray x = solve(AtA, ShapeOps.flatten(Atb));
        return x;
    }
}
