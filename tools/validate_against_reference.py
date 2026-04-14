#!/usr/bin/env python3
"""
Validation harness: runs NumPy operations and prints reference values
that the Java tests compare against for numerical parity.

Usage: python3 tools/validate_against_reference.py
"""
import numpy as np

np.set_printoptions(precision=8, suppress=True)
SEED = 42

def p(label, val):
    """Print first 4 values for comparison."""
    flat = np.asarray(val).flatten()
    print(f"{label}: {flat[:4].tolist()}")

def main():
    rng = np.random.default_rng(SEED)

    # ── Stage 1: Array creation ──
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    p("arange_6", a)

    z = np.zeros((2, 3), dtype=np.float32)
    p("zeros_2x3", z)

    eye = np.eye(3, dtype=np.float32)
    p("eye_3", eye)

    ls = np.linspace(0, 1, 5)
    p("linspace_0_1_5", ls)

    # ── Stage 2: Element-wise ops ──
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    p("add", x + y)
    p("mul", x * y)
    p("div", x / y)
    p("sqrt", np.sqrt(x))
    p("exp", np.exp(x))
    p("log", np.log(x))
    p("tanh", np.tanh(x))
    p("sin", np.sin(x))
    p("cos", np.cos(x))
    p("abs", np.abs(np.array([-1.0, -2.0, 3.0, -4.0], dtype=np.float32)))
    p("clip", np.clip(x, 1.5, 3.5))
    p("relu", np.maximum(x - 2, 0))

    # GELU (tanh approximation)
    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    p("gelu", gelu(x))

    # Sigmoid
    p("sigmoid", 1 / (1 + np.exp(-x)))

    # ── Stage 3: Reductions ──
    a6 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    print(f"sum: {np.sum(a6)}")
    print(f"prod: {np.prod(a6)}")
    print(f"mean: {np.mean(a6)}")
    print(f"std: {np.std(a6)}")
    print(f"var: {np.var(a6)}")
    print(f"max: {np.max(a6)}")
    print(f"min: {np.min(a6)}")
    print(f"argmax: {np.argmax(a6)}")
    print(f"argmin: {np.argmin(a6)}")
    p("cumsum", np.cumsum(a6))
    p("cumprod", np.cumprod(a6))

    # Axis reductions
    m = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    p("sum_axis0", np.sum(m, axis=0))
    p("sum_axis1", np.sum(m, axis=1))
    p("max_axis1", np.max(m, axis=1))
    p("argmax_axis1", np.argmax(m, axis=1))

    # ── Stage 4: Shape ops ──
    r = np.arange(12, dtype=np.float32).reshape(3, 4)
    p("reshape_3x4", r)
    p("transpose", r.T)
    p("flatten", r.flatten())

    # ── Stage 5: Matmul ──
    A = np.array([[1, 2], [3, 4]], dtype=np.float32)
    B = np.array([[5, 6], [7, 8]], dtype=np.float32)
    p("matmul", A @ B)
    p("dot_1d", np.array([np.dot(np.array([1,2,3], dtype=np.float32),
                                  np.array([4,5,6], dtype=np.float32))]))
    p("outer", np.outer(np.array([1,2], dtype=np.float32),
                         np.array([3,4,5], dtype=np.float32)))

    # ── Stage 6: LinAlg ──
    M = np.array([[4, 7], [2, 6]], dtype=np.float32)
    print(f"det: {np.linalg.det(M)}")
    p("inv", np.linalg.inv(M))
    print(f"norm: {np.linalg.norm(np.array([3, 4], dtype=np.float32))}")

    b_vec = np.array([1, 2], dtype=np.float32)
    p("solve", np.linalg.solve(M, b_vec))

    spd = np.array([[4, 2], [2, 3]], dtype=np.float32)
    p("cholesky", np.linalg.cholesky(spd))

    Q, R = np.linalg.qr(A.astype(np.float64))
    p("qr_Q", Q.astype(np.float32))
    p("qr_R", R.astype(np.float32))

    # ── Stage 7: FFT ──
    sig = np.array([1, 2, 3, 4], dtype=np.float32)
    fft_out = np.fft.fft(sig)
    p("fft_real", fft_out.real.astype(np.float32))
    p("fft_imag", fft_out.imag.astype(np.float32))
    p("fftfreq", np.fft.fftfreq(4, 1.0))

    # ── Stage 8: ML ops ──
    logits = np.array([[2.0, 1.0, 0.1]], dtype=np.float32)
    # Softmax
    def softmax(x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)
    p("softmax", softmax(logits))

    # Log softmax
    def log_softmax(x):
        m = np.max(x, axis=-1, keepdims=True)
        e = np.exp(x - m)
        return x - m - np.log(np.sum(e, axis=-1, keepdims=True))
    p("log_softmax", log_softmax(logits))

    # Greedy search
    print(f"greedy: {np.argmax(logits, axis=-1).tolist()}")

    # Top-K
    k = 2
    sorted_logits = np.sort(logits, axis=-1)
    threshold = sorted_logits[0, -k]
    topk = np.where(logits >= threshold, logits, -np.inf)
    p("topk_2", topk)

    # Layer norm
    x_ln = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    gamma = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    beta = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    mean_ln = np.mean(x_ln, axis=-1, keepdims=True)
    var_ln = np.var(x_ln, axis=-1, keepdims=True)
    ln_out = (x_ln - mean_ln) / np.sqrt(var_ln + 1e-5) * gamma + beta
    p("layer_norm", ln_out)

    print("\n=== All reference values generated ===")

if __name__ == "__main__":
    main()
