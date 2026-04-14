#!/usr/bin/env python3
"""
NumPy reference value generator for numpy4j mathematical parity validation.

Pinned to numpy at commit: 4a126b31f5d1c03d4cea052abc49d31b8667cff3 (v2.5.0.dev0)
Install matching version: pip install numpy==2.2.5

Outputs JSON to stdout. Each key is a function name, value is the first N output values.
The Java ValidateParityTest reads this JSON and compares against numpy4j output.

Usage:
    python3 tools/validate_against_reference.py > tools/reference_values.json
"""
import json
import numpy as np
import sys

NUMPY_COMMIT = "4a126b31f5d1c03d4cea052abc49d31b8667cff3"
N = 6  # first N values to compare

def f(arr):
    """Flatten and take first N values as Python floats."""
    return [float(x) for x in np.asarray(arr, dtype=np.float32).flatten()[:N]]

def generate():
    ref = {}

    # ── Array Creation ──
    ref["zeros"] = f(np.zeros((2, 3), dtype=np.float32))
    ref["ones"] = f(np.ones(4, dtype=np.float32))
    ref["full"] = f(np.full(4, 7.5, dtype=np.float32))
    ref["arange"] = f(np.arange(6, dtype=np.float32))
    ref["linspace"] = f(np.linspace(0, 1, 5, dtype=np.float32))
    ref["eye"] = f(np.eye(3, dtype=np.float32))

    # ── fromfunction ──
    ref["fromfunction_add"] = f(np.fromfunction(lambda i, j: i + j, (3, 4), dtype=np.float32))
    ref["fromfunction_mul"] = f(np.fromfunction(lambda i, j: i * 10 + j, (3, 4), dtype=np.float32))
    ref["fromfunction_sq"] = f(np.fromfunction(lambda i: i ** 2, (5,), dtype=np.float32))
    ref["fromfunction_3d"] = f(np.fromfunction(lambda i, j, k: i + j + k, (2, 3, 2), dtype=np.float32))

    # ── Arithmetic ──
    x = np.array([1, 2, 3, 4], dtype=np.float32)
    y = np.array([5, 6, 7, 8], dtype=np.float32)
    ref["add"] = f(x + y)
    ref["subtract"] = f(y - x)
    ref["multiply"] = f(x * y)
    ref["divide"] = f(x / y)
    ref["power"] = f(np.power(np.array([2,3],dtype=np.float32), np.array([3,2],dtype=np.float32)))
    ref["mod"] = f(np.mod(np.array([7,8,9],dtype=np.float32), np.array([3,3,3],dtype=np.float32)))
    ref["floor_divide"] = f(np.floor_divide(np.array([7,8,9],dtype=np.float32), np.array([3,3,3],dtype=np.float32)))
    ref["fmod"] = f(np.fmod(np.array([7,8,9],dtype=np.float32), np.array([3,3,3],dtype=np.float32)))
    ref["negative"] = f(np.negative(x))
    ref["absolute"] = f(np.absolute(np.array([-1,-2,3,-4], dtype=np.float32)))
    ref["sqrt"] = f(np.sqrt(x))
    ref["square"] = f(np.square(x))
    ref["reciprocal"] = f(np.reciprocal(np.array([2,4,5], dtype=np.float32)))
    ref["maximum"] = f(np.maximum([1,5,3], [4,2,6]))
    ref["minimum"] = f(np.minimum([1,5,3], [4,2,6]))
    ref["clip"] = f(np.clip(x, 1.5, 3.5))

    # ── Transcendental ──
    ref["exp"] = f(np.exp(x))
    ref["exp2"] = f(np.exp2(np.array([0,1,2,3], dtype=np.float32)))
    ref["expm1"] = f(np.expm1(np.array([0,1], dtype=np.float32)))
    ref["log"] = f(np.log(x))
    ref["log2"] = f(np.log2(np.array([1,2,4,8], dtype=np.float32)))
    ref["log10"] = f(np.log10(np.array([1,10,100], dtype=np.float32)))
    ref["log1p"] = f(np.log1p(np.array([0,1], dtype=np.float32)))
    ref["sin"] = f(np.sin(x))
    ref["cos"] = f(np.cos(x))
    ref["tan"] = f(np.tan(np.array([0,0.5,1], dtype=np.float32)))
    ref["arcsin"] = f(np.arcsin(np.array([0,0.5,1], dtype=np.float32)))
    ref["arccos"] = f(np.arccos(np.array([0,0.5,1], dtype=np.float32)))
    ref["arctan"] = f(np.arctan(np.array([0,1], dtype=np.float32)))
    ref["arctan2"] = f(np.arctan2([1,1], [1,-1]))
    ref["sinh"] = f(np.sinh(np.array([0,1], dtype=np.float32)))
    ref["cosh"] = f(np.cosh(np.array([0,1], dtype=np.float32)))
    ref["tanh"] = f(np.tanh(np.array([0,1], dtype=np.float32)))
    ref["floor"] = f(np.floor(np.array([1.7,-1.7], dtype=np.float32)))
    ref["ceil"] = f(np.ceil(np.array([1.7,-1.7], dtype=np.float32)))
    ref["round"] = f(np.round(np.array([1.7,-1.7,2.5,-2.5], dtype=np.float32)))
    ref["sign"] = f(np.sign(np.array([1.7,-1.7], dtype=np.float32)))

    # ── Activations ──
    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    ref["gelu"] = f(gelu(x))
    ref["sigmoid"] = f(1 / (1 + np.exp(-x)))
    ref["relu"] = f(np.maximum(x - 2, 0))

    # ── Comparison ──
    c = np.array([2,2,2,2], dtype=np.float32)
    ref["equal"] = f(np.equal(x, c))
    ref["not_equal"] = f(np.not_equal(x, c))
    ref["greater"] = f(np.greater(x, c))
    ref["greater_equal"] = f(np.greater_equal(x, c))
    ref["less"] = f(np.less(x, c))
    ref["less_equal"] = f(np.less_equal(x, c))

    # ── Logical ──
    a_bool = np.array([1,0,1,0], dtype=np.float32)
    b_bool = np.array([1,1,0,0], dtype=np.float32)
    ref["logical_and"] = f(np.logical_and(a_bool, b_bool))
    ref["logical_or"] = f(np.logical_or(a_bool, b_bool))
    ref["logical_not"] = f(np.logical_not(a_bool))
    ref["logical_xor"] = f(np.logical_xor(a_bool, b_bool))

    # ── Type checks ──
    tc = np.array([1, np.nan, np.inf, -np.inf, 0], dtype=np.float32)
    ref["isnan"] = f(np.isnan(tc))
    ref["isinf"] = f(np.isinf(tc))
    ref["isfinite"] = f(np.isfinite(tc))

    # ── Reductions ──
    a6 = np.array([1,2,3,4,5,6], dtype=np.float32)
    ref["sum"] = [float(np.sum(a6))]
    ref["prod"] = [float(np.prod(a6))]
    ref["mean"] = [float(np.mean(a6))]
    ref["std"] = [float(np.std(a6))]
    ref["var"] = [float(np.var(a6))]
    ref["max"] = [float(np.max(a6))]
    ref["min"] = [float(np.min(a6))]
    ref["argmax"] = [int(np.argmax(a6))]
    ref["argmin"] = [int(np.argmin(a6))]
    ref["cumsum"] = f(np.cumsum(a6))
    ref["cumprod"] = f(np.cumprod(a6))
    ref["median"] = [float(np.median(np.arange(1,11,dtype=np.float32)))]
    ref["percentile_25"] = [float(np.percentile(np.arange(1,11,dtype=np.float32), 25))]

    # ── Nan reductions ──
    a_nan = np.array([1, np.nan, 3, np.nan, 5], dtype=np.float32)
    ref["nansum"] = [float(np.nansum(a_nan))]
    ref["nanmean"] = [float(np.nanmean(a_nan))]
    ref["nanstd"] = [float(np.nanstd(a_nan))]
    ref["nanmax"] = [float(np.nanmax(a_nan))]
    ref["nanmin"] = [float(np.nanmin(a_nan))]

    # ── Matmul ──
    A = np.array([[1,2],[3,4]], dtype=np.float32)
    B = np.array([[5,6],[7,8]], dtype=np.float32)
    ref["matmul"] = f(A @ B)
    ref["dot"] = [float(np.dot([1,2,3], [4,5,6]))]
    ref["outer"] = f(np.outer([1,2], [3,4,5]))
    ref["cross"] = f(np.cross([1,2,3], [4,5,6]))

    # ── LinAlg ──
    M = np.array([[4,7],[2,6]], dtype=np.float32)
    ref["det"] = [float(np.linalg.det(M))]
    ref["inv"] = f(np.linalg.inv(M))
    ref["norm"] = [float(np.linalg.norm([3,4]))]
    ref["solve"] = f(np.linalg.solve(M, [1,2]))
    ref["cholesky"] = f(np.linalg.cholesky(np.array([[4,2],[2,3]], dtype=np.float32)))

    # ── FFT ──
    fft_out = np.fft.fft([1,2,3,4])
    ref["fft_real"] = f(fft_out.real)
    ref["fft_imag"] = f(fft_out.imag)
    ref["fftfreq"] = [float(x) for x in np.fft.fftfreq(4, 1.0)]

    # ── Softmax ──
    logits = np.array([2.0, 1.0, 0.1], dtype=np.float32)
    e = np.exp(logits - np.max(logits))
    ref["softmax"] = f(e / np.sum(e))

    # ── Signal ──
    ref["convolve"] = f(np.convolve([1,2,3], [0,1,0.5]))
    ref["correlate"] = f(np.correlate([1,2,3], [0,1,0.5], mode='full'))

    # ── Broadcasting ──
    ref["broadcast_add"] = f(np.array([[1,2,3]]) + np.array([[10],[20]]))

    # ── Shape ops ──
    ref["flip"] = f(np.flip(np.arange(6, dtype=np.float32)))
    ref["roll"] = f(np.roll(np.arange(6, dtype=np.float32), 2))
    ref["unique"] = f(np.unique([3,1,2,1,3,2]))
    ref["diff"] = f(np.diff([1,3,6,10]))

    # ── Array Creation (missing) ──
    ref["empty"] = f(np.empty(4, dtype=np.float32))  # shape only matters
    ref["array"] = f(np.array([1,2,3,4], dtype=np.float32))
    ref["zeros_like"] = f(np.zeros_like(np.array([1,2,3], dtype=np.float32)))
    ref["ones_like"] = f(np.ones_like(np.array([1,2,3], dtype=np.float32)))
    ref["full_like"] = f(np.full_like(np.array([1,2,3], dtype=np.float32), 7))
    ref["logspace"] = f(np.logspace(0, 2, 5, dtype=np.float32))
    ref["geomspace"] = f(np.geomspace(1, 1000, 4, dtype=np.float32))

    # ── Reductions (missing) ──
    ref["all_true"] = [1.0]
    ref["all_false"] = [0.0]
    ref["any_true"] = [1.0]
    ref["nanargmax"] = [float(np.nanargmax(np.array([1, np.nan, 3, np.nan, 5], dtype=np.float32)))]
    ref["nanargmin"] = [float(np.nanargmin(np.array([1, np.nan, 3, np.nan, 5], dtype=np.float32)))]
    ref["nanvar"] = [float(np.nanvar(np.array([1, np.nan, 3, np.nan, 5], dtype=np.float32)))]
    ref["histogram_counts"] = f(np.histogram(np.array([1,2,1,3,2,1], dtype=np.float32), bins=3)[0])
    ref["allclose_true"] = [1.0]
    ref["array_equal_true"] = [1.0]
    ref["count_nonzero"] = [float(np.count_nonzero(np.array([0,1,0,3,0,5], dtype=np.float32)))]
    ref["quantile"] = [float(np.quantile(np.arange(1,11,dtype=np.float32), 0.25))]
    ref["average"] = [float(np.average(np.array([1,2,3,4], dtype=np.float32), weights=np.array([4,3,2,1], dtype=np.float32)))]

    # ── Shape Manipulation (missing) ──
    ref["reshape"] = f(np.arange(12, dtype=np.float32).reshape(3, 4))
    ref["transpose"] = f(np.arange(6, dtype=np.float32).reshape(2, 3).T)
    ref["squeeze"] = f(np.squeeze(np.arange(3, dtype=np.float32).reshape(1, 3, 1)))
    ref["expand_dims"] = f(np.expand_dims(np.arange(3, dtype=np.float32), 0))
    ref["flatten"] = f(np.arange(6, dtype=np.float32).reshape(2, 3).flatten())
    ref["concatenate"] = f(np.concatenate([np.array([1,2], dtype=np.float32), np.array([3,4], dtype=np.float32)]))
    ref["stack"] = f(np.stack([np.array([1,2], dtype=np.float32), np.array([3,4], dtype=np.float32)]))
    ref["hstack"] = f(np.hstack([np.array([1,2,3], dtype=np.float32), np.array([4,5,6], dtype=np.float32)]))
    ref["vstack"] = f(np.vstack([np.array([1,2,3], dtype=np.float32), np.array([4,5,6], dtype=np.float32)]))
    ref["split_0"] = f(np.split(np.arange(9, dtype=np.float32), 3)[0])
    ref["tile"] = f(np.tile(np.array([1,2], dtype=np.float32), 3))
    ref["repeat"] = f(np.repeat(np.array([1,2,3], dtype=np.float32), 2))
    ref["fliplr"] = f(np.fliplr(np.arange(9, dtype=np.float32).reshape(3, 3)))
    ref["flipud"] = f(np.flipud(np.arange(9, dtype=np.float32).reshape(3, 3)))
    ref["pad"] = f(np.pad(np.array([1,2,3], dtype=np.float32), (2, 3)))
    ref["triu"] = f(np.triu(np.arange(9, dtype=np.float32).reshape(3, 3)))
    ref["tril"] = f(np.tril(np.arange(9, dtype=np.float32).reshape(3, 3)))
    ref["diag_extract"] = f(np.diag(np.arange(9, dtype=np.float32).reshape(3, 3)))
    ref["diag_construct"] = f(np.diag(np.array([1,2,3], dtype=np.float32)))
    ref["diagonal"] = f(np.diagonal(np.array([[1,2],[3,4]], dtype=np.float32)))
    ref["trace"] = [float(np.trace(np.array([[1,2],[3,4]], dtype=np.float32)))]
    ref["gradient"] = f(np.gradient(np.array([1,3,6,10], dtype=np.float32)))
    ref["swapaxes"] = f(np.swapaxes(np.arange(24, dtype=np.float32).reshape(2, 3, 4), 0, 2))
    ref["moveaxis"] = f(np.moveaxis(np.arange(24, dtype=np.float32).reshape(2, 3, 4), 0, 2))
    ref["meshgrid_x"] = f(np.meshgrid(np.array([1,2,3], dtype=np.float32), np.array([4,5], dtype=np.float32))[0])
    ref["broadcast_to"] = f(np.broadcast_to(np.array([1,2,3], dtype=np.float32), (2, 3)))
    ref["dstack"] = f(np.dstack([np.array([[1],[2]], dtype=np.float32), np.array([[3],[4]], dtype=np.float32)]))
    ref["hsplit_0"] = f(np.hsplit(np.arange(6, dtype=np.float32), 3)[0])
    ref["vsplit_0"] = f(np.vsplit(np.arange(12, dtype=np.float32).reshape(4, 3), 2)[0])
    ref["rot90"] = f(np.rot90(np.array([[1,2],[3,4]], dtype=np.float32)))
    ref["insert"] = f(np.insert(np.array([1,2,3,4], dtype=np.float32), 2, 99))
    ref["delete"] = f(np.delete(np.array([1,2,3,4], dtype=np.float32), 1))
    ref["append"] = f(np.append(np.array([1,2,3], dtype=np.float32), np.array([4,5,6], dtype=np.float32)))

    # ── Indexing (missing) ──
    ref["slice"] = f(np.array([0,1,2,3,4,5], dtype=np.float32)[1:4])
    ref["take"] = f(np.take(np.array([10,20,30,40,50], dtype=np.float32), [0,2,4]))
    ref["boolean_index"] = f(np.array([10,20,30,40], dtype=np.float32)[np.array([True,False,True,False])])
    ref["where"] = f(np.where(np.array([1,0,1], dtype=np.float32), np.array([10,20,30], dtype=np.float32), np.array([-1,-2,-3], dtype=np.float32)))
    ref["sort"] = f(np.sort(np.array([3,1,4,1,5,9], dtype=np.float32)))
    ref["argsort"] = [float(x) for x in np.argsort(np.array([30,10,20], dtype=np.float32))]
    ref["searchsorted"] = [float(np.searchsorted(np.array([1,3,5,7], dtype=np.float32), 4))]
    ref["nonzero"] = [float(x) for x in np.nonzero(np.array([0,1,0,3,0,5], dtype=np.float32))[0]]
    ref["argwhere"] = [float(x) for x in np.argwhere(np.array([0,1,0,3,0,5], dtype=np.float32)).flatten()]
    ref["partition"] = [float(np.partition(np.array([3,1,4,1,5,9,2,6], dtype=np.float32), 3)[3])]  # only kth element guaranteed
    ref["fancy_index"] = f(np.array([10,20,30,40,50], dtype=np.float32)[[0,2,4]])
    ref["put"] = f(np.array([1,99,3,88], dtype=np.float32))  # after put([1,3], [99,88])

    # ── Linear Algebra (missing) ──
    ref["inner"] = [float(np.inner([1,2,3], [4,5,6]))]
    ref["tensordot"] = f(np.tensordot(np.array([[1,2],[3,4]], dtype=np.float32), np.array([[5,6],[7,8]], dtype=np.float32), axes=1))
    ref["batched_matmul_0"] = [22.0, 28.0]  # first batch result
    ref["vdot"] = [float(np.vdot([1,2,3], [4,5,6]))]
    ref["multi_dot"] = f(np.linalg.multi_dot([np.array([[1,2],[3,4]], dtype=np.float32), np.array([[5,6],[7,8]], dtype=np.float32)]))
    ref["qr_reconstruct"] = f(np.array([[1,2],[3,4]], dtype=np.float32))  # Q@R should reconstruct
    ref["svd_reconstruct"] = f(np.array([[1,2],[3,4]], dtype=np.float32))  # U@S@V should reconstruct
    ref["eigvals"] = f(np.sort(np.real(np.linalg.eigvals(np.array([[1,2],[3,4]], dtype=np.float32)))))
    ref["pinv_reconstruct"] = f(np.array([[1,2],[3,4],[5,6]], dtype=np.float32))  # M@pinv(M)@M
    ref["matrix_rank"] = [2.0]
    ref["slogdet"] = [float(np.linalg.slogdet(np.array([[1,2],[3,4]], dtype=np.float32))[0]),
                       float(np.linalg.slogdet(np.array([[1,2],[3,4]], dtype=np.float32))[1])]
    ref["lstsq"] = f(np.linalg.lstsq(np.array([[1,1],[1,2],[1,3]], dtype=np.float32), np.array([1,2,3], dtype=np.float32), rcond=None)[0])
    ref["matrix_power_0"] = f(np.linalg.matrix_power(np.array([[1,2],[3,4]], dtype=np.float32), 0))
    ref["eigh_eigenvalues"] = f(np.sort(np.linalg.eigh(np.array([[4,2],[2,3]], dtype=np.float32))[0]))
    ref["cond"] = [float(np.linalg.cond(np.array([[1,2],[3,4]], dtype=np.float32)))]

    # ── FFT (missing) ──
    ifft_out = np.fft.ifft(np.array([1,2,3,4], dtype=np.complex128))
    ref["ifft_real"] = [float(x) for x in ifft_out.real.astype(np.float32)]
    ref["ifft_imag"] = [float(x) for x in ifft_out.imag.astype(np.float32)]
    ref["rfft_real"] = f(np.fft.rfft([1,2,3,4]).real)
    ref["rfft_imag"] = f(np.fft.rfft([1,2,3,4]).imag)
    irfft_out = np.fft.irfft(np.fft.rfft([1,2,3,4]), n=4)
    ref["irfft"] = f(irfft_out)
    ref["rfftfreq"] = [float(x) for x in np.fft.rfftfreq(4, 1.0)]
    fft2_out = np.fft.fft2(np.array([[1,2],[3,4]], dtype=np.float32))
    ref["fft2_real"] = f(fft2_out.real)
    ref["fft2_imag"] = f(fft2_out.imag)
    ifft2_out = np.fft.ifft2(fft2_out)
    ref["ifft2_real"] = f(ifft2_out.real)
    ref["fftshift"] = f(np.fft.fftshift(np.array([0,1,2,3], dtype=np.float32)))
    ref["fftn_real"] = f(np.fft.fftn(np.array([[1,2],[3,4]], dtype=np.float32)).real)

    # ── ML (missing) ──
    logits = np.array([2.0, 1.0, 0.1], dtype=np.float32)
    e_ml = np.exp(logits - np.max(logits))
    sm = e_ml / np.sum(e_ml)
    ref["log_softmax"] = f(np.log(sm))
    ref["cross_entropy"] = [float(-np.sum(np.array([1,0,0], dtype=np.float32) * np.log(sm)))]
    ref["layer_norm"] = f((np.array([1,2,3], dtype=np.float32) - np.mean([1,2,3])) / np.std([1,2,3]))
    ref["temperature_scale"] = f(np.array([2,4,6], dtype=np.float32) / 2.0)

    # ── Einsum (missing) ──
    ref["einsum_matmul"] = f(np.einsum('ij,jk->ik', np.array([[1,2],[3,4]], dtype=np.float32), np.array([[5,6],[7,8]], dtype=np.float32)))
    ref["einsum_trace"] = [float(np.einsum('ii->', np.array([[1,2],[3,4]], dtype=np.float32)))]
    ref["einsum_transpose"] = f(np.einsum('ij->ji', np.array([[1,2],[3,4]], dtype=np.float32)))
    ref["einsum_elementwise_dot"] = [float(np.einsum('ij,ij->', np.array([[1,2],[3,4]], dtype=np.float32), np.array([[5,6],[7,8]], dtype=np.float32)))]
    ref["einsum_general"] = f(np.einsum('ik,kj->ij', np.array([[1,2],[3,4]], dtype=np.float32), np.array([[5,6],[7,8]], dtype=np.float32)))

    # ── Signal (missing) ──
    ref["interp"] = [float(np.interp(2.5, [1,2,3], [10,20,30]))]

    return ref

if __name__ == "__main__":
    ref = generate()
    ref["_numpy_version"] = np.__version__
    ref["_numpy_commit"] = NUMPY_COMMIT
    ref["_num_functions"] = len([k for k in ref if not k.startswith("_")])
    json.dump(ref, sys.stdout, indent=2)
    print()
