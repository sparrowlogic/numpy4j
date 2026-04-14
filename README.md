# numpy4j

[![CI](https://github.com/sparrowlogic/numpy4j/actions/workflows/ci.yml/badge.svg)](https://github.com/sparrowlogic/numpy4j/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sparrowlogic/numpy4j/branch/main/graph/badge.svg)](https://codecov.io/gh/sparrowlogic/numpy4j)
[![Java 26](https://img.shields.io/badge/Java-26-blue)](https://openjdk.org/projects/jdk/26/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A 1:1 NumPy port for Java 26 using FFM (Foreign Function & Memory API), SIMD (Vector API preview), and AMX offloading on
Apple Silicon.

## Requirements

- Java 26+ (with `--enable-preview`)
- Maven 3.9+
- macOS arm64 for AMX acceleration (falls back to SIMD on other platforms)

## Build & Test

```bash
mvn clean test
```

499 test runs across 14 test classes, including 201 parameterized parity checks against NumPy reference values.
CI runs on both `ubuntu-latest` and `macos-latest` (GitHub Actions).

## Code Quality

Zero checkstyle violations enforced via [K-SAUR](checkstyle.xml) (Karl's Style and Usage Rules) — a strict
ruleset covering naming, braces, complexity, imports, and Javadoc. The only suppression is `module-info.java`.

## Architecture

| Layer    | Technology                                          | Purpose                                       |
|----------|-----------------------------------------------------|-----------------------------------------------|
| Storage  | `MemorySegment` + `Arena` (FFM)                     | Off-heap tensor memory, zero GC pressure      |
| SIMD     | `jdk.incubator.vector` (Vector API)                 | Element-wise ops, reductions, dot products    |
| AMX      | Accelerate.framework `cblas_sgemm` via FFM `Linker` | Matrix multiply on Apple M-series coprocessor |
| Fallback | Pure Java                                           | Cross-platform compatibility                  |

## Project Structure

```
src/main/java/com/sparrowlogic/numpy4j/
├── NdArray.java          # Core n-dimensional array (numpy.ndarray)
├── NdArrayFactory.java   # Array creation (zeros, ones, arange, linspace, eye)
├── DType.java            # Data types (FLOAT32, FLOAT64, INT32, INT64, BOOL)
├── ShapeUtils.java       # Shape/stride/broadcast utilities
├── SimdOps.java          # Low-level SIMD kernels (Vector API)
├── Ufunc.java            # Element-wise universal functions
├── Reductions.java       # Reduction operations (sum, mean, std, etc.)
├── ShapeOps.java         # Shape manipulation (reshape, transpose, etc.)
├── IndexOps.java         # Indexing, slicing, sorting
├── MatMul.java           # Matrix multiplication (AMX + SIMD)
├── LinAlg.java           # Linear algebra (numpy.linalg)
├── FFT.java              # Fast Fourier Transform (numpy.fft)
├── ML.java               # ML-specific ops (softmax, conv1d, beam search)
├── Random.java           # Random number generation (PCG64)
├── Einsum.java           # Einstein summation (numpy.einsum)
└── SignalOps.java        # Signal processing (convolve, correlate, interp)

src/test/java/com/sparrowlogic/numpy4j/
├── NdArrayTest.java          # ↔ NdArray.java
├── NdArrayFactoryTest.java   # ↔ NdArrayFactory.java
├── UfuncTest.java            # ↔ Ufunc.java
├── ReductionsTest.java       # ↔ Reductions.java
├── ShapeOpsTest.java         # ↔ ShapeOps.java
├── IndexOpsTest.java         # ↔ IndexOps.java
├── MatMulTest.java           # ↔ MatMul.java
├── LinAlgTest.java           # ↔ LinAlg.java
├── FFTTest.java              # ↔ FFT.java
├── MLTest.java               # ↔ ML.java
├── RandomTest.java           # ↔ Random.java
├── EinsumTest.java           # ↔ Einsum.java
├── SignalOpsTest.java        # ↔ SignalOps.java
└── ValidateParityTest.java   # Cross-validated against NumPy reference values
```

## NumPy Parity Checklist

### Array Creation (`NdArrayFactory`)

```java
import com.sparrowlogic.numpy4j.NdArrayFactory;
```

| NumPy Function                 | numpy4j Method                  | Status | Tested |
|--------------------------------|---------------------------------|--------|--------|
| `numpy.zeros`                  | `NdArrayFactory.zeros()`        | ✅      | ✅      |
| `numpy.ones`                   | `NdArrayFactory.ones()`         | ✅      | ✅      |
| `numpy.full`                   | `NdArrayFactory.full()`         | ✅      | ✅      |
| `numpy.empty`                  | `NdArrayFactory.empty()`        | ✅      | ✅      |
| `numpy.arange`                 | `NdArrayFactory.arange()`       | ✅      | ✅      |
| `numpy.linspace`               | `NdArrayFactory.linspace()`     | ✅      | ✅      |
| `numpy.eye` / `numpy.identity` | `NdArrayFactory.eye()`          | ✅      | ✅      |
| `numpy.array`                  | `NdArrayFactory.array()`        | ✅      | ✅      |
| `numpy.zeros_like`             | `NdArrayFactory.zerosLike()`    | ✅      | ✅      |
| `numpy.ones_like`              | `NdArrayFactory.onesLike()`     | ✅      | ✅      |
| `numpy.full_like`              | `NdArrayFactory.fullLike()`     | ✅      | ✅      |
| `numpy.empty_like`             | `NdArrayFactory.emptyLike()`    | ✅      | ✅      |
| `numpy.fromfunction`           | `NdArrayFactory.fromfunction()` | ✅      | ✅      |
| `numpy.logspace`               | `NdArrayFactory.logspace()`     | ✅      | ✅      |
| `numpy.geomspace`              | `NdArrayFactory.geomspace()`    | ✅      | ✅      |

### Element-wise Operations (`Ufunc`)

```java
import com.sparrowlogic.numpy4j.Ufunc;
```

| NumPy Function                       | numpy4j Method         | Status | Tested |
|--------------------------------------|------------------------|--------|--------|
| `numpy.add`                          | `Ufunc.add()`          | ✅      | ✅      |
| `numpy.subtract`                     | `Ufunc.subtract()`     | ✅      | ✅      |
| `numpy.multiply`                     | `Ufunc.multiply()`     | ✅      | ✅      |
| `numpy.divide` / `numpy.true_divide` | `Ufunc.divide()`       | ✅      | ✅      |
| `numpy.power`                        | `Ufunc.power()`        | ✅      | ✅      |
| `numpy.mod` / `numpy.remainder`      | `Ufunc.mod()`          | ✅      | ✅      |
| `numpy.negative`                     | `Ufunc.neg()`          | ✅      | ✅      |
| `numpy.absolute` / `numpy.abs`       | `Ufunc.abs()`          | ✅      | ✅      |
| `numpy.sqrt`                         | `Ufunc.sqrt()`         | ✅      | ✅      |
| `numpy.square`                       | `Ufunc.square()`       | ✅      | ✅      |
| `numpy.exp`                          | `Ufunc.exp()`          | ✅      | ✅      |
| `numpy.exp2`                         | `Ufunc.exp2()`         | ✅      | ✅      |
| `numpy.expm1`                        | `Ufunc.expm1()`        | ✅      | ✅      |
| `numpy.log`                          | `Ufunc.log()`          | ✅      | ✅      |
| `numpy.log2`                         | `Ufunc.log2()`         | ✅      | ✅      |
| `numpy.log10`                        | `Ufunc.log10()`        | ✅      | ✅      |
| `numpy.log1p`                        | `Ufunc.log1p()`        | ✅      | ✅      |
| `numpy.sin`                          | `Ufunc.sin()`          | ✅      | ✅      |
| `numpy.cos`                          | `Ufunc.cos()`          | ✅      | ✅      |
| `numpy.tan`                          | `Ufunc.tan()`          | ✅      | ✅      |
| `numpy.arcsin`                       | `Ufunc.arcsin()`       | ✅      | ✅      |
| `numpy.arccos`                       | `Ufunc.arccos()`       | ✅      | ✅      |
| `numpy.arctan`                       | `Ufunc.arctan()`       | ✅      | ✅      |
| `numpy.arctan2`                      | `Ufunc.arctan2()`      | ✅      | ✅      |
| `numpy.sinh`                         | `Ufunc.sinh()`         | ✅      | ✅      |
| `numpy.cosh`                         | `Ufunc.cosh()`         | ✅      | ✅      |
| `numpy.tanh`                         | `Ufunc.tanh()`         | ✅      | ✅      |
| `numpy.floor`                        | `Ufunc.floor()`        | ✅      | ✅      |
| `numpy.ceil`                         | `Ufunc.ceil()`         | ✅      | ✅      |
| `numpy.round` / `numpy.around`       | `Ufunc.round()`        | ✅      | ✅      |
| `numpy.sign`                         | `Ufunc.sign()`         | ✅      | ✅      |
| `numpy.clip`                         | `Ufunc.clip()`         | ✅      | ✅      |
| `numpy.maximum`                      | `Ufunc.maximum()`      | ✅      | ✅      |
| `numpy.minimum`                      | `Ufunc.minimum()`      | ✅      | ✅      |
| `numpy.reciprocal`                   | `Ufunc.reciprocal()`   | ✅      | ✅      |
| `numpy.floor_divide`                 | `Ufunc.floorDivide()`  | ✅      | ✅      |
| `numpy.fmod`                         | `Ufunc.fmod()`         | ✅      | ✅      |

### Comparison & Logical (`Ufunc`)

```java
import com.sparrowlogic.numpy4j.Ufunc;
```

| NumPy Function        | numpy4j Method         | Status | Tested |
|-----------------------|------------------------|--------|--------|
| `numpy.equal`         | `Ufunc.equal()`        | ✅      | ✅      |
| `numpy.not_equal`     | `Ufunc.notEqual()`     | ✅      | ✅      |
| `numpy.greater`       | `Ufunc.greater()`      | ✅      | ✅      |
| `numpy.greater_equal` | `Ufunc.greaterEqual()` | ✅      | ✅      |
| `numpy.less`          | `Ufunc.less()`         | ✅      | ✅      |
| `numpy.less_equal`    | `Ufunc.lessEqual()`    | ✅      | ✅      |
| `numpy.logical_and`   | `Ufunc.logicalAnd()`   | ✅      | ✅      |
| `numpy.logical_or`    | `Ufunc.logicalOr()`    | ✅      | ✅      |
| `numpy.logical_not`   | `Ufunc.logicalNot()`   | ✅      | ✅      |
| `numpy.logical_xor`   | `Ufunc.logicalXor()`   | ✅      | ✅      |
| `numpy.isnan`         | `Ufunc.isnan()`        | ✅      | ✅      |
| `numpy.isinf`         | `Ufunc.isinf()`        | ✅      | ✅      |
| `numpy.isfinite`      | `Ufunc.isfinite()`     | ✅      | ✅      |

### Activations (ML-specific, `Ufunc` / `ML`)

```java
import com.sparrowlogic.numpy4j.Ufunc;
import com.sparrowlogic.numpy4j.ML;
```

| Function           | numpy4j Method    | Status | Tested |
|--------------------|-------------------|--------|--------|
| GELU (tanh approx) | `Ufunc.gelu()`    | ✅      | ✅      |
| Sigmoid            | `Ufunc.sigmoid()` | ✅      | ✅      |
| ReLU               | `Ufunc.relu()`    | ✅      | ✅      |
| Softmax            | `ML.softmax()`    | ✅      | ✅      |
| Log-Softmax        | `ML.logSoftmax()` | ✅      | ✅      |

### Reductions (`Reductions`)

```java
import com.sparrowlogic.numpy4j.Reductions;
```

| NumPy Function                     | numpy4j Method            | Status | Tested |
|------------------------------------|---------------------------|--------|--------|
| `numpy.sum`                        | `Reductions.sum()`        | ✅      | ✅      |
| `numpy.prod`                       | `Reductions.prod()`       | ✅      | ✅      |
| `numpy.mean`                       | `Reductions.mean()`       | ✅      | ✅      |
| `numpy.std`                        | `Reductions.std()`        | ✅      | ✅      |
| `numpy.var`                        | `Reductions.var()`        | ✅      | ✅      |
| `numpy.min` / `numpy.amin`         | `Reductions.min()`        | ✅      | ✅      |
| `numpy.max` / `numpy.amax`         | `Reductions.max()`        | ✅      | ✅      |
| `numpy.argmin`                     | `Reductions.argmin()`     | ✅      | ✅      |
| `numpy.argmax`                     | `Reductions.argmax()`     | ✅      | ✅      |
| `numpy.cumsum`                     | `Reductions.cumsum()`     | ✅      | ✅      |
| `numpy.cumprod`                    | `Reductions.cumprod()`    | ✅      | ✅      |
| `numpy.all`                        | `Reductions.all()`        | ✅      | ✅      |
| `numpy.any`                        | `Reductions.any()`        | ✅      | ✅      |
| `numpy.nansum`                     | `Reductions.nansum()`     | ✅      | ✅      |
| `numpy.nanmean`                    | `Reductions.nanmean()`    | ✅      | ✅      |
| `numpy.nanstd`                     | `Reductions.nanstd()`     | ✅      | ✅      |
| `numpy.nanmax`                     | `Reductions.nanmax()`     | ✅      | ✅      |
| `numpy.nanmin`                     | `Reductions.nanmin()`     | ✅      | ✅      |
| `numpy.nanargmax`                  | `Reductions.nanargmax()`  | ✅      | ✅      |
| `numpy.nanargmin`                  | `Reductions.nanargmin()`  | ✅      | ✅      |
| `numpy.median`                     | `Reductions.median()`     | ✅      | ✅      |
| `numpy.percentile`                 | `Reductions.percentile()` | ✅      | ✅      |
| `numpy.histogram`                  | `Reductions.histogram()`  | ✅      | ✅      |
| `numpy.allclose`                   | `Reductions.allclose()`   | ✅      | ✅      |
| `numpy.array_equal`                | `Reductions.arrayEqual()` | ✅      | ✅      |
| `numpy.count_nonzero`              | `Reductions.countNonzero()` | ✅      | ✅      |
| `numpy.quantile`                   | `Reductions.quantile()`     | ✅      | ✅      |
| `numpy.average`                    | `Reductions.average()`      | ✅      | ✅      |
| `numpy.nanvar`                     | `Reductions.nanvar()`     | ✅      | ✅      |
| Axis-aware sum/max/min/mean/argmax | ✅                         | ✅      | ✅      |

### Shape Manipulation (`ShapeOps`)

```java
import com.sparrowlogic.numpy4j.ShapeOps;
```

| NumPy Function                  | numpy4j Method             | Status | Tested |
|---------------------------------|----------------------------|--------|--------|
| `numpy.reshape`                 | `ShapeOps.reshape()`       | ✅      | ✅      |
| `numpy.transpose`               | `ShapeOps.transpose()`     | ✅      | ✅      |
| `numpy.squeeze`                 | `ShapeOps.squeeze()`       | ✅      | ✅      |
| `numpy.expand_dims`             | `ShapeOps.expandDims()`    | ✅      | ✅      |
| `numpy.flatten` / `numpy.ravel` | `ShapeOps.flatten()`       | ✅      | ✅      |
| `numpy.concatenate`             | `ShapeOps.concatenate()`   | ✅      | ✅      |
| `numpy.stack`                   | `ShapeOps.stack()`         | ✅      | ✅      |
| `numpy.hstack`                  | `ShapeOps.hstack()`        | ✅      | ✅      |
| `numpy.vstack`                  | `ShapeOps.vstack()`        | ✅      | ✅      |
| `numpy.split`                   | `ShapeOps.split()`         | ✅      | ✅      |
| `numpy.tile`                    | `ShapeOps.tile()`          | ✅      | ✅      |
| `numpy.repeat`                  | `ShapeOps.repeat()`        | ✅      | ✅      |
| `numpy.flip`                    | `ShapeOps.flip()`          | ✅      | ✅      |
| `numpy.fliplr`                  | `ShapeOps.fliplr()`        | ✅      | ✅      |
| `numpy.flipud`                  | `ShapeOps.flipud()`        | ✅      | ✅      |
| `numpy.roll`                    | `ShapeOps.roll()`          | ✅      | ✅      |
| `numpy.pad`                     | `ShapeOps.pad()`           | ✅      | ✅      |
| `numpy.triu`                    | `ShapeOps.triu()`          | ✅      | ✅      |
| `numpy.tril`                    | `ShapeOps.tril()`          | ✅      | ✅      |
| `numpy.diag` (extract)          | `ShapeOps.diag()`          | ✅      | ✅      |
| `numpy.diag` (construct)        | `ShapeOps.diagConstruct()` | ✅      | ✅      |
| `numpy.diagonal`                | `ShapeOps.diagonal()`      | ✅      | ✅      |
| `numpy.trace`                   | `ShapeOps.trace()`         | ✅      | ✅      |
| `numpy.unique`                  | `ShapeOps.unique()`        | ✅      | ✅      |
| `numpy.diff`                    | `ShapeOps.diff()`          | ✅      | ✅      |
| `numpy.gradient`                | `ShapeOps.gradient()`      | ✅      | ✅      |
| `numpy.swapaxes`                | `ShapeOps.swapaxes()`      | ✅      | ✅      |
| `numpy.moveaxis`                | `ShapeOps.moveaxis()`      | ✅      | ✅      |
| `numpy.meshgrid`                | `ShapeOps.meshgrid()`      | ✅      | ✅      |
| `numpy.broadcast_to`            | `ShapeOps.broadcastTo()`   | ✅      | ✅      |
| `numpy.dstack`                  | `ShapeOps.dstack()`        | ✅      | ✅      |
| `numpy.hsplit` / `numpy.vsplit` | `ShapeOps.hsplit()` / `vsplit()` | ✅ | ✅      |
| `numpy.rot90`                   | `ShapeOps.rot90()`         | ✅      | ✅      |
| `numpy.insert` / `numpy.delete` | `ShapeOps.insert()` / `delete()` | ✅ | ✅      |
| `numpy.append`                  | `ShapeOps.append()`        | ✅      | ✅      |

### Indexing (`IndexOps`)

```java
import com.sparrowlogic.numpy4j.IndexOps;
```

| NumPy Function               | numpy4j Method            | Status | Tested |
|------------------------------|---------------------------|--------|--------|
| Slice (`a[start:stop:step]`) | `IndexOps.slice()`        | ✅      | ✅      |
| `numpy.take`                 | `IndexOps.take()`         | ✅      | ✅      |
| Boolean indexing             | `IndexOps.booleanIndex()` | ✅      | ✅      |
| `numpy.where`                | `IndexOps.where()`        | ✅      | ✅      |
| `numpy.sort`                 | `IndexOps.sort()`         | ✅      | ✅      |
| `numpy.argsort`              | `IndexOps.argsort()`      | ✅      | ✅      |
| `numpy.searchsorted`         | `IndexOps.searchsorted()` | ✅      | ✅      |
| `numpy.nonzero`              | `IndexOps.nonzero()`      | ✅      | ✅      |
| `numpy.argwhere`             | `IndexOps.argwhere()`     | ✅      | ✅      |
| `numpy.partition`            | `IndexOps.partition()`    | ✅      | ✅      |
| `numpy.argpartition`         | `IndexOps.argpartition()` | ✅      | ✅      |
| `numpy.put`                  | `IndexOps.put()`          | ✅      | ✅      |
| Fancy indexing (int array)   | `IndexOps.fancyIndex()`   | ✅      | ✅      |

### Linear Algebra (`MatMul` / `LinAlg`)

```java
import com.sparrowlogic.numpy4j.MatMul;
import com.sparrowlogic.numpy4j.LinAlg;
```

| NumPy Function                 | numpy4j Method           | Status | Tested |
|--------------------------------|--------------------------|--------|--------|
| `numpy.dot`                    | `MatMul.dot()`           | ✅      | ✅      |
| `numpy.matmul` / `@`           | `MatMul.matmul()`        | ✅      | ✅      |
| `numpy.outer`                  | `MatMul.outer()`         | ✅      | ✅      |
| `numpy.inner`                  | `MatMul.inner()`         | ✅      | ✅      |
| `numpy.tensordot`              | `MatMul.tensordot()`     | ✅      | ✅      |
| `numpy.cross`                  | `MatMul.cross()`         | ✅      | ✅      |
| Batched matmul (3D)            | `MatMul.batchedMatmul()` | ✅      | ✅      |
| `numpy.linalg.norm`            | `LinAlg.norm()`          | ✅      | ✅      |
| `numpy.linalg.inv`             | `LinAlg.inv()`           | ✅      | ✅      |
| `numpy.linalg.det`             | `LinAlg.det()`           | ✅      | ✅      |
| `numpy.linalg.solve`           | `LinAlg.solve()`         | ✅      | ✅      |
| `numpy.linalg.cholesky`        | `LinAlg.cholesky()`      | ✅      | ✅      |
| `numpy.linalg.qr`              | `LinAlg.qr()`            | ✅      | ✅      |
| `numpy.linalg.svd`             | `LinAlg.svd()`           | ✅      | ✅      |
| `numpy.linalg.eig` / `eigvals` | `LinAlg.eigvals()`       | ✅      | ✅      |
| `numpy.linalg.pinv`            | `LinAlg.pinv()`          | ✅      | ✅      |
| `numpy.linalg.matrix_rank`     | `LinAlg.matrixRank()`    | ✅      | ✅      |
| `numpy.linalg.slogdet`         | `LinAlg.slogdet()`       | ✅      | ✅      |
| `numpy.linalg.lstsq`           | `LinAlg.lstsq()`         | ✅      | ✅      |
| `numpy.linalg.matrix_power`    | `LinAlg.matrixPower()`   | ✅      | ✅      |
| `numpy.linalg.eigh`            | `LinAlg.eigh()`          | ✅      | ✅      |
| `numpy.linalg.cond`            | `LinAlg.cond()`          | ✅      | ✅      |
| `numpy.linalg.multi_dot`       | `MatMul.multiDot()`      | ✅      | ✅      |
| `numpy.vdot`                   | `MatMul.vdot()`          | ✅      | ✅      |

### FFT (`FFT`)

```java
import com.sparrowlogic.numpy4j.FFT;
```

| NumPy Function       | numpy4j Method   | Status | Tested |
|----------------------|------------------|--------|--------|
| `numpy.fft.fft`      | `FFT.fft()`      | ✅      | ✅      |
| `numpy.fft.ifft`     | `FFT.ifft()`     | ✅      | ✅      |
| `numpy.fft.rfft`     | `FFT.rfft()`     | ✅      | ✅      |
| `numpy.fft.irfft`    | `FFT.irfft()`    | ✅      | ✅      |
| `numpy.fft.fftfreq`  | `FFT.fftfreq()`  | ✅      | ✅      |
| `numpy.fft.rfftfreq` | `FFT.rfftfreq()` | ✅      | ✅      |
| `numpy.fft.fft2`     | `FFT.fft2()`     | ✅      | ✅      |
| `numpy.fft.ifft2`    | `FFT.ifft2()`    | ✅      | ✅      |
| `numpy.fft.fftn`     | `FFT.fftn()`     | ✅      | ✅      |
| `numpy.fft.fftshift` | `FFT.fftshift()` | ✅      | ✅      |

### ML Operations (`ML`)

```java
import com.sparrowlogic.numpy4j.ML;
```

| Function                  | numpy4j Method          | Status | Tested |
|---------------------------|-------------------------|--------|--------|
| Softmax                   | `ML.softmax()`          | ✅      | ✅      |
| Log-Softmax               | `ML.logSoftmax()`       | ✅      | ✅      |
| Cross-Entropy Loss        | `ML.crossEntropy()`     | ✅      | ✅      |
| Layer Normalization       | `ML.layerNorm()`        | ✅      | ✅      |
| Conv1d                    | `ML.conv1d()`           | ✅      | ✅      |
| Greedy Search             | `ML.greedySearch()`     | ✅      | ✅      |
| Beam Search               | `ML.beamSearch()`       | ✅      | ✅      |
| Top-K Filtering           | `ML.topK()`             | ✅      | ✅      |
| Top-P (Nucleus) Filtering | `ML.topP()`             | ✅      | ✅      |
| Temperature Scaling       | `ML.temperatureScale()` | ✅      | ✅      |

### Random (`Random`)

```java
import com.sparrowlogic.numpy4j.Random;
```

| NumPy Function              | numpy4j Method         | Status | Tested |
|-----------------------------|------------------------|--------|--------|
| `numpy.random.default_rng`  | `new Random(seed)`     | ✅      | ✅      |
| `Generator.uniform`         | `Random.uniform()`     | ✅      | ✅      |
| `Generator.normal`          | `Random.normal()`      | ✅      | ✅      |
| `Generator.integers`        | `Random.randint()`     | ✅      | ✅      |
| `Generator.choice`          | `Random.choice()`      | ✅      | ✅      |
| `Generator.shuffle`         | `Random.shuffle()`     | ✅      | ✅      |
| `Generator.permutation`     | `Random.permutation()` | ✅      | ✅      |
| `Generator.standard_normal` | `Random.standardNormal()` | ✅      | ✅      |
| `Generator.exponential`     | `Random.exponential()`    | ✅      | ✅      |
| `Generator.poisson`         | `Random.poisson()`        | ✅      | ✅      |
| `Generator.binomial`        | `Random.binomial()`       | ✅      | ✅      |

### Einsum (`Einsum`)

```java
import com.sparrowlogic.numpy4j.Einsum;
```

| Pattern                      | Status | Tested |
|------------------------------|--------|--------|
| `ij,jk->ik` (matmul)         | ✅      | ✅      |
| `ii->` (trace)               | ✅      | ✅      |
| `ij->ji` (transpose)         | ✅      | ✅      |
| `ij,ij->` (element-wise dot) | ✅      | ✅      |
| General binary contraction   | ✅      | ✅      |

### Signal Processing (`SignalOps`)

```java
import com.sparrowlogic.numpy4j.SignalOps;
```

| NumPy Function    | numpy4j Method          | Status | Tested |
|-------------------|-------------------------|--------|--------|
| `numpy.convolve`  | `SignalOps.convolve()`  | ✅      | ✅      |
| `numpy.correlate` | `SignalOps.correlate()` | ✅      | ✅      |
| `numpy.interp`    | `SignalOps.interp()`    | ✅      | ✅      |

---

## Parity Summary

| Category             | Implemented | Not Yet | Coverage |
|----------------------|-------------|---------|----------|
| Array Creation       | 15          | 0       | 100%     |
| Element-wise Ops     | 38          | 0       | 100%     |
| Comparison & Logical | 13          | 0       | 100%     |
| Reductions           | 29          | 0       | 100%     |
| Shape Manipulation   | 37          | 0       | 100%     |
| Indexing             | 13          | 0       | 100%     |
| Linear Algebra       | 24          | 0       | 100%     |
| FFT                  | 10          | 0       | 100%     |
| ML Ops               | 10          | 0       | 100%     |
| Random               | 11          | 0       | 100%     |
| Einsum               | 5           | 0       | 100%     |
| Signal               | 3           | 0       | 100%     |
| **Total**            | **208**     | **0**   | **100%** |

## Validation

All numerical outputs are validated against NumPy reference values generated by `tools/validate_against_reference.py`.

| Tolerance | Rationale                          |
|-----------|------------------------------------|
| `< 0.001` | FP32 rounding — all ops must match |

## License

MIT
