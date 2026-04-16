package com.sparrowlogic.numpy4j;

import org.jspecify.annotations.Nullable;
import java.util.Arrays;

/**
 * ML-specific operations: softmax, layer_norm, conv1d, greedy/beam search,
 * top-k/top-p sampling, temperature scaling, cross-entropy.
     */
public final class ML {
    private ML() {
    }

    // ── Softmax / LogSoftmax ──

    /**
     * Softmax along last axis. Numerically stable (subtract max).
     *
     * @param a input array
     * @return result array
     */
    public static NdArray softmax(final NdArray a) {
        return softmaxImpl(a, false);
    }

    /**
     * logSoftmax operation.
     *
     * @param a input array
     * @return result array
     */
    public static NdArray logSoftmax(final NdArray a) {
        return softmaxImpl(a, true);
    }

    private static NdArray softmaxImpl(final NdArray a, final boolean log) {
        NdArray c = a.contiguous();
        int lastDim = a.shape(a.ndim() - 1);
        long nSlices = a.size() / lastDim;
        NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, a.shape());

        if (!log && AccelerateOps.isAvailable() && out.data().isNative()) {
            for (long i = 0; i < a.size(); i++) {
                out.flatSetFloat(i, c.flatGetFloat(i));
            }
            AccelerateOps.softmaxRows(out.data(), (int) nSlices, lastDim);
            return out;
        }

        softmaxScalar(c, out, nSlices, lastDim, log);
        return out;
    }

    private static void softmaxScalar(final NdArray c, final NdArray out,
                                      final long nSlices, final int lastDim, final boolean log) {
        for (long s = 0; s < nSlices; s++) {
            long base = s * lastDim;
            float max = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < lastDim; i++) {
                max = Math.max(max, c.flatGetFloat(base + i));
            }
            float sum = 0;
            for (int i = 0; i < lastDim; i++) {
                float e = (float) Math.exp(c.flatGetFloat(base + i) - max);
                out.flatSetFloat(base + i, e);
                sum += e;
            }
            if (log) {
                float logSum = (float) Math.log(sum);
                for (int i = 0; i < lastDim; i++) {
                    out.flatSetFloat(base + i, c.flatGetFloat(base + i) - max - logSum);
                }
            } else {
                for (int i = 0; i < lastDim; i++) {
                    out.flatSetFloat(base + i, out.flatGetFloat(base + i) / sum);
                }
            }
        }
    }

    /**
     * Cross-entropy loss: -sum(target * log(pred)) / n
     *
     * @param logits input logits
     * @param targets target distribution
     * @return the computed value
     */
    public static float crossEntropy(final NdArray logits, final NdArray targets) {
        NdArray logProbs = logSoftmax(logits);
        NdArray c = logProbs.contiguous();
        NdArray t = targets.contiguous();
        float loss = 0;
        for (long i = 0; i < logits.size(); i++) {
            loss -= t.flatGetFloat(i) * c.flatGetFloat(i);
        }
        return loss / (logits.size() / logits.shape(logits.ndim() - 1));
    }

    // ── Layer Norm ──

    /**
     * Layer normalization along last axis: (x - mean) / sqrt(var + eps) * gamma + beta
     *
     * @param a input array
     * @param gamma scale parameter
     * @param beta shift parameter
     * @param eps epsilon for numerical stability
     * @return result array
     */
    public static NdArray layerNorm(final NdArray a, final NdArray gamma, final NdArray beta, final float eps) {
        NdArray c = a.contiguous();
        int lastDim = a.shape(a.ndim() - 1);
        long nSlices = a.size() / lastDim;
        NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, a.shape());
        NdArray cg = gamma.contiguous();
        NdArray cb = beta.contiguous();

        for (long s = 0; s < nSlices; s++) {
            long base = s * lastDim;
            float mean = 0;
            for (int i = 0; i < lastDim; i++) {
                mean += c.flatGetFloat(base + i);
            }
            mean /= lastDim;
            float var = 0;
            for (int i = 0; i < lastDim; i++) {
                float d = c.flatGetFloat(base + i) - mean;
                var += d * d;
            }
            var /= lastDim;
            float invStd = (float) (1.0 / Math.sqrt(var + eps));
            for (int i = 0; i < lastDim; i++) {
                float normalized = (c.flatGetFloat(base + i) - mean) * invStd;
                out.flatSetFloat(base + i, normalized * cg.flatGetFloat(i) + cb.flatGetFloat(i));
            }
        }
        return out;
    }

    // ── Conv1d ──

    /**
     * 1D convolution: input [batch, inCh, L], weight [outCh, inCh, K], bias [outCh]
     *
     * @param input input tensor
     * @param weight convolution weights
     * @param bias optional bias (nullable)
     * @param stride convolution stride
     * @param padding convolution padding
     * @return result array
     */
    public static NdArray conv1d(
            final NdArray input,
            final NdArray weight,
            final @Nullable NdArray bias,
            final int stride,
            final int padding
    ) {
        NdArray ci = input.contiguous();
        NdArray cw = weight.contiguous();
        int batch = ci.shape(0);
        int inCh = ci.shape(1);
        int inLen = ci.shape(2);
        int outCh = cw.shape(0);
        int kernelSize = cw.shape(2);
        int outL = (inLen + 2 * padding - kernelSize) / stride + 1;

        NdArray out = NdArrayFactory.zeros(input.arena(), DType.FLOAT32, batch, outCh, outL);
        NdArray cb = bias != null ? bias.contiguous() : null;

        for (int b = 0; b < batch; b++) {
            for (int oc = 0; oc < outCh; oc++) {
                float biasVal = cb != null ? cb.flatGetFloat(oc) : 0f;
                Conv1dParams p = new Conv1dParams(ci, cw, out, inCh, inLen, kernelSize, outCh, outL, stride, padding);
                conv1dOutputChannel(p, b, oc, biasVal);
            }
        }
        return out;
    }

    // ── Decoding algorithms ──

    private static void conv1dOutputChannel(
            final Conv1dParams p, final int b, final int oc, final float biasVal
    ) {
        for (int ol = 0; ol < p.outL; ol++) {
            float sum = biasVal;
            for (int ic = 0; ic < p.inCh; ic++) {
                sum += conv1dKernel(p, b, oc, ol, ic);
            }
            p.out.flatSetFloat((long) b * p.outCh * p.outL + (long) oc * p.outL + ol, sum);
        }
    }

    private static float conv1dKernel(
            final Conv1dParams p, final int b, final int oc, final int ol, final int ic
    ) {
        float sum = 0;
        for (int k = 0; k < p.kernelSize; k++) {
            int il = ol * p.stride - p.padding + k;
            if (il >= 0 && il < p.inLen) {
                float iv = p.ci.flatGetFloat((long) b * p.inCh * p.inLen + (long) ic * p.inLen + il);
                float wv = p.cw.flatGetFloat((long) oc * p.inCh * p.kernelSize + (long) ic * p.kernelSize + k);
                sum += iv * wv;
            }
        }
        return sum;
    }

    /**
     * Greedy search: argmax at each position along last axis. Returns INT32 indices.
     *
     * @param logits input logits
     * @return result array
     */
    public static NdArray greedySearch(final NdArray logits) {
        return Reductions.argmax(logits, logits.ndim() - 1);
    }

    /**
     * Top-K: keep only top k logits, set rest to -inf. Returns modified logits.
     *
     * @param logits input logits
     * @param k number of top elements
     * @return result array
     */
    public static NdArray topK(final NdArray logits, final int k) {
        NdArray c = logits.contiguous();
        int lastDim = logits.shape(logits.ndim() - 1);
        long nSlices = logits.size() / lastDim;
        NdArray out = NdArrayFactory.empty(logits.arena(), DType.FLOAT32, logits.shape());

        float[] buf = new float[lastDim];
        for (long s = 0; s < nSlices; s++) {
            long base = s * lastDim;
            for (int i = 0; i < lastDim; i++) {
                buf[i] = c.flatGetFloat(base + i);
            }
            float[] sorted = buf.clone();
            Arrays.sort(sorted);
            float threshold = sorted[lastDim - k];
            for (int i = 0; i < lastDim; i++) {
                out.flatSetFloat(base + i, buf[i] >= threshold ? buf[i] : Float.NEGATIVE_INFINITY);
            }
        }
        return out;
    }

    /**
     * Top-P (nucleus): keep smallest set of logits whose cumulative prob >= p.
     *
     * @param logits input logits
     * @param p cumulative probability threshold
     * @return result array
     */
    public static NdArray topP(final NdArray logits, final float p) {
        NdArray probs = softmax(logits);
        NdArray cp = probs.contiguous();
        NdArray c = logits.contiguous();
        int lastDim = logits.shape(logits.ndim() - 1);
        long nSlices = logits.size() / lastDim;
        NdArray out = NdArrayFactory.empty(logits.arena(), DType.FLOAT32, logits.shape());

        Integer[] indices = new Integer[lastDim];
        float[] probBuf = new float[lastDim];
        for (long s = 0; s < nSlices; s++) {
            long base = s * lastDim;
            for (int i = 0; i < lastDim; i++) {
                indices[i] = i;
                probBuf[i] = cp.flatGetFloat(base + i);
            }
            Arrays.sort(indices, (a, b) -> Float.compare(probBuf[b], probBuf[a]));
            float cumSum = 0;
            boolean[] keep = new boolean[lastDim];
            for (final int idx : indices) {
                keep[idx] = true;
                cumSum += probBuf[idx];
                if (cumSum >= p) {
                    break;
                }
            }
            for (int i = 0; i < lastDim; i++) {
                out.flatSetFloat(base + i, keep[i] ? c.flatGetFloat(base + i) : Float.NEGATIVE_INFINITY);
            }
        }
        return out;
    }

    /**
     * Temperature scaling: logits / temperature
     *
     * @param logits input logits
     * @param temperature temperature scaling factor
     * @return result array
     */
    public static NdArray temperatureScale(final NdArray logits, final float temperature) {
        return Ufunc.divScalar(logits, temperature);
    }

    /**
     * Beam search — returns top-k token sequences. Simplified single-step version.
     *
     * @param logits input logits
     * @param beamWidth beam width
     * @return result array
     */
    public static int[][] beamSearch(final NdArray logits, final int beamWidth) {
        NdArray c = logits.contiguous();
        int vocabSize = logits.shape(logits.ndim() - 1);
        // Single step: find top beamWidth tokens
        float[] scores = new float[vocabSize];
        for (int i = 0; i < vocabSize; i++) {
            scores[i] = c.flatGetFloat(i);
        }

        Integer[] indices = new Integer[vocabSize];
        for (int i = 0; i < vocabSize; i++) {
            indices[i] = i;
        }
        Arrays.sort(indices, (a, b) -> Float.compare(scores[b], scores[a]));

        int[][] beams = new int[beamWidth][1];
        for (int i = 0; i < beamWidth; i++) {
            beams[i][0] = indices[i];
        }
        return beams;
    }

    private record Conv1dParams(NdArray ci, NdArray cw, NdArray out,
                                int inCh, int inLen, int kernelSize, int outCh, int outL,
                                int stride, int padding) {
    }
}
