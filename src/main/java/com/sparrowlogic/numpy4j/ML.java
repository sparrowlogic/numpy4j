package com.sparrowlogic.numpy4j;

import java.lang.foreign.ValueLayout;
import java.util.Arrays;

/**
 * ML-specific operations: softmax, layer_norm, conv1d, greedy/beam search,
 * top-k/top-p sampling, temperature scaling, cross-entropy.
 */
public final class ML {
    private ML() {}

    // ── Softmax / LogSoftmax ──

    /** Softmax along last axis. Numerically stable (subtract max). */
    public static NdArray softmax(NdArray a) {
        return softmaxImpl(a, false);
    }

    public static NdArray logSoftmax(NdArray a) {
        return softmaxImpl(a, true);
    }

    private static NdArray softmaxImpl(NdArray a, boolean log) {
        NdArray c = a.contiguous();
        int lastDim = a.shape(a.ndim() - 1);
        long nSlices = a.size() / lastDim;
        NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, a.shape());

        for (long s = 0; s < nSlices; s++) {
            long base = s * lastDim;
            // Find max for numerical stability
            float max = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < lastDim; i++) max = Math.max(max, c.flatGetFloat(base + i));
            // Compute exp(x - max) and sum
            float sum = 0;
            for (int i = 0; i < lastDim; i++) {
                float e = (float) Math.exp(c.flatGetFloat(base + i) - max);
                out.flatSetFloat(base + i, e);
                sum += e;
            }
            // Normalize
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
        return out;
    }

    /** Cross-entropy loss: -sum(target * log(pred)) / n */
    public static float crossEntropy(NdArray logits, NdArray targets) {
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

    /** Layer normalization along last axis: (x - mean) / sqrt(var + eps) * gamma + beta */
    public static NdArray layerNorm(NdArray a, NdArray gamma, NdArray beta, float eps) {
        NdArray c = a.contiguous();
        int lastDim = a.shape(a.ndim() - 1);
        long nSlices = a.size() / lastDim;
        NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, a.shape());
        NdArray cg = gamma.contiguous(), cb = beta.contiguous();

        for (long s = 0; s < nSlices; s++) {
            long base = s * lastDim;
            float mean = 0;
            for (int i = 0; i < lastDim; i++) mean += c.flatGetFloat(base + i);
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

    /** 1D convolution: input [batch, inCh, L], weight [outCh, inCh, K], bias [outCh] */
    public static NdArray conv1d(NdArray input, NdArray weight, NdArray bias, int stride, int padding) {
        NdArray ci = input.contiguous(), cw = weight.contiguous();
        int batch = ci.shape(0), inCh = ci.shape(1), L = ci.shape(2);
        int outCh = cw.shape(0), K = cw.shape(2);
        int outL = (L + 2 * padding - K) / stride + 1;

        NdArray out = NdArrayFactory.zeros(input.arena(), DType.FLOAT32, batch, outCh, outL);
        NdArray cb = bias != null ? bias.contiguous() : null;

        for (int b = 0; b < batch; b++) {
            for (int oc = 0; oc < outCh; oc++) {
                float biasVal = cb != null ? cb.flatGetFloat(oc) : 0f;
                for (int ol = 0; ol < outL; ol++) {
                    float sum = biasVal;
                    for (int ic = 0; ic < inCh; ic++) {
                        for (int k = 0; k < K; k++) {
                            int il = ol * stride - padding + k;
                            if (il >= 0 && il < L) {
                                float iv = ci.flatGetFloat((long) b * inCh * L + (long) ic * L + il);
                                float wv = cw.flatGetFloat((long) oc * inCh * K + (long) ic * K + k);
                                sum += iv * wv;
                            }
                        }
                    }
                    out.flatSetFloat((long) b * outCh * outL + (long) oc * outL + ol, sum);
                }
            }
        }
        return out;
    }

    // ── Decoding algorithms ──

    /** Greedy search: argmax at each position along last axis. Returns INT32 indices. */
    public static NdArray greedySearch(NdArray logits) {
        return Reductions.argmax(logits, logits.ndim() - 1);
    }

    /** Top-K: keep only top k logits, set rest to -inf. Returns modified logits. */
    public static NdArray topK(NdArray logits, int k) {
        NdArray c = logits.contiguous();
        int lastDim = logits.shape(logits.ndim() - 1);
        long nSlices = logits.size() / lastDim;
        NdArray out = NdArrayFactory.empty(logits.arena(), DType.FLOAT32, logits.shape());

        float[] buf = new float[lastDim];
        for (long s = 0; s < nSlices; s++) {
            long base = s * lastDim;
            for (int i = 0; i < lastDim; i++) buf[i] = c.flatGetFloat(base + i);
            float[] sorted = buf.clone();
            Arrays.sort(sorted);
            float threshold = sorted[lastDim - k];
            for (int i = 0; i < lastDim; i++) {
                out.flatSetFloat(base + i, buf[i] >= threshold ? buf[i] : Float.NEGATIVE_INFINITY);
            }
        }
        return out;
    }

    /** Top-P (nucleus): keep smallest set of logits whose cumulative prob >= p. */
    public static NdArray topP(NdArray logits, float p) {
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
            for (int i = 0; i < lastDim; i++) { indices[i] = i; probBuf[i] = cp.flatGetFloat(base + i); }
            Arrays.sort(indices, (a, b) -> Float.compare(probBuf[b], probBuf[a]));
            float cumSum = 0;
            boolean[] keep = new boolean[lastDim];
            for (int idx : indices) {
                keep[idx] = true;
                cumSum += probBuf[idx];
                if (cumSum >= p) break;
            }
            for (int i = 0; i < lastDim; i++) {
                out.flatSetFloat(base + i, keep[i] ? c.flatGetFloat(base + i) : Float.NEGATIVE_INFINITY);
            }
        }
        return out;
    }

    /** Temperature scaling: logits / temperature */
    public static NdArray temperatureScale(NdArray logits, float temperature) {
        return Ufunc.divScalar(logits, temperature);
    }

    /** Beam search — returns top-k token sequences. Simplified single-step version. */
    public static int[][] beamSearch(NdArray logits, int beamWidth) {
        NdArray c = logits.contiguous();
        int vocabSize = logits.shape(logits.ndim() - 1);
        // Single step: find top beamWidth tokens
        float[] scores = new float[vocabSize];
        for (int i = 0; i < vocabSize; i++) scores[i] = c.flatGetFloat(i);

        Integer[] indices = new Integer[vocabSize];
        for (int i = 0; i < vocabSize; i++) indices[i] = i;
        Arrays.sort(indices, (a, b) -> Float.compare(scores[b], scores[a]));

        int[][] beams = new int[beamWidth][1];
        for (int i = 0; i < beamWidth; i++) beams[i][0] = indices[i];
        return beams;
    }
}
