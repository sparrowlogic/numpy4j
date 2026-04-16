package com.sparrowlogic.numpy4j;

import org.jspecify.annotations.Nullable;

/**
 * Basic einsum implementation supporting common patterns:
 * - "ij,jk->ik" (matmul)
 * - "ii->" (trace)
 * - "ij->ji" (transpose)
 * - "ij,ij->" (element-wise dot)
 */
public final class Einsum {
    private Einsum() {
    }

    /**
     * Einstein summation ({@code numpy.einsum}).
     *
     * <p>Supported patterns include matmul ({@code "ij,jk->ik"}), trace ({@code "ii->"}),
     * transpose ({@code "ij->ji"}), element-wise dot ({@code "ij,ij->"}),
     * and general binary contractions.</p>
     *
     * @param subscripts einsum subscript string (e.g. {@code "ij,jk->ik"})
     * @param operands   one or two input arrays
     * @return the result of the einsum contraction
     * @throws UnsupportedOperationException if the pattern is not supported
     */
    public static NdArray einsum(final String subscripts, final NdArray... operands) {
        String[] parts = subscripts.split("->");
        String lhs = parts[0];
        String rhs = parts.length > 1 ? parts[1] : "";
        String[] inputs = lhs.split(",");

        if (operands.length == 1) {
            return einsumUnary(inputs[0], rhs, operands[0]);
        }
        if (operands.length == 2) {
            return einsumBinary(inputs[0], inputs[1], rhs, operands[0], operands[1]);
        }
        throw new UnsupportedOperationException("Einsum supports 1 or 2 operands");
    }

    private static NdArray einsumUnary(final String input, final String output, final NdArray a) {
        if (isTrace(input, output)) {
            NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, 1);
            out.flatSetFloat(0, ShapeOps.trace(a));
            return out;
        }
        if (isTranspose(input, output)) {
            return ShapeOps.transpose(a).contiguous();
        }
        throw new UnsupportedOperationException("Unsupported unary einsum: " + input + "->" + output);
    }

    private static boolean isTrace(final String input, final String output) {
        return input.length() == 2 && input.charAt(0) == input.charAt(1) && output.isEmpty();
    }

    private static boolean isTranspose(final String input, final String output) {
        return input.length() == 2 && output.length() == 2
                && input.charAt(0) == output.charAt(1) && input.charAt(1) == output.charAt(0);
    }

    private static NdArray einsumBinary(
            final String inA,
            final String inB,
            final String output,
            final NdArray a,
            final NdArray b
    ) {
        NdArray fast = tryFastBinary(inA, inB, output, a, b);
        return fast != null ? fast : generalBinaryEinsum(inA, inB, output, a, b);
    }

    private static @Nullable NdArray tryFastBinary(final String inA, final String inB, final String output,
                                         final NdArray a, final NdArray b) {
        if ("ij".equals(inA) && "jk".equals(inB) && "ik".equals(output)) {
            return MatMul.matmul(a, b);
        }
        return inA.equals(inB) && output.isEmpty() ? elementWiseDot(a, b) : null;
    }

    private static NdArray elementWiseDot(final NdArray a, final NdArray b) {
        NdArray prod = Ufunc.multiply(a, b);
        NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, 1);
        out.flatSetFloat(0, Reductions.sum(prod));
        return out;
    }

    private static NdArray generalBinaryEinsum(
            final String inA,
            final String inB,
            final String output,
            final NdArray a,
            final NdArray b
    ) {
        EinsumContext ctx = buildContext(inA, inB, output, a, b);
        NdArray out = NdArrayFactory.zeros(a.arena(), DType.FLOAT32, ctx.outShape);

        for (long oi = 0; oi < ctx.outSize; oi++) {
            float sum = contractedSum(ctx, oi);
            out.flatSetFloat(oi, sum);
        }
        return out;
    }

    private static float contractedSum(final EinsumContext ctx, final long oi) {
        int[] outIdx = ShapeUtils.unravelIndex(oi, ctx.outShape);
        java.util.Map<Character, Integer> idxMap = new java.util.HashMap<>();
        for (int i = 0; i < ctx.output.length(); i++) {
            idxMap.put(ctx.output.charAt(i), outIdx[i]);
        }

        float sum = 0;
        for (long ci = 0; ci < ctx.contractedTotal; ci++) {
            int[] cIdx = ShapeUtils.unravelIndex(ci, ctx.contractedSizes);
            for (int i = 0; i < ctx.contracted.size(); i++) {
                idxMap.put(ctx.contracted.get(i), cIdx[i]);
            }
            long aFlat = computeFlat(idxMap, ctx.inA, ctx.aStrides);
            long bFlat = computeFlat(idxMap, ctx.inB, ctx.bStrides);
            sum += ctx.ca.flatGetFloat(aFlat) * ctx.cb.flatGetFloat(bFlat);
        }
        return sum;
    }

    private static long computeFlat(final java.util.Map<Character, Integer> idxMap,
                                    final String subscript, final int[] strides) {
        long flat = 0;
        for (int i = 0; i < subscript.length(); i++) {
            flat += (long) idxMap.get(subscript.charAt(i)) * strides[i];
        }
        return flat;
    }

    private static EinsumContext buildContext(final String inA, final String inB, final String output,
                                             final NdArray a, final NdArray b) {
        java.util.Map<Character, Integer> indexSizes = new java.util.LinkedHashMap<>();
        for (int i = 0; i < inA.length(); i++) {
            indexSizes.put(inA.charAt(i), a.shape(i));
        }
        for (int i = 0; i < inB.length(); i++) {
            indexSizes.put(inB.charAt(i), b.shape(i));
        }

        int[] outShape = new int[output.length()];
        for (int i = 0; i < output.length(); i++) {
            outShape[i] = indexSizes.get(output.charAt(i));
        }

        java.util.List<Character> contracted = new java.util.ArrayList<>();
        for (final char c : indexSizes.keySet()) {
            if (output.indexOf(c) < 0) {
                contracted.add(c);
            }
        }

        int[] contractedSizes = new int[contracted.size()];
        for (int i = 0; i < contracted.size(); i++) {
            contractedSizes[i] = indexSizes.get(contracted.get(i));
        }

        return new EinsumContext(inA, inB, output, outShape, contracted, contractedSizes,
                ShapeUtils.size(outShape), ShapeUtils.size(contractedSizes),
                a.contiguous(), b.contiguous(),
                ShapeUtils.cStrides(a.shape()), ShapeUtils.cStrides(b.shape()));
    }

    private record EinsumContext(
            String inA, String inB, String output,
            int[] outShape,
            java.util.List<Character> contracted,
            int[] contractedSizes,
            long outSize, long contractedTotal,
            NdArray ca, NdArray cb,
            int[] aStrides, int[] bStrides
    ) {
    }
}
