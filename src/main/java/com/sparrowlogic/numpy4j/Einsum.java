package com.sparrowlogic.numpy4j;

/**
 * Basic einsum implementation supporting common patterns:
 * - "ij,jk->ik" (matmul)
 * - "ii->" (trace)
 * - "ij->ji" (transpose)
 * - "ij,ij->" (element-wise dot)
 */
public final class Einsum {
    private Einsum() {}

    public static NdArray einsum(String subscripts, NdArray... operands) {
        String[] parts = subscripts.split("->");
        String lhs = parts[0];
        String rhs = parts.length > 1 ? parts[1] : "";
        String[] inputs = lhs.split(",");

        if (operands.length == 1) return einsumUnary(inputs[0], rhs, operands[0]);
        if (operands.length == 2) return einsumBinary(inputs[0], inputs[1], rhs, operands[0], operands[1]);
        throw new UnsupportedOperationException("Einsum supports 1 or 2 operands");
    }

    private static NdArray einsumUnary(String input, String output, NdArray a) {
        // "ii->" → trace
        if (input.length() == 2 && input.charAt(0) == input.charAt(1) && output.isEmpty()) {
            NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, 1);
            out.flatSetFloat(0, ShapeOps.trace(a));
            return out;
        }
        // "ij->ji" → transpose
        if (input.length() == 2 && output.length() == 2
                && input.charAt(0) == output.charAt(1) && input.charAt(1) == output.charAt(0)) {
            return ShapeOps.transpose(a).contiguous();
        }
        throw new UnsupportedOperationException("Unsupported unary einsum: " + input + "->" + output);
    }

    private static NdArray einsumBinary(String inA, String inB, String output, NdArray a, NdArray b) {
        // "ij,jk->ik" → matmul
        if (inA.equals("ij") && inB.equals("jk") && output.equals("ik")) {
            return MatMul.matmul(a, b);
        }
        // "ij,ij->" → element-wise sum of products
        if (inA.equals(inB) && output.isEmpty()) {
            NdArray prod = Ufunc.multiply(a, b);
            NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, 1);
            out.flatSetFloat(0, Reductions.sum(prod));
            return out;
        }
        // General case: brute-force contraction
        return generalBinaryEinsum(inA, inB, output, a, b);
    }

    private static NdArray generalBinaryEinsum(String inA, String inB, String output, NdArray a, NdArray b) {
        // Build index map: each unique letter → range
        java.util.LinkedHashMap<Character, Integer> indexSizes = new java.util.LinkedHashMap<>();
        for (int i = 0; i < inA.length(); i++) indexSizes.put(inA.charAt(i), a.shape(i));
        for (int i = 0; i < inB.length(); i++) indexSizes.put(inB.charAt(i), b.shape(i));

        // Output shape
        int[] outShape = new int[output.length()];
        for (int i = 0; i < output.length(); i++) outShape[i] = indexSizes.get(output.charAt(i));

        // Contracted indices = all indices not in output
        java.util.List<Character> contracted = new java.util.ArrayList<>();
        for (char c : indexSizes.keySet()) if (output.indexOf(c) < 0) contracted.add(c);

        NdArray out = NdArrayFactory.zeros(a.arena(), DType.FLOAT32, outShape);
        NdArray ca = a.contiguous(), cb = b.contiguous();
        int[] aStrides = ShapeUtils.cStrides(a.shape());
        int[] bStrides = ShapeUtils.cStrides(b.shape());
        int[] oStrides = ShapeUtils.cStrides(outShape);

        // Iterate over all output indices
        long outSize = ShapeUtils.size(outShape);
        int[] contractedSizes = new int[contracted.size()];
        for (int i = 0; i < contracted.size(); i++) contractedSizes[i] = indexSizes.get(contracted.get(i));
        long contractedTotal = ShapeUtils.size(contractedSizes);

        for (long oi = 0; oi < outSize; oi++) {
            int[] outIdx = ShapeUtils.unravelIndex(oi, outShape);
            java.util.HashMap<Character, Integer> idxMap = new java.util.HashMap<>();
            for (int i = 0; i < output.length(); i++) idxMap.put(output.charAt(i), outIdx[i]);

            float sum = 0;
            for (long ci = 0; ci < contractedTotal; ci++) {
                int[] cIdx = ShapeUtils.unravelIndex(ci, contractedSizes);
                for (int i = 0; i < contracted.size(); i++) idxMap.put(contracted.get(i), cIdx[i]);

                long aFlat = 0;
                for (int i = 0; i < inA.length(); i++) aFlat += (long) idxMap.get(inA.charAt(i)) * aStrides[i];
                long bFlat = 0;
                for (int i = 0; i < inB.length(); i++) bFlat += (long) idxMap.get(inB.charAt(i)) * bStrides[i];

                sum += ca.flatGetFloat(aFlat) * cb.flatGetFloat(bFlat);
            }
            out.flatSetFloat(oi, sum);
        }
        return out;
    }
}
