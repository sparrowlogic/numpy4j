package com.sparrowlogic.numpy4j;

import org.junit.jupiter.api.Test;
import java.lang.foreign.Arena;
import static org.junit.jupiter.api.Assertions.*;

public class ShapeOpsTest {
    private static final float T = 0.001f;
    private static void assertArr(String l, NdArray a, float[] e) { float[] ac = a.toFloatArray(); for (int i = 0; i < e.length; i++) assertEquals(e[i], ac[i], T, l+"["+i+"]"); }

    @Test void reshape() { try (var a = Arena.ofConfined()) { assertArrayEquals(new int[]{3,4}, ShapeOps.reshape(NdArrayFactory.arange(a, 12), 3, 4).shape()); } }
    @Test void reshapeInfer() { try (var a = Arena.ofConfined()) { assertArrayEquals(new int[]{3,4}, ShapeOps.reshape(NdArrayFactory.arange(a, 12), 3, -1).shape()); } }
    @Test void transpose() { try (var a = Arena.ofConfined()) { NdArray t = ShapeOps.transpose(ShapeOps.reshape(NdArrayFactory.arange(a, 6), 2, 3)).contiguous(); assertArr("T", t, new float[]{0,3,1,4,2,5}); } }
    @Test void squeeze() { try (var a = Arena.ofConfined()) { assertArrayEquals(new int[]{3}, ShapeOps.squeeze(ShapeOps.reshape(NdArrayFactory.arange(a, 3), 1, 3, 1)).shape()); } }
    @Test void expandDims() { try (var a = Arena.ofConfined()) { assertArrayEquals(new int[]{1,3}, ShapeOps.expandDims(NdArrayFactory.arange(a, 3), 0).shape()); } }
    @Test void flatten() { try (var a = Arena.ofConfined()) { assertArrayEquals(new int[]{6}, ShapeOps.flatten(ShapeOps.reshape(NdArrayFactory.arange(a, 6), 2, 3)).shape()); } }
    @Test void concatenate() { try (var a = Arena.ofConfined()) { assertArr("cat", ShapeOps.concatenate(new NdArray[]{NdArrayFactory.array(a, new float[]{1,2}), NdArrayFactory.array(a, new float[]{3,4})}, 0), new float[]{1,2,3,4}); } }
    @Test void stack() { try (var a = Arena.ofConfined()) { assertArrayEquals(new int[]{2,2}, ShapeOps.stack(new NdArray[]{NdArrayFactory.array(a, new float[]{1,2}), NdArrayFactory.array(a, new float[]{3,4})}, 0).shape()); } }
    @Test void tile() { try (var a = Arena.ofConfined()) { assertArr("tile", ShapeOps.tile(NdArrayFactory.array(a, new float[]{1,2}), 3), new float[]{1,2,1,2,1,2}); } }
    @Test void repeat() { try (var a = Arena.ofConfined()) { assertArr("rep", ShapeOps.repeat(NdArrayFactory.array(a, new float[]{1,2,3}), 2, 0), new float[]{1,1,2,2,3,3}); } }
    @Test void flip() { try (var a = Arena.ofConfined()) { assertArr("flip", ShapeOps.flip(NdArrayFactory.arange(a, 6)), new float[]{5,4,3,2,1,0}); } }
    @Test void fliplr() { try (var a = Arena.ofConfined()) { assertArr("fliplr", ShapeOps.fliplr(ShapeOps.reshape(NdArrayFactory.arange(a, 9), 3, 3)), new float[]{2,1,0,5,4,3}); } }
    @Test void flipud() { try (var a = Arena.ofConfined()) { assertArr("flipud", ShapeOps.flipud(ShapeOps.reshape(NdArrayFactory.arange(a, 9), 3, 3)), new float[]{6,7,8,3,4,5}); } }
    @Test void roll() { try (var a = Arena.ofConfined()) { assertArr("roll", ShapeOps.roll(NdArrayFactory.arange(a, 6), 2), new float[]{4,5,0,1,2,3}); } }
    @Test void pad() { try (var a = Arena.ofConfined()) { NdArray p = ShapeOps.pad(NdArrayFactory.array(a, new float[]{1,2,3}), 2, 3, 0f); assertEquals(8, p.size()); assertArr("pad", p, new float[]{0,0,1,2,3,0}); } }
    @Test void triu() { try (var a = Arena.ofConfined()) { assertArr("triu", ShapeOps.triu(ShapeOps.reshape(NdArrayFactory.arange(a, 9), 3, 3)), new float[]{0,1,2,0,4,5}); } }
    @Test void tril() { try (var a = Arena.ofConfined()) { assertArr("tril", ShapeOps.tril(ShapeOps.reshape(NdArrayFactory.arange(a, 9), 3, 3)), new float[]{0,0,0,3,4,0}); } }
    @Test void trace() { try (var a = Arena.ofConfined()) { assertEquals(5f, ShapeOps.trace(NdArrayFactory.array(a, new float[]{1,2,3,4}, 2, 2)), T); } }
    @Test void diagonal() { try (var a = Arena.ofConfined()) { assertArr("diag", ShapeOps.diagonal(NdArrayFactory.array(a, new float[]{1,2,3,4}, 2, 2)), new float[]{1,4}); } }
    @Test void diagExtract() { try (var a = Arena.ofConfined()) { assertArr("d3", ShapeOps.diag(ShapeOps.reshape(NdArrayFactory.arange(a, 9), 3, 3)), new float[]{0,4,8}); } }
    @Test void diagConstruct() { try (var a = Arena.ofConfined()) { assertArr("dc", ShapeOps.diagConstruct(NdArrayFactory.array(a, new float[]{1,2,3})), new float[]{1,0,0,0,2,0}); } }
    @Test void unique() { try (var a = Arena.ofConfined()) { assertArr("u", ShapeOps.unique(NdArrayFactory.array(a, new float[]{3,1,2,1,3,2})), new float[]{1,2,3}); } }
    @Test void diff() { try (var a = Arena.ofConfined()) { assertArr("diff", ShapeOps.diff(NdArrayFactory.array(a, new float[]{1,3,6,10})), new float[]{2,3,4}); } }
    @Test void gradient() { try (var a = Arena.ofConfined()) { assertArr("grad", ShapeOps.gradient(NdArrayFactory.array(a, new float[]{1,3,6,10})), new float[]{2,2.5f,3.5f,4}); } }
    @Test void split() { try (var a = Arena.ofConfined()) { NdArray[] p = ShapeOps.split(NdArrayFactory.arange(a, 9), 3); assertArr("s0", p[0], new float[]{0,1,2}); } }
    @Test void hstack() { try (var a = Arena.ofConfined()) { assertArr("hs", ShapeOps.hstack(NdArrayFactory.array(a, new float[]{1,2,3}), NdArrayFactory.array(a, new float[]{4,5,6})), new float[]{1,2,3,4,5,6}); } }
    @Test void vstack() { try (var a = Arena.ofConfined()) { assertArrayEquals(new int[]{2,3}, ShapeOps.vstack(NdArrayFactory.array(a, new float[]{1,2,3}), NdArrayFactory.array(a, new float[]{4,5,6})).shape()); } }
    @Test void swapaxes() { try (var a = Arena.ofConfined()) { assertArr("sw", ShapeOps.swapaxes(ShapeOps.reshape(NdArrayFactory.arange(a, 24), 2, 3, 4), 0, 2).contiguous(), new float[]{0,12,4,16,8,20}); } }
    @Test void moveaxis() { try (var a = Arena.ofConfined()) { assertArr("mv", ShapeOps.moveaxis(ShapeOps.reshape(NdArrayFactory.arange(a, 24), 2, 3, 4), 0, 2).contiguous(), new float[]{0,12,1,13,2,14}); } }
    @Test void meshgrid() { try (var a = Arena.ofConfined()) { NdArray[] g = ShapeOps.meshgrid(NdArrayFactory.array(a, new float[]{1,2,3}), NdArrayFactory.array(a, new float[]{4,5})); assertArr("X", g[0], new float[]{1,2,3,1,2,3}); } }
    @Test void broadcastTo() { try (var a = Arena.ofConfined()) { assertArr("bc", ShapeOps.broadcastTo(NdArrayFactory.array(a, new float[]{1,2,3}), 2, 3), new float[]{1,2,3,1,2,3}); } }
    @Test void rot90() { try (var a = Arena.ofConfined()) { assertArr("rot", ShapeOps.rot90(NdArrayFactory.array(a, new float[]{1,2,3,4}, 2, 2)), new float[]{2,4,1,3}); } }
    @Test void insertOp() { try (var a = Arena.ofConfined()) { assertArr("ins", ShapeOps.insert(NdArrayFactory.array(a, new float[]{1,2,3,4}), 2, 99f), new float[]{1,2,99,3,4}); } }
    @Test void deleteOp() { try (var a = Arena.ofConfined()) { assertArr("del", ShapeOps.delete(NdArrayFactory.array(a, new float[]{1,2,3,4}), 1), new float[]{1,3,4}); } }
    @Test void appendOp() { try (var a = Arena.ofConfined()) { assertArr("app", ShapeOps.append(NdArrayFactory.array(a, new float[]{1,2,3}), NdArrayFactory.array(a, new float[]{4,5,6})), new float[]{1,2,3,4,5,6}); } }
    @Test void hsplitOp() { try (var a = Arena.ofConfined()) { assertArr("hs0", ShapeOps.hsplit(NdArrayFactory.arange(a, 6), 3)[0], new float[]{0,1}); } }
    @Test void vsplitOp() { try (var a = Arena.ofConfined()) { assertArr("vs0", ShapeOps.vsplit(ShapeOps.reshape(NdArrayFactory.arange(a, 12), 4, 3), 2)[0], new float[]{0,1,2,3,4,5}); } }
}
