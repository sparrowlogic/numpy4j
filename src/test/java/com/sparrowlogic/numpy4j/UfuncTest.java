package com.sparrowlogic.numpy4j;

import org.junit.jupiter.api.Test;
import java.lang.foreign.Arena;
import static org.junit.jupiter.api.Assertions.*;

public class UfuncTest {
    private static final float T = 0.001f;
    private static void assertArr(String l, NdArray a, float[] e) { float[] ac = a.toFloatArray(); for (int i = 0; i < e.length; i++) assertEquals(e[i], ac[i], T, l+"["+i+"]"); }

    @Test void add() { try (var a = Arena.ofConfined()) { assertArr("add", Ufunc.add(NdArrayFactory.array(a, new float[]{1,2,3,4}), NdArrayFactory.array(a, new float[]{5,6,7,8})), new float[]{6,8,10,12}); } }
    @Test void subtract() { try (var a = Arena.ofConfined()) { assertArr("sub", Ufunc.subtract(NdArrayFactory.array(a, new float[]{5,6}), NdArrayFactory.array(a, new float[]{1,2})), new float[]{4,4}); } }
    @Test void multiply() { try (var a = Arena.ofConfined()) { assertArr("mul", Ufunc.multiply(NdArrayFactory.array(a, new float[]{1,2,3,4}), NdArrayFactory.array(a, new float[]{5,6,7,8})), new float[]{5,12,21,32}); } }
    @Test void divide() { try (var a = Arena.ofConfined()) { assertArr("div", Ufunc.divide(NdArrayFactory.array(a, new float[]{1,2}), NdArrayFactory.array(a, new float[]{5,6})), new float[]{0.2f,0.33333334f}); } }
    @Test void neg() { try (var a = Arena.ofConfined()) { assertArr("neg", Ufunc.neg(NdArrayFactory.array(a, new float[]{1,-2})), new float[]{-1,2}); } }
    @Test void abs() { try (var a = Arena.ofConfined()) { assertArr("abs", Ufunc.abs(NdArrayFactory.array(a, new float[]{-1,-2,3})), new float[]{1,2,3}); } }
    @Test void sqrt() { try (var a = Arena.ofConfined()) { assertArr("sqrt", Ufunc.sqrt(NdArrayFactory.array(a, new float[]{1,4,9})), new float[]{1,2,3}); } }
    @Test void exp() { try (var a = Arena.ofConfined()) { assertArr("exp", Ufunc.exp(NdArrayFactory.array(a, new float[]{0,1})), new float[]{1f,2.7182817f}); } }
    @Test void log() { try (var a = Arena.ofConfined()) { assertArr("log", Ufunc.log(NdArrayFactory.array(a, new float[]{1,2.7182817f})), new float[]{0f,1f}); } }
    @Test void sin() { try (var a = Arena.ofConfined()) { assertEquals(1f, Ufunc.sin(NdArrayFactory.array(a, new float[]{(float)(Math.PI/2)})).flatGetFloat(0), T); } }
    @Test void cos() { try (var a = Arena.ofConfined()) { assertEquals(1f, Ufunc.cos(NdArrayFactory.array(a, new float[]{0})).flatGetFloat(0), T); } }
    @Test void tan() { try (var a = Arena.ofConfined()) { assertArr("tan", Ufunc.tan(NdArrayFactory.array(a, new float[]{0,0.5f,1})), new float[]{0f,0.5463025f,1.5574077f}); } }
    @Test void tanh() { try (var a = Arena.ofConfined()) { assertArr("tanh", Ufunc.tanh(NdArrayFactory.array(a, new float[]{0,1})), new float[]{0f,0.7615942f}); } }
    @Test void power() { try (var a = Arena.ofConfined()) { assertArr("pow", Ufunc.power(NdArrayFactory.array(a, new float[]{2,3}), NdArrayFactory.array(a, new float[]{3,2})), new float[]{8,9}); } }
    @Test void mod() { try (var a = Arena.ofConfined()) { assertArr("mod", Ufunc.mod(NdArrayFactory.array(a, new float[]{7,8,9}), NdArrayFactory.array(a, new float[]{3,3,3})), new float[]{1,2,0}); } }
    @Test void clip() { try (var a = Arena.ofConfined()) { assertArr("clip", Ufunc.clip(NdArrayFactory.array(a, new float[]{1,2,3,4}), 1.5f, 3.5f), new float[]{1.5f,2,3,3.5f}); } }
    @Test void maximum() { try (var a = Arena.ofConfined()) { assertArr("max", Ufunc.maximum(NdArrayFactory.array(a, new float[]{1,5,3}), NdArrayFactory.array(a, new float[]{4,2,6})), new float[]{4,5,6}); } }
    @Test void minimum() { try (var a = Arena.ofConfined()) { assertArr("min", Ufunc.minimum(NdArrayFactory.array(a, new float[]{1,5,3}), NdArrayFactory.array(a, new float[]{4,2,6})), new float[]{1,2,3}); } }
    @Test void gelu() { try (var a = Arena.ofConfined()) { assertArr("gelu", Ufunc.gelu(NdArrayFactory.array(a, new float[]{1,2})), new float[]{0.8411920f,1.9545977f}); } }
    @Test void sigmoid() { try (var a = Arena.ofConfined()) { assertEquals(0.5f, Ufunc.sigmoid(NdArrayFactory.array(a, new float[]{0})).flatGetFloat(0), T); } }
    @Test void relu() { try (var a = Arena.ofConfined()) { assertArr("relu", Ufunc.relu(NdArrayFactory.array(a, new float[]{-1,0,1,2})), new float[]{0,0,1,2}); } }
    @Test void reciprocal() { try (var a = Arena.ofConfined()) { assertArr("recip", Ufunc.reciprocal(NdArrayFactory.array(a, new float[]{2,4,5})), new float[]{0.5f,0.25f,0.2f}); } }
    @Test void square() { try (var a = Arena.ofConfined()) { assertArr("sq", Ufunc.square(NdArrayFactory.array(a, new float[]{1,2,3,4})), new float[]{1,4,9,16}); } }
    @Test void exp2() { try (var a = Arena.ofConfined()) { assertArr("exp2", Ufunc.exp2(NdArrayFactory.array(a, new float[]{0,1,2,3})), new float[]{1,2,4,8}); } }
    @Test void floorDivide() { try (var a = Arena.ofConfined()) { assertArr("fd", Ufunc.floorDivide(NdArrayFactory.array(a, new float[]{7,8,9}), NdArrayFactory.array(a, new float[]{3,3,3})), new float[]{2,2,3}); } }
    @Test void fmodOp() { try (var a = Arena.ofConfined()) { assertArr("fmod", Ufunc.fmod(NdArrayFactory.array(a, new float[]{7,8,9}), NdArrayFactory.array(a, new float[]{3,3,3})), new float[]{1,2,0}); } }
    @Test void floor() { try (var a = Arena.ofConfined()) { assertArr("floor", Ufunc.floor(NdArrayFactory.array(a, new float[]{1.7f,-1.7f})), new float[]{1,-2}); } }
    @Test void ceil() { try (var a = Arena.ofConfined()) { assertArr("ceil", Ufunc.ceil(NdArrayFactory.array(a, new float[]{1.7f,-1.7f})), new float[]{2,-1}); } }
    @Test void round() { try (var a = Arena.ofConfined()) { assertArr("round", Ufunc.round(NdArrayFactory.array(a, new float[]{1.7f,-1.7f})), new float[]{2,-2}); } }
    @Test void sign() { try (var a = Arena.ofConfined()) { assertArr("sign", Ufunc.sign(NdArrayFactory.array(a, new float[]{1.7f,-1.7f})), new float[]{1,-1}); } }
    @Test void sinh() { try (var a = Arena.ofConfined()) { assertArr("sinh", Ufunc.sinh(NdArrayFactory.array(a, new float[]{0,1})), new float[]{0f,1.1752012f}); } }
    @Test void cosh() { try (var a = Arena.ofConfined()) { assertArr("cosh", Ufunc.cosh(NdArrayFactory.array(a, new float[]{0,1})), new float[]{1f,1.5430807f}); } }
    @Test void arcsin() { try (var a = Arena.ofConfined()) { assertArr("asin", Ufunc.arcsin(NdArrayFactory.array(a, new float[]{0,0.5f,1})), new float[]{0f,0.5235988f,1.5707963f}); } }
    @Test void arccos() { try (var a = Arena.ofConfined()) { assertArr("acos", Ufunc.arccos(NdArrayFactory.array(a, new float[]{0,0.5f,1})), new float[]{1.5707964f,1.0471976f,0f}); } }
    @Test void arctan() { try (var a = Arena.ofConfined()) { assertArr("atan", Ufunc.arctan(NdArrayFactory.array(a, new float[]{0,1})), new float[]{0f,0.7853982f}); } }
    @Test void arctan2() { try (var a = Arena.ofConfined()) { assertArr("atan2", Ufunc.arctan2(NdArrayFactory.array(a, new float[]{1,1}), NdArrayFactory.array(a, new float[]{1,-1})), new float[]{0.7853982f,2.3561945f}); } }
    @Test void log2() { try (var a = Arena.ofConfined()) { assertArr("log2", Ufunc.log2(NdArrayFactory.array(a, new float[]{1,2,4,8})), new float[]{0,1,2,3}); } }
    @Test void log10() { try (var a = Arena.ofConfined()) { assertArr("log10", Ufunc.log10(NdArrayFactory.array(a, new float[]{1,10,100})), new float[]{0,1,2}); } }
    @Test void log1p() { try (var a = Arena.ofConfined()) { assertArr("log1p", Ufunc.log1p(NdArrayFactory.array(a, new float[]{0,1})), new float[]{0f,0.6931472f}); } }
    @Test void expm1() { try (var a = Arena.ofConfined()) { assertArr("expm1", Ufunc.expm1(NdArrayFactory.array(a, new float[]{0,1})), new float[]{0f,1.7182819f}); } }
    @Test void equal() { try (var a = Arena.ofConfined()) { assertArr("eq", Ufunc.equal(NdArrayFactory.array(a, new float[]{1,2,3}), NdArrayFactory.array(a, new float[]{2,2,2})), new float[]{0,1,0}); } }
    @Test void notEqual() { try (var a = Arena.ofConfined()) { assertArr("ne", Ufunc.notEqual(NdArrayFactory.array(a, new float[]{1,2,3}), NdArrayFactory.array(a, new float[]{2,2,2})), new float[]{1,0,1}); } }
    @Test void greater() { try (var a = Arena.ofConfined()) { assertArr("gt", Ufunc.greater(NdArrayFactory.array(a, new float[]{1,2,3}), NdArrayFactory.array(a, new float[]{2,2,2})), new float[]{0,0,1}); } }
    @Test void greaterEqual() { try (var a = Arena.ofConfined()) { assertArr("ge", Ufunc.greaterEqual(NdArrayFactory.array(a, new float[]{1,2,3}), NdArrayFactory.array(a, new float[]{2,2,2})), new float[]{0,1,1}); } }
    @Test void less() { try (var a = Arena.ofConfined()) { assertArr("lt", Ufunc.less(NdArrayFactory.array(a, new float[]{1,2,3}), NdArrayFactory.array(a, new float[]{2,2,2})), new float[]{1,0,0}); } }
    @Test void lessEqual() { try (var a = Arena.ofConfined()) { assertArr("le", Ufunc.lessEqual(NdArrayFactory.array(a, new float[]{1,2,3}), NdArrayFactory.array(a, new float[]{2,2,2})), new float[]{1,1,0}); } }
    @Test void logicalAnd() { try (var a = Arena.ofConfined()) { assertArr("and", Ufunc.logicalAnd(NdArrayFactory.array(a, new float[]{1,0,1,0}), NdArrayFactory.array(a, new float[]{1,1,0,0})), new float[]{1,0,0,0}); } }
    @Test void logicalOr() { try (var a = Arena.ofConfined()) { assertArr("or", Ufunc.logicalOr(NdArrayFactory.array(a, new float[]{1,0,1,0}), NdArrayFactory.array(a, new float[]{1,1,0,0})), new float[]{1,1,1,0}); } }
    @Test void logicalNot() { try (var a = Arena.ofConfined()) { assertArr("not", Ufunc.logicalNot(NdArrayFactory.array(a, new float[]{1,0,1,0})), new float[]{0,1,0,1}); } }
    @Test void logicalXor() { try (var a = Arena.ofConfined()) { assertArr("xor", Ufunc.logicalXor(NdArrayFactory.array(a, new float[]{1,0,1,0}), NdArrayFactory.array(a, new float[]{1,1,0,0})), new float[]{0,1,1,0}); } }
    @Test void isnan() { try (var a = Arena.ofConfined()) { assertArr("isnan", Ufunc.isnan(NdArrayFactory.array(a, new float[]{1,Float.NaN,0})), new float[]{0,1,0}); } }
    @Test void isinf() { try (var a = Arena.ofConfined()) { assertArr("isinf", Ufunc.isinf(NdArrayFactory.array(a, new float[]{1,Float.POSITIVE_INFINITY,Float.NEGATIVE_INFINITY})), new float[]{0,1,1}); } }
    @Test void isfinite() { try (var a = Arena.ofConfined()) { assertArr("isfinite", Ufunc.isfinite(NdArrayFactory.array(a, new float[]{1,Float.NaN,Float.POSITIVE_INFINITY})), new float[]{1,0,0}); } }
    @Test void broadcastAdd() { try (var a = Arena.ofConfined()) { assertArr("bc", Ufunc.add(NdArrayFactory.array(a, new float[]{1,2,3}, 1, 3), NdArrayFactory.array(a, new float[]{10,20}, 2, 1)), new float[]{11,12,13,21,22,23}); } }
    @Test void broadcastMul() { try (var a = Arena.ofConfined()) { assertArr("bcm", Ufunc.multiply(NdArrayFactory.array(a, new float[]{1,2,3}, 1, 3), NdArrayFactory.array(a, new float[]{10,20}, 2, 1)), new float[]{10,20,30,20,40,60}); } }
    @Test void scalarOps() { try (var a = Arena.ofConfined()) { NdArray x = NdArrayFactory.array(a, new float[]{2,4,6}); assertArr("addS", Ufunc.addScalar(x, 1), new float[]{3,5,7}); assertArr("mulS", Ufunc.mulScalar(x, 2), new float[]{4,8,12}); } }
}
