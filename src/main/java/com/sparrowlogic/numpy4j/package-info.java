/**
 * numpy4j — a 1:1 NumPy port for Java 26 using FFM, SIMD Vector API, and AMX offloading.
 *
 * <p>Core entry points:</p>
 * <ul>
 *   <li>{@link com.sparrowlogic.numpy4j.NdArray} — N-dimensional array</li>
 *   <li>{@link com.sparrowlogic.numpy4j.NdArrayFactory} — Array creation (zeros, ones, arange, etc.)</li>
 *   <li>{@link com.sparrowlogic.numpy4j.Ufunc} — Element-wise operations</li>
 *   <li>{@link com.sparrowlogic.numpy4j.Reductions} — Sum, mean, std, min, max, etc.</li>
 *   <li>{@link com.sparrowlogic.numpy4j.ShapeOps} — Reshape, transpose, concatenate, etc.</li>
 *   <li>{@link com.sparrowlogic.numpy4j.MatMul} — Matrix multiplication (AMX + SIMD)</li>
 *   <li>{@link com.sparrowlogic.numpy4j.LinAlg} — Linear algebra (numpy.linalg)</li>
 * </ul>
 */
@NullMarked
package com.sparrowlogic.numpy4j;

import org.jspecify.annotations.NullMarked;
