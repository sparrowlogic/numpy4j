package com.sparrowlogic.numpy4j;

import java.lang.foreign.ValueLayout;
import java.util.Arrays;

/**
 * Shape manipulation ops matching NumPy: reshape, transpose, squeeze,
 * unsqueeze, expand_dims, flatten, ravel, concatenate, stack, split,
 * tile, repeat, broadcast_to.
 */
public final class ShapeOps {
    private ShapeOps() {
    }

    /**
     * View-based reshape (must be contiguous). Infers -1 dimension.
     */
    public static NdArray reshape(final NdArray a, final int... newShape) {
        // Resolve -1
        long known = 1;
        int negIdx = -1;
        int[] resolvedShape = newShape;
        for (int i = 0; i < resolvedShape.length; i++) {
            if (resolvedShape[i] == -1) {
                negIdx = i;
            } else {
                known *= resolvedShape[i];
            }
        }
        if (negIdx >= 0) {
            resolvedShape = resolvedShape.clone();
            resolvedShape[negIdx] = (int) (a.size() / known);
        }
        if (ShapeUtils.size(resolvedShape) != a.size()) {
            throw new IllegalArgumentException("Cannot reshape " + Arrays.toString(a.shape())
                                                       + " to " + Arrays.toString(resolvedShape));
        }
        NdArray c = a.contiguous();
        return new NdArray(c.data(), c.arena(), c.dtype(), resolvedShape, ShapeUtils.cStrides(resolvedShape), 0);
    }

    /**
     * Transpose (reverse axes) — returns a view.
     */
    public static NdArray transpose(final NdArray a) {
        int[] axes = new int[a.ndim()];
        for (int i = 0; i < axes.length; i++) {
            axes[i] = axes.length - 1 - i;
        }
        return transpose(a, axes);
    }

    /**
     * Transpose with explicit axis permutation — returns a view.
     */
    public static NdArray transpose(final NdArray a, final int... axes) {
        int[] oldShape = a.shape();
        int[] oldStrides = a.strides();
        int[] newShape = new int[axes.length];
        int[] newStrides = new int[axes.length];
        for (int i = 0; i < axes.length; i++) {
            newShape[i] = oldShape[axes[i]];
            newStrides[i] = oldStrides[axes[i]];
        }
        return new NdArray(a.data(), a.arena(), a.dtype(), newShape, newStrides, a.offset());
    }

    /**
     * Remove all size-1 dimensions.
     */
    public static NdArray squeeze(final NdArray a) {
        int[] shape = a.shape();
        int[] strides = a.strides();
        int count = 0;
        for (final int s : shape) {
            if (s != 1) {
                count++;
            }
        }
        int[] ns = new int[count];
        int[] nst = new int[count];
        int j = 0;
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] != 1) {
                ns[j] = shape[i];
                nst[j] = strides[i];
                j++;
            }
        }
        return new NdArray(a.data(), a.arena(), a.dtype(), ns, nst, a.offset());
    }

    /**
     * Add a size-1 dimension at the given axis.
     */
    public static NdArray expandDims(final NdArray a, final int axis) {
        int ax = axis < 0 ? axis + a.ndim() + 1 : axis;
        int[] shape = a.shape();
        int[] strides = a.strides();
        int[] ns = new int[shape.length + 1];
        int[] nst = new int[strides.length + 1];
        for (int i = 0, j = 0; i < ns.length; i++) {
            if (i == ax) {
                ns[i] = 1;
                nst[i] = 0;
            } else {
                ns[i] = shape[j];
                nst[i] = strides[j];
                j++;
            }
        }
        return new NdArray(a.data(), a.arena(), a.dtype(), ns, nst, a.offset());
    }

    /**
     * Flatten to 1D (contiguous copy).
     */
    public static NdArray flatten(final NdArray a) {
        return reshape(a, (int) a.size());
    }

    /**
     * Ravel — same as flatten for C-contiguous.
     */
    public static NdArray ravel(final NdArray a) {
        return flatten(a);
    }

    /**
     * Concatenate arrays along an axis.
     */
    public static NdArray concatenate(final NdArray[] arrays, final int axis) {
        int ax = axis < 0 ? axis + arrays[0].ndim() : axis;
        int[] baseShape = arrays[0].shape();
        int totalAxis = 0;
        for (final NdArray a : arrays) {
            totalAxis += a.shape(ax);
        }

        int[] outShape = baseShape.clone();
        outShape[ax] = totalAxis;
        NdArray out = NdArrayFactory.empty(arrays[0].arena(), DType.FLOAT32, outShape);

        int axisOffset = 0;
        for (final NdArray a : arrays) {
            NdArray c = a.contiguous();
            copySlice(c, out, ax, axisOffset);
            axisOffset += a.shape(ax);
        }
        return out;
    }

    /**
     * Stack arrays along a new axis.
     */
    public static NdArray stack(final NdArray[] arrays, final int axis) {
        NdArray[] expanded = new NdArray[arrays.length];
        for (int i = 0; i < arrays.length; i++) {
            expanded[i] = expandDims(arrays[i], axis);
        }
        return concatenate(expanded, axis);
    }

    /**
     * Tile: repeat array along each axis.
     */
    public static NdArray tile(final NdArray a, final int... reps) {
        NdArray result = a;
        for (int d = 0; d < reps.length; d++) {
            if (reps[d] > 1) {
                NdArray[] copies = new NdArray[reps[d]];
                Arrays.fill(copies, result);
                result = concatenate(copies, d);
            }
        }
        return result;
    }

    /**
     * Repeat each element along axis.
     */
    public static NdArray repeat(final NdArray a, final int repeats, final int axis) {
        int ax = axis < 0 ? axis + a.ndim() : axis;
        int[] inShape = a.shape();
        int[] outShape = inShape.clone();
        outShape[ax] = inShape[ax] * repeats;
        NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, outShape);
        NdArray c = a.contiguous();

        long outSize = ShapeUtils.size(outShape);
        int[] outStrides = ShapeUtils.cStrides(outShape);
        int[] inStrides = ShapeUtils.cStrides(inShape);

        for (long oi = 0; oi < outSize; oi++) {
            int[] outIdx = ShapeUtils.unravelIndex(oi, outShape);
            int[] inIdx = outIdx.clone();
            inIdx[ax] = outIdx[ax] / repeats;
            long flatIn = ShapeUtils.flatIndex(inIdx, inStrides);
            out.data().setAtIndex(
                    ValueLayout.JAVA_FLOAT, oi,
                    c.data().getAtIndex(ValueLayout.JAVA_FLOAT, flatIn)
            );
        }
        return out;
    }

    // ── Flip / Roll / Pad ──

    /**
     * Reverse elements of 1D array. For nD, reverses along all axes.
     */
    public static NdArray flip(final NdArray a) {
        NdArray c = a.contiguous();
        NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, a.shape());
        long n = a.size();
        for (long i = 0; i < n; i++) {
            out.flatSetFloat(i, c.flatGetFloat(n - 1 - i));
        }
        return out;
    }

    /**
     * Flip left-right (reverse axis 1).
     */
    public static NdArray fliplr(final NdArray a) {
        return flipAxis(a, 1);
    }

    /**
     * Flip up-down (reverse axis 0).
     */
    public static NdArray flipud(final NdArray a) {
        return flipAxis(a, 0);
    }

    private static NdArray flipAxis(final NdArray a, final int axis) {
        NdArray c = a.contiguous();
        int[] shape = a.shape();
        NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, shape);
        long size = a.size();
        int[] strides = ShapeUtils.cStrides(shape);
        for (long i = 0; i < size; i++) {
            int[] idx = ShapeUtils.unravelIndex(i, shape);
            int[] srcIdx = idx.clone();
            srcIdx[axis] = shape[axis] - 1 - idx[axis];
            long srcFlat = ShapeUtils.flatIndex(srcIdx, strides);
            out.flatSetFloat(i, c.flatGetFloat(srcFlat));
        }
        return out;
    }

    /**
     * Roll elements by shift positions.
     */
    public static NdArray roll(final NdArray a, final int shift) {
        NdArray c = a.contiguous();
        int n = (int) a.size();
        NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, a.shape());
        int s = ((shift % n) + n) % n;
        for (int i = 0; i < n; i++) {
            out.flatSetFloat(i, c.flatGetFloat((i - s + n) % n));
        }
        return out;
    }

    /**
     * Pad 1D array with constant value.
     */
    public static NdArray pad(final NdArray a, final int padBefore, final int padAfter, final float value) {
        NdArray c = a.contiguous();
        int n = (int) a.size();
        NdArray out = NdArrayFactory.full(a.arena(), DType.FLOAT32, value, n + padBefore + padAfter);
        for (int i = 0; i < n; i++) {
            out.flatSetFloat(padBefore + i, c.flatGetFloat(i));
        }
        return out;
    }

    // ── Triu / Tril / Diag / Trace / Diagonal ──

    /**
     * Upper triangular.
     */
    public static NdArray triu(final NdArray a) {
        NdArray c = a.contiguous();
        int m = a.shape(0);
        int n = a.shape(1);
        NdArray out = NdArrayFactory.zeros(a.arena(), DType.FLOAT32, m, n);
        for (int i = 0; i < m; i++) {
            for (int j = i; j < n; j++) {
                out.flatSetFloat((long) i * n + j, c.flatGetFloat((long) i * n + j));
            }
        }
        return out;
    }

    /**
     * Lower triangular.
     */
    public static NdArray tril(final NdArray a) {
        NdArray c = a.contiguous();
        int m = a.shape(0);
        int n = a.shape(1);
        NdArray out = NdArrayFactory.zeros(a.arena(), DType.FLOAT32, m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j <= i; j++) {
                out.flatSetFloat((long) i * n + j, c.flatGetFloat((long) i * n + j));
            }
        }
        return out;
    }

    /**
     * Extract diagonal from 2D matrix.
     */
    public static NdArray diag(final NdArray a) {
        return diagonal(a);
    }

    /**
     * Extract diagonal.
     */
    public static NdArray diagonal(final NdArray a) {
        NdArray c = a.contiguous();
        int n = Math.min(a.shape(0), a.shape(1));
        int cols = a.shape(1);
        NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, n);
        for (int i = 0; i < n; i++) {
            out.flatSetFloat(i, c.flatGetFloat((long) i * cols + i));
        }
        return out;
    }

    /**
     * Construct diagonal matrix from 1D vector.
     */
    public static NdArray diagConstruct(final NdArray v) {
        NdArray c = v.contiguous();
        int n = (int) v.size();
        NdArray out = NdArrayFactory.zeros(v.arena(), DType.FLOAT32, n, n);
        for (int i = 0; i < n; i++) {
            out.flatSetFloat((long) i * n + i, c.flatGetFloat(i));
        }
        return out;
    }

    /**
     * Trace: sum of diagonal elements.
     */
    public static float trace(final NdArray a) {
        NdArray d = diagonal(a);
        return Reductions.sum(d);
    }

    // ── Unique / Diff / Gradient ──

    /**
     * Sorted unique elements.
     */
    public static NdArray unique(final NdArray a) {
        float[] data = a.toFloatArray();
        java.util.Set<Float> set = new java.util.TreeSet<>();
        for (final float f : data) {
            set.add(f);
        }
        float[] result = new float[set.size()];
        int i = 0;
        for (final float f : set) {
            result[i++] = f;
        }
        return NdArrayFactory.array(a.arena(), result);
    }

    /**
     * First-order difference.
     */
    public static NdArray diff(final NdArray a) {
        NdArray c = a.contiguous();
        int n = (int) a.size();
        NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, n - 1);
        for (int i = 0; i < n - 1; i++) {
            out.flatSetFloat(i, c.flatGetFloat(i + 1) - c.flatGetFloat(i));
        }
        return out;
    }

    /**
     * Numerical gradient (central differences, forward/backward at edges).
     */
    public static NdArray gradient(final NdArray a) {
        NdArray c = a.contiguous();
        int n = (int) a.size();
        NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, n);
        if (n == 1) {
            out.flatSetFloat(0, 0f);
            return out;
        }
        out.flatSetFloat(0, c.flatGetFloat(1) - c.flatGetFloat(0));
        for (int i = 1; i < n - 1; i++) {
            out.flatSetFloat(i, (c.flatGetFloat(i + 1) - c.flatGetFloat(i - 1)) / 2f);
        }
        out.flatSetFloat(n - 1, c.flatGetFloat(n - 1) - c.flatGetFloat(n - 2));
        return out;
    }

    // ── Split / Hstack / Vstack ──

    /**
     * Split array into equal parts.
     */
    public static NdArray[] split(final NdArray a, final int nSections) {
        int n = (int) a.size();
        int sectionSize = n / nSections;
        NdArray c = a.contiguous();
        NdArray[] result = new NdArray[nSections];
        for (int i = 0; i < nSections; i++) {
            NdArray part = NdArrayFactory.empty(a.arena(), DType.FLOAT32, sectionSize);
            for (int j = 0; j < sectionSize; j++) {
                part.flatSetFloat(j, c.flatGetFloat((long) i * sectionSize + j));
            }
            result[i] = part;
        }
        return result;
    }

    /**
     * Horizontal stack (concatenate along axis 0 for 1D).
     */
    public static NdArray hstack(final NdArray a, final NdArray b) {
        return concatenate(new NdArray[]{a, b}, 0);
    }

    /**
     * Vertical stack (stack as rows).
     */
    public static NdArray vstack(final NdArray a, final NdArray b) {
        NdArray ra = a.ndim() == 1 ? reshape(a, 1, (int) a.size()) : a;
        NdArray rb = b.ndim() == 1 ? reshape(b, 1, (int) b.size()) : b;
        return concatenate(new NdArray[]{ra, rb}, 0);
    }

    // ── Swapaxes / Moveaxis ──

    /**
     * Swap two axes (returns a view).
     */
    public static NdArray swapaxes(final NdArray a, final int axis1, final int axis2) {
        int[] axes = new int[a.ndim()];
        for (int i = 0; i < axes.length; i++) {
            axes[i] = i;
        }
        axes[axis1] = axis2;
        axes[axis2] = axis1;
        return transpose(a, axes);
    }

    /**
     * Move axis from source to destination position.
     */
    public static NdArray moveaxis(final NdArray a, final int source, final int destination) {
        int ndim = a.ndim();
        int src = source < 0 ? source + ndim : source;
        int dst = destination < 0 ? destination + ndim : destination;
        int[] axes = new int[ndim];
        // Build permutation: remove source, insert at destination
        int j = 0;
        for (int i = 0; i < ndim; i++) {
            if (i == dst) {
                axes[i] = src;
            } else {
                if (j == src) {
                    j++;
                }
                axes[i] = j++;
            }
        }
        return transpose(a, axes);
    }

    /**
     * Meshgrid: create coordinate matrices from coordinate vectors.
     */
    public static NdArray[] meshgrid(final NdArray x, final NdArray y) {
        NdArray cx = x.contiguous();
        NdArray cy = y.contiguous();
        int nx = (int) x.size();
        int ny = (int) y.size();
        NdArray xGrid = NdArrayFactory.empty(x.arena(), DType.FLOAT32, ny, nx);
        NdArray yGrid = NdArrayFactory.empty(x.arena(), DType.FLOAT32, ny, nx);
        for (int i = 0; i < ny; i++) {
            for (int j = 0; j < nx; j++) {
                xGrid.flatSetFloat((long) i * nx + j, cx.flatGetFloat(j));
                yGrid.flatSetFloat((long) i * nx + j, cy.flatGetFloat(i));
            }
        }
        return new NdArray[]{xGrid, yGrid};
    }

    // ── Broadcast / Stack / Split extras ──

    /**
     * numpy.broadcast_to — broadcast array to target shape.
     */
    public static NdArray broadcastTo(final NdArray a, final int... targetShape) {
        int[] bStrides = ShapeUtils.broadcastStrides(a.shape(), a.strides(), targetShape);
        long outSize = ShapeUtils.size(targetShape);
        NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, targetShape);
        for (long i = 0; i < outSize; i++) {
            int[] idx = ShapeUtils.unravelIndex(i, targetShape);
            long srcFlat = a.offset();
            for (int d = 0; d < idx.length; d++) {
                srcFlat += (long) idx[d] * bStrides[d];
            }
            out.flatSetFloat(i, a.data().getAtIndex(ValueLayout.JAVA_FLOAT, srcFlat));
        }
        return out;
    }

    /**
     * numpy.dstack — stack along third axis.
     */
    public static NdArray dstack(final NdArray a, final NdArray b) {
        NdArray ra = a.ndim() == 1 ? reshape(a, 1, (int) a.size(), 1) : (a.ndim() == 2 ? reshape(
                a,
                a.shape(0),
                a.shape(1),
                1
        ) : a);
        NdArray rb = b.ndim() == 1 ? reshape(b, 1, (int) b.size(), 1) : (b.ndim() == 2 ? reshape(
                b,
                b.shape(0),
                b.shape(1),
                1
        ) : b);
        return concatenate(new NdArray[]{ra, rb}, 2);
    }

    /**
     * numpy.hsplit — split along axis 0 for 1D, axis 1 for 2D+.
     */
    public static NdArray[] hsplit(final NdArray a, final int nSections) {
        if (a.ndim() == 1) {
            return split(a, nSections);
        }
        int axis = 1;
        int secSize = a.shape(axis) / nSections;
        NdArray[] result = new NdArray[nSections];
        for (int i = 0; i < nSections; i++) {
            result[i] = IndexOps.slice(a, axis, i * secSize, (i + 1) * secSize, 1).contiguous();
        }
        return result;
    }

    /**
     * numpy.vsplit — split along axis 0.
     */
    public static NdArray[] vsplit(final NdArray a, final int nSections) {
        int secSize = a.shape(0) / nSections;
        NdArray[] result = new NdArray[nSections];
        for (int i = 0; i < nSections; i++) {
            result[i] = IndexOps.slice(a, 0, i * secSize, (i + 1) * secSize, 1).contiguous();
        }
        return result;
    }

    /**
     * numpy.rot90 — rotate 2D array 90 degrees counter-clockwise.
     */
    public static NdArray rot90(final NdArray a) {
        NdArray t = transpose(a).contiguous();
        return flipud(t);
    }

    /**
     * numpy.insert — insert value at index in 1D array.
     */
    public static NdArray insert(final NdArray a, final int index, final float value) {
        NdArray c = a.contiguous();
        int n = (int) a.size();
        NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, n + 1);
        for (int i = 0; i < index; i++) {
            out.flatSetFloat(i, c.flatGetFloat(i));
        }
        out.flatSetFloat(index, value);
        for (int i = index; i < n; i++) {
            out.flatSetFloat(i + 1, c.flatGetFloat(i));
        }
        return out;
    }

    /**
     * numpy.delete — remove element at index from 1D array.
     */
    public static NdArray delete(final NdArray a, final int index) {
        NdArray c = a.contiguous();
        int n = (int) a.size();
        NdArray out = NdArrayFactory.empty(a.arena(), DType.FLOAT32, n - 1);
        for (int i = 0, j = 0; i < n; i++) {
            if (i != index) {
                out.flatSetFloat(j++, c.flatGetFloat(i));
            }
        }
        return out;
    }

    /**
     * numpy.append — append values to end of 1D array.
     */
    public static NdArray append(final NdArray a, final NdArray values) {
        return concatenate(new NdArray[]{flatten(a), flatten(values)}, 0);
    }

    // ── Internal ──

    private static void copySlice(final NdArray src, final NdArray dst, final int axis, final int axisOffset) {
        long srcSize = src.size();
        int[] srcShape = src.shape();
        int[] srcStrides = ShapeUtils.cStrides(srcShape);
        int[] dstStrides = ShapeUtils.cStrides(dst.shape());

        for (long i = 0; i < srcSize; i++) {
            int[] idx = ShapeUtils.unravelIndex(i, srcShape);
            int[] dstIdx = idx.clone();
            dstIdx[axis] += axisOffset;
            long flatDst = ShapeUtils.flatIndex(dstIdx, dstStrides);
            dst.data().setAtIndex(
                    ValueLayout.JAVA_FLOAT, flatDst,
                    src.data().getAtIndex(ValueLayout.JAVA_FLOAT, i)
            );
        }
    }
}
