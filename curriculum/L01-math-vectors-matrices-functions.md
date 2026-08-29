# L01 — Foundations: Vectors, Matrices & Functions

**Phase:** Foundations  
**Sequence:** 1  
**Estimated effort:** 2.5–3 hours

> A compact textbook/work packet: explanation first, formulas with derivations, then implementation and evaluation. It is intentionally not a complete linear-algebra course.

## 1. Why this matters for AI

Machine learning repeatedly does something like:

**represent something numerically → transform the representation → measure the result → adjust parameters.**

Linear algebra gives us the language for numerical representations and transformations.

If you understand what vectors, dot products and matrices actually do, later code such as `x @ W + b`, embedding similarity, attention scores and neural-network layers becomes ordinary mathematics rather than framework magic.

## 2. Scalars, vectors, matrices and tensors

A scalar is one number: $$x\in\mathbb{R}.$$ A vector is an ordered collection of numbers, for example $$x=\begin{bmatrix}2\\-1\\4\end{bmatrix}\in\mathbb{R}^3.$$ A matrix is a rectangular array, $$A\in\mathbb{R}^{m\times n}.$$ In ML it may represent data, parameters, a batch, or a transformation. A tensor is a multidimensional numerical array; for this course, reason primarily in terms of **shape and axes**.

A vector can be viewed as a numerical representation or, when useful, as a geometric object with magnitude and direction. An embedding is a vector, but its coordinates are learned rather than necessarily human-interpretable features.

## 3. Vectors: the operations you will keep seeing

For vectors $x,y$ of equal length, addition is componentwise. Scalar multiplication multiplies every component.

The dot product is

$$x^Ty=\sum_{i=1}^{n}x_i y_i.$$

For $x=[2,3]^T$ and $y=[4,-1]^T$:

$$x^Ty=2(4)+3(-1)=5.$$

The useful intuition is **weighted sum**. That is already the core operation of a linear model:

$$\hat y=w^Tx+b.$$

The Euclidean/L2 norm is

$$\|x\|_2=\sqrt{x^Tx}=\sqrt{\sum_i x_i^2}.$$

For nonzero vectors, cosine similarity is

$$\operatorname{cos\_sim}(x,y)=\frac{x^Ty}{\|x\|_2\|y\|_2}.$$

It measures angular alignment. Same direction gives 1, perpendicular vectors give 0, and opposite directions give -1. It deliberately removes magnitude; `[1,1]` and `[100,100]` have cosine similarity 1.

## 4. Why the dot product has a geometric meaning

For vectors with angle $\theta$ between them,

$$x^Ty=\|x\|_2\|y\|_2\cos\theta.$$

Derivation: the law of cosines gives

$$\|x-y\|^2=\|x\|^2+\|y\|^2-2\|x\|\|y\|\cos\theta.$$

Meanwhile,

$$\begin{aligned}\|x-y\|^2&=(x-y)^T(x-y)\\&=x^Tx-2x^Ty+y^Ty\\&=\|x\|^2+\|y\|^2-2x^Ty.\end{aligned}$$

Comparing the equations gives the dot-product/angle identity. Dividing by the two nonzero norms gives cosine similarity.

## 5. Matrix-vector multiplication

Let

$$A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix},\qquad x=\begin{bmatrix}2\\1\\0\end{bmatrix}.$$

Then

$$Ax=\begin{bmatrix}1(2)+2(1)+3(0)\\4(2)+5(1)+6(0)\end{bmatrix}=\begin{bmatrix}4\\13\end{bmatrix}.$$

**Each output is the dot product of one row of $A$ with $x$.**

The shape rule is

$$(m\times n)(n\times p)=(m\times p).$$

The inner dimensions must match; the result keeps the outer dimensions. This is a mathematical constraint, not just API syntax.

## 6. Matrix-matrix multiplication

For compatible matrices,

$$(AB)_{ij}=\sum_k A_{ik}B_{kj}.$$

Again, each output element is a dot product. This differs from elementwise multiplication, where $C_{ij}=A_{ij}B_{ij}$.

Matrix multiplication is generally not commutative:

$$AB\ne BA,$$

but it is associative:

$$A(BC)=(AB)C.$$

The associativity matters because matrices can represent transformations applied in sequence.

## 7. Matrices as transformations

A matrix defines a function

$$f(x)=Ax.$$

It can represent scaling, rotation, reflection or projection. It is linear because

$$A(x+y)=Ax+Ay$$

and

$$A(cx)=cAx.$$

A neural-network layer commonly starts with

$$z=Wx+b.$$

$Wx$ is linear; $Wx+b$ is **affine**. The bias permits a shift, so the mapping is not strictly linear even though ML libraries commonly call this a linear layer.

## 8. Composition explains neural-network depth

Let

$$f(x)=Ax+a,\qquad g(x)=Bx+b.$$

Then

$$g(f(x))=B(Ax+a)+b=(BA)x+(Ba+b).$$

So affine transformations compose into another affine transformation.

Therefore

$$W_2(W_1x+b_1)+b_2=(W_2W_1)x+(W_2b_1+b_2).$$

A stack of affine layers without nonlinearities can therefore collapse into one affine transformation. A nonlinear activation between them prevents that collapse and is essential to the expressive power of ordinary feed-forward neural networks.

## 9. Functions and parameters

A function maps inputs to outputs:

$$f:X\rightarrow Y.$$

ML usually uses a parameterized function:

$$f_\theta(x).$$

Keep these separate:

- input $x$ = data;
- parameters $\theta$ = values training can adjust;
- output = prediction or representation;
- target $y$ = desired answer when one exists;
- loss = numerical objective used to judge the result during training.

For linear regression,

$$\hat y=w^Tx+b.$$

The architecture determines a family of computations; learned parameters select one particular function from that family.

## 10. Why batches are matrices

With $N$ examples and $d$ features, a common representation is

$$X\in\mathbb{R}^{N\times d}.$$

With $w\in\mathbb{R}^d$,

$$Xw\in\mathbb{R}^{N}$$

gives one prediction per example for a linear model. Vectorized libraries perform the same underlying algebra efficiently over the batch.

## 11. Worked neural-layer example

Let

$$x=\begin{bmatrix}2\\1\\0\end{bmatrix},\quad W=\begin{bmatrix}1&0&2\\-1&3&1\end{bmatrix},\quad b=\begin{bmatrix}1\\-2\end{bmatrix}.$$

First,

$$Wx=\begin{bmatrix}2\\1\end{bmatrix}.$$

Then,

$$z=Wx+b=\begin{bmatrix}3\\-1\end{bmatrix}.$$

With ReLU,

$$\operatorname{ReLU}(z)=\begin{bmatrix}3\\0\end{bmatrix}.$$

So the mechanism is: **weighted sums → bias shift → nonlinear transformation**.

## 12. Debugging habit

For code such as

```python
hidden = x @ W + b
```

ask: What are the shapes? What does each axis represent? Why are the inner dimensions compatible? What output shape should result? What is the bias shape? Is broadcasting involved? Is there a nonlinearity afterward?

Start with mathematics, not framework documentation.

## 13. Common misconceptions

- A vector is not necessarily a 3-D arrow.
- Matrix multiplication is not matching-cell multiplication.
- Transpose is not inverse; an inverse satisfies $A^{-1}A=I$ when it exists.
- $Wx+b$ is affine, not strictly linear.
- Cosine similarity intentionally removes magnitude.
- More embedding dimensions do not automatically mean better embeddings.
- Array dimensionality is not the same thing as mathematical tensor rank.

## 14. Implementation experiment

Use **Python + NumPy only**. Create `experiments/foundations/math01_linear_algebra.py`.

1. Predict the Section 5 matrix-vector result by hand, then verify it.
2. Verify the shapes of $(3\times4)(4\times2)$, $(5\times3)(3)$ and $(2\times5)(5\times7)$.
3. Deliberately perform an invalid multiplication and explain the error.
4. Implement cosine similarity and test identical, orthogonal, opposite and same-direction/different-magnitude vectors.
5. Verify $g(f(x))=(BA)x+(Ba+b)$ with `assert_allclose`.
6. Implement the neural-layer example and verify every intermediate value.

## 15. Exercises

1. Calculate the L2 norm of $[1,2,2]$.
2. Calculate the dot product of $[1,2,3]$ and $[4,0,-1]$.
3. Calculate cosine similarity for $[1,0]$ and $[1,1]$.
4. Give the output shape of $(3\times4)(4\times2)$ and explain why.
5. Explain matrix-vector multiplication using dot products.
6. Show why two affine layers without activation collapse into one affine layer.
7. Explain $N$ and $d$ in $X\in\mathbb{R}^{N\times d}$.
8. Explain exactly what cosine similarity discards.
9. If `x.shape == (32, 128)` and `W.shape == (128, 64)`, what is `x @ W`'s shape? What does each dimension mean?
10. Diagnose a matrix-shape error without trial-and-error changes.

## 16. Questions you must be able to answer

- Why does a dot product behave like a weighted sum?
- Why does matrix-vector multiplication produce multiple dot products?
- Why must inner dimensions match?
- Why is matrix multiplication useful for batches?
- What does a matrix mean as a transformation?
- Why is $Wx+b$ affine?
- Why do nonlinear activations matter?
- Why does cosine similarity ignore magnitude?
- How do tensor shapes help debug AI systems?

## 17. Acceptance criteria

Pass only when you can calculate the basic operations by hand, reason about shapes before running code, explain matrix multiplication as dot products, derive the cosine identity, explain affine composition, connect `x @ W + b` to the mathematics, and verify the experiments with assertions.

**Evidence:** NumPy experiment, hand calculations, automated assertions, and written explanations of cosine similarity and affine composition.

## References and proof sources

1. Goodfellow, Bengio & Courville, *Deep Learning*, Chapter 2, **Linear Algebra**.  
   https://www.deeplearningbook.org/contents/linear_algebra.html
2. Gilbert Strang, MIT, *Lecture Notes for Linear Algebra*.  
   https://math.mit.edu/~gs/LectureNotes/
3. Gilbert Strang, *Introduction to Linear Algebra*, 6th-edition resources.  
   https://math.mit.edu/~gs/linearalgebra/ila6/indexila6.html

**Proofs:** the dot-product/angle identity is derived from the law of cosines and the algebraic dot-product definition; affine composition is derived by direct substitution and distribution. The lesson therefore does not rely on unexplained formula memorization.
