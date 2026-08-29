# L01 — Foundations: Vectors, Matrices & Functions

**Phase:** Foundations  
**Sequence:** 1  
**Estimated effort:** 2.5–3 hours

> This is a compact textbook/work packet: explanations first, formulas second, implementation third. It is intentionally not a complete linear-algebra textbook.

## 1. Why this matters for AI

Machine learning repeatedly performs the same broad operation:

**represent something numerically → transform the representation → measure the result → adjust parameters.**

Linear algebra gives us the language for the numerical representation and transformation parts.

If you understand what a vector, matrix, dot product and matrix multiplication actually do, later code such as `x @ W + b`, embedding similarity, attention scores and neural-network layers becomes ordinary mathematics rather than framework magic.

---

## 2. Scalars, vectors, matrices and tensors

### Scalar

A scalar is one number:

$$x \in \mathbb{R}.$$

Examples include a loss value, a learning rate or one model parameter.

### Vector

A vector is an ordered collection of numbers:

$$x = \begin{bmatrix}2\\-1\\4\end{bmatrix} \in \mathbb{R}^3.$$

In engineering, think of it as a fixed-length numerical representation. Geometrically, it can also have magnitude and direction. Both interpretations are useful, but neither should be assumed universally.

A feature vector might be `[age, income, number_of_orders]`. An embedding is also a vector, but its coordinates are learned rather than being manually assigned human features.

### Matrix

A matrix is a rectangular array:

$$A \in \mathbb{R}^{m\times n}.$$

It has $m$ rows and $n$ columns.

Depending on context, a matrix may represent a dataset, a batch of vectors, model parameters, or a transformation.

### Tensor

In ML libraries, a tensor is generally a multidimensional numerical array with a shape and dtype. A matrix is a 2-D array; an image batch might have four axes.

For this course, **shape is the important engineering concept**. Do not confuse the number of array axes with the technical mathematical meaning of tensor rank.

---

## 3. Vectors: the operations that keep reappearing

Let

$$x=\begin{bmatrix}2\\3\end{bmatrix},\qquad y=\begin{bmatrix}4\\-1\end{bmatrix}.$$

### Addition

Add corresponding components:

$$x+y=\begin{bmatrix}6\\2\end{bmatrix}.$$

### Scalar multiplication

For scalar $c$:

$$cx=\begin{bmatrix}2c\\3c\end{bmatrix}.$$

### Dot product

The dot product is

$$x^Ty=\sum_{i=1}^{n}x_i y_i.$$

For the example:

$$x^Ty=2(4)+3(-1)=5.$$

The most useful intuition is **weighted sum**: multiply each component by its corresponding weight, then add.

That is already the core operation of a linear model:

$$\hat y=w^Tx+b.$$

A neural layer performs many such weighted sums at once.

### L2 norm

The Euclidean norm is

$$\|x\|_2=\sqrt{x^Tx}=\sqrt{\sum_i x_i^2}.$$

For `[3,4]`, the norm is 5.

A norm measures magnitude. L2 is one particular norm.

### Cosine similarity

For nonzero vectors:

$$\operatorname{cos\_sim}(x,y)=\frac{x^Ty}{\|x\|_2\|y\|_2}.$$

It measures angular alignment. Same direction gives 1; perpendicular nonzero vectors give 0; opposite directions give -1.

The important point is that normalization removes magnitude. Vectors `[1,1]` and `[100,100]` have cosine similarity 1 even though their lengths differ by a factor of 100.

This is why cosine similarity can be useful for comparing embeddings when direction is more relevant than magnitude. It is not universally the correct similarity measure.

---

## 4. Why the dot product has a geometric meaning

For vectors with angle $\theta$ between them,

$$x^Ty=\|x\|_2\|y\|_2\cos\theta.$$

Therefore,

$$\cos\theta=\frac{x^Ty}{\|x\|_2\|y\|_2}.$$

### Derivation

The law of cosines gives

$$\|x-y\|^2=\|x\|^2+\|y\|^2-2\|x\|\|y\|\cos\theta.$$

But

$$\begin{aligned}
\|x-y\|^2
&=(x-y)^T(x-y)\\
&=x^Tx-2x^Ty+y^Ty\\
&=\|x\|^2+\|y\|^2-2x^Ty.
\end{aligned}$$

Equating the two expressions gives

$$x^Ty=\|x\|\|y\|\cos\theta.$$

So cosine similarity is not an arbitrary ML trick: it is the normalized dot product.

---

## 5. Matrix-vector multiplication: what is actually happening?

Take

$$A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix},\qquad
x=\begin{bmatrix}2\\1\\0\end{bmatrix}.$$

$A$ is $2\times3$ and $x$ has length 3, so the operation is valid and produces a length-2 vector.

$$Ax=
\begin{bmatrix}
1(2)+2(1)+3(0)\\
4(2)+5(1)+6(0)
\end{bmatrix}
=
\begin{bmatrix}4\\13\end{bmatrix}.$$

The key interpretation is:

> **Each output is the dot product of one row of $A$ with $x$.**

That is why a matrix can perform many weighted sums in one operation.

### Shape rule

$$ (m\times n)(n\times p)=(m\times p). $$

The inner dimensions must match; the result keeps the outer dimensions.

For example:

$$ (3\times4)(4\times2)\rightarrow(3\times2). $$

A shape is not just bookkeeping. It tells you what the operation means and whether the data representation is compatible.

---

## 6. Matrix-matrix multiplication

For compatible matrices:

$$ (AB)_{ij}=\sum_k A_{ik}B_{kj}. $$

Again, each output element is a dot product.

Do not confuse this with elementwise multiplication:

$$C_{ij}=A_{ij}B_{ij}.$$

These are different operations.

Matrix multiplication is generally not commutative:

$$AB\ne BA.$$

But it is associative:

$$A(BC)=(AB)C.$$

The associativity becomes important when matrices represent transformations applied one after another.

---

## 7. A matrix as a transformation

A matrix defines a function

$$f(x)=Ax.$$

For example, a matrix can scale, rotate, reflect or project vectors.

The function is linear because

$$A(x+y)=Ax+Ay$$

and

$$A(cx)=cAx.$$

This gives a better mental model than "matrix = table": a matrix can encode a rule for transforming every vector in its input space.

### Affine transformations

A neural-network layer commonly begins with

$$z=Wx+b.$$

$Wx$ is linear; $Wx+b$ is **affine**.

The bias $b$ is important because it allows the transformation to shift the output rather than only scale/rotate/etc. around the origin.

Calling $Wx+b$ a "linear layer" is common ML terminology, but mathematically the mapping is affine.

---

## 8. Composition explains why matrix multiplication appears in networks

Suppose

$$f(x)=Ax+a$$

and

$$g(x)=Bx+b.$$

Applying $f$ and then $g$ gives

$$\begin{aligned}
g(f(x))
&=B(Ax+a)+b\\
&=(BA)x+(Ba+b).
\end{aligned}$$

So affine transformations compose into another affine transformation.

This gives an important neural-network fact.

Consider two layers with no activation between them:

$$W_2(W_1x+b_1)+b_2.$$

Rearrange:

$$ (W_2W_1)x+(W_2b_1+b_2). $$

That is just another affine function.

Therefore, stacking affine transformations alone does not provide the expressive benefit of a deep nonlinear network. Nonlinear activation functions prevent the whole stack from collapsing into one affine transformation.

This is one of the most important reasons neural networks contain nonlinearities.

---

## 9. Functions and parameters: the ML viewpoint

A function maps an input to an output:

$$f:X\rightarrow Y.$$

ML normally uses a parameterized function:

$$f_\theta(x).$$

Separate these concepts:

- **input $x$** — data supplied to the model;
- **parameters $\theta$** — values that training can adjust;
- **output $f_\theta(x)$** — prediction or representation;
- **target $y$** — desired answer when a supervised target exists;
- **loss** — numerical measure used by training to judge the result.

For linear regression:

$$\hat y=w^Tx+b.$$

The parameters are $w$ and $b$.

This distinction becomes fundamental later: architecture defines the computation family; learned parameters determine the particular function instantiated by a trained model.

---

## 10. Why batches are matrices

Suppose we have $N$ examples and each has $d$ features:

$$X\in\mathbb{R}^{N\times d}.$$

Each row can represent one example.

With

$$w\in\mathbb{R}^{d},$$

we can compute all linear predictions together:

$$Xw\in\mathbb{R}^{N}.$$

Conceptually this is:

```python
for row in X:
    prediction = row @ w
```

but vectorized numerical libraries can execute the batch operation efficiently.

The same shape reasoning will later apply to neural-network activations, token sequences, attention matrices and embedding tables.

---

## 11. A useful debugging habit

When you encounter:

```python
hidden = x @ W + b
```

do not start by asking which framework API is involved.

Ask:

1. What is the shape of `x`?
2. What does each axis represent?
3. What is the shape of `W`?
4. Why are those shapes compatible?
5. What shape should `x @ W` have?
6. What is the shape of `b`?
7. Is broadcasting being used?
8. Is there a nonlinear function afterward?

This turns many "AI bugs" into ordinary dimensional-analysis problems.

---

## 12. Worked example: a tiny neural layer

Suppose

$$x=\begin{bmatrix}2\\1\\0\end{bmatrix},
\quad
W=\begin{bmatrix}1&0&2\\-1&3&1\end{bmatrix},
\quad
b=\begin{bmatrix}1\\-2\end{bmatrix}.$$

First compute $Wx$:

$$Wx=\begin{bmatrix}
1(2)+0(1)+2(0)\\
-1(2)+3(1)+1(0)
\end{bmatrix}
=\begin{bmatrix}2\\1\end{bmatrix}.$$

Then add the bias:

$$z=Wx+b=\begin{bmatrix}3\\-1\end{bmatrix}.$$

If the next operation is ReLU,

$$\operatorname{ReLU}(z)=\begin{bmatrix}3\\0\end{bmatrix}.$$

Notice what happened:

**input → weighted sums → bias shift → nonlinear transformation.**

That pattern will reappear constantly in neural networks.

---

## 13. Common misconceptions

**A vector is always a geometric arrow.**  
No. It is an ordered numerical object. Geometry is a useful interpretation.

**Matrix multiplication multiplies corresponding cells.**  
No. It uses dot products of rows and columns.

**Transpose and inverse are the same.**  
No. Transpose swaps axes. An inverse, when it exists, satisfies $A^{-1}A=I$.

**$Wx+b$ is linear.**  
Strictly, it is affine.

**Cosine similarity measures magnitude.**  
It intentionally removes magnitude from the comparison.

**More embedding dimensions automatically means better embeddings.**  
No. Quality depends on training, representation, model and task.

**Tensor rank means number of dimensions.**  
Not in the strict mathematical sense. Use shape/axes for array dimensionality.

---

## 14. Implementation experiment

Use **Python + NumPy only**.

Create `experiments/foundations/math01_linear_algebra.py`.

### Experiment A — predict before running

Calculate by hand:

$$A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix},\quad
x=\begin{bmatrix}2\\1\\0\end{bmatrix}.$$

Predict $Ax$. Then verify with NumPy.

### Experiment B — shapes

Verify the result shapes for:

- $(3\times4)(4\times2)$
- $(5\times3)(3)$
- $(2\times5)(5\times7)$

Then deliberately perform an invalid multiplication and explain the error.

### Experiment C — cosine similarity

Implement cosine similarity and test:

- identical direction;
- orthogonal vectors;
- opposite direction;
- same direction but different magnitude.

### Experiment D — affine composition

Choose compatible $A,B,a,b,x$ and verify:

$$g(f(x))=(BA)x+(Ba+b).$$

Use `numpy.testing.assert_allclose` or an equivalent tolerance-based assertion.

### Experiment E — tiny neural layer

Implement the worked example from Section 12. Verify each intermediate value, not only the final result.

---

## 15. Exercises

1. Calculate the L2 norm of $[1,2,2]$.
2. Calculate the dot product of $[1,2,3]$ and $[4,0,-1]$.
3. Calculate cosine similarity for $[1,0]$ and $[1,1]$.
4. State whether each multiplication is valid and give its output shape:
   - $(3\times4)(4\times2)$
   - $(3\times4)(3\times2)$
   - $(5\times2)(2)$
5. Explain matrix-vector multiplication without using the phrase "matrix multiplication" as your explanation.
6. Show algebraically why two affine layers without an activation collapse into one affine layer.
7. Explain what $N$ and $d$ mean in $X\in\mathbb{R}^{N\times d}$.
8. Explain exactly what cosine similarity throws away.
9. Given `x.shape == (32, 128)` and `W.shape == (128, 64)`, what is the shape of `x @ W`? What do the dimensions represent?
10. If `x @ W` fails because the inner dimensions differ, explain the likely conceptual mistake before changing code.

---

## 16. Questions you must be able to answer

- Why does a dot product behave like a weighted sum?
- Why does matrix-vector multiplication produce multiple dot products?
- Why must inner dimensions match?
- Why is matrix multiplication useful for batches?
- What does a matrix mean when interpreted as a transformation?
- Why is $Wx+b$ affine?
- Why do nonlinear activations matter?
- Why does cosine similarity ignore magnitude?
- What does each dimension of a tensor mean in a concrete ML operation?
- How would you diagnose a tensor-shape error without trial and error?

---

## 17. Acceptance criteria

Pass only when you can:

- calculate the basic operations by hand;
- reason about shapes before running code;
- explain matrix multiplication as dot products;
- derive the cosine identity;
- explain affine composition;
- connect `x @ W + b` to the mathematics;
- implement and verify the experiments with assertions;
- explain all of the above without relying on framework vocabulary.

### Evidence

- NumPy experiment committed to the repository.
- Hand calculations for the worked examples.
- Automated numerical assertions.
- Short explanation of affine composition and cosine similarity.
- Answers to the questions above.

## References and proof sources

1. Goodfellow, Bengio & Courville, *Deep Learning*, Chapter 2, **Linear Algebra**. The chapter explicitly focuses on the subset of linear algebra needed for deep learning.  
   https://www.deeplearningbook.org/contents/linear_algebra.html

2. Gilbert Strang, MIT, *Lecture Notes for Linear Algebra*. See the sections on linear combinations, dot products, matrix-vector multiplication and matrix-matrix multiplication.  
   https://math.mit.edu/~gs/LectureNotes/

3. Gilbert Strang, *Introduction to Linear Algebra*, 6th-edition resources.  
   https://math.mit.edu/~gs/linearalgebra/ila6/indexila6.html

**Proofs used in this lesson:** the dot-product/angle identity is derived directly from the law of cosines and the algebraic definition of the dot product above; affine composition is derived by substitution and distribution. These derivations are included so the central claims are not presented as unsupported facts.
