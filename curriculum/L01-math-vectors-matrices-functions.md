# L01 — Foundations: Vectors, Matrices & Functions

**Phase:** Foundations  
**Sequence:** 1  
**Style:** Compact textbook + work packet  
**Estimated effort:** 2.5–3 hours

## 1. Why this matters for AI

A large amount of machine learning can be reduced to a recurring pattern:

**represent data numerically → transform the representation → measure an objective → adjust parameters.**

Linear algebra supplies much of the language for the first two steps. Later, the same ideas appear as embeddings, neural-network layers, attention matrices, logits, projections and batched tensor operations.

The goal here is not to reproduce a university linear-algebra course. It is to build enough understanding that expressions such as `x @ W + b` stop looking like framework magic.

## 2. Scalars, vectors, matrices and tensors

### Scalar

A scalar is a single number.

$$x \in \mathbb{R}$$

### Vector

A vector is an ordered collection of numbers. For example:

$$x = \begin{bmatrix}2\\-1\\4\end{bmatrix} \in \mathbb{R}^3$$

There are two useful ways to think about a vector: as an ordered numerical representation and, when useful, as a geometric object with magnitude and direction. An embedding is a vector, but its coordinates do not normally correspond to human-readable features.

### Matrix

A matrix is a rectangular array. For example:

$$A = \begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}$$

This matrix has 2 rows and 3 columns, so its shape is $2\times3$.

A matrix can represent data, model parameters, or a transformation. Its meaning depends on context.

### Tensor

In ML software, a tensor is generally a multidimensional numerical array with a shape and data type. A matrix is a two-dimensional case.

Do not equate "number of axes" with the strict mathematical definition of tensor rank. For engineering work, first ask what each axis represents and what its shape is.

## 3. Vectors: the operations that keep reappearing

Let

$$x=\begin{bmatrix}2\\3\end{bmatrix},\qquad y=\begin{bmatrix}4\\-1\end{bmatrix}$$

Add vectors componentwise and multiply a vector by a scalar componentwise.

The dot product is the sum of pairwise products:

$$x^T y=\sum_{i=1}^{n}x_i y_i$$

For our example:

$$x^T y=2(4)+3(-1)=5$$

The useful ML intuition is **weighted sum**: multiply each input component by a corresponding weight, then add.

That is already the core calculation of a linear model:

$$\hat y=w^T x+b$$

The Euclidean/L2 norm is

$$\lVert x\rVert_2=\sqrt{x^T x}=\sqrt{\sum_i x_i^2}$$

For nonzero vectors, cosine similarity is

$$\mathrm{cos\_sim}(x,y)=\frac{x^T y}{\lVert x\rVert_2\lVert y\rVert_2}$$

It measures angular alignment. Same direction gives 1, perpendicular vectors give 0, and opposite directions give -1. It deliberately removes magnitude; $[1,1]$ and $[100,100]$ have cosine similarity 1.

## 4. Why the dot product has a geometric meaning

For nonzero vectors with angle $\theta$ between them:

$$x^T y=\lVert x\rVert_2\lVert y\rVert_2\cos\theta$$

The law of cosines gives

$$\lVert x-y\rVert_2^2=\lVert x\rVert_2^2+\lVert y\rVert_2^2-2\lVert x\rVert_2\lVert y\rVert_2\cos\theta$$

while direct expansion gives

$$\begin{aligned}\lVert x-y\rVert_2^2&=(x-y)^T(x-y)\\&=x^Tx-2x^Ty+y^Ty\\&=\lVert x\rVert_2^2+\lVert y\rVert_2^2-2x^Ty\end{aligned}$$

Equating them gives the dot-product/angle identity. Dividing by the nonzero norms gives cosine similarity.

## 5. Matrix-vector multiplication

Take

$$A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix},\qquad x=\begin{bmatrix}2\\1\\0\end{bmatrix}$$

Then

$$Ax=\begin{bmatrix}1(2)+2(1)+3(0)\\4(2)+5(1)+6(0)\end{bmatrix}=\begin{bmatrix}4\\13\end{bmatrix}$$

Each output is the dot product of one row of $A$ with $x$.

The shape rule is

$$(m\times n)(n\times p)=(m\times p)$$

The inner dimensions must match; the result keeps the outer dimensions.

## 6. Matrix-matrix multiplication

For compatible matrices,

$$(AB)_{ij}=\sum_k A_{ik}B_{kj}$$

Again, each output element is a dot product. This differs from elementwise multiplication, where $C_{ij}=A_{ij}B_{ij}$.

Matrix multiplication is generally not commutative:

$$AB\ne BA$$

but it is associative:

$$A(BC)=(AB)C$$

## 7. Matrices as transformations

A matrix defines a function

$$f(x)=Ax$$

It can represent scaling, rotation, reflection or projection. It is linear because

$$A(x+y)=Ax+Ay$$

and

$$A(cx)=cAx$$

A neural-network layer commonly starts with

$$z=Wx+b$$

$Wx$ is linear; $Wx+b$ is **affine**. The bias permits a shift, so the full mapping is not strictly linear when $b\ne0$.

## 8. Composition explains neural-network depth

Let

$$f(x)=Ax+a,\qquad g(x)=Bx+b$$

Then

$$g(f(x))=B(Ax+a)+b=(BA)x+(Ba+b)$$

So affine transformations compose into another affine transformation.

Therefore

$$W_2(W_1x+b_1)+b_2=(W_2W_1)x+(W_2b_1+b_2)$$

A stack of affine layers without nonlinearities can therefore collapse into one affine transformation. A nonlinear activation between them prevents that collapse and is essential to the expressive power of ordinary feed-forward neural networks.

## 9. Functions and parameters

A function maps inputs to outputs:

$$f:X\rightarrow Y$$

ML usually uses a parameterized function:

$$f_\theta(x)$$

Keep these separate: input $x$ is data; parameters $\theta$ are values training can adjust; output is the prediction or representation; target $y$ is the desired answer when one exists; loss is a numerical objective used to judge the result.

For linear regression,

$$\hat y=w^T x+b$$

The architecture determines a family of computations; learned parameters select one particular function from that family.

## 10. Why batches are matrices

With $N$ examples and $d$ features, a common representation is

$$X\in\mathbb{R}^{N\times d}$$

With $w\in\mathbb{R}^{d}$,

$$Xw\in\mathbb{R}^{N}$$

gives one prediction per example for a linear model.

## 11. Worked neural-layer example

Let

$$x=\begin{bmatrix}2\\1\\0\end{bmatrix},\quad W=\begin{bmatrix}1&0&2\\-1&3&1\end{bmatrix},\quad b=\begin{bmatrix}1\\-2\end{bmatrix}$$

First,

$$Wx=\begin{bmatrix}2\\1\end{bmatrix}$$

Then,

$$z=Wx+b=\begin{bmatrix}3\\-1\end{bmatrix}$$

Define ReLU componentwise by

$$\mathrm{ReLU}(z_i)=\max(0,z_i)$$

so

$$\mathrm{ReLU}(z)=\begin{bmatrix}3\\0\end{bmatrix}$$

The mechanism is: **weighted sums → bias shift → nonlinear transformation**.

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

1. Predict the Section 5 result by hand, then verify it.
2. Verify the shapes of $(3\times4)(4\times2)$, $(5\times3)(3)$ and $(2\times5)(5\times7)$.
3. Deliberately perform an invalid multiplication and explain the error.
4. Implement cosine similarity and test identical, orthogonal, opposite and same-direction/different-magnitude vectors.
5. Verify $g(f(x))=(BA)x+(Ba+b)$ with numerical assertions.
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
4. GitHub Docs, *Writing mathematical expressions*. GitHub renders Markdown mathematics with MathJax and supports `$...$`, `$$...$$`, and `math` fenced blocks.  
   https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions

**Proofs:** the dot-product/angle identity is derived from the law of cosines and the algebraic dot-product definition; affine composition is derived by direct substitution and distribution.
