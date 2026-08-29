# L01 — Math 1: Vectors, Matrices & Functions

**Phase:** Foundations  
**Week:** 1  
**Sequence:** 1  
**Estimated effort:** 2.5 hours

## Purpose

Machine learning is largely computation over numerical representations. This lesson establishes the minimum linear-algebra vocabulary needed to read neural-network equations, tensor code, embeddings, attention, and optimization code without treating the mathematics as framework magic.

The goal is **working fluency**, not a complete linear-algebra course.

## Objectives

By the end, you should be able to:

- distinguish scalars, vectors, matrices, and higher-order tensors;
- interpret a vector as both data and a geometric object when appropriate;
- compute dot products, norms, and cosine similarity;
- perform matrix-vector and matrix-matrix multiplication and explain their dimensions;
- understand a matrix as a linear transformation;
- explain composition through matrix multiplication;
- understand functions as mappings from inputs to outputs and identify parameters versus inputs;
- connect these ideas to linear regression, neural-network layers, and embeddings;
- implement the core operations in NumPy and verify hand calculations.

## Prerequisites

Basic algebra, summation notation, exponentiation, and familiarity with Python syntax. No prior ML knowledge is required.

---

## 1. The numerical objects

### Scalar

A scalar is a single number, such as

\[
x = 3.5.
\]

### Vector

A vector is an ordered collection of numbers. In ML it commonly represents a feature vector, parameter vector, embedding, or intermediate activation:

\[
x = \begin{bmatrix}2\\-1\\4\end{bmatrix} \in \mathbb{R}^3.
\]

The notation \(\mathbb{R}^3\) means a vector with three real-valued components.

Do not assume every vector is physically a geometric arrow. In ML, a vector often represents data in a chosen coordinate system. The geometric interpretation becomes useful when operations such as dot products and norms are relevant.

### Matrix

A matrix is a rectangular array of numbers:

\[
A \in \mathbb{R}^{m\times n}.
\]

It has \(m\) rows and \(n\) columns.

A matrix can represent a batch of vectors, a collection of features, or a linear transformation. Its meaning depends on the model and the chosen convention.

### Tensor

A tensor is a generalization of these arrays to more than two dimensions. In practical ML code, "tensor" often means a multidimensional numerical array with a defined shape and dtype. A scalar, vector, and matrix can therefore be viewed as tensors of rank/order 0, 1, and 2 respectively, although terminology around tensor rank/order can vary; do not confuse array dimensionality with tensor rank in the strict mathematical sense.

For this course, **shape** is the important engineering property.

---

## 2. Vectors: dot products, length and direction

For

\[
x = (x_1,\ldots,x_n), \quad y=(y_1,\ldots,y_n),
\]

the dot product is

\[
x\cdot y = \sum_{i=1}^{n}x_i y_i.
\]

Example:

\[
[2,3]\cdot[4,-1] = 2(4)+3(-1)=5.
\]

### Why it matters

Dot products appear everywhere:

- linear models compute weighted sums;
- neural layers use matrix multiplication, which is many dot products;
- similarity methods compare vectors using dot products or normalized variants;
- attention uses dot products between queries and keys.

### Norm

The Euclidean/L2 norm is

\[
\|x\|_2 = \sqrt{x\cdot x} = \sqrt{\sum_i x_i^2}.
\]

For \([3,4]\), the norm is 5.

A norm measures magnitude/length. Other norms exist; do not treat "norm" as synonymous with L2.

### Cosine similarity

For nonzero vectors,

\[
\operatorname{cos\_sim}(x,y)=\frac{x\cdot y}{\|x\|_2\|y\|_2}.
\]

It measures the cosine of the angle between the vectors. It is 1 for vectors pointing in the same direction, -1 for opposite directions, and 0 for orthogonal nonzero vectors.

**Important:** cosine similarity measures orientation after normalization; it deliberately removes the effect of vector magnitude.

### Geometric identity

For an angle \(\theta\) between vectors,

\[
x\cdot y = \|x\|\|y\|\cos\theta.
\]

This is why the normalized dot product gives cosine similarity.

---

## 3. Matrices and shape discipline

Let

\[
A \in \mathbb{R}^{m\times n}, \qquad x\in\mathbb{R}^{n}.
\]

Then \(Ax\) is defined and produces a vector in \(\mathbb{R}^{m}\).

Component-wise:

\[
(Ax)_i = \sum_{j=1}^{n} A_{ij}x_j.
\]

Each output component is a dot product between one row of \(A\) and \(x\).

### Shape rule

For matrix multiplication,

\[
(m\times n)(n\times p)=(m\times p).
\]

The **inner dimensions must match**. The result takes the outer dimensions.

This is not cosmetic. Shape errors often reveal a misunderstanding of what a model operation means.

Example:

\[
A\in\mathbb{R}^{2\times3},\quad x\in\mathbb{R}^{3}
\Rightarrow Ax\in\mathbb{R}^{2}.
\]

### Matrix-matrix multiplication

For

\[
A\in\mathbb{R}^{m\times n},\quad B\in\mathbb{R}^{n\times p},
\]

\[
(AB)_{ij}=\sum_{k=1}^{n}A_{ik}B_{kj}.
\]

Matrix multiplication is generally **not commutative**:

\[
AB\neq BA
\]

when both products are defined.

---

## 4. A matrix as a linear transformation

A matrix \(A\) defines a function

\[
f(x)=Ax.
\]

It is linear because

\[
A(x+y)=Ax+Ay
\]

and

\[
A(cx)=cAx.
\]

A matrix can therefore transform coordinates: scaling, rotation, reflection, projection, or combinations of these operations are examples of linear transformations.

### Why this matters for AI

A neural-network affine layer is typically written as

\[
z = Wx+b.
\]

The matrix multiplication \(Wx\) is the linear part; adding \(b\) makes the overall mapping **affine**, not linear in the strict mathematical sense.

That distinction matters because later we add nonlinear activation functions. A stack of affine transformations alone can still be collapsed into one affine transformation; nonlinearities are what give a standard feed-forward network substantially greater expressive power.

---

## 5. Composition

Suppose

\[
f(x)=Ax, \qquad g(x)=Bx.
\]

Applying \(f\) and then \(g\) gives

\[
g(f(x))=B(Ax)=(BA)x.
\]

Therefore matrix multiplication represents composition of linear transformations.

This is the conceptual bridge to neural networks:

\[
x \rightarrow W_1x+b_1 \rightarrow \sigma(\cdot) \rightarrow W_2h+b_2 \rightarrow \cdots
\]

A framework may express this as layers, but underneath it is numerical transformation and composition.

---

## 6. Functions: the model-level view

A function maps an input to an output:

\[
f:X\rightarrow Y.
\]

In ML, distinguish:

- **input/data** — what the model receives;
- **parameters** — values learned or chosen by the training procedure;
- **output/prediction** — what the model computes;
- **objective/loss** — a numerical measure used to guide parameter selection during training.

For example, linear regression can be written

\[
\hat y = w^Tx+b.
\]

Here \(x\) is the input, \(w,b\) are parameters, and \(\hat y\) is the prediction.

The same basic pattern will recur throughout the course: **represent data numerically, apply parameterized functions, measure error, adjust parameters.**

---

## 7. ML connection: a batch is a matrix

Suppose there are \(N\) examples and each has \(d\) features. A common representation is

\[
X\in\mathbb{R}^{N\times d}.
\]

Each row is one example.

A weight vector can be represented as

\[
w\in\mathbb{R}^{d}.
\]

Then

\[
Xw\in\mathbb{R}^{N}
\]

gives one scalar prediction per example for a linear model (before any task-specific output transformation).

This is one reason vectorization matters: instead of writing a Python loop over examples, numerical libraries can execute the same algebra over the whole batch efficiently.

---

## 8. Derivations / proofs worth knowing

### 8.1 Dot product and angle

The law of cosines applied to the triangle formed by \(x\), \(y\), and \(x-y\) gives

\[
\|x-y\|^2=\|x\|^2+\|y\|^2-2\|x\|\|y\|\cos\theta.
\]

Expanding the left side using the dot product gives

\[
\|x-y\|^2=x\cdot x+y\cdot y-2x\cdot y.
\]

Equating the two expressions yields

\[
x\cdot y=\|x\|\|y\|\cos\theta.
\]

You should understand this derivation, not memorize only the final formula.

### 8.2 Associativity of matrix multiplication

For compatible matrices,

\[
((AB)C)_{ij}=\sum_k\sum_l A_{ik}B_{kl}C_{lj}
\]

and

\[
(A(BC))_{ij}=\sum_l\sum_k A_{ik}B_{kl}C_{lj}.
\]

Because finite sums can be regrouped/reordered,

\[
(AB)C=A(BC).
\]

Associativity is what lets us reason about a sequence of transformations without changing the result by changing the grouping.

### 8.3 Why affine layers compose to another affine layer

Let

\[
f(x)=A x+a,
\qquad g(x)=B x+b.
\]

Then

\[
g(f(x))=B(Ax+a)+b=(BA)x+(Ba+b).
\]

So the composition is again affine.

This is the precise reason that simply stacking linear/affine layers without nonlinear activation does not create a fundamentally more general class of functions than one affine layer.

---

## 9. Common misconceptions

- **"A vector is always a point in space."** No. It is an ordered numerical object; geometric interpretation is context-dependent.
- **"A matrix is just a table."** It is an array, but in ML its transformation interpretation is often the more useful one.
- **"Matrix multiplication means multiplying corresponding cells."** That is elementwise multiplication, not matrix multiplication.
- **"Transpose is the same as inverse."** No. \(A^T\) simply swaps rows and columns. An inverse, when it exists, satisfies \(A^{-1}A=I\).
- **"All matrix multiplication is commutative."** False.
- **"A neural-network layer is linear."** \(Wx+b\) is affine; adding a nonlinear activation makes the layer composition nonlinear.
- **"Cosine similarity measures how large two vectors are."** It measures angular alignment after normalizing magnitude away.
- **"Tensor rank means number of dimensions."** In everyday ML libraries, people often use "rank" informally for number of dimensions/axes. In mathematics, tensor rank has a different technical meaning. Use shape/number of axes when discussing array dimensions.

---

## 10. Implementation / experiment

Use **NumPy only**. Do not use PyTorch for this lesson.

Create `experiments/foundations/math01_linear_algebra.py` or an equivalent notebook that:

1. constructs vectors and matrices;
2. computes dot products and L2 norms manually and with NumPy;
3. computes cosine similarity;
4. performs matrix-vector and matrix-matrix multiplication;
5. prints shapes before and after each operation;
6. verifies the affine-composition identity numerically;
7. demonstrates a deliberately invalid matrix multiplication and explains the shape error;
8. verifies one hand-worked example with assertions.

### Minimum verification example

Use

\[
A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix},
\quad
x=\begin{bmatrix}2\\1\\0\end{bmatrix}.
\]

Compute \(Ax\) by hand before running the code.

Then choose arbitrary compatible \(A,B,x,a,b\) and verify numerically that

\[
g(f(x))=(BA)x+(Ba+b).
\]

Do not merely print values. Assert equality within a sensible floating-point tolerance.

---

## Exercises

1. Given \(x=[1,2,2]\), calculate its L2 norm.
2. Calculate the dot product and cosine similarity of \([1,0]\) and \([1,1]\).
3. Determine whether each multiplication is valid and give the output shape:
   - \((3\times4)(4\times2)\)
   - \((3\times4)(3\times2)\)
   - \((5\times2)(2)\)
4. Explain why \(AB\neq BA\) in general.
5. Show algebraically that a composition of two affine functions is affine.
6. Explain what each dimension means in a batch matrix \(X\in\mathbb{R}^{N\times d}\).

---

## Questions you should be able to answer

- What is the difference between a scalar, vector, matrix, and tensor?
- Why does \((m\times n)(n\times p)\) produce \((m\times p)\)?
- What does each row of a matrix-vector multiplication do?
- Why is matrix multiplication useful for batches?
- What does a matrix mean as a transformation?
- Why is \(Wx+b\) affine rather than linear?
- Why are nonlinear activation functions necessary in a standard deep feed-forward network?
- What information does cosine similarity discard?
- Why can a shape error reveal a conceptual error rather than merely a coding typo?

---

## Acceptance criteria

The lesson is complete only when you can:

- derive the main formulas above without copying them;
- perform shape reasoning without trial-and-error execution;
- explain matrix multiplication as dot products and as transformation composition;
- implement the required operations in NumPy;
- verify at least one identity with an automated assertion;
- explain the difference between linear, affine, and nonlinear mappings;
- answer the questions above without relying on framework terminology.

## Required evidence

- NumPy implementation/notebook committed to GitHub.
- Hand calculations for the minimum example.
- Automated assertions for numerical verification.
- Short written explanation of the affine-composition and cosine-similarity derivations.

## Evaluation criteria

| Dimension | What is being checked |
|---|---|
| Conceptual understanding | Correct interpretation of vectors, matrices, functions, shapes and transformations |
| Implementation correctness | NumPy operations and assertions are correct |
| AI relevance | Correctly connects the math to ML layers, batches and representations |
| Engineering quality | Reproducible, readable experiment with meaningful assertions |
| Experimental methodology | Hand prediction precedes execution; results are verified |
| Reasoning & tradeoffs | Can explain why the operations matter rather than reciting formulas |

## References

1. Ian Goodfellow, Yoshua Bengio, Aaron Courville, *Deep Learning*, MIT Press — Chapter 2, Linear Algebra. The online textbook is freely available: https://www.deeplearningbook.org/contents/linear_algebra.html
2. Gilbert Strang, *Introduction to Linear Algebra*, MIT-hosted materials. See vectors, dot products, matrices, linear combinations and matrix multiplication: https://math.mit.edu/~gs/LectureNotes/
3. Gilbert Strang, *Introduction to Linear Algebra*, 6th edition outline and chapter structure: https://math.mit.edu/~gs/linearalgebra/ila6/ila6outline.pdf

**Source note:** The mathematical identities in this lesson are standard linear-algebra results. The derivations shown above are included so the lesson does not depend on an unsupported claim or an appeal to framework behavior.
