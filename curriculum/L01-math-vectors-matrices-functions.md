# L01 — Foundations: Vectors, Matrices & Functions

**Phase:** Foundations  
**Sequence:** 1  
**Format:** Compact textbook + multi-module work packet  
**Target effort:** ~5–6 hours across modules  

> **Purpose:** rebuild the mathematical intuition needed for AI. This is not an engineering-math course. You should be able to understand the ideas from this document even if your last comfortable experience with mathematics was around school level.

---

## How to use this lesson

Do not try to memorize the notation first.

For each module:

1. understand the problem in plain English;
2. look at the visual/model;
3. work through the tiny examples;
4. learn the notation;
5. connect it to AI;
6. implement it;
7. explain it back without looking.

If a formula feels like magic, stop. The formula is supposed to be a compact way of writing something you already understand.

---

# Module 1 — Numbers, variables and functions

## 1.1 Why do we need variables?

Suppose a taxi charges ₹100 to start and ₹20 per kilometre.

For 3 km:

$$100 + 20(3) = 160$$

For 10 km:

$$100 + 20(10) = 300$$

Instead of writing a new calculation for every distance, call the distance $x$:

$$\mathrm{cost}(x)=100+20x$$

That is the basic idea of a **function**: a rule that takes an input and produces an output.

```text
input x ─────► [ rule: 100 + 20x ] ─────► output
  3                                      160
 10                                      300
```

The notation

$$f:X\rightarrow Y$$

simply says: function $f$ maps inputs from $X$ to outputs in $Y$.

You do not need to make this more mysterious than that.

## 1.2 Why functions matter in AI

A model is ultimately a function.

For example:

$$f_\theta(x)=\hat y$$

means that a function, controlled by parameters $\theta$, takes input $x$ and produces prediction $\hat y$.

A neural network is a complicated function. Training means changing $\theta$ so that the function behaves better on the task.

That single idea will recur throughout the entire course.

## 1.3 Composition: functions can be chained

Suppose:

- function $f$ converts Celsius to Fahrenheit;
- function $g$ converts Fahrenheit to a display string.

You can do:

$$x\rightarrow f(x)\rightarrow g(f(x))$$

This is **composition**.

Neural networks do the same thing repeatedly:

```text
input
  ↓
function 1
  ↓
function 2
  ↓
function 3
  ↓
output
```

Later, "deep" networks will largely mean that we compose many parameterized transformations, with nonlinear operations between them.

## 1.4 Example: recommendation

Imagine a book recommender has three input numbers:

```text
likes mystery       = 0.9
likes history       = 0.7
likes romance       = 0.2
```

Those numbers are a representation of the user. A model can transform that representation into a score for a candidate book.

The important idea is not the specific numbers. It is:

**real-world thing → numerical representation → mathematical function → prediction/score.**

---

# Module 2 — Vectors: numbers that belong together

## 2.1 The problem vectors solve

Suppose you want to describe a house using:

```text
area       = 1200 sq ft
bedrooms   = 3
age        = 8 years
```

Put those numbers together:

$$x=\begin{bmatrix}1200\\3\\8\end{bmatrix}$$

That is a **vector**.

A vector is simply an ordered collection of numbers.

The order matters. These two are different representations:

$$\begin{bmatrix}1200\\3\\8\end{bmatrix}
\neq
\begin{bmatrix}3\\1200\\8\end{bmatrix}$$

because the positions have different meanings.

## 2.2 The visual intuition

A two-number vector can also be drawn as an arrow from the origin.

```text
          y
          ↑
       3  |        ● (2,3)
          |       /
       2  |      /
          |     /
       1  |    /
          |   /
       0  +----------------→ x
             0  1  2  3
```

For two dimensions, the arrow picture is useful.

For a 768-dimensional embedding, drawing 768 axes is impossible. The vector is still an ordered list of numbers; the geometric picture becomes an intuition rather than a literal drawing we can inspect directly.

## 2.3 Vector addition

Take:

$$a=\begin{bmatrix}2\\1\end{bmatrix},\qquad b=\begin{bmatrix}3\\4\end{bmatrix}$$

Add corresponding positions:

$$a+b=\begin{bmatrix}2+3\\1+4\end{bmatrix}=\begin{bmatrix}5\\5\end{bmatrix}$$

Geometrically, adding vectors is equivalent to following one arrow and then the other.

This is useful because many physical quantities—position changes, velocity changes, forces—can naturally be represented as vectors.

## 2.4 Scaling a vector

Multiply by a number:

$$2\begin{bmatrix}2\\1\end{bmatrix}=\begin{bmatrix}4\\2\end{bmatrix}$$

The direction stays the same and the length doubles when the scalar is positive.

A negative scalar reverses the direction.

This simple operation becomes important when weights and learned parameters scale signals inside neural networks.

## 2.5 AI examples

Vectors appear everywhere:

- a row of tabular features;
- a word or sentence embedding;
- an image after flattening or feature extraction;
- a user's preference representation;
- a model's parameter vector;
- a gradient vector.

Do not conclude that every vector coordinate has a human interpretation. In learned representations, individual dimensions often do not correspond neatly to concepts like "happy" or "sports".

---

# Module 3 — Dot products: the weighted-score machine

This is one of the most important calculations in AI.

## 3.1 Start with something familiar

Suppose your restaurant score is:

```text
food quality     weight 5
service          weight 3
ambience         weight 2
```

A restaurant has scores:

```text
food             8
service          6
ambience         9
```

A weighted score is:

$$8(5)+6(3)+9(2)=76$$

That calculation is a **dot product**.

Write the numbers as vectors:

$$x=\begin{bmatrix}8\\6\\9\end{bmatrix},\qquad w=\begin{bmatrix}5\\3\\2\end{bmatrix}$$

Then:

$$w^T x=8(5)+6(3)+9(2)=76$$

The superscript $T$ means **transpose**. It changes a column vector into a row vector so the multiplication is defined.

## 3.2 The formula

For two vectors of the same length:

$$x^Ty=\sum_{i=1}^{n}x_i y_i$$

Read it in English:

> multiply corresponding numbers and add all the results.

That is all a dot product is at this level.

## 3.3 Why AI loves dot products

A neuron does essentially this:

$$z=w^Tx+b$$

So:

```text
inputs × learned weights
        ↓
   add everything
        ↓
      + bias
        ↓
      output
```

The weights tell the model how strongly each input contributes to the score.

This is why understanding dot products is much more important for AI than memorizing the symbol $x^Ty$.

## 3.4 Similarity intuition

Now suppose two vectors describe things:

$$x=\begin{bmatrix}1\\1\end{bmatrix},\qquad y=\begin{bmatrix}1\\1\end{bmatrix}$$

Their dot product is positive and large relative to their sizes.

If:

$$y=\begin{bmatrix}-1\\-1\end{bmatrix}$$

then:

$$x^Ty=-2$$

They point in opposite directions.

If:

$$y=\begin{bmatrix}1\\-1\end{bmatrix}$$

then:

$$x^Ty=0$$

They are perpendicular.

The dot product therefore contains information about alignment—but its raw value also depends on vector magnitude.

## 3.5 Cosine similarity

To focus on direction rather than magnitude, normalize the dot product:

$$\mathrm{cos\_sim}(x,y)=\frac{x^Ty}{\lVert x\rVert_2\lVert y\rVert_2}$$

where:

$$\lVert x\rVert_2=\sqrt{\sum_i x_i^2}$$

This produces a value between -1 and 1 for ordinary nonzero real vectors.

```text
same direction       →  +1
perpendicular        →   0
opposite direction   →  -1
```

Example:

$$x=\begin{bmatrix}1\\1\end{bmatrix},\qquad y=\begin{bmatrix}100\\100\end{bmatrix}$$

They have very different magnitudes, but exactly the same direction, so their cosine similarity is 1.

This idea later becomes useful for understanding embedding similarity and retrieval.

## 3.6 Real-world applications

Dot products appear in:

- weighted scoring systems;
- linear regression;
- classification layers;
- recommendation systems;
- information retrieval;
- embedding similarity;
- attention mechanisms.

The same arithmetic keeps appearing because it is an efficient way to measure or combine aligned numerical information.

---

# Module 4 — Matrices: organized collections and transformations

## 4.1 Start with a spreadsheet

You already know what this looks like:

```text
             area   bedrooms   age
house A      1200      3        8
house B      1800      4        5
house C       900      2       20
```

That is a matrix-like arrangement of numbers.

Mathematically:

$$X=\begin{bmatrix}
1200&3&8\\
1800&4&5\\
900&2&20
\end{bmatrix}$$

It has 3 rows and 3 columns, so its shape is $3\times3$.

The meaning of each axis matters:

- rows = houses;
- columns = features.

In AI, a matrix often represents a batch of examples or a set of model parameters.

## 4.2 Matrix-vector multiplication

Take:

$$A=\begin{bmatrix}
1&2&3\\
4&5&6
\end{bmatrix},\qquad
x=\begin{bmatrix}
2\\
1\\
0
\end{bmatrix}$$

Each row of $A$ performs a dot product with $x$:

$$Ax=\begin{bmatrix}
1(2)+2(1)+3(0)\\
4(2)+5(1)+6(0)
\end{bmatrix}
=\begin{bmatrix}
4\\
13
\end{bmatrix}$$

**This is the visual idea to remember:**

```text
             x
             ↓
row 1  ──dot─→ output 1
row 2  ──dot─→ output 2
```

A matrix-vector multiplication is therefore not a mysterious new operation. It is many dot products performed together.

## 4.3 Why shapes matter

If:

$$A\in\mathbb{R}^{m\times n}$$

and

$$x\in\mathbb{R}^{n}$$

then:

$$Ax\in\mathbb{R}^{m}$$

The number of columns in $A$ must match the number of entries in $x$.

For example:

```text
(2 × 3) · (3 × 1) = (2 × 1)   ✓
(2 × 3) · (2 × 1) = invalid   ✗
```

This is one of the most useful debugging habits in AI engineering: **look at shapes before looking at code.**

## 4.4 Matrix-matrix multiplication

Suppose:

$$A\in\mathbb{R}^{m\times n},\qquad B\in\mathbb{R}^{n\times p}$$

Then:

$$AB\in\mathbb{R}^{m\times p}$$

Each output element is a dot product:

$$(AB)_{ij}=\sum_k A_{ik}B_{kj}$$

So matrix multiplication is essentially **a grid of dot products**.

## 4.5 Do not confuse multiplication types

Elementwise multiplication:

$$C_{ij}=A_{ij}B_{ij}$$

Matrix multiplication:

$$(AB)_{ij}=\sum_k A_{ik}B_{kj}$$

They are different operations.

This distinction is responsible for a remarkable number of bugs in numerical code.

---

# Module 5 — Matrices as machines that transform vectors

## 5.1 A matrix can be more than a table

A matrix can take one vector and produce another:

$$f(x)=Ax$$

Think of $A$ as a machine:

```text
vector x ──► [ MATRIX A ] ──► new vector Ax
```

Depending on $A$, it can stretch, shrink, rotate, reflect, project, mix or otherwise transform the vector.

## 5.2 A simple scaling example

Take:

$$A=\begin{bmatrix}2&0\\0&3\end{bmatrix}$$

and:

$$x=\begin{bmatrix}1\\2\end{bmatrix}$$

Then:

$$Ax=\begin{bmatrix}2\\6\end{bmatrix}$$

The first coordinate was doubled and the second tripled.

The matrix encoded the transformation.

## 5.3 Why this matters for neural networks

A neural-network layer often contains:

$$z=Wx+b$$

The matrix $W$ mixes and scales the input values.

The vector $b$ shifts the result.

Then an activation function such as ReLU may change it nonlinearly.

So a simplified neural layer is:

```text
input vector
     ↓
   W × x
     ↓
   + b
     ↓
   activation
     ↓
output vector
```

That is already enough to demystify a large amount of neural-network code.

## 5.4 Linear vs affine

A strictly linear transformation must satisfy:

$$f(x+y)=f(x)+f(y)$$

and:

$$f(cx)=cf(x)$$

For $f(x)=Wx$, this works.

But:

$$f(x)=Wx+b$$

is generally **affine**, not linear, when $b\ne0$.

Why care? Because mathematical terminology matters when we later reason about what transformations can or cannot express.

---

# Module 6 — Why nonlinearities make neural networks interesting

Suppose we stack two affine transformations:

$$f(x)=W_1x+b_1$$

followed by:

$$g(x)=W_2f(x)+b_2$$

Substitute $f(x)$:

$$g(f(x))=W_2(W_1x+b_1)+b_2$$

Distribute:

$$=(W_2W_1)x+(W_2b_1+b_2)$$

That is still just one affine transformation.

So merely stacking linear/affine layers does not create the kind of nonlinear function class we associate with deep neural networks.

Add a nonlinear activation:

$$h(x)=\mathrm{ReLU}(W_1x+b_1)$$

and the collapse argument no longer applies in the same way.

## ReLU

ReLU means rectified linear unit:

$$\mathrm{ReLU}(z)=\max(0,z)$$

For a vector, apply it component by component.

Example:

$$z=\begin{bmatrix}3\\-1\\0.5\end{bmatrix}$$

then:

$$\mathrm{ReLU}(z)=\begin{bmatrix}3\\0\\0.5\end{bmatrix}$$

The operation is simple. Its importance comes from inserting nonlinearity into the network.

---

# Module 7 — Functions, parameters and batches in real AI code

## 7.1 A linear model

A simple model is:

$$\hat y=w^Tx+b$$

Here:

- $x$ = input data;
- $w$ = learned weights;
- $b$ = learned bias;
- $\hat y$ = prediction.

The model is not learning a new mathematical rule every time. It is learning the **parameters** of a parameterized function.

## 7.2 Batches

Suppose we have 32 examples and each has 128 features.

A common representation is:

$$X\in\mathbb{R}^{32\times128}$$

If the weight matrix maps 128 features to 64 outputs:

$$W\in\mathbb{R}^{128\times64}$$

then:

$$XW\in\mathbb{R}^{32\times64}$$

Read this as:

```text
32 examples
   ×
128 input features
   ↓
128 → 64 learned transformation
   ↓
32 examples × 64 output features
```

This is why tensor shapes are not bookkeeping trivia. They tell you what computation is actually being performed.

## 7.3 A worked neural layer

Let:

$$x=\begin{bmatrix}2\\1\\0\end{bmatrix},\qquad
W=\begin{bmatrix}1&0&2\\-1&3&1\end{bmatrix},\qquad
b=\begin{bmatrix}1\\-2\end{bmatrix}$$

First:

$$Wx=\begin{bmatrix}2\\1\end{bmatrix}$$

Then:

$$z=Wx+b=\begin{bmatrix}3\\-1\end{bmatrix}$$

Then:

$$\mathrm{ReLU}(z)=\begin{bmatrix}3\\0\end{bmatrix}$$

Do not memorize the result. Be able to reconstruct it.

---

# Module 8 — The four mental models to keep

### Vector

**A bundle of numbers with an order and meaning.**

### Dot product

**Multiply corresponding values and add them. Often acts as a weighted score or alignment measure.**

### Matrix

**A rectangular collection of numbers that can also represent a transformation.**

### Function

**A rule that maps inputs to outputs. AI models are parameterized functions.**

If those four ideas become intuitive, much of the notation that follows in the course becomes less intimidating.

---

# Real-world application map

| Concept | Everyday interpretation | AI use |
|---|---|---|
| Vector | A bundle of measurements | Features, embeddings, parameters |
| Dot product | Weighted score | Neurons, retrieval, similarity |
| Norm | Size/length of a vector | Normalization, similarity |
| Matrix | Structured table / transformation | Weights, batches, projections |
| Matrix multiplication | Many weighted combinations | Neural layers, attention |
| Function | Input → output rule | Model |
| Composition | Chain of transformations | Deep networks |
| Nonlinearity | A rule that bends/changes behavior | Activations |

---

# Experiment — Make the math visible in Python

Use **Python + NumPy** only.

Create:

`experiments/foundations/math01_linear_algebra.py`

## Required experiments

### A. Vectors

Create two vectors and verify addition, subtraction, scalar multiplication and norms.

### B. Dot product

Calculate a weighted restaurant score by hand, then reproduce it with NumPy.

### C. Cosine similarity

Test:

- identical direction;
- opposite direction;
- perpendicular vectors;
- same direction but very different magnitude.

Predict the result before executing.

### D. Matrix multiplication

Reproduce the worked matrix-vector example manually and with NumPy.

Then deliberately try an incompatible shape and explain why it fails.

### E. Neural layer

Implement:

```python
z = x @ W + b
output = np.maximum(z, 0)
```

Print every intermediate shape and value.

### F. Batch reasoning

Create an array with shape `(32, 128)` and a weight matrix `(128, 64)`. Verify that the result is `(32, 64)`.

Do not just print the shape. Explain what the three numbers mean.

---

# Exercises

1. Explain a vector without using the word "array".
2. Give two real-world examples where a vector is useful.
3. Calculate:

$$\begin{bmatrix}2\\3\end{bmatrix}^T
\begin{bmatrix}4\\5\end{bmatrix}$$

4. Explain that calculation in plain English.
5. Why does cosine similarity treat $[1,1]$ and $[100,100]$ as pointing in the same direction?
6. Calculate the result of:

$$\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}
\begin{bmatrix}2\\1\\0\end{bmatrix}$$

7. Why is $(2\times3)(2\times1)$ invalid?
8. Explain matrix multiplication as "many dot products".
9. Explain the difference between elementwise multiplication and matrix multiplication.
10. Why is $Wx+b$ affine rather than strictly linear?
11. Why does adding ReLU between affine layers change what the network can represent?
12. If `X.shape == (32, 128)` and `W.shape == (128, 64)`, explain every dimension of `X @ W`.

---

# Questions you must be able to answer

- What problem does a vector solve?
- What is a dot product actually calculating?
- Why is a dot product useful for weighted scoring?
- What does cosine similarity remove from the comparison?
- Why must matrix multiplication dimensions line up?
- Why can matrix multiplication be understood as many dot products?
- How can a matrix act as a transformation?
- What is the difference between linear and affine?
- Why do neural networks need nonlinearities?
- What does `X @ W` mean when `X` is a batch?
- What do the axes of a tensor mean?

---

# Acceptance criteria

Pass only when you can:

- calculate the basic vector and dot-product operations by hand;
- explain the calculations without relying on notation;
- reason about matrix shapes before running code;
- explain matrix-vector multiplication as multiple dot products;
- distinguish matrix multiplication from elementwise multiplication;
- explain $Wx+b$ as an affine transformation;
- explain why nonlinear activation changes the situation;
- implement and test the examples with NumPy;
- inspect `X @ W + b` and predict its output shape and meaning.

**Required evidence:** Python experiment, assertions, hand calculations, and short written explanations for dot product, cosine similarity, matrix multiplication and affine transformation.

---

# References and proof sources

1. Goodfellow, Bengio & Courville, *Deep Learning*, Chapter 2 — Linear Algebra. This chapter deliberately focuses on the linear-algebra topics needed for deep learning.  
   https://www.deeplearningbook.org/contents/linear_algebra.html
2. Gilbert Strang, MIT, *Lecture Notes for Linear Algebra*. Relevant sections include vectors, dot products, matrix-vector multiplication and matrix-matrix multiplication.  
   https://math.mit.edu/~gs/LectureNotes/
3. Gilbert Strang, *Introduction to Linear Algebra*, 6th edition resources.  
   https://math.mit.edu/~gs/linearalgebra/ila6/indexila6.html
4. GitHub Docs, *Writing mathematical expressions*. GitHub Markdown supports LaTeX math through MathJax and supports `$...$`, `$$...$$` and `math` fenced blocks.  
   https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions

**Proofs used in this lesson:** dot-product identities follow from the definition of the dot product and Euclidean geometry; affine composition follows by substitution and distributivity; the neural-layer calculations are direct matrix multiplication and componentwise application of ReLU.
