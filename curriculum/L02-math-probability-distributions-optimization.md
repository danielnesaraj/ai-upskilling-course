# L02 — Math 2: Probability, Distributions & Optimization Intuition

**Phase:** Foundations  
**Week:** 1  
**Sequence:** 2  
**Estimated effort:** 2.5 hours

## Purpose

ML systems operate under uncertainty and are trained by optimizing numerical objectives. This lesson establishes the probability and optimization vocabulary needed to understand loss functions, likelihood, sampling, gradients, and later LLM training/inference.

The goal is **engineering fluency**, not a full probability or calculus course.

## Objectives

By the end, you should be able to:

- distinguish probability, random variables, distributions, samples, parameters, and observations;
- calculate expectation and variance for simple discrete random variables;
- understand conditional probability and Bayes' rule;
- distinguish probability mass functions from probability density functions;
- understand why a continuous density value is not itself a probability;
- explain a loss function as an objective over model predictions;
- understand derivatives and gradients as local sensitivity information;
- explain gradient descent and learning rate;
- numerically estimate a derivative with finite differences;
- connect probability and optimization to ML training and inference.

## Prerequisites

L01: vectors, matrices, functions, and basic algebra.

---

## 1. Probability as a model of uncertainty

Probability provides a mathematical way to represent uncertainty about outcomes of a random process or uncertain quantities.

For an event \(A\),

\[
0\le P(A)\le1.
\]

For mutually exclusive events \(A\) and \(B\),

\[
P(A\cup B)=P(A)+P(B).
\]

For any event,

\[
P(A^c)=1-P(A).
\]

These are properties of probability measures, not empirical guarantees about any particular finite dataset.

### Random variable

A random variable maps outcomes of a random experiment to numerical values. It is **not necessarily random in the sense of changing every time you look at it**; the randomness is part of the probabilistic model.

For example, if a model represents whether a request succeeds, a Bernoulli random variable \(X\) can take values 1 or 0.

---

## 2. Distributions

A probability distribution describes how probability is assigned to possible values of a random variable.

### Discrete: PMF

For a discrete random variable, a probability mass function (PMF) gives

\[
p_X(x)=P(X=x).
\]

The probabilities sum to one:

\[
\sum_x p_X(x)=1.
\]

Example: a fair six-sided die has

\[
P(X=k)=\frac16,\quad k\in\{1,2,3,4,5,6\}.
\]

### Continuous: PDF

A continuous random variable can be described by a probability density function (PDF) \(f_X(x)\). Probabilities are obtained by integrating density over an interval:

\[
P(a\le X\le b)=\int_a^b f_X(x)\,dx.
\]

For a continuous variable,

\[
P(X=x)=0
\]

for an individual point under the usual continuous-distribution model, even though the density at \(x\) may be positive.

**Critical distinction:** a PDF value is a density, not a probability. It can exceed 1 when the density is concentrated over a sufficiently small interval.

---

## 3. Expectation: the average implied by a distribution

For a discrete random variable,

\[
E[X]=\sum_x xP(X=x).
\]

For a continuous random variable,

\[
E[X]=\int x f_X(x)\,dx
\]

when the expectation exists.

Expectation is a weighted average over the distribution.

### Example

For a fair die,

\[
E[X]=\frac{1+2+3+4+5+6}{6}=3.5.
\]

This does **not** mean a die roll can produce 3.5. It means 3.5 is the long-run average under the model.

### Linearity of expectation

For random variables with defined expectations,

\[
E[aX+bY+c]=aE[X]+bE[Y]+c.
\]

Importantly, this does **not** require \(X\) and \(Y\) to be independent.

---

## 4. Variance: how spread out a variable is

Variance is

\[
\operatorname{Var}(X)=E[(X-E[X])^2].
\]

It measures expected squared deviation from the mean.

The standard deviation is

\[
\sigma_X=\sqrt{\operatorname{Var}(X)}.
\]

### Useful identity

\[
\operatorname{Var}(X)=E[X^2]-(E[X])^2.
\]

#### Derivation

Let \(\mu=E[X]\). Then

\[
\begin{aligned}
\operatorname{Var}(X)
&=E[(X-\mu)^2]\\
&=E[X^2-2\mu X+\mu^2]\\
&=E[X^2]-2\mu E[X]+\mu^2\\
&=E[X^2]-2\mu^2+\mu^2\\
&=E[X^2]-\mu^2.
\end{aligned}
\]

This identity is worth being able to derive because it appears in statistics, optimization, normalization, and ML analysis.

---

## 5. Conditional probability

Conditional probability asks: **what is the probability of A given that B is known to have occurred?**

For \(P(B)>0\),

\[
P(A\mid B)=\frac{P(A\cap B)}{P(B)}.
\]

Rearranging gives the product rule:

\[
P(A\cap B)=P(A\mid B)P(B).
\]

Similarly,

\[
P(A\cap B)=P(B\mid A)P(A).
\]

Equating these produces Bayes' rule:

\[
P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}.
\]

### Why this matters in AI

Many ML problems can be expressed probabilistically. For example, a classifier may model or approximate

\[
P(Y\mid X=x),
\]

rather than directly producing an unqualified statement of certainty.

Do not automatically interpret every model output as a calibrated probability. Whether an output is probabilistically meaningful depends on the model, training objective, output parameterization, and calibration.

---

## 6. Independence

Two events are independent if

\[
P(A\cap B)=P(A)P(B).
\]

When \(P(B)>0\), this is equivalent to

\[
P(A\mid B)=P(A).
\]

Independence is a strong assumption. It should not be confused with events merely being unrelated in ordinary language.

For random variables, independence means their joint distribution factorizes appropriately; for two variables,

\[
p(x,y)=p(x)p(y)
\]

in the discrete case, with analogous factorization for continuous distributions.

---

## 7. Loss functions: turning error into a number

An ML model often has parameters \(\theta\) and produces predictions

\[
\hat y=f_\theta(x).
\]

A loss function measures error for an example:

\[
\ell(\hat y,y).
\]

A training objective commonly aggregates losses over a dataset:

\[
J(\theta)=\frac{1}{N}\sum_{i=1}^{N}\ell(f_\theta(x_i),y_i).
\]

Training then seeks parameters that make the objective small, subject to the chosen optimization procedure and any regularization or constraints.

The exact objective matters. "The model learns" is not a mechanism; the optimization objective and update procedure are part of the mechanism.

### Loss versus metric

A **loss** is an objective used by a training procedure; a **metric** is a measure used to assess behavior. They can be the same quantity, but they do not have to be.

For example, a classifier might be trained with cross-entropy while reported with accuracy, precision, recall, or another metric.

---

## 8. Derivatives: local sensitivity

For a scalar function \(f(x)\), the derivative at \(x\) is

\[
f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h},
\]

when the limit exists.

Intuitively, it describes the local rate at which the output changes as the input changes.

For example,

\[
f(x)=x^2 \Rightarrow f'(x)=2x.
\]

At \(x=3\), the derivative is 6.

### Finite-difference approximation

For small \(h\),

\[
f'(x)\approx\frac{f(x+h)-f(x)}{h}.
\]

A more accurate symmetric approximation is

\[
f'(x)\approx\frac{f(x+h)-f(x-h)}{2h}.
\]

These are numerical approximations, not exact derivatives.

---

## 9. Gradients

For a scalar-valued function of multiple variables,

\[
f:\mathbb{R}^n\rightarrow\mathbb{R},
\]

the gradient is

\[
\nabla f(x)=
\begin{bmatrix}
\frac{\partial f}{\partial x_1}\\
\vdots\\
\frac{\partial f}{\partial x_n}
\end{bmatrix}.
\]

The gradient points in the direction of greatest local increase of a differentiable scalar function under the standard Euclidean inner product. Therefore \(-\nabla f\) is a local descent direction when the gradient is nonzero.

This gives the core idea behind gradient descent.

---

## 10. Gradient descent

Given objective \(J(\theta)\), the basic gradient-descent update is

\[
\theta_{t+1}=\theta_t-\eta\nabla J(\theta_t),
\]

where \(\eta>0\) is the learning rate.

### Why subtract?

A first-order Taylor approximation gives

\[
J(\theta+\Delta)\approx J(\theta)+\nabla J(\theta)^T\Delta.
\]

Choose

\[
\Delta=-\eta\nabla J(\theta).
\]

Then

\[
J(\theta+\Delta)\approx J(\theta)-\eta\|\nabla J(\theta)\|^2.
\]

For sufficiently small positive \(\eta\), this predicts a decrease in the objective locally whenever the gradient is nonzero.

This is a **local approximation**, not a guarantee that every update decreases the true objective. Large learning rates, nonconvex objectives, numerical issues, and other factors can cause increases or instability.

### Learning rate intuition

- Too small: updates can be slow.
- Reasonably chosen: progress can be efficient.
- Too large: updates can overshoot or become unstable.

There is no universal correct learning rate.

---

## 11. Why this becomes ML training

For a model with millions or billions of parameters, we cannot manually choose each parameter. Instead:

1. parameters define a function;
2. the function produces predictions;
3. a loss/objective measures error;
4. differentiation tells us how the objective changes with parameters;
5. an optimizer uses gradient information to update parameters;
6. repeated updates produce a trained parameter set.

Backpropagation, introduced later, is an efficient application of the chain rule for computing gradients through compositions of functions.

An optimizer is not the same thing as backpropagation:

- **backpropagation** computes gradients efficiently;
- **the optimizer** decides how to use those gradients to update parameters.

This distinction becomes important in neural-network debugging.

---

## 12. Batch, stochastic and mini-batch intuition

The full-dataset objective may be

\[
J(\theta)=\frac1N\sum_{i=1}^N \ell_i(\theta).
\]

Computing its exact gradient can be expensive for large datasets.

A mini-batch uses a subset \(B\):

\[
J_B(\theta)=\frac1{|B|}\sum_{i\in B}\ell_i(\theta).
\]

Its gradient is an estimate of the full-dataset gradient under appropriate sampling assumptions.

This creates a tradeoff:

- larger batches usually give a less noisy estimate per update but require more computation/memory per update;
- smaller batches provide cheaper, noisier updates and may require more updates.

Do not reduce this to "SGD is always faster" or "bigger batches are always better." The practical result depends on hardware, data, optimizer, learning-rate schedule, model, and workload.

---

## 13. A simple optimization example

Consider

\[
f(w)=(w-3)^2.
\]

Its derivative is

\[
f'(w)=2(w-3).
\]

Gradient descent gives

\[
w_{t+1}=w_t-\eta\,2(w_t-3).
\]

With \(w_0=0\) and \(\eta=0.1\):

\[
w_1=0.6,
\]

then

\[
w_2=1.08,
\]

and the sequence moves toward the minimum at \(w=3\).

This toy problem is deliberately simple. The point is to make the update mechanics visible before introducing neural networks.

---

## 14. Derivations / proofs worth knowing

### 14.1 Bayes' rule

Start with the product rule:

\[
P(A\cap B)=P(A\mid B)P(B).
\]

Also,

\[
P(A\cap B)=P(B\mid A)P(A).
\]

Therefore,

\[
P(A\mid B)P(B)=P(B\mid A)P(A).
\]

Divide by \(P(B)>0\):

\[
P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}.
\]

### 14.2 Variance identity

Derived in Section 4:

\[
\operatorname{Var}(X)=E[X^2]-(E[X])^2.
\]

You should be able to reproduce the expansion.

### 14.3 Gradient descent direction

From the first-order approximation,

\[
f(x+\Delta)\approx f(x)+\nabla f(x)^T\Delta.
\]

For a fixed step length \(\|\Delta\|\), the inner product is minimized by choosing \(\Delta\) opposite the gradient. Thus \(-\nabla f\) is a steepest-descent direction under the Euclidean norm.

This explains the sign of the gradient-descent update; it is not merely a memorized rule.

---

## 15. Common misconceptions

- **"Probability and frequency are identical."** Probability is a model/measure; observed frequency is data. Frequencies can estimate probabilities under suitable assumptions and increasing data, but they are not definitionally identical.
- **"A PDF value is a probability."** It is a density. Integrate over a region to obtain probability.
- **"Expected value is a value you should expect to observe."** It is a weighted average; it may not be an attainable outcome.
- **"Bayes' rule says reverse the probability."** No. It includes the prior and normalization term.
- **"A gradient points downhill."** The gradient points toward greatest local increase; the negative gradient is the local descent direction.
- **"Gradient descent guarantees lower loss after every step."** No. The local approximation supports small enough steps under suitable differentiability conditions; practical training can increase loss.
- **"Backpropagation is the optimizer."** No. Backpropagation computes gradients; optimizers use gradients to update parameters.
- **"Training loss is the same as generalization."** No. A model can fit training data well and still perform poorly on unseen data.
- **"Randomness means the model is broken."** Not necessarily. Sampling and stochastic optimization intentionally introduce randomness in many systems.
- **"Temperature is the same thing as model uncertainty."** No. In generative-model inference, temperature modifies sampling/logit scaling; it is not a general-purpose measure of epistemic or aleatoric uncertainty.

---

## 16. Implementation / experiment

Use Python + NumPy + Matplotlib. Do not use an ML framework yet.

Create `experiments/foundations/math02_probability_optimization.py` or an equivalent notebook that:

1. defines a small discrete distribution and verifies probabilities sum to 1;
2. computes expectation and variance manually and with code;
3. verifies `Var(X) = E[X**2] - E[X]**2`;
4. demonstrates Bayes' rule using a small finite example;
5. defines \(f(w)=(w-3)^2\);
6. computes the exact derivative \(2(w-3)\);
7. estimates the derivative numerically using a central finite difference;
8. compares the numerical derivative with the exact derivative for several \(h\) values;
9. runs gradient descent from several starting points;
10. deliberately chooses a learning rate that is too large and records what happens;
11. plots the objective and the parameter trajectory.

### Required prediction before execution

Before running gradient descent, predict:

- what happens for a small learning rate;
- what happens for a moderate learning rate;
- what happens when the learning rate is too large;
- why the sign of the update changes on opposite sides of the minimum.

Then test the predictions.

---

## Exercises

1. A random variable takes values 0 and 1 with probabilities 0.7 and 0.3. Compute its expectation and variance.
2. Explain why \(E[X+Y]=E[X]+E[Y]\) does not require independence.
3. Derive Bayes' rule from the definition of conditional probability.
4. Give an example where a density value exceeds 1 without violating probability axioms.
5. For \(f(w)=(w-4)^2\), derive the gradient-descent update.
6. Explain why a finite-difference derivative is an approximation.
7. Explain the difference between a gradient and an optimizer.
8. Explain why a mini-batch gradient is useful but noisy.
9. Given a loss that decreases on the training set but increases on validation data, what phenomenon might you suspect and what experiment would you run next?

---

## Questions you should be able to answer

- What is a random variable?
- What is the difference between a PMF and a PDF?
- Why is a PDF value not itself a probability?
- What does expectation mean?
- Why can the expected value be impossible as an observed outcome?
- What does variance measure?
- Can expectation be distributed over a sum without independence?
- What does \(P(A\mid B)\) mean?
- Derive Bayes' rule.
- What is a loss function?
- What is a gradient?
- Why does gradient descent subtract the gradient?
- What does the learning rate control?
- Why does a mini-batch gradient differ from the full-dataset gradient?
- What exactly does backpropagation compute?
- What exactly does an optimizer do?

---

## Acceptance criteria

The lesson is complete only when you can:

- calculate expectation and variance for simple discrete distributions;
- explain PMF versus PDF correctly;
- derive Bayes' rule and the variance identity;
- explain gradient descent from the first-order approximation;
- numerically verify a derivative with finite differences;
- demonstrate stable and unstable learning-rate behavior;
- distinguish backpropagation, gradients, loss, and optimization;
- explain why mini-batch training is an approximation to the full objective gradient.

## Required evidence

- Reproducible Python/NumPy experiment committed to GitHub.
- Hand derivations for Bayes' rule, variance identity, and gradient-descent direction.
- Numerical derivative comparison.
- Plot showing at least one stable and one unstable/poorly chosen learning-rate run.
- Short written interpretation of the observed behavior.

## Evaluation criteria

| Dimension | What is being checked |
|---|---|
| Conceptual understanding | Correct probability and optimization mental models |
| Implementation correctness | Numerical calculations and gradient experiment are correct |
| AI relevance | Correct connection to ML loss, training, mini-batches and later LLM training |
| Engineering quality | Reproducible experiment and clear assertions/plots |
| Experimental methodology | Predictions are made before execution and compared with observations |
| Reasoning & tradeoffs | Can explain learning-rate and batch-size tradeoffs without slogans |

## References

1. Ian Goodfellow, Yoshua Bengio, Aaron Courville, *Deep Learning*, MIT Press — Chapters 3 (Probability and Information Theory) and 4 (Numerical Computation): https://www.deeplearningbook.org/
2. Stanford CS229 course materials — probability, supervised learning, linear regression, gradient descent and optimization topics: https://cs229.stanford.edu/summer2020/syllabus.html
3. MIT OpenCourseWare / Gilbert Strang linear-algebra materials — supporting mathematical foundations: https://math.mit.edu/~gs/LectureNotes/

**Source note:** Probability identities and calculus results used here are standard mathematical results. The lesson includes their derivations where they are central to later AI mechanisms. Optimization statements are intentionally qualified: gradient descent is a local first-order method, not a universal guarantee of monotonic improvement.
