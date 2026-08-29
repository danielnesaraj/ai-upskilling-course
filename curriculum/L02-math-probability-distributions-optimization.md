# L02 — Foundations: Probability, Distributions & Optimization

**Phase:** Foundations  
**Sequence:** 2  
**Estimated effort:** 2.5–3 hours

## Purpose

ML systems represent uncertainty and optimize numerical objectives. This lesson builds the probability, calculus and optimization vocabulary needed later for loss functions, likelihood, sampling, gradients, backpropagation and LLM training.

This is not a complete probability or calculus course. It is the minimum foundation needed to reason about AI systems rather than merely call training APIs.

## Objectives

You should be able to:

- distinguish probability, random variables, distributions, samples and observations;
- distinguish PMFs from PDFs;
- calculate expectation and variance for simple discrete variables;
- derive conditional probability and Bayes' rule;
- understand derivatives and gradients as local sensitivity;
- explain why gradient descent subtracts the gradient;
- numerically estimate a derivative;
- distinguish gradients, backpropagation, loss functions and optimizers;
- explain why mini-batch gradients are useful estimates rather than exact full-dataset gradients.

## 1. Probability: representing uncertainty

Probability is a mathematical measure used to represent uncertainty about events or quantities.

For an event $A$:

$$0\le P(A)\le1$$

For mutually exclusive events $A$ and $B$:

$$P(A\cup B)=P(A)+P(B)$$

For any event:

$$P(A^c)=1-P(A)$$

These are properties of the probability model. They are not statements that a finite sample must exactly match the probabilities.

### Random variable

A random variable maps outcomes of an underlying experiment or uncertain process to numerical values. The word "random" refers to the probabilistic model, not necessarily to a value changing every time it is inspected.

For example, let $X=1$ mean a request succeeds and $X=0$ mean it fails. Then $X$ is a Bernoulli random variable.

---

## 2. Distributions

A probability distribution specifies how probability is assigned across possible values of a random variable.

### Discrete: PMF

For a discrete variable, the probability mass function is

$$p_X(x)=P(X=x)$$

and

$$\sum_x p_X(x)=1$$

For a fair die:

$$P(X=k)=\frac{1}{6},\qquad k\in\{1,2,3,4,5,6\}$$

### Continuous: PDF

A continuous random variable can be described by a probability density function $f_X(x)$. Probability comes from integrating density over an interval:

$$P(a\le X\le b)=\int_a^b f_X(x)\,dx$$

For a standard continuous distribution,

$$P(X=x)=0$$

for any single point, even though $f_X(x)$ may be positive.

**Critical distinction:** $f_X(x)$ is a density, not a probability. A density can exceed 1; the integral over the relevant domain must still equal 1.

---

## 3. Expectation: the distribution's weighted average

For a discrete variable:

$$E[X]=\sum_x xP(X=x)$$

For a continuous variable, when the expectation exists:

$$E[X]=\int x f_X(x)\,dx$$

Think of expectation as the average implied by the entire distribution.

For a fair die:

$$E[X]=\frac{1+2+3+4+5+6}{6}=3.5$$

A die never produces 3.5. The expected value is a weighted average, not necessarily an attainable outcome.

### Linearity of expectation

For random variables with defined expectations:

$$E[aX+bY+c]=aE[X]+bE[Y]+c$$

No independence assumption is required.

---

## 4. Variance: expected squared deviation

Variance measures the expected squared distance from the mean:

$$\mathrm{Var}(X)=E[(X-E[X])^2]$$

Standard deviation is

$$\sigma_X=\sqrt{\mathrm{Var}(X)}$$

A useful identity is

$$\mathrm{Var}(X)=E[X^2]-(E[X])^2$$

### Derivation

Let $\mu=E[X]$:

$$\begin{aligned}\mathrm{Var}(X)
&=E[(X-\mu)^2]\\
&=E[X^2-2\mu X+\mu^2]\\
&=E[X^2]-2\mu E[X]+\mu^2\\
&=E[X^2]-2\mu^2+\mu^2\\
&=E[X^2]-\mu^2
\end{aligned}$$

The important skill is not memorizing the shortcut; it is being able to derive it.

---

## 5. Conditional probability and Bayes' rule

Conditional probability asks: **what is the probability of $A$ given that $B$ is known?**

For $P(B)>0$:

$$P(A\mid B)=\frac{P(A\cap B)}{P(B)}$$

Rearrange:

$$P(A\cap B)=P(A\mid B)P(B)$$

The same intersection can be written the other way:

$$P(A\cap B)=P(B\mid A)P(A)$$

Equating them gives Bayes' rule:

$$P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}$$

The formula says that the updated probability depends on three things: the likelihood $P(B\mid A)$, the prior $P(A)$, and the normalizing evidence $P(B)$.

### Why this matters in AI

Many ML systems model conditional quantities such as

$$P(Y\mid X=x)$$

A model output should not automatically be treated as a calibrated probability. That interpretation depends on the model's output parameterization, training objective and calibration behavior.

---

## 6. Independence

Two events are independent when

$$P(A\cap B)=P(A)P(B)$$

For $P(B)>0$, this is equivalent to

$$P(A\mid B)=P(A)$$

For two discrete random variables, independence means

$$p(x,y)=p(x)p(y)$$

Independence is a mathematical assumption, not a synonym for "these things seem unrelated."

---

## 7. Loss functions: turning behavior into an objective

A parameterized model produces

$$\hat y=f_\theta(x)$$

A per-example loss measures discrepancy between prediction and target:

$$\ell(\hat y,y)$$

A common dataset objective is

$$J(\theta)=\frac{1}{N}\sum_{i=1}^{N}\ell(f_\theta(x_i),y_i)$$

Training attempts to find parameters that make the chosen objective good under the chosen optimization procedure and constraints.

This is an important conceptual correction: **"the model learns" is not a mechanism.** The model computes a function, the loss defines an objective, differentiation provides gradient information, and an optimizer changes parameters.

### Loss versus metric

A loss is an objective used during a training procedure. A metric is a measure used to assess behavior. They may be the same quantity, but they need not be.

A classifier, for example, could train using cross-entropy while being evaluated with accuracy or another metric.

---

## 8. Derivatives: local change

For a scalar function $f(x)$, the derivative is

$$f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}$$

when the limit exists.

It measures the local rate at which the output changes as the input changes.

For

$$f(x)=x^2$$

we have

$$f'(x)=2x$$

so at $x=3$, the derivative is 6.

### Finite differences

A forward finite difference approximates the derivative:

$$f'(x)\approx\frac{f(x+h)-f(x)}{h}$$

A central finite difference is generally more accurate for sufficiently small $h$:

$$f'(x)\approx\frac{f(x+h)-f(x-h)}{2h}$$

These are numerical approximations. Making $h$ smaller does not indefinitely improve floating-point calculations because rounding and cancellation eventually matter.

---

## 9. Gradients

For a scalar-valued function

$$f:\mathbb{R}^n\rightarrow\mathbb{R}$$

the gradient is the vector of partial derivatives:

$$\nabla f(x)=\begin{bmatrix}\frac{\partial f}{\partial x_1}\\\vdots\\\frac{\partial f}{\partial x_n}\end{bmatrix}$$

Under the ordinary Euclidean inner product, the gradient gives the direction of greatest local increase for a differentiable scalar function. Therefore the negative gradient is a local descent direction.

This is the mathematical basis of gradient descent.

---

## 10. Gradient descent

Given objective $J(\theta)$, basic gradient descent uses

$$\theta_{t+1}=\theta_t-\eta\nabla J(\theta_t)$$

where $\eta>0$ is the learning rate.

### Why subtract?

A first-order Taylor approximation gives

$$J(\theta+\Delta)\approx J(\theta)+\nabla J(\theta)^T\Delta$$

Choose

$$\Delta=-\eta\nabla J(\theta)$$

Then

$$J(\theta+\Delta)\approx J(\theta)-\eta\lVert\nabla J(\theta)\rVert_2^2$$

The change is negative in this approximation when $\eta>0$ and the gradient is nonzero.

This is a **local first-order argument**, not a universal guarantee that every actual update reduces the objective. Learning rate, curvature, numerical effects and nonconvexity all matter.

### Learning-rate intuition

- Too small: progress can be unnecessarily slow.
- Appropriate: progress can be efficient.
- Too large: updates can overshoot, oscillate or diverge.

There is no universal correct learning rate.

---

## 11. From calculus to ML training

For a model with many parameters:

1. parameters determine a function;
2. the function produces outputs;
3. the loss turns behavior into a scalar objective;
4. differentiation tells us how that objective changes with parameters;
5. an optimizer uses gradient information to update parameters;
6. repeated updates produce a new parameter set.

### Backpropagation is not the optimizer

Backpropagation is an efficient application of the chain rule for computing gradients through a composition of functions.

An optimizer takes gradient information and decides how to update parameters.

This distinction matters when debugging training:

- wrong gradient → investigate the computation graph/backpropagation or derivative implementation;
- sensible gradient but bad updates → investigate optimizer, learning rate, scaling, etc.;
- sensible training but poor validation behavior → investigate data, objective, regularization or generalization.

---

## 12. Full-batch and mini-batch gradients

A full dataset objective might be

$$J(\theta)=\frac{1}{N}\sum_{i=1}^{N}\ell_i(\theta)$$

and its exact gradient is

$$\nabla J(\theta)=\frac{1}{N}\sum_{i=1}^{N}\nabla\ell_i(\theta)$$

A mini-batch $B$ uses

$$J_B(\theta)=\frac{1}{|B|}\sum_{i\in B}\ell_i(\theta)$$

The mini-batch gradient is generally an estimate of the full-dataset gradient. Under suitable random sampling assumptions, it can be an unbiased estimator; its variance depends on the sampling and data.

This creates a practical tradeoff:

- larger batches generally reduce gradient noise per update but require more memory and computation per update;
- smaller batches are cheaper per update and noisier.

Do not reduce this to "SGD is always faster" or "larger batches are always better." The result depends on hardware, model, optimizer, data and workload.

---

## 13. Worked optimization example

Consider

$$f(w)=(w-3)^2$$

Then

$$f'(w)=2(w-3)$$

and gradient descent becomes

$$w_{t+1}=w_t-2\eta(w_t-3)$$

With $w_0=0$ and $\eta=0.1$:

$$w_1=0.6$$

then

$$w_2=1.08$$

The sequence moves toward the minimum at $w=3$.

The point of this example is not that real neural-network optimization is this simple. It is to make the update mechanics visible before introducing high-dimensional parameter spaces.

---

## 14. Proofs and derivations

### Bayes' rule

Start with

$$P(A\cap B)=P(A\mid B)P(B)$$

and

$$P(A\cap B)=P(B\mid A)P(A)$$

Equate them and divide by $P(B)>0$:

$$P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}$$

### Variance identity

The expansion in Section 4 proves

$$\mathrm{Var}(X)=E[X^2]-(E[X])^2$$

### Gradient-descent direction

From

$$f(x+\Delta)\approx f(x)+\nabla f(x)^T\Delta$$

the change is controlled by the inner product $\nabla f(x)^T\Delta$. For a fixed Euclidean step length, the most negative value occurs when $\Delta$ points opposite the gradient. Hence $-\nabla f$ is a steepest-descent direction.

The derivation explains the sign in the update rather than asking you to memorize it.

---

## 15. Common misconceptions

- **Probability and frequency are identical.** No. Probability is part of a model; frequency is observed data.
- **A PDF value is a probability.** No. It is a density; integrate it over an interval to obtain probability.
- **Expected value is necessarily an observable value.** No.
- **Bayes' rule simply reverses a probability.** No. It includes the prior and evidence terms.
- **The gradient points downhill.** No. The gradient points toward greatest local increase; the negative gradient is the descent direction.
- **Gradient descent guarantees lower loss after every update.** No.
- **Backpropagation is an optimizer.** No. It computes gradients; an optimizer uses them.
- **Training loss equals generalization performance.** No.
- **A smaller finite-difference step is always better.** No; floating-point error eventually matters.
- **Temperature is a general uncertainty measure.** No. In generative-model sampling, temperature modifies the sampling distribution/logit scaling; it is not by itself a measure of epistemic or aleatoric uncertainty.

---

## 16. Experiment

Use **Python + NumPy + Matplotlib**. Do not use an ML framework yet.

Create `experiments/foundations/math02_probability_optimization.py`.

### A — probability

Define a small discrete distribution and assert that probabilities sum to 1.

### B — expectation and variance

Compute expectation and variance manually. Also verify numerically that

$$\mathrm{Var}(X)=E[X^2]-(E[X])^2$$

### C — Bayes

Create a small finite example and calculate $P(A\mid B)$ both directly and through Bayes' rule. Assert that the results agree.

### D — derivative check

For $f(w)=(w-3)^2$, compare the exact derivative $2(w-3)$ with central finite differences for several $h$ values. Explain why the approximation eventually stops improving as $h$ becomes extremely small.

### E — gradient descent

Run gradient descent from multiple starting points. Try at least one sensible learning rate and one deliberately excessive learning rate. Record the parameter and objective trajectories.

### Required prediction before execution

Predict what should happen before running the code:

- small learning rate;
- moderate learning rate;
- excessive learning rate;
- starting on either side of the minimum.

Then compare predictions with observations.

---

## 17. Exercises

1. A random variable takes values 0 and 1 with probabilities 0.7 and 0.3. Compute expectation and variance.
2. Explain why $E[X+Y]=E[X]+E[Y]$ does not require independence.
3. Derive Bayes' rule from conditional probability.
4. Give an example where a density value exceeds 1 without violating probability axioms.
5. For $f(w)=(w-4)^2$, derive the gradient-descent update.
6. Explain why a finite-difference derivative is an approximation.
7. Explain the difference between a gradient and an optimizer.
8. Explain why a mini-batch gradient is useful but noisy.
9. If training loss decreases while validation loss increases, what phenomenon might you suspect? What experiment would you run next?

## 18. Questions you must be able to answer

- What is a random variable?
- What is the difference between a PMF and a PDF?
- Why is a PDF value not itself a probability?
- What does expectation mean?
- Why can expected value be impossible as an observed outcome?
- What does variance measure?
- Can expectation distribute over a sum without independence?
- What does $P(A\mid B)$ mean?
- Can you derive Bayes' rule?
- What is a loss function?
- What is a gradient?
- Why does gradient descent subtract the gradient?
- What does learning rate control?
- Why does a mini-batch gradient differ from the full-dataset gradient?
- What exactly does backpropagation compute?
- What exactly does an optimizer do?

## 19. Acceptance criteria

Pass only when you can calculate expectation and variance for simple distributions, explain PMF versus PDF, derive Bayes' rule and the variance identity, explain gradient descent from the first-order approximation, numerically verify a derivative, demonstrate stable and poorly chosen learning-rate behavior, and distinguish backpropagation, gradients, losses and optimizers.

### Required evidence

- Reproducible Python/NumPy experiment committed to GitHub.
- Hand derivations for Bayes' rule, variance identity and gradient-descent direction.
- Numerical derivative comparison.
- Plot showing stable and poorly chosen learning-rate behavior.
- Short written interpretation of the observations.

## References and proof sources

1. Ian Goodfellow, Yoshua Bengio, Aaron Courville, *Deep Learning*, Chapters 3 and 4.  
   https://www.deeplearningbook.org/
2. Stanford CS229 course materials, including probability and optimization foundations.  
   https://cs229.stanford.edu/
3. MIT OpenCourseWare, calculus and linear-algebra materials for supporting mathematical foundations.  
   https://ocw.mit.edu/

**Source note:** central mathematical claims are derived in the lesson where practical. Optimization claims are deliberately qualified: gradient descent is a local first-order method, not a universal guarantee of monotonic improvement.
