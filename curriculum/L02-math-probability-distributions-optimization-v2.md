# L02 — Foundations: Probability, Distributions & Optimization

**Phase:** Foundations  
**Sequence:** 2  
**Estimated effort:** 2.5–3 hours

> Compact textbook/work packet. The goal is to understand the machinery that later becomes loss functions, likelihoods, sampling, gradients and training—not to finish a university probability or calculus course.

## 1. Why probability and optimization matter

Two questions appear throughout AI:

1. **What should the model believe about uncertain outcomes?**
2. **How do we change model parameters so its behavior improves?**

Probability gives us a language for uncertainty. Optimization gives us a procedure for choosing parameters that minimize an objective.

A simplified training loop is:

**parameters → predictions → loss → gradients → parameter update → repeat.**

Probability appears when we model distributions or likelihoods. Optimization appears when we adjust parameters.

---

## 2. Probability is a mathematical model of uncertainty

A probability measure assigns numbers between 0 and 1 to events.

$$0\le P(A)\le1$$

The probability of the entire sample space is 1. For disjoint events,

$$P(A\cup B)=P(A)+P(B).$$

For an event's complement,

$$P(A^c)=1-P(A).$$

These are mathematical properties of the model. An observed frequency in a finite dataset is not literally the same thing as probability, although frequencies can be used to estimate probabilities under appropriate assumptions.

### Random variable

A random variable maps outcomes of a random process to numerical values.

For example, define $X=1$ if a request succeeds and $X=0$ otherwise. The randomness belongs to the modeled process; it does not mean the variable must change every time you inspect it.

---

## 3. Distributions: where probability lives

A distribution tells us how probability is assigned across possible values.

### Discrete variables: probability mass function

For a discrete variable,

$$p_X(x)=P(X=x).$$

The masses sum to 1:

$$\sum_x p_X(x)=1.$$

For a fair die,

$$P(X=k)=\frac16,\qquad k\in\{1,2,3,4,5,6\}.$$

### Continuous variables: probability density function

A continuous variable can have density $f_X(x)$. Probability is obtained by integrating density over an interval:

$$P(a\le X\le b)=\int_a^b f_X(x)\,dx.$$

A crucial distinction:

> **A density value is not itself a probability.**

For an ordinary continuous distribution, the probability of any single exact point is zero:

$$P(X=x)=0.$$

Yet $f_X(x)$ can be positive, and a density can even exceed 1. What matters is the area under the density curve.

---

## 4. Expectation: the average implied by a distribution

For a discrete variable,

$$E[X]=\sum_x xP(X=x).$$

For a continuous variable,

$$E[X]=\int x f_X(x)\,dx,$$

when the expectation exists.

Think of expectation as a weighted average where the weights come from the distribution.

For a fair die:

$$E[X]=\frac{1+2+3+4+5+6}{6}=3.5.$$

The die never rolls 3.5. The expected value is a property of the distribution, not necessarily an outcome that can occur.

### Linearity of expectation

$$E[aX+bY+c]=aE[X]+bE[Y]+c.$$

This does **not** require independence between $X$ and $Y$.

That fact is useful because many ML quantities are sums or averages of random contributions.

---

## 5. Variance: measuring spread

Variance is the expected squared deviation from the mean:

$$\mathrm{Var}(X)=E[(X-E[X])^2].$$

It measures how far values tend to lie from the mean, with larger deviations receiving more weight because they are squared.

Standard deviation is the square root of variance:

$$\sigma=\sqrt{\mathrm{Var}(X)}.$$

### Why square the deviation?

If we averaged the signed deviation $X-E[X]$, positive and negative deviations would cancel. Squaring makes every deviation nonnegative and gives disproportionately more weight to large deviations.

### Derive the useful identity

Let $\mu=E[X]$.

Start from the definition:

$$\begin{aligned}
\mathrm{Var}(X)
&=E[(X-\mu)^2]\\
&=E[X^2-2\mu X+\mu^2]\\
&=E[X^2]-2\mu E[X]+\mu^2\\
&=E[X^2]-2\mu^2+\mu^2\\
&=E[X^2]-\mu^2.
\end{aligned}$$

Since $\mu=E[X]$,

$$\mathrm{Var}(X)=E[X^2]-(E[X])^2.$$

The point is to understand the derivation, not merely memorize the shortcut.

---

## 6. Conditional probability: probability after receiving information

Conditional probability asks:

> How should the probability of $A$ change once we know $B$ happened?

For $P(B)>0$,

$$P(A\mid B)=\frac{P(A\cap B)}{P(B)}.$$

Rearrange:

$$P(A\cap B)=P(A\mid B)P(B).$$

We can also write

$$P(A\cap B)=P(B\mid A)P(A).$$

Equating them:

$$P(A\mid B)P(B)=P(B\mid A)P(A).$$

Therefore,

$$P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}.$$

This is Bayes' rule.

### The intuition

Bayes' rule separates three ideas:

- $P(A)$ = prior belief;
- $P(B\mid A)$ = how expected the evidence is if $A$ is true;
- $P(A\mid B)$ = updated belief after seeing evidence $B$.

The denominator $P(B)$ normalizes the result so the resulting probability is valid.

Do not reduce Bayes' rule to "reverse the conditional probability." It is not generally true that $P(A\mid B)=P(B\mid A)$.

---

## 7. Independence

Events $A$ and $B$ are independent when

$$P(A\cap B)=P(A)P(B).$$

If $P(B)>0$, this is equivalent to

$$P(A\mid B)=P(A).$$

Intuitively, learning that $B$ happened does not change the probability of $A$.

For random variables, independence means their joint distribution factorizes. In the discrete case:

$$p(x,y)=p(x)p(y).$$

Independence is a mathematical assumption, not a synonym for "these things seem unrelated."

---

## 8. From probability to ML predictions

A supervised classifier may model something like

$$P(Y\mid X=x).$$

That notation means the model assigns a conditional distribution over possible labels given an input.

But do not automatically assume every number emitted by a model is a calibrated probability. Whether an output has probabilistic meaning depends on the model, output parameterization, training objective and calibration.

This distinction matters later when we evaluate confidence and uncertainty.

---

## 9. Loss functions: turning behavior into a number

Let a model with parameters $\theta$ produce

$$\hat y=f_\theta(x).$$

A per-example loss measures how undesirable that prediction is:

$$\ell(\hat y,y).$$

For a dataset of $N$ examples, a common objective is the mean loss:

$$J(\theta)=\frac1N\sum_{i=1}^{N}\ell(f_\theta(x_i),y_i).$$

Now the learning problem becomes:

> Find parameter values that make $J(\theta)$ small.

That sentence is the bridge from ML to optimization.

### Loss is not automatically the evaluation metric

A training procedure might optimize cross-entropy while we report accuracy, precision/recall, calibration or task-specific quality. A metric and a loss can coincide, but they serve different roles.

---

## 10. Derivatives: how sensitive is the objective?

For a scalar function $f(x)$, the derivative is

$$f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h},$$

when the limit exists.

Intuitively, it is the local slope: how much does the output change for a tiny change in the input?

For

$$f(x)=x^2,$$

we have

$$f'(x)=2x.$$

At $x=3$, the local slope is 6.

### Finite differences

For small $h$,

$$f'(x)\approx\frac{f(x+h)-f(x)}{h}.$$

A symmetric approximation is usually more accurate for smooth functions:

$$f'(x)\approx\frac{f(x+h)-f(x-h)}{2h}.$$

These are approximations. They are useful for checking an analytical or automatic derivative.

---

## 11. Gradients: derivatives with many inputs

For

$$f:\mathbb{R}^n\rightarrow\mathbb{R},$$

the gradient is

$$\nabla f(x)=\begin{bmatrix}
\frac{\partial f}{\partial x_1}\\
\vdots\\
\frac{\partial f}{\partial x_n}
\end{bmatrix}.$$

Each component answers:

> If I change this particular input slightly while holding the others fixed, how does the scalar output change locally?

The gradient points in the direction of greatest local increase under the standard Euclidean geometry. Therefore $-\nabla f$ is a local descent direction.

This is the key fact that makes gradient-based optimization possible.

---

## 12. Gradient descent: the mechanism

Given objective $J(\theta)$, basic gradient descent uses

$$\theta_{t+1}=\theta_t-\eta\nabla J(\theta_t),$$

where $\eta>0$ is the learning rate.

### Why subtract the gradient?

Use a first-order Taylor approximation:

$$J(\theta+\Delta)\approx J(\theta)+\nabla J(\theta)^T\Delta.$$

Choose

$$\Delta=-\eta\nabla J(\theta).$$

Then

$$J(\theta+\Delta)\approx J(\theta)-\eta\|\nabla J(\theta)\|^2.$$

For sufficiently small positive $\eta$, this predicts a decrease when the gradient is nonzero.

Important qualification: this is a **local approximation**, not a universal guarantee. A learning rate that is too large can overshoot or destabilize optimization.

### Learning-rate intuition

- too small → slow progress;
- appropriate → efficient progress;
- too large → overshooting or instability.

There is no universal best learning rate.

---

## 13. A complete toy optimization example

Take

$$f(w)=(w-3)^2.$$

The derivative is

$$f'(w)=2(w-3).$$

Start at $w_0=0$ with $\eta=0.1$.

The update is

$$w_{t+1}=w_t-0.1\cdot2(w_t-3).$$

First step:

$$w_1=0-0.1(-6)=0.6.$$

Second:

$$w_2=0.6-0.1(-4.8)=1.08.$$

The values move toward the minimum at $w=3$.

The important thing is not the arithmetic. It is the mechanism:

**current parameter → gradient at current parameter → signed step → new parameter.**

---

## 14. Backpropagation is not the optimizer

A neural network is a composition of many functions. We need gradients of the final loss with respect to many parameters.

**Backpropagation** efficiently computes those gradients using the chain rule through the computational graph.

**An optimizer** uses those gradients to update parameters.

These are different jobs.

For example, SGD and Adam are optimization algorithms; backpropagation is the gradient-computation procedure.

This distinction becomes extremely important when debugging training. A wrong gradient and a bad optimizer setting are different failure modes.

---

## 15. Full-batch vs mini-batch training

A dataset objective might be

$$J(\theta)=\frac1N\sum_{i=1}^{N}\ell_i(\theta).$$

Computing the exact gradient over a huge dataset for every update is expensive.

A mini-batch $B$ uses

$$J_B(\theta)=\frac1{|B|}\sum_{i\in B}\ell_i(\theta).$$

Its gradient can be used as an estimate of the full-dataset gradient under suitable sampling assumptions.

This creates a tradeoff:

- larger batches usually give less noisy gradient estimates per update but cost more memory/computation;
- smaller batches are cheaper per update but produce noisier estimates.

Do not memorize "small batch good" or "large batch good." The practical result depends on the model, hardware, optimizer, learning-rate schedule and workload.

---

## 16. Common misconceptions

**Probability = observed frequency.**  
No. Probability is part of a mathematical model; frequency is an observed statistic.

**A PDF value is a probability.**  
No. Density must be integrated over an interval to obtain probability.

**Expected value is an outcome you should see.**  
No. It is a weighted average and may be unattainable.

**Bayes' rule just reverses a conditional.**  
No. The prior and normalization term matter.

**The gradient points downhill.**  
No. The gradient points toward greatest local increase; the negative gradient is the descent direction.

**Gradient descent always decreases loss.**  
No. The argument is local and depends on step size and assumptions.

**Backpropagation is the optimizer.**  
No. Backpropagation computes gradients; the optimizer updates parameters.

**Training loss equals generalization.**  
No. A model can fit training data well while performing poorly on unseen data.

**A smaller finite-difference step is always better.**  
No. Floating-point rounding and cancellation eventually matter.

---

## 17. Implementation experiments

Use Python + NumPy. PyTorch is optional only for the final gradient comparison.

Create `experiments/foundations/math02_probability_optimization.py`.

### Experiment A — probability distribution

Represent a fair die as a PMF. Verify the probabilities sum to 1 and compute expectation and variance from definitions.

### Experiment B — finite differences

For $f(x)=x^2$, compare the analytical derivative $2x$ with finite-difference estimates for several values of $h$.

Observe what happens when $h$ is extremely large and extremely small.

### Experiment C — gradient descent

Implement gradient descent for $f(w)=(w-3)^2$.

Try several learning rates. Record whether the sequence converges, moves slowly, oscillates or diverges.

### Experiment D — automatic differentiation

Reproduce the same derivative using PyTorch autograd. Compare the result against the analytical derivative and finite difference.

The point is not to learn the PyTorch API yet. The point is to verify that automatic differentiation produces the mathematical gradient you expect.

### Required prediction before execution

Predict what should happen before running the code:

- small learning rate;
- moderate learning rate;
- excessive learning rate;
- starting on either side of the minimum.

Then compare predictions with observations.

---

## 18. Exercises

1. A random variable takes values 0 and 1 with probabilities 0.7 and 0.3. Compute expectation and variance.
2. Explain why $E[X+Y]=E[X]+E[Y]$ does not require independence.
3. Derive Bayes' rule from conditional probability.
4. Give an example where a density value exceeds 1 without violating probability axioms.
5. For $f(w)=(w-4)^2$, derive the gradient-descent update.
6. Explain why a finite-difference derivative is an approximation.
7. Explain the difference between a gradient and an optimizer.
8. Explain why a mini-batch gradient can be useful but noisy.
9. Given training loss decreasing while validation loss increases, what phenomenon might you suspect and what experiment would you run next?

---

## 19. Questions you must be able to answer

- What is a random variable?
- What is the difference between PMF and PDF?
- Why is a PDF value not itself a probability?
- What does expectation mean?
- Why can the expected value be impossible as an observed outcome?
- What does variance measure?
- Why is variance based on squared deviation?
- Does linearity of expectation require independence?
- What does $P(A\mid B)$ mean?
- Derive Bayes' rule.
- What is a loss function?
- What is a gradient?
- Why does gradient descent subtract the gradient?
- What does the learning rate control?
- Why does a mini-batch gradient differ from the full-dataset gradient?
- What exactly does backpropagation compute?
- What exactly does an optimizer do?

## 20. Acceptance criteria

The lesson passes when you can calculate expectation and variance for simple distributions, explain PMF versus PDF, derive Bayes' rule and the variance identity, explain gradient descent from the first-order approximation, numerically verify a derivative, demonstrate stable and unstable learning-rate behavior, and distinguish loss, gradients, backpropagation and optimization.

**Required evidence:** reproducible Python experiment, hand derivations, numerical derivative comparison, gradient-descent plots, and a short interpretation of observations.

## References and proof sources

1. Ian Goodfellow, Yoshua Bengio, Aaron Courville, *Deep Learning*, Chapter 3 (Probability and Information Theory) and Chapter 4 (Numerical Computation). MIT Press.  
   https://www.deeplearningbook.org/
2. Stanford CS229, course materials on probability, supervised learning and optimization.  
   https://cs229.stanford.edu/
3. MIT OpenCourseWare, Gilbert Strang's linear algebra and related mathematical materials.  
   https://ocw.mit.edu/
4. GitHub Docs, *Writing mathematical expressions*. GitHub documents `$...$`, `$$...$$`, and ` ```math ` syntax for mathematical expressions in Markdown.  
   https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions

**Proof/derivation note:** Bayes' rule follows algebraically from the definition of conditional probability. The variance identity follows by expanding $(X-\mu)^2$ and applying linearity of expectation. The gradient-descent direction follows from the first-order Taylor approximation and the Euclidean inner product.
