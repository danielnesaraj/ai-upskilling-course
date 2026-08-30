# L02 — Foundations: Probability, Distributions, Derivatives & Optimization

**Phase:** Foundations  
**Sequence:** 2  
**Format:** Compact textbook + multi-module work packet  
**Target effort:** ~5–6 hours across modules  

> **Purpose:** make uncertainty, change and optimization intuitive enough that later concepts such as loss functions, likelihoods, backpropagation and model training do not feel like mathematical witchcraft.

---

## How to use this lesson

The order matters:

**story → picture → tiny example → notation → calculation → AI connection → experiment.**

Do not memorize Bayes' rule, variance or gradient descent before you understand what each is trying to accomplish.

---

# Module 1 — Probability: reasoning when you do not know the outcome

## 1.1 Start with a prediction

Suppose you ask:

> Will it rain tomorrow?

You do not know the outcome yet, but you can still represent uncertainty.

You might say there is a 70% chance of rain.

Probability gives us a formal language for statements like this.

A probability is between 0 and 1:

$$0\le P(A)\le1$$

where $A$ is an event.

Think of the endpoints:

```text
0%                         50%                         100%
|---------------------------|----------------------------|
impossible              uncertain                    certain
```

The number is part of a mathematical model of uncertainty. It is not automatically the same thing as an observed frequency.

## 1.2 Events

Suppose we roll a die.

Event $A$ = "the result is even".

The possible outcomes are:

$$\{1,2,3,4,5,6\}$$

The event contains:

$$A=\{2,4,6\}$$

For a fair die:

$$P(A)=\frac36=\frac12$$

The probability of the entire set of possible outcomes is 1.

## 1.3 Why this matters in AI

AI systems constantly deal with uncertain outcomes:

- Is this email spam?
- Which class does this image belong to?
- Which token is likely next?
- How likely is a retrieved document to be relevant?
- How uncertain is a prediction?

Probability gives us vocabulary for these questions.

But be careful: an AI system outputting `0.9` does **not by itself prove** that the system's event has a true 90% probability. Calibration and the meaning of the output depend on the model and training setup.

---

# Module 2 — Distributions: where the probability is spread

## 2.1 Imagine a die rolled thousands of times

For a fair die, you expect roughly equal numbers of 1s through 6s over many trials.

A **probability distribution** describes how probability is allocated across possible outcomes.

For a fair die:

$$P(X=k)=\frac16,\qquad k\in\{1,2,3,4,5,6\}$$

Here $X$ is a random variable representing the die result.

## 2.2 Random variable does not mean "a variable that changes whenever I look"

A random variable is a mathematical function that assigns a numerical value to an outcome of a random process.

For example:

```text
coin outcome       X
-----------       ---
Heads              1
Tails               0
```

The randomness belongs to the process being modelled.

## 2.3 Discrete distributions

For a discrete random variable, we can assign a probability mass to each possible value:

$$p_X(x)=P(X=x)$$

All the masses add to 1:

$$\sum_x p_X(x)=1$$

Example:

```text
value       0     1     2
probability .2    .5    .3
```

Check:

$$0.2+0.5+0.3=1$$

## 2.4 Continuous distributions

Now consider a person's height.

There are enormously many possible values. Instead of assigning a separate positive probability to every exact real number, we can describe a **density**.

For density $f_X(x)$:

$$P(a\le X\le b)=\int_a^b f_X(x)\,dx$$

The important visual idea is:

```text
        density
          ↑
          |       /\
          |      /  \
          |_____/____\________→ x
                a    b

      probability = area between a and b
```

A density value is **not itself a probability**. The area under the density over an interval is the probability.

For an ordinary continuous random variable:

$$P(X=x)=0$$

for any exact point $x$.

A density can even be greater than 1. That is not a contradiction because density is not probability; its integral over the full domain is 1.

---

# Module 3 — Expected value: the weighted average

## 3.1 A familiar average

Suppose your marks are:

```text
70, 80, 90
```

The ordinary average is:

$$\frac{70+80+90}{3}=80$$

Now suppose some outcomes are more likely than others.

For a random variable, the expected value is a **probability-weighted average**.

For a discrete variable:

$$E[X]=\sum_x xP(X=x)$$

## 3.2 Dice example

For a fair die:

$$E[X]=1\left(\frac16\right)+2\left(\frac16\right)+\cdots+6\left(\frac16\right)$$

Therefore:

$$E[X]=\frac{1+2+3+4+5+6}{6}=3.5$$

A die never actually lands on 3.5.

That is important:

> **An expected value is a property of a distribution, not necessarily an outcome that can occur.**

## 3.3 Why expectation matters in AI

Many ML quantities are averages:

- average loss over examples;
- average reward;
- expected return;
- expected error;
- expected value of a random variable.

The same mathematical idea keeps reappearing: **weight possible values by how likely they are, then add them.**

## 3.4 Linearity of expectation

Expectation has a particularly useful property:

$$E[aX+bY+c]=aE[X]+bE[Y]+c$$

No independence assumption is required for this identity.

This is one reason expectations are so convenient in probability and ML mathematics.

---

# Module 4 — Variance: how spread out are the outcomes?

## 4.1 Why average alone is not enough

Consider two students:

```text
Student A:  49, 50, 51
Student B:  10, 50, 90
```

Both average 50.

But Student B's results are much more spread out.

We need another number to describe that spread.

## 4.2 Deviation from the mean

For a value $x$, the deviation from mean $\mu$ is:

$$x-\mu$$

For values 49, 50, 51 with mean 50:

```text
49 → -1
50 →  0
51 → +1
```

If we simply average the signed deviations, they cancel:

$$\frac{-1+0+1}{3}=0$$

That tells us nothing about the spread.

## 4.3 Square the deviations

Squaring gives:

```text
(-1)² = 1
 0²  = 0
(+1)² = 1
```

Now they cannot cancel.

Variance is the expected squared deviation from the mean:

$$\mathrm{Var}(X)=E[(X-E[X])^2]$$

Standard deviation is:

$$\sigma=\sqrt{\mathrm{Var}(X)}$$

The square root puts the measure back in the same units as $X$.

## 4.4 Useful identity — derive it, do not memorize it blindly

Let:

$$\mu=E[X]$$

Start with the definition:

$$\begin{aligned}
\mathrm{Var}(X)
&=E[(X-\mu)^2]\\
&=E[X^2-2\mu X+\mu^2]\\
&=E[X^2]-2\mu E[X]+\mu^2\\
&=E[X^2]-2\mu^2+\mu^2\\
&=E[X^2]-\mu^2.
\end{aligned}$$

Therefore:

$$\mathrm{Var}(X)=E[X^2]-(E[X])^2$$

The derivation uses only algebra plus linearity of expectation.

## 4.5 Why variance appears in ML

Variance is used to reason about:

- spread in data;
- noisy estimates;
- distributions of model outputs;
- uncertainty measures;
- optimization behavior;
- initialization and normalization techniques.

Do not jump to "high variance is bad." Whether variance is useful or harmful depends on what quantity we are measuring and why.

---

# Module 5 — Conditional probability: what changes when we learn something?

## 5.1 A simple example

Suppose 100 employees are classified as:

```text
                 Remote   Office
Junior             30       20
Senior             10       40
```

Suppose we randomly select an employee.

What is the probability they are senior?

There are 50 senior employees out of 100:

$$P(\mathrm{Senior})=0.5$$

Now suppose you learn that the employee works remotely.

Among the 40 remote employees, only 10 are senior:

$$P(\mathrm{Senior}\mid\mathrm{Remote})=\frac{10}{40}=0.25$$

The new information changed the probability.

## 5.2 Conditional probability formula

For $P(B)>0$:

$$P(A\mid B)=\frac{P(A\cap B)}{P(B)}$$

Read this as:

> probability of $A$ given that $B$ is known to have happened.

The denominator $P(B)$ effectively says: **we are now restricting attention to cases where $B$ occurred.**

## 5.3 Bayes' rule

From the definition:

$$P(A\cap B)=P(A\mid B)P(B)$$

We can also write:

$$P(A\cap B)=P(B\mid A)P(A)$$

Therefore:

$$P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}$$

This is Bayes' rule.

The useful mental model is:

```text
prior belief
     ↓
observe evidence
     ↓
how compatible is the evidence with the hypothesis?
     ↓
normalize
     ↓
updated belief
```

## 5.4 Medical-test example

Suppose a disease affects 1% of a population.

A test is positive for 99% of people who have the disease.

Suppose it is also positive for 5% of people who do not have it.

If someone tests positive, the question is **not** simply "the test is 99% accurate, therefore they have a 99% chance of disease."

Let:

- $D$ = disease;
- $+$ = positive test.

Then:

$$P(D\mid +)=\frac{P(+\mid D)P(D)}{P(+)}$$

This is the classic reason base rates matter.

## 5.5 AI connection

Conditional probability is fundamental to statements such as:

$$P(Y\mid X)$$

and later:

$$P(\text{next token}\mid\text{previous tokens})$$

The notation is simple once the intuition is clear: **what probability are we assigning after conditioning on information we already have?**

---

# Module 6 — Independence: when information does not change anything

Two events are independent when:

$$P(A\cap B)=P(A)P(B)$$

If $P(B)>0$, that is equivalent to:

$$P(A\mid B)=P(A)$$

In plain English:

> learning that $B$ happened gives us no information about whether $A$ happened.

For random variables, independence means the joint distribution factorizes. In the discrete case:

$$p(x,y)=p(x)p(y)$$

Do not use "independent" to mean "seems unrelated." It is a mathematical property.

This distinction becomes important later when we discuss naive Bayes, probabilistic models and assumptions made to make inference tractable.

---

# Module 7 — From probability to a machine-learning objective

## 7.1 A model makes a prediction

Let the model be:

$$\hat y=f_\theta(x)$$

We need a way to say how good or bad that prediction is.

A **loss function** gives us a number:

$$\ell(\hat y,y)$$

For a dataset with $N$ examples, a common objective is the average loss:

$$J(\theta)=\frac1N\sum_{i=1}^{N}\ell(f_\theta(x_i),y_i)$$

Now the problem becomes:

> Find parameter values $\theta$ that make $J(\theta)$ small.

This is optimization.

## 7.2 Loss is not automatically the same as a metric

Training might optimize cross-entropy while evaluation reports accuracy, F1, calibration or a task-specific metric.

A loss is an objective used by the training procedure. A metric is used to measure performance. Sometimes they are related; they are not automatically interchangeable.

---

# Module 8 — Derivatives: measuring change

This is where many people who hated high-school calculus get lost. Do not start with the limit notation.

Start with a question:

> If I change the input a tiny amount, how much does the output change?

## 8.1 Slope you already know

For a straight line:

$$y=2x$$

increase $x$ by 1 and $y$ increases by 2.

The slope is 2.

Now consider:

$$y=x^2$$

The slope changes depending on where you are.

```text
 y
 ↑
 |                 /
 |              __/
 |           __/
 |        __/
 |____ __/____________→ x
```

Near $x=1$, the curve is not as steep as near $x=5$.

The derivative tells us the **local slope**.

## 8.2 Formal definition

For a scalar function:

$$f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}$$

The fraction is the slope between two nearby points. Taking the limit as the gap approaches zero gives the slope at the point, when the limit exists.

## 8.3 Example

For:

$$f(x)=x^2$$

we get:

$$f'(x)=2x$$

At $x=3$:

$$f'(3)=6$$

Interpretation:

> around $x=3$, a small increase in $x$ produces roughly six times that small increase in $f(x)$.

The word **roughly** matters because this is a local approximation.

## 8.4 Finite differences

We can approximate a derivative numerically:

$$f'(x)\approx\frac{f(x+h)-f(x)}{h}$$

A symmetric version is:

$$f'(x)\approx\frac{f(x+h)-f(x-h)}{2h}$$

This is extremely useful for checking derivative implementations.

---

# Module 9 — Gradients: derivatives when there are many inputs

Suppose:

$$f(x,y)=x^2+3y^2$$

There are now two inputs.

We can ask:

- how sensitive is $f$ to $x$?
- how sensitive is $f$ to $y$?

Those are partial derivatives:

$$\frac{\partial f}{\partial x}=2x$$

and:

$$\frac{\partial f}{\partial y}=6y$$

Put them together:

$$\nabla f(x,y)=\begin{bmatrix}2x\\6y\end{bmatrix}$$

That vector is the **gradient**.

## 9.1 Visual intuition: a landscape

Imagine $f(x,y)$ is a landscape.

- $x$ and $y$ tell you where you stand.
- $f(x,y)$ tells you the height.
- The gradient tells you the direction of steepest local increase.

```text
                 uphill
                    ↗
                   /
             ●----/       gradient
            /\
           /  \
          /    \
      ----      ----
             position
```

The negative gradient points toward steepest local decrease under the standard Euclidean geometry.

This is the key intuition behind gradient descent.

## 9.2 Why this matters for neural networks

A neural network may have millions or billions of parameters.

The loss is one scalar:

$$J(\theta)$$

The gradient tells us how that scalar changes with respect to every parameter:

$$\nabla_\theta J$$

That is the information needed to decide how to change the parameters.

---

# Module 10 — Gradient descent: walking downhill

Suppose you are standing on a hill and want to reach a valley.

If you know the local downhill direction, you can take a small step that way.

Mathematically:

$$\theta_{t+1}=\theta_t-\eta\nabla J(\theta_t)$$

where $\eta>0$ is the learning rate.

Read it literally:

```text
current parameters
       ↓
calculate gradient
       ↓
point downhill = negative gradient
       ↓
take a step of size η
       ↓
new parameters
```

## 10.1 Why subtract?

A first-order Taylor approximation says:

$$J(\theta+\Delta)\approx J(\theta)+\nabla J(\theta)^T\Delta$$

Choose:

$$\Delta=-\eta\nabla J(\theta)$$

Then:

$$J(\theta+\Delta)\approx J(\theta)-\eta\lVert\nabla J(\theta)\rVert_2^2$$

For a sufficiently small positive learning rate, the local approximation predicts a decrease when the gradient is nonzero.

This is **not** a guarantee that every real training step decreases the loss.

## 10.2 Learning rate

The learning rate controls step size.

```text
η too small       → tiny steps → slow
η reasonable      → useful steps
η too large       → overshoot / instability possible
```

There is no universally correct learning rate. It depends on the objective, parameterization, optimizer and training setup.

---

# Module 11 — A toy training loop you can calculate by hand

Take:

$$J(w)=(w-3)^2$$

The minimum is obviously at $w=3$ because the square is zero there.

Derivative:

$$J'(w)=2(w-3)$$

Start at:

$$w_0=0$$

Use:

$$\eta=0.1$$

Update rule:

$$w_{t+1}=w_t-0.1\cdot2(w_t-3)$$

First step:

$$w_1=0-0.1(-6)=0.6$$

Second:

$$w_2=0.6-0.1(-4.8)=1.08$$

The sequence moves toward 3.

The mechanism is the important part:

**parameter → derivative → signed correction → new parameter.**

## Try different learning rates

Run the experiment with:

- $\eta=0.01$;
- $\eta=0.1$;
- $\eta=0.5$;
- $\eta=1.0$;
- $\eta=1.1$.

Do not guess which one works. Predict first, then observe.

For this particular quadratic, the update can be simplified to:

$$w_{t+1}-3=(1-2\eta)(w_t-3)$$

so convergence from any starting point occurs when:

$$|1-2\eta|<1$$

which gives:

$$0<\eta<1$$

This is a property of this particular objective and update rule, not a universal learning-rate rule for neural networks.

---

# Module 12 — Backpropagation is not gradient descent

This distinction is critical.

A neural network is a composition of functions:

```text
x
 ↓
layer 1
 ↓
activation
 ↓
layer 2
 ↓
loss
```

We need the derivative of the final loss with respect to many parameters.

**Backpropagation** efficiently computes those gradients through the computational graph using the chain rule.

**An optimizer** uses the gradients to update parameters.

So:

```text
backpropagation → computes gradients
optimizer       → uses gradients to update parameters
```

SGD and Adam are optimizers. Backpropagation is not an optimizer.

This distinction will matter when we debug training later.

---

# Module 13 — Full-batch and mini-batch training

Suppose the objective is:

$$J(\theta)=\frac1N\sum_{i=1}^{N}\ell_i(\theta)$$

The exact dataset gradient uses all $N$ examples.

For a mini-batch $B$:

$$J_B(\theta)=\frac1{|B|}\sum_{i\in B}\ell_i(\theta)$$

The gradient of the mini-batch objective is often used as an estimate of the full-data gradient. Under appropriate sampling assumptions, the mini-batch gradient can be an unbiased estimator of the full-dataset gradient.

Tradeoff:

```text
small batch → cheaper update / noisier estimate
large batch → more computation / often less noisy estimate
```

There is no universally optimal batch size.

---

# The whole story in one picture

```text
          DATA
            │
            ▼
      model prediction
            │
            ▼
       calculate loss
            │
            ▼
       compute gradient
       (backpropagation)
            │
            ▼
      optimizer update
            │
            ▼
      new parameters
            │
            └───────────────► repeat
```

Probability helps us describe uncertainty and distributions.

Expectation and variance help us reason about averages and spread.

Derivatives tell us how things change.

Gradients tell us how a scalar objective changes with many parameters.

Optimization uses those gradients to improve the parameters.

That is the mathematical skeleton of model training.

---

# Real-world application map

| Concept | Everyday example | AI connection |
|---|---|---|
| Probability | Chance of rain | Uncertain predictions |
| Distribution | Outcomes of repeated dice rolls | Modelled data/token distributions |
| Expectation | Probability-weighted average | Expected loss/reward |
| Variance | Consistency vs spread | Noise and uncertainty |
| Conditional probability | Risk after new information | $P(Y\mid X)$ |
| Bayes | Updating belief after evidence | Probabilistic inference |
| Derivative | Slope of a road | Sensitivity of loss |
| Gradient | Direction of steepest uphill | Parameter sensitivity |
| Gradient descent | Walking downhill | Training |
| Backpropagation | Efficiently tracing influence backward | Gradient computation |
| Mini-batch | Sampling part of a population | Efficient stochastic training |

---

# Experiments

Use **Python + NumPy**. Use PyTorch only for the final automatic-differentiation comparison.

Create:

`experiments/foundations/math02_probability_optimization.py`

## Experiment A — Build a PMF

Represent a small discrete distribution.

Verify:

$$\sum_x p(x)=1$$

Compute its expectation and variance in two ways:

1. from the definitions;
2. using the variance shortcut.

The answers should agree up to floating-point precision.

## Experiment B — Bayes with counts

Use the employee table from Module 5.

Calculate the conditional probabilities from raw counts rather than memorized formulas.

Then verify Bayes' rule numerically.

## Experiment C — Finite differences

For:

$$f(x)=x^2$$

compare:

$$2x$$

with forward and symmetric finite differences.

Try several values of $h$.

Observe both approximation error and numerical issues when $h$ becomes extremely small.

## Experiment D — Gradient descent

Implement gradient descent for:

$$J(w)=(w-3)^2$$

Try multiple learning rates.

Record:

- starting value;
- learning rate;
- first several parameter values;
- loss values;
- whether it converges;
- whether it oscillates;
- whether it diverges.

## Experiment E — Automatic differentiation

Use PyTorch autograd to calculate the derivative of:

$$f(w)=(w-3)^2$$

Compare the result with the analytical derivative.

The purpose is **not** to learn the PyTorch API yet. It is to prove that automatic differentiation is producing the mathematical derivative you expect.

### Required prediction

Before executing each experiment, write down what you expect to happen.

Then compare prediction vs observation.

That prediction step is part of the learning, not optional paperwork.

---

# Exercises

1. A variable takes values 0 and 1 with probabilities 0.7 and 0.3. Calculate its expectation.
2. Calculate its variance from the definition.
3. Verify the variance shortcut on the same example.
4. Explain why expectation is a weighted average.
5. Explain why $E[X+Y]=E[X]+E[Y]$ does not require independence.
6. Derive Bayes' rule from the definition of conditional probability.
7. Explain the difference between probability density and probability.
8. Explain why an expected value can be a number that never occurs.
9. For $f(w)=(w-4)^2$, derive the derivative.
10. Calculate one gradient-descent update from $w=0$ with $\eta=0.1$.
11. Explain why the negative gradient is used for minimization.
12. Explain why gradient descent does not guarantee a lower loss for an arbitrary step size.
13. Explain the difference between backpropagation and an optimizer.
14. Explain why mini-batch gradients can be noisy.
15. A training loss falls while validation loss rises. What might this indicate? What experiment would you run before drawing a conclusion?

---

# Questions you must be able to answer

- What does probability represent?
- What is a distribution?
- What is the difference between a PMF and a PDF?
- Why is a density not itself a probability?
- What does expectation mean intuitively?
- Why can expected value be an impossible outcome?
- What does variance measure?
- Why do we square deviations?
- What does conditional probability mean in plain English?
- Why does Bayes' rule need the prior and denominator?
- What does independence actually mean?
- What does a derivative tell us?
- What does a gradient tell us?
- Why does subtracting the gradient tend to move downhill locally?
- What does the learning rate control?
- Why can a learning rate be too large?
- What does backpropagation do?
- What does an optimizer do?
- Why use mini-batches?

---

# Acceptance criteria

Pass only when you can:

- calculate expectation and variance from a small distribution;
- derive and explain the variance shortcut;
- calculate conditional probability from counts;
- derive Bayes' rule;
- explain a PDF without calling density a probability;
- explain a derivative as local change/slope;
- calculate a gradient for a simple function;
- derive the basic gradient-descent update;
- explain why the learning rate matters;
- distinguish backpropagation from optimization;
- run the experiments and compare predictions with observations;
- explain the entire training loop without hiding behind framework terminology.

**Required evidence:** Python experiments, numerical assertions, prediction-vs-observation notes, analytical calculations, and a written explanation of Bayes, variance, gradients and gradient descent.

---

# References and proof sources

1. Goodfellow, Bengio & Courville, *Deep Learning*, Chapters 3–4: Probability and Information Theory; Numerical Computation.  
   https://www.deeplearningbook.org/
2. OpenStax, *Introductory Statistics*, probability and descriptive-statistics material.  
   https://openstax.org/details/books/introductory-statistics
3. MIT OpenCourseWare, calculus resources for derivatives and multivariable calculus.  
   https://ocw.mit.edu/search/?q=calculus
4. Christopher Bishop, *Pattern Recognition and Machine Learning*, probabilistic modelling and optimization foundations.  
   https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/

**Proofs/derivations in this lesson:** variance expansion follows from algebra and linearity of expectation; Bayes' rule follows by equating the two factorizations of the joint probability; the gradient-descent step follows from a first-order Taylor approximation; the toy quadratic's convergence condition is derived directly from its scalar recurrence.

**Formatting reference:** GitHub's official documentation confirms that Markdown math is rendered with MathJax and supports `$...$`, `$$...$$` and `math` fenced blocks.  
https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions
