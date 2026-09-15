---
layout: study-chapter
title: "Chapter 6. Graph SLAM and loop closure"
description: "A constraint-based representation in which past trajectory can still be corrected."
importance: 6
category: SLAM
series: slam_history
permalink: /study/slam/history/06-graph-slam/
---

> **Goal:** Separate the roles of front-end, back-end and factor.  
> **Workload:** 15 minutes. Uses the residual concept from Chapter 5.

## 1. Even a trajectory you have already travelled can be revised

A filter, as in Chapter 2, throws the past away. It keeps only the current state, so once pose $x_1$ has been folded into the estimate, there is no $x_1$ left to correct.

Graph SLAM keeps all of them. Every pose stays a variable you can still edit:

```text
  filter                          graph
  ───────────────                 ────────────────────────────
  keeps: x_3 only                 keeps: x_0, x_1, x_2, x_3
  past is gone                    all still editable

  new information                 new information can revise
  updates x_3                     any pose, including old ones
```

The measurements become **constraints** between those variables. "Odometry says $x_1$ is 1 m past $x_0$" is a constraint linking $x_0$ and $x_1$. A revisit constraint links $x_0$ and $x_3$ directly. The front-end produces these measurements; the back-end finds the set of poses that keeps as many of them as happy as possible.

"As happy as possible" is exactly a least-squares problem. Assuming Gaussian noise, the MAP estimate reduces to:

$$
X^*=\arg\min_X\sum_k r_k(X)^T\Omega_k r_k(X)
$$

Read it as: for every constraint $k$, measure how badly it is violated ($r_k$), weight that by how much you trust it ($\Omega_k$), and pick the poses $X$ that minimise the total. $\Omega_k$ is the information matrix, the inverse of the covariance — so a _confident_ measurement has a _large_ $\Omega$ and pulls harder. For the detailed probabilistic derivation, see the [KRoC Back-end lecture](https://drive.google.com/file/d/1FGnya__7ZQYsgE7CRhRjggeU2fIQ3izH/view).

## 2. What happens when you add a revisit constraint

```text
x0 ── x1 ── x2 ── x3
└─────────────────┘
    revisit constraint
```

When a relation between $x_0$ and $x_3$ is added to a graph that previously held only consecutive motion measurements, **the intermediate poses change too**. This is the part people expect least, so here it is with numbers.

Odometry says each step is 1 m, so the chain reads 0, 1, 2, 3. Then a revisit constraint says $x_3$ is really only 2.7 m from $x_0$ — a 0.3 m disagreement. Give all four measurements the same weight and minimise:

$$
E=(x_1-1)^2+(x_2-x_1-1)^2+(x_3-x_2-1)^2+(x_3-2.7)^2
$$

```text
  before                 the naive fix              what the graph does
  ────────────           ─────────────────          ────────────────────
  x0 = 0.000             x0 = 0.000                 x0 = 0.000
  x1 = 1.000             x1 = 1.000                 x1 = 0.925
  x2 = 2.000             x2 = 2.000                 x2 = 1.850
  x3 = 3.000             x3 = 2.700 ← just moved    x3 = 2.775
                                      the endpoint
  loop error 0.300       steps: 1.0, 1.0, 0.7       steps: .925, .925, .925

                         residuals                  residuals
                         odom  0, 0, -0.3           odom  -0.075 ×3
                         loop  0                    loop  +0.075
                         Σr²  = 0.090               Σr²  = 0.0225
```

The naive fix satisfies the loop exactly but tells an absurd story: the robot moved exactly 1 m twice and then suddenly 0.7 m. Its total squared violation is four times worse.

Two things are worth noticing in the right-hand column. The 0.3 m got spread across every step rather than dumped on the last one. And the graph did **not** fully honour the loop either — it landed at 2.775, leaving 0.075 of loop residual. That is correct behaviour: the loop closure is just one more noisy measurement, not a decree. Weight it more heavily (a smaller covariance, a larger $\Omega$) and the solution slides towards 2.7; weight it less and it slides back towards 3.0.

Question 2 below works through the same kind of problem step by step.

It is not a matter of simply attaching the last point. A graph of purely relative measurements with no absolute reference gives the same residual if the whole thing is translated and rotated together. That degree of freedom is handled by fixing a reference pose or adding a suitable prior. Stachniss's _Graph-Based SLAM and Sparsity_ explains the relationship between the graph and the matrix; the [recommended-materials guide](https://gisbi-kim.github.io/post/slam-textbooks/) suggests a reading order.

## 3. A wrong loop can be optimised too

The solver has no idea what a building looks like. It minimises whatever you hand it, and it will do so obediently even when the input is nonsense.

Suppose a first-floor corridor and a second-floor corridor look alike, and the front-end links them as the same place:

```text
  what is true                     what the constraint says

  floor 2   ────────────           "x_50 and x_120 are the same place"
                                              ↓
  floor 1   ────────────           the solver folds the building in half
                                   to make that true

                                   result: one corridor, low cost,
                                           completely wrong map
```

The cost went _down_. By its own scoring the solver improved the answer. A low final cost is evidence that the constraints agree with each other, not evidence that they are true.

So when studying a system, split it with two questions: **“why was this constraint created?”** is a front-end question, and **“how were the given constraints reconciled?”** is a back-end question. Most catastrophic SLAM failures are the first kind wearing the second kind's clothes.

When studying a system, split it with two questions: “why was this constraint created?” is a front-end question, and “how were the given constraints reconciled?” is a back-end question.

## Check questions

### Question 1 — Concept

If a single wrong loop closure comes in, does applying a robust kernel fully solve the problem? Answer from both the front-end and the back-end perspective.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

There is no guarantee it is fully solved. A robust kernel reduces the influence of factors with large residuals, but if a false loop has a small residual near the initial estimate, or an excessively large information matrix, it can act more strongly than the correct constraints. At the front-end, follow descriptor retrieval with geometric verification and temporal/spatial consistency checks; at the back-end you can use a robust loss, switchable constraints, DCS or graph consistency tests. You need both the stage that rejects wrong loops in the first place and the stage that reduces the influence of the outliers that get through.

</details>

### Question 2 — Math and reasoning

In a one-dimensional pose graph, fix $x_0=0$ with odometry measurements $x_1-x_0=1$ and $x_2-x_1=1$, and a loop measurement $x_2-x_0=1.8$. With all information weights equal, find the least-squares solution $x_1,x_2$.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

The objective is

$$
E=(x_1-1)^2+(x_2-x_1-1)^2+(x_2-1.8)^2.
$$

Setting the partial derivatives to zero,

$$
2x_1-x_2=0,\qquad -x_1+2x_2=2.8.
$$

The first gives $x_2=2x_1$; substituting into the second gives $3x_1=2.8$. Therefore

$$
x_1\approx0.9333,\qquad x_2\approx1.8667.
$$

You can see that the $0.2m$ loop error is not applied to the last pose alone but distributed across both odometry segments.

</details>

## Original reading

- KRoC Back-end: the part going from probability to least squares. Local copy: `_resource/slam/kroc2026/04-backend-younggun-cho.pdf`.
- For Stachniss (2016), use the description and links in the recommended guide. The old direct PDF address now returns 404, so it is not included in the local resources.

<aside class="study-summary" markdown="1">
## What you learned

<dl>
  <dt>Factor graph</dt><dd>A graph whose variables are connected by measurement or prior constraints.</dd>
  <dt>Front-end</dt><dd>The component that turns sensor data into proposed correspondences and constraints.</dd>
  <dt>Back-end</dt><dd>The component that optimises the variables to best satisfy all weighted constraints.</dd>
  <dt>Loop closure</dt><dd>A revisit constraint that lets the back-end redistribute accumulated error across past poses.</dd>
  <dt>Robust kernel</dt><dd>A loss function that reduces, but cannot guarantee removal of, the influence of outliers.</dd>
</dl>
</aside>
