---
layout: study-chapter
title: "Chapter 3. FastSLAM and conditional independence"
description: "Splitting one large joint estimation problem into a path problem and landmark problems."
importance: 3
category: SLAM
series: slam_history
permalink: /study/slam/history/03-fastslam/
---

> **Goal:** Explain the core of FastSLAM through the condition “given the path”.  
> **Workload:** 10–15 minutes. Read after Chapter 2.

## 1. Do we have to estimate every point at once?

Chapter 2 ended with a problem: every landmark is correlated with every other one, so the covariance matrix has to store all those pairs. With 1,000 landmarks that is a $2001 \times 2001$ matrix, and updating it costs roughly the square of that.

But look again at _why_ they were correlated. Pillars A and B were wrong together only because they were both measured from the same uncertain pose. So ask a hypothetical question:

```text
"Suppose someone told me the exact path I took.
 Would A and B still be correlated?"

  → No. With the pose known exactly, A's position comes only from
    A's own measurement, and B's only from B's. Nothing is shared.
```

That is the whole trick of [FastSLAM](https://www.cs.cmu.edu/~thrun/papers/montemerlo.fastslam-tr.html) (2002). **Once the path is given**, the landmark estimates decouple, and 1,000 landmarks become 1,000 tiny independent problems instead of one huge coupled one.

Nobody hands us the true path, of course. So FastSLAM guesses it — many times over. Each guess is a particle, and inside each particle the path is treated as known, which makes the cheap decoupled version legal there. This works under model assumptions such as static landmarks and conditionally independent observations.

$$
p(x_{0:t},m\mid z_{1:t},u_{1:t})
=p(x_{0:t}\mid z_{1:t},u_{1:t})
\prod_j p(m_j\mid x_{0:t},z_{1:t})
$$

This notation is a study-friendly form that omits the condition that data association is given. It also assumes the observation noise of different landmarks is independent.

## 2. A particle is not a single point on the map

A common misreading is that a particle is a guessed _position_. It is not. **Each particle carries a whole path guess and its own private map** built on that guess.

```text
particle 1   path: turned at x=4.8   its map: door at 9.8,  corner at 14.9
particle 2   path: turned at x=5.0   its map: door at 10.0, corner at 15.0
particle 3   path: turned at x=5.3   its map: door at 10.3, corner at 15.3
```

Three particles, three different maps. None of them is "the map" yet.

Now a measurement arrives: the door is 10.0 m away. Score each particle by how well its own map predicted that.

```text
              predicted    measured    miss      weight
 particle 1      9.8         10.0      0.2       low
 particle 2     10.0         10.0      0.0       high
 particle 3     10.3         10.0      0.3       low
```

Particle 2 gets the large weight. Resampling then copies the high-weight particles and drops the low-weight ones, so the next round is mostly made of paths near "turned at x = 5.0". The path was never solved directly — it was decided by which guess kept predicting well.

The maps are not unconditionally independent. The decomposition is legal only because the path is conditioned on _within each particle_.

## 3. A small thought experiment

Here is where particles beat a single Gaussian outright. There are two identical doors, at x = 10 and x = 30, and the robot is in front of one of them but does not know which.

An EKF stores one mean and one variance, so the best it can say is the average:

```text
  EKF:        x = 20.0,  σ = 10     ← "probably around 20"

                 10          20          30
                 │           ▲           │
               door        the EKF's    door
                           answer
                     nothing is actually here
```

The one place the EKF points at is the one place the robot certainly is not. Particles have no such problem, because they are just a list:

```text
  particles:  [10, 10, 10, 30, 30, 30]   ← "either 10 or 30, about 50/50"
```

That list says something an EKF cannot express: two answers, both live, nothing in between.

But if the next observation also fails to distinguish the two doors, keeping several hypotheses does not solve the problem by itself. You also have to think about the number of hypotheses, the computational cost, and the diversity that survives resampling.

## Check questions

### Question 1 — Concept

When we say landmarks are independent in FastSLAM, exactly what condition is attached? Why is it wrong to drop that condition and say simply “map points are independent of each other”?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Under model assumptions such as static landmarks, conditionally independent observation noise and known data association, the landmark posteriors separate conditionally **once the robot path is given**. If the path is uncertain, the same path error affects several landmarks in common, so marginally they depend on each other. FastSLAM represents path hypotheses as particles and keeps a per-landmark estimator inside each particle in order to exploit this structure.

</details>

### Question 2 — Math and reasoning

Three particles have unnormalised weights $[0.1,0.3,0.6]$. With the effective sample size defined as $N_{eff}=1/\sum_i w_i^2$, compute it, and decide whether to resample if the threshold is $N/2$.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

The weights already sum to 1, so

$$
N_{eff}=\frac{1}{0.1^2+0.3^2+0.6^2}
=\frac{1}{0.46}\approx2.17.
$$

With $N=3$ the threshold is $1.5$, so under this rule we do not resample. That said, $N_{eff}$ alone does not tell you whether the hypotheses are geometrically diverse enough. Resampling too often causes particle impoverishment; resampling too late means you keep spending computation on low-weight particles.

</details>

## Original reading

- Montemerlo et al. (2002), FastSLAM: read around Figure 1 and the posterior factorisation. [Original PDF](https://www.cs.cmu.edu/~thrun/papers/montemerlo.fastslam-tr.pdf).
- From the next chapter we pause to set up the geometry needed by both filtering and optimisation.

<aside class="study-summary" markdown="1">
## What you learned

<dl>
  <dt>FastSLAM</dt><dd>A Rao–Blackwellised particle filter that samples robot paths and estimates landmarks inside each path hypothesis.</dd>
  <dt>Conditional independence</dt><dd>Landmark estimates separate only after the robot path is given.</dd>
  <dt>Particle weight</dt><dd>The relative likelihood of one path hypothesis given the observations.</dd>
  <dt>Resampling</dt><dd>Replicating likely particles and discarding unlikely ones to focus computation.</dd>
  <dt>Particle impoverishment</dt><dd>Loss of hypothesis diversity caused by repeated or premature resampling.</dd>
</dl>
</aside>
