---
layout: study-chapter
title: "Chapter 5. ICP and point cloud registration"
description: "Understanding iterative optimisation through the problem of aligning two observations."
importance: 5
category: SLAM
series: slam_history
permalink: /study/slam/history/05-registration-icp/
---

> **Goal:** Separate correspondence from pose update.  
> **Workload:** 15 minutes. Uses the frame transforms from Chapter 4.

## 1. Two point clouds that do not line up

The robot scans a corner, drives forward a bit, and scans it again. Both scans show the same corner, but written in the sensor frame of two different positions — so the two clouds do not overlap on screen. The question ICP answers is: **what rotation and translation would slide the second cloud onto the first?** And that answer is the robot's motion.

```text
   scan at t=0 (target)      scan at t=1 (source)     what we want

        ┌────                      ┌────                  ┌────
        │                          │                     ▒│        overlaid
        │                     ▒    │                      │
                                                       R, t = the motion
```

A representative objective for point-to-point ICP is

$$
\min_{R,t}\sum_i\|Rp_i+t-q_{c(i)}\|^2
$$

where $c(i)$ is the index of the target point matched to source point $p_i$. Read the expression inside the norm as “where the transform puts my point, minus where its partner actually is” — the leftover gap, squared and summed.

There is a chicken-and-egg problem hiding here, and it is worth naming. To know which points are partners you need the transform; to find the transform you need the partners. ICP breaks the loop by alternating and hoping it converges:

```text
  guess a transform (often: assume the robot barely moved)
    ↓
  ① for each source point, call the nearest target point its partner
    ↓
  ② hold those partners fixed, solve for the transform that best fits them
    ↓
  repeat ① with the improved transform
```

That "hoping it converges" is not a figure of speech — §2 is about when it does not. Grisetti's _From Least-Squares to ICP_, recommended in [Giseop Kim's guide to SLAM back-end material](https://gisbi-kim.github.io/post/slam-textbooks/), connects residuals and Jacobians on this small problem.

## 2. Is reducing the error enough?

No. A low error means "the points line up", and points can line up while the pose is wrong.

Take a corridor with pillars every 5 m, and suppose the robot really moved 5 m. If ICP matches each pillar to the _next_ pillar along instead of to itself, everything fits beautifully:

```text
  true motion 5 m                 ICP's answer 0 m

  pillar at  0  5  10  15         pillar at  0  5  10  15
  scan sees     5  10  15         matched to 0  5  10
                ↑                            every pair lines up
       correct: shifted by 5      to within a few mm — error ≈ 0
```

The residual is near zero and the answer is off by a full 5 m. The optimiser did its job; the correspondences lied to it. That is why the initial guess, the overlap region and outlier rejection matter as much as the solver.

The second failure is different: sometimes the data genuinely cannot tell you. Point-to-plane measures the distance from a point to its corresponding plane _along the plane's normal_, and if the robot only sees one flat wall:

```text
                   wall
    ─────────────────────────────────
              ↑ this motion changes the distance   → observable
              ← → this motion does not             → invisible

    sliding along the wall leaves every residual exactly as it was
```

The filter is not being stupid; the measurement contains no information about that direction. Always ask which directions your measurements actually constrain. For how registration developed, read on into the [KRoC registration lecture](https://drive.google.com/file/d/1fwaHF77iwTxcDVl1s1h8BcCZE0EvPmUC/view).

## 3. One calculation in one dimension

Strip everything away: no rotation, correspondences already known, one axis. Source `[0, 1, 2]`, target `[2, 3, 4]`, residual $r_i=p_i+t-q_i$.

```text
  source   target   needed shift
    0        2          +2
    1        3          +2
    2        4          +2
                     ─────────
                 t = 2, residual 0
```

Every point wants the same shift, so $t=2$ and the fit is perfect.

Now corrupt one correspondence — say the third source point gets matched to a target at 40 instead of 4. The least-squares answer is the average of what each pair wants:

```text
    0        2          +2
    1        3          +2
    2       40         +38    ← one bad match
                     ─────────
              t = (2 + 2 + 38)/3 = 14
```

$t=14$. Not 2, not even close, and the two good correspondences that agreed perfectly have been overruled by a single bad one. Squaring is what does it: being wrong by 12 costs $144$, so the optimiser will happily ruin two good fits to reduce that one.

The arithmetic was exact throughout. The input was wrong. This is why robust losses (which cap what a single outlier can cost) and correspondence validation (which drops it before it enters) are both needed — they attack the same problem from two sides. Robust losses and correspondence validation complement each other.

## Check questions

### Question 1 — Concept

ICP converged with a low residual, but the estimated pose is wrong. Give three possible causes and a diagnostic for each.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

First, the iterative scheme may have converged to a bad local minimum, so change the initial guess or compare against a global registration result. Second, wrong correspondences and outliers may dominate the objective, so inspect the distribution of correspondence distances and normal angles, and the inlier ratio. Third, it may be a degeneracy where a particular motion direction is unobserved, as on a plane or in a long corridor, so look at the eigenvalues and condition number of the Hessian or normal matrix. Low overlap or temporal distortion is also possible, so visualise the raw scan and the deskewed result.

</details>

### Question 2 — Math

Take the point-to-plane ICP residual as $r_i=n_i^T(Rp_i+t-q_i)$. Applying a small left rotation $R'\approx(I+[\delta\theta]_\times)R$ and a translation increment $\delta t$ near the current estimate, derive one row of the Jacobian with respect to $\delta\xi=[\delta\theta^T,\delta t^T]^T$.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Writing $a_i=Rp_i$,

$$
R'p_i\approx a_i+[\delta\theta]_\times a_i
=a_i-[a_i]_\times\delta\theta.
$$

Therefore

$$
r_i'\approx r_i+n_i^T\left(-[a_i]_\times\delta\theta+\delta t\right)
$$

and the Jacobian is

$$
J_i=\begin{bmatrix}-n_i^T[a_i]_\times & n_i^T\end{bmatrix}.
$$

With a right perturbation or a different pose convention the rotation part of the Jacobian takes a different form. The key is to match the convention used in the derivation to the one used in the code.

</details>

## Original reading

- KRoC registration lecture: the part linking ICP to robust, global and learning-based approaches. Local copy: `_resource/slam/kroc2026/06-registration-hyungtae-lim.pdf`.
- For Grisetti (2016), use the description and links in the recommended guide. The old direct PDF address now returns 404, so it is not included in the local resources.
