---
layout: study-chapter
title: "Chapter 2. Probability and EKF-SLAM"
description: "Tracking uncertainty instead of a single right answer."
importance: 2
category: SLAM
series: slam_history
permalink: /study/slam/history/02-probability-ekf/
---

> **Goal:** Understand prediction and correction, and the correlation between robot and map.  
> **Workload:** 15 minutes. Helpful if you know what mean and variance are.

## 1. One coordinate is not enough

A record that a wheel rolled 1 m does not guarantee 1 m of actual motion, because of slip and sensor error. So instead of storing one number, we store a number **plus how sure we are of it**.

```text
what a plain estimate says          what a probabilistic estimate says

  x = 2.0                             x = 2.0,  variance 0.25
                                      (standard deviation 0.5 m)

  "the robot is at 2.0"               "the robot is most likely near 2.0,
                                       but 1.5 or 2.5 would not surprise me"
```

The two numbers do different jobs. The mean is the answer; the variance is how much the next measurement is allowed to move that answer. A prediction with variance 0.25 will be dragged a long way by a good measurement, and one with variance 0.0001 will barely budge. **How far it gets dragged is decided by a single number called the Kalman gain**, and §5 derives it.

EKF-SLAM linearises a nonlinear model around the current estimate and handles the robot state and the landmarks together; it is the representative filtering approach. The following equations show the structure of the problem.

$$
x_t=f(x_{t-1},u_t)+w_t, \qquad z_t=h(x_t,m_j)+v_t
$$

Here $u_t$ is the motion input, $z_t$ is the observation of landmark $m_j$, and $w_t,v_t$ are the noise terms put into the model. **Prediction** applies the motion model; **correction** uses the difference between the predicted and the actual observation. The basics of frames and models are covered in [Solà's Course on SLAM](https://upcommons.upc.edu/handle/2117/337287).

## 2. Why do map points become linked to each other?

Because every landmark is written down _from_ the robot's pose, an error in the pose is copied into all of them at once.

Work it through. The robot believes it is at x = 5.0, but it is really at x = 5.2 — it is 0.2 m off without knowing it. From there it measures two pillars.

```text
                    measured      robot thinks         truth
                    distance      pillar is at      pillar is at
  pillar A            +3.0          5.0+3.0 = 8.0     5.2+3.0 = 8.2
  pillar B            +7.0          5.0+7.0 = 12.0    5.2+7.0 = 12.2
                                          ↑                 ↑
                                    both wrong by the same 0.2
```

The distances themselves were measured perfectly. **Both pillars are still wrong, by exactly the same amount and in the same direction**, because they inherited the robot's 0.2 m.

That is what **correlation** means here, and it is useful rather than merely annoying. Suppose the robot later drives back and re-measures pillar A with a good sensor, learning that A is really at 8.2. It has learned three things at once:

```text
  pillar A is at 8.2     →  so my pose was 0.2 m short
                         →  so pillar B is at 12.2 as well
```

The correction to B arrives without B ever being re-observed. The filter can only do this if it remembered that A, B and the pose share an error — and that memory is stored in the off-diagonal entries of the covariance matrix. The diagonal entries hold the variance of each variable on its own; the off-diagonal entries hold exactly this "they are wrong together" relationship. This coupling is useful, but it becomes expensive on a large map. The [Introduction of the original FastSLAM paper](https://www.cs.cmu.edu/~thrun/papers/montemerlo.fastslam-tr.pdf) takes the scalability problem of the earlier EKF approach as its starting point.

## 3. What “linearising” actually means

§5 will compute a gain from $P$ and $R$ alone. That algebra only works if the models are
**linear** — if the measurement is literally $z = Hx + v$ for some matrix $H$.
Real sensors are not. A range measurement to a landmark at $(m_x,m_y)$ is

$$
h(x_r,y_r)=\sqrt{(m_x-x_r)^2+(m_y-y_r)^2}
$$

That square root is not a matrix multiplication and never will be.
The **E** in EKF is the fix: pretend it is one, near where we currently think we are.

```text
   the true h()                 what the EKF uses

        ╱                            ╱
      ╱                            ╱     a straight line,
    ╱   curved                   ╱       tangent at x̂
  ╱                            ╱
 ────────────────────        ────────────────────
        x̂                          x̂
                              exact at x̂, wrong elsewhere
```

Formally it is a first-order Taylor expansion around the current estimate $\hat x$:

$$
h(x)\;\approx\;h(\hat x)\;+\;H\,(x-\hat x),
\qquad H=\left.\frac{\partial h}{\partial x}\right|_{\hat x}
$$

So, to answer the question directly: **linearising is not itself a matrix multiplication —
it is taking a derivative.** What it _produces_ is a matrix, $H$ (the Jacobian),
and once you have $H$ everything downstream is matrix multiplication.

```text
   nonlinear h()  ──[ take ∂h/∂x at x̂ ]──▶  matrix H  ──▶  K = PHᵀ(HPHᵀ+R)⁻¹
                        ← 이 단계가 linearisation          ← 여기부터가 행렬 연산
```

Work the range example. Robot at $(0,0)$, landmark at $(3,4)$, so $h=5$:

$$
H=\frac{\partial h}{\partial (x_r,y_r)}
=\left[-\frac{m_x-x_r}{h},\;-\frac{m_y-y_r}{h}\right]
=[-0.6,\;-0.8]
$$

That is just the **unit vector pointing from the landmark to the robot** —
"move 1 m towards the landmark and the range drops by 1 m", split across the two axes.
Jacobians in SLAM usually have a plain geometric reading like this.

Now check how good the straight line is. Move the robot along $x$ and compare:

```text
      δx      true h()    linear     error
    0.05       4.9702     4.9700    0.0002
    0.10       4.9406     4.9400    0.0006
    0.50       4.7170     4.7000    0.0170
    1.00       4.4721     4.4000    0.0721
    2.00       4.1231     3.8000    0.3231
    3.00       4.0000     3.2000    0.8000
```

**Excellent nearby, useless far away.** And "far" is measured in units of your own uncertainty —
if $\sigma$ is 5 cm you live in the top rows; if $\sigma$ is 2 m you live in the bottom ones.

That is the whole practical story of the EKF:

```text
   small uncertainty  →  the tangent is a good stand-in  →  filter behaves
   large uncertainty  →  the tangent is a bad stand-in   →  filter drifts,
                                                            becomes overconfident,
                                                            can diverge outright
```

There is a second, subtler cost. A Gaussian pushed through a curved function
does not come out Gaussian, but the EKF assumes it does:

```text
   σ = 0.1 m   true: mean 5.0005, std 0.060, skew +0.07   ← near enough
   σ = 1.0 m   true: mean 5.0651, std 0.586, skew +0.63   ← visibly skewed
               EKF thinks: mean 5.0000, std 0.600
```

At $\sigma=1$ m the real distribution is lopsided and its mean has shifted by 6.5 cm,
while the filter still reports a clean symmetric Gaussian centred on the wrong value.
**The filter is not merely imprecise here — it is confidently wrong**, and nothing
in its own covariance warns it.

This is why good initialisation matters so much in EKF-SLAM, why later methods
relinearise (Chapter 7's iSAM) or avoid a single linearisation point altogether
(Chapter 3's particles), and why iterated filters re-do the expansion several times
per update (FAST-LIO's iterated EKF, Part 2).

---

## 4. What is actually in the state — and what the matrix looks like

A natural guess is that the covariance relates **the robot's poses over time** to
**the landmarks seen at each time**. That is not what EKF-SLAM does.

```text
   what EKF-SLAM keeps                what it does NOT keep
   ─────────────────────────          ────────────────────────────
   x = [ x_t          ]  ← one         [ x_0, x_1, x_2, … , x_t ]
       [ m_1          ]    pose,        the whole trajectory
       [ m_2          ]    the
       [ …            ]    current      past poses are marginalised
       [ m_n          ]    one          out and gone
```

The state holds **one robot pose — the current one — plus every landmark.**
When the robot moves, $x_{t-1}$ is not appended; it is replaced.
Its information does not vanish, though: it gets folded into the covariance,
which is exactly why the landmarks end up correlated.

(Chapter 6 takes the other road. Graph SLAM keeps $x_0 … x_t$ as variables,
so old poses stay editable. That is the real filtering-versus-smoothing split.)

For a 2D robot with $n$ landmarks the matrix is $(3+2n)\times(3+2n)$:

```text
              x_r  y_r  θ_r │ m1x m1y │ m2x m2y
            ┌───────────────┼─────────┼─────────┐
      x_r   │               │         │         │
      y_r   │  P_rr    3×3  │  P_rm1  │  P_rm2  │   ← robot ↔ landmark
      θ_r   │               │         │         │
            ├───────────────┼─────────┼─────────┤
      m1x   │               │         │         │
      m1y   │  P_m1r        │ P_m1m1  │ P_m1m2  │   ← landmark ↔ landmark
            ├───────────────┼─────────┼─────────┤
      m2x   │               │         │         │
      m2y   │  P_m2r        │ P_m2m1  │ P_m2m2  │
            └───────────────┴─────────┴─────────┘

   one pose block, n landmark blocks.  no time axis anywhere.
```

The diagonal blocks are each variable's own uncertainty.
**The off-diagonal blocks are §2's story written down as numbers.**

Take §2's pillars in 1D and actually fill the matrix in. The robot's position estimate
has variance $0.25$ ($\sigma=0.5$ m), and each range measurement has variance $0.01$ ($\sigma=0.1$ m).
A landmark is initialised as $\hat A = \hat x_r + z_A$, so its error is
the robot's error _plus_ the measurement's:

$$
\operatorname{Var}(A)=0.25+0.01=0.26,\qquad
\operatorname{Cov}(x_r,A)=\operatorname{Var}(x_r)=0.25
$$

$$
\operatorname{Cov}(A,B)=\operatorname{Var}(x_r)=0.25
$$

```text
             x_r     A       B
          ┌─────────────────────┐
    x_r   │ 0.25   0.25   0.25  │
    A     │ 0.25   0.26   0.25  │
    B     │ 0.25   0.25   0.26  │
          └─────────────────────┘

   every off-diagonal entry = 0.25 = the robot's own variance
```

That is the whole point in one picture. **The number the pillars share is the robot's error.**
Nothing else could have put it there — the two range measurements were independent.

As correlation coefficients it is even starker:

```text
             x_r     A       B
    x_r    1.000   0.981   0.981
    A      0.981   1.000   0.962
    B      0.981   0.962   1.000
```

$\rho(A,B)=0.962$. Two pillars 4 m apart, measured independently, are almost
perfectly correlated — because almost all of their uncertainty is one shared quantity.

Now re-measure only A, precisely ($z=8.2$, $R=10^{-4}$), and run one EKF update
with $H=\begin{bmatrix}0&1&0\end{bmatrix}$:

```text
   innovation  8.2 − 8.0 = 0.2
   K = P Hᵀ (H P Hᵀ + R)⁻¹ = [ 0.961,  1.000,  0.961 ]ᵀ
                                 ↑        ↑       ↑
                              robot      A       B
                                              never observed,
                                              yet its gain is 0.961

                    before      after      truth
        robot        5.00       5.192       5.2
        A            8.00       8.200       8.2
        B           12.00      12.192      12.2

   σ   robot         0.50       0.099
       A             0.51       0.010
       B             0.51       0.140
```

**B moved 0.192 m and its σ dropped from 0.51 to 0.14 without being observed at all.**
The correction travelled through the off-diagonal 0.25. That entry is not bookkeeping
overhead — it is the mechanism that makes loop closure work at all (Chapter 6).

And it is also the cost. With $n$ landmarks the matrix has $O(n^2)$ entries and the update
touches all of them, which is precisely the scalability wall
[FastSLAM](https://www.cs.cmu.edu/~thrun/papers/montemerlo.fastslam-tr.pdf) attacks in Chapter 3.

---

## 5. A one-dimensional correction example

To simplify, assume a reference point on the map is known. The robot predicts it is at 2.0 m with variance 0.25, then a sensor measures 2.4 m with variance 0.04.

```text
  prediction    x = 2.0,  variance P = 0.25   (σ = 0.5 m)
  measurement   z = 2.4,  variance R = 0.04   (σ = 0.2 m)

  they disagree by 0.4 m.  which one do we believe?
```

Neither, entirely. We move **part way** from the prediction towards the measurement:

$$
\hat x = x + K\,(z - x)
$$

$(z-x)$ is the disagreement, called the **innovation**. $K$ is how much of it we accept:

```text
  K = 0    ignore the measurement entirely, keep x = 2.0
  K = 1    accept the measurement entirely, x = 2.4
  K = 0.5  split the difference, x = 2.2
```

So the whole question is what $K$ should be. The answer is called the **Kalman gain**:

$$
K=\frac{P}{P+R}
$$

Read it as **“what share of the total uncertainty is mine?”**
$P+R$ is all the doubt in the room; $P$ is my part of it.
If most of the doubt is mine, I should move most of the way to the measurement.

The extremes confirm this is the right shape.

```text
  R → 0     a perfect sensor     K → 1     take the measurement whole
  R → ∞     a useless sensor     K → 0     ignore it, keep my prediction
  P → 0     a perfect prediction K → 0     nothing to correct
  P = R     equally trustworthy  K = 0.5   split evenly
```

Now the numbers:

$$
K=\frac{0.25}{0.25+0.04}\approx0.862,\qquad
\hat x=2.0+0.862\,(2.4-2.0)\approx2.345
$$

Read $K \approx 0.86$ as “move 86% of the way from the prediction to the measurement”.
The gap is $0.4$, so we move $0.86 \times 0.4 \approx 0.35$ and land at 2.345 —
much closer to the measurement than to the prediction.

That is the right behaviour, because variance 0.04 (σ = 0.2 m) is a far tighter claim than variance 0.25 (σ = 0.5 m).

The uncertainty shrinks too, and by the same gain:

$$
P^{+}=(1-K)\,P=(1-0.862)\times0.25\approx0.034
$$

```text
  before    P = 0.25    (σ = 0.50 m)
  after     P⁺= 0.034   (σ = 0.19 m)

  0.034 is smaller than 0.25  and  smaller than 0.04
```

That last line is worth pausing on. **The result is more certain than either input.**
Two independent opinions that roughly agree are together stronger than either alone —
which is exactly why fusing sensors is worth doing at all.

Change the numbers and the filter changes its mind accordingly:

```text
    P            R            K       x̂       P⁺      who wins
   0.25         0.04         0.86     2.345   0.034    measurement
   0.25         0.25         0.50     2.200   0.125    a tie, split evenly
   0.25         4.00         0.06     2.024   0.235    prediction
```

Nobody tuned those weights by hand. **They fall out of the two variances.**
That is the appeal of the whole approach: you state how much you trust each source,
and the blending ratio follows. In real EKF-SLAM the observation may be a range and bearing rather than the position itself, so Jacobians and correlations have to be included.

## Check questions

### Question 1 — Concept

What goes wrong in EKF-SLAM if you keep all the cross-covariances between the robot pose and the landmarks at zero?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Landmarks observed from the same uncertain robot pose share a common position error. Forcing the cross-covariance to zero throws that correlation away and blocks the information a landmark re-observation should pass on to the robot pose and to other landmarks. The filter then interprets the data as containing more independent measurements than it really does and becomes overconfident, which is an inconsistency. Approximations to save computation are legitimate, but you have to say which correlations you discarded and how you manage consistency.

</details>

### Question 2 — Math

A one-dimensional state has prior $x\sim\mathcal N(2.0,0.25)$ and measurement model $z=x+v$ with $v\sim\mathcal N(0,0.04)$. Given the actual measurement $z=2.4$, find the Kalman gain, the posterior mean and the posterior variance.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

The measurement Jacobian is $H=1$, so

$$
K=\frac{0.25}{0.25+0.04}\approx0.8621
$$

and therefore

$$
\hat x^+=2.0+0.8621(2.4-2.0)\approx2.3448,
$$

$$
P^+=(1-K)0.25\approx0.0345.
$$

Because the measurement variance is smaller than the prior variance, the mean moves a long way towards the measurement and the posterior uncertainty decreases. In a real EKF, $H$ is a Jacobian linearised at the current estimate, so the initial estimate and linearisation error also have to be taken into account.

</details>

## Original reading

- Course on SLAM: read only the motion/observation model sections. Local copy: `_resource/slam/foundations/sola2017-course-on-slam.pdf`.
- FastSLAM (2002): the explanation of the EKF's limits in the Introduction. The next chapter continues from there with conditional independence.
