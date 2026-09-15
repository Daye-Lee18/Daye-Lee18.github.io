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

EKF-SLAM estimates the robot's state and the positions of landmarks **together**. It repeatedly performs two steps: predict where the robot has moved, then use a sensor observation to correct both the robot state and the map.

### Prediction: where should the robot be now?

$$
x_t=f(x_{t-1},u_t)+w_t
$$

- $x_{t-1}$ is the previous robot state.
- $u_t$ is a motion input, such as wheel odometry or an IMU reading.
- $f$ is the motion model that predicts the next state.
- $w_t$ represents motion uncertainty, such as wheel slip or an imperfect model.
- $x_t$ is the predicted current state.

In plain language, the equation says: **use the previous state and the measured motion to predict the current state, while admitting that the prediction is not exact.**

### Correction: does the sensor agree with the prediction?

$$
z_t=h(x_t,m_j)+v_t
$$

- $m_j$ is the position of landmark $j$ in the map.
- $h$ predicts how that landmark should appear from the current robot state.
- $z_t$ is what the sensor actually observes.
- $v_t$ represents sensor noise.

For example, the current state and map may predict that a wall should be 3.0 m away, while the LiDAR measures 3.3 m. The difference between the predicted observation and the real observation is called the **innovation** (or measurement residual). EKF-SLAM uses it to correct the robot state and the landmark estimates, weighting the correction by how uncertain the prediction and measurement are.

```text
previous state + motion input
              ↓
    predict current state
              ↓
predicted observation ↔ actual observation
              ↓
   correct robot state and map
```

Real motion and sensor models are usually nonlinear. The EKF makes them manageable by approximating each model as linear **near the current estimate**. This local approximation is what “linearisation” means; if the current estimate is far from the truth, the approximation can be poor and the filter can become inaccurate or overconfident. The basics of frames and models are covered in [Solà's Course on SLAM](https://upcommons.upc.edu/handle/2117/337287).

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

### When does a Kalman correction happen?

The robot does not have to complete a loop before using a Kalman correction. It only has to observe something that the estimator can associate with an existing landmark.

```text
time t₁: observe pillar A for the first time
         → add A to the state and initialise its uncertainty

time t₂: move a short distance and observe A again
         → ordinary EKF correction

time t₃: travel around the building and recognise A again
         → the same update can provide loop-closure information
```

At $t_1$, there is no previous estimate of A to correct. The first observation initialises it. At $t_2$ and $t_3$, A already exists in the state, so the filter compares the observation predicted from its current state with the new sensor observation. The elapsed time is not what determines whether the Kalman update applies; successful data association is.

### The correction below, step by step

The following calculation deliberately uses a simplified sensor that directly measures A's global coordinate. That is why $H=\begin{bmatrix}0&1&0\end{bmatrix}$. A real LiDAR usually measures relative range or bearing, in which case $h(x,m_A)$ and $H$ also depend on the robot pose. The simplified model lets us focus on how covariance carries A's information to the robot and B.

#### Step 1 — Begin with the predicted state and covariance

After the motion prediction, but before the new measurement, the joint state is

$$
\hat x^-=\begin{bmatrix}5.0&8.0&12.0\end{bmatrix}^T,
$$

and $P^-$ is the covariance matrix shown above. The superscript $-$ means “before correction”.

#### Step 2 — Receive the new measurement

The precise sensor reports

$$
z=8.2,\qquad R=10^{-4}.
$$

The value $8.2$ comes from the new sensor measurement; the Kalman filter does not generate it.

#### Step 3 — Predict the measurement

The filter asks what the sensor should read if the current state were correct:

$$
\hat z=H\hat x^-
=\begin{bmatrix}0&1&0\end{bmatrix}
 \begin{bmatrix}5.0&8.0&12.0\end{bmatrix}^T
=8.0.
$$

#### Step 4 — Compute the innovation

$$
y=z-\hat z=8.2-8.0=0.2.
$$

The $0.2$ is therefore the disagreement between the new measurement and the predicted measurement. It is an input to the correction, not the result of the correction.

#### Step 5 — Compute the Kalman gain

First compute the innovation covariance and then the gain:

$$
S=HP^-H^T+R=0.26+0.0001=0.2601,
$$

$$
K=P^-H^TS^{-1}
=\frac{1}{0.2601}
\begin{bmatrix}0.25\\0.26\\0.25\end{bmatrix}
\approx
\begin{bmatrix}0.961\\1.000\\0.961\end{bmatrix}.
$$

A and the robot/B receive different gains according to their covariance with the measured variable A.

#### Step 6 — Correct the state and covariance

The state correction is

$$
\hat x^+=\hat x^-+Ky.
$$

Numerically,

$$
Ky\approx
\begin{bmatrix}0.961\\1.000\\0.961\end{bmatrix}(0.2)
=\begin{bmatrix}0.192\\0.200\\0.192\end{bmatrix},
$$

so

$$
\hat x^+\approx
\begin{bmatrix}5.192\\8.200\\12.192\end{bmatrix}.
$$

The covariance is also corrected, for example with

$$
P^+=(I-KH)P^-.
$$

The whole calculation can be summarised as follows:

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

### Question 3 — Concept

A teammate explains that the EKF-SLAM state vector holds every pose from $x_0$ to $x_t$
along with the landmarks. Where is this wrong, and which method does that description actually fit?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

EKF-SLAM keeps **one** robot pose — the current one — plus all landmarks:
$x=[x_t, m_1, \dots, m_n]$. On each motion step the previous pose is not appended, it is replaced;
past poses are marginalised out. Their information is not discarded, but it survives only
as correlations inside the covariance, not as recoverable variables. The practical consequence is
that a filter cannot go back and revise $x_{5}$ once it has moved on.

The description fits **Graph SLAM / smoothing** (Chapter 6), where $x_0 \dots x_t$ all remain
variables and stay editable, which is what makes loop closure able to bend the whole trajectory.
This is the real filtering-versus-smoothing distinction — not "old versus new".

</details>

### Question 4 — Math

A 2D robot is at $(1,1)$ and a landmark at $(4,5)$. For the range measurement
$h=\sqrt{(m_x-x_r)^2+(m_y-y_r)^2}$, find $H=\partial h/\partial(x_r,y_r)$.
Then predict the change in range if the robot moves 0.1 m **straight towards** the landmark,
and if it moves 0.1 m **perpendicular** to that direction. Interpret both.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$h=\sqrt{3^2+4^2}=5$, and

$$
H=\left[-\frac{m_x-x_r}{h},\;-\frac{m_y-y_r}{h}\right]=[-0.6,\;-0.8]
$$

The unit vector towards the landmark is $(0.6,0.8)$, so moving 0.1 m along it is
$\delta=(0.06,0.08)$ and

$$
H\delta = (-0.6)(0.06)+(-0.8)(0.08) = -0.1
$$

The range drops by exactly 0.1 m, which is what "moving 0.1 m closer" has to mean —
a useful sanity check on any Jacobian you derive.

Perpendicular motion is $\delta=(-0.08,0.06)$, giving $H\delta = 0$: to first order the range
does not change at all. The true change is $+0.001$ m, the second-order term the linearisation drops.

The geometric reading is the point: $H$ is just $-1\times$ the unit vector from landmark to robot.
**A range measurement constrains only the along-the-ray direction and says nothing about
the perpendicular one.** That is why a single range is never enough to fix a 2D position,
and it is the same observability argument that reappears for walls in Chapter 5.

</details>

### Question 5 — Math and reasoning

For the range model, linearisation error at $\delta$ from the expansion point is roughly
$\tfrac12 h''\delta^2$. If your position uncertainty $\sigma$ doubles, roughly how much does the
linearisation error grow? Verify against these measured values and say what it implies
for filter initialisation.

```text
   δ = 0.25 → error 0.00412
   δ = 0.50 → error 0.01699
   δ = 1.00 → error 0.07214
   δ = 2.00 → error 0.32311
```

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

The leading error term is quadratic in $\delta$, so doubling $\delta$ should multiply the error by
about 4. The measurements give ratios of $4.12$, $4.25$ and $4.48$ — close to 4, drifting upward
because higher-order terms start contributing as $\delta$ grows.

Since the filter effectively evaluates over a region of size $\sim\sigma$, **halving your uncertainty
cuts linearisation error by about four times.** The implications are all one-directional:

- A good initial estimate is worth far more than it looks; error falls quadratically, not linearly.
- A filter that is already uncertain linearises badly, which makes its next estimate worse,
  which makes it linearise worse still — EKF divergence is this feedback loop.
- Feeding in a rough prior (wheel odometry, IMU propagation, GNSS) early is not merely a convenience;
  it changes which row of that table you operate on.

</details>

### Question 6 — Math

A 2D EKF-SLAM system has $n$ landmarks. How many entries does the covariance matrix hold?
Evaluate for $n=10$, $100$, $1000$, and explain why this is the wall that Chapter 3 attacks.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

The state is $3+2n$ (robot $x,y,\theta$ plus $x,y$ per landmark), so the covariance is
$(3+2n)\times(3+2n)$:

```text
   n =   10  →   23 ×   23 =        529
   n =  100  →  203 ×  203 =     41,209
   n = 1000  → 2003 × 2003 =  4,012,009
```

Growth is $O(n^2)$ in memory. Worse, a landmark observation updates the **whole** matrix,
because the correction has to propagate along every correlation — the very mechanism §4 showed
moving pillar B. So per-update cost is $O(n^2)$ too, and it applies on every step, not occasionally.

Note that this cost is not incidental; it is the direct price of the property that makes EKF-SLAM
work. Chapter 3's FastSLAM buys it back by conditioning on the robot path, which makes the
landmarks conditionally independent so that no $n \times n$ block has to be maintained at all.

</details>

### Question 7 — Math

Using §4's setup — robot variance $\sigma_r^2$, independent range noise $\sigma_m^2$ per landmark,
two landmarks initialised from the same pose — show that
$\rho(A,B)=\sigma_r^2/(\sigma_r^2+\sigma_m^2)$, then evaluate it as $\sigma_m^2 \to 0$ and as
$\sigma_m^2 \to \infty$. What does each limit mean physically?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

With $\hat A=\hat x_r+z_A$ and $\hat B=\hat x_r+z_B$, the errors are $e_A=e_r+e_{z_A}$ and
$e_B=e_r+e_{z_B}$ with all three independent. So

$$
\operatorname{Cov}(A,B)=\operatorname{Var}(e_r)=\sigma_r^2,
\qquad
\operatorname{Var}(A)=\operatorname{Var}(B)=\sigma_r^2+\sigma_m^2
$$

$$
\rho(A,B)=\frac{\sigma_r^2}{\sigma_r^2+\sigma_m^2}
$$

With $\sigma_r^2=0.25,\ \sigma_m^2=0.01$ this is $0.25/0.26\approx0.962$, matching §4.

```text
   σ_m² → 0     ρ → 1        a perfect sensor
   σ_m² → ∞     ρ → 0        a useless sensor
```

$\rho \to 1$: with a perfect sensor the _only_ thing either landmark estimate is unsure about is
the robot's pose, so they are the same uncertainty wearing two labels. Learn one and you learn the other.

$\rho \to 0$: the landmark estimates are dominated by their own measurement noise, and the shared
pose error is negligible in comparison, so correcting one tells you almost nothing about the other.

**Better sensors produce more correlated maps, not less.** That is counterintuitive until you see
that correlation here measures how much of the uncertainty is _shared_, not how large it is.

</details>

### Question 8 — Systems and debugging

An EKF-SLAM run reports steadily shrinking covariance while the actual trajectory error grows.
The filter is confident and wrong. List what you would check, and how you would detect this
automatically rather than by eye.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

This is filter **inconsistency**: reported uncertainty no longer bounds true error. Candidates:

- **Linearisation error** (§3). Large uncertainty makes bad Jacobians, and the resulting
  overconfidence shrinks covariance further — the divergence loop of Question 5.
- **Wrong data association.** Matching an observation to the wrong landmark injects a
  confident, false constraint; the filter gains certainty from information that is not real.
- **Understated $R$ or $Q$.** Tuning noise down makes the filter look smooth and precise
  in demos while destroying its ability to admit error.
- **Discarded correlations.** Zeroing cross-covariances for speed (Question 1) makes the filter
  count dependent measurements as independent evidence.
- **Unmodelled effects** — a moving "landmark", clock skew, unestimated bias — all appear as
  information the filter has no way to discount.

To catch it automatically, compare reported covariance against actual error rather than reading plots:

```text
   NEES  needs ground truth   (x−x̂)ᵀ P⁻¹ (x−x̂)   should sit near the state dimension
   NIS   no ground truth      yᵀ S⁻¹ y            should sit near the measurement dimension
```

Both have known chi-square bounds, so a run can be flagged automatically. **NIS is the one that
works on a real robot**, since it only needs innovations and their predicted covariance.
Consistently small NIS means the filter is claiming more precision than its own residuals justify —
exactly the failure described here.

</details>

## Original reading

- Course on SLAM: read only the motion/observation model sections. Local copy: `_resource/slam/foundations/sola2017-course-on-slam.pdf`.
- FastSLAM (2002): the explanation of the EKF's limits in the Introduction. The next chapter continues from there with conditional independence.

<aside class="study-summary" markdown="1">
## What you learned

<dl>
  <dt>Prediction</dt><dd>Propagating the state and its uncertainty through the motion model.</dd>
  <dt>Correction</dt><dd>Combining a measurement with the prediction according to their uncertainties.</dd>
  <dt>Covariance</dt><dd>A matrix describing uncertainty and the correlations between robot and landmark errors.</dd>
  <dt>EKF-SLAM</dt><dd>A Gaussian filter that jointly estimates the robot pose and landmark positions after linearising nonlinear models.</dd>
  <dt>Consistency</dt><dd>The requirement that reported uncertainty honestly reflects the estimator's actual error.</dd>
</dl>
</aside>
