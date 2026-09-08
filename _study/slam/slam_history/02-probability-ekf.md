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

The two numbers do different jobs. The mean is the answer; the variance is how much the next measurement is allowed to move that answer. A prediction with variance 0.25 will be dragged a long way by a good measurement, and one with variance 0.0001 will barely budge. That is the whole idea a Kalman filter runs on, and §3 does the arithmetic.

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
  pillar A            +3.0          5.0+3.0 = 8.0     5.2+3.0 = 8.4
  pillar B            +7.0          5.0+7.0 = 12.0    5.2+7.0 = 12.4
                                          ↑                 ↑
                                    both wrong by the same 0.2
```

The distances themselves were measured perfectly. Both pillars are still wrong, by exactly the same amount and in the same direction, because they inherited the robot's 0.2 m.

That is what correlation means here, and it is useful rather than merely annoying. Suppose the robot later drives back and re-measures pillar A with a good sensor, learning that A is really at 8.4. It has learned three things at once:

```text
  pillar A is at 8.4     →  so my pose was 0.2 m short
                         →  so pillar B is at 12.4 as well
```

The correction to B arrives without B ever being re-observed. The filter can only do this if it remembered that A, B and the pose share an error — and that memory is stored in the off-diagonal entries of the covariance matrix. The diagonal entries hold the variance of each variable on its own; the off-diagonal entries hold exactly this "they are wrong together" relationship. This coupling is useful, but it becomes expensive on a large map. The [Introduction of the original FastSLAM paper](https://www.cs.cmu.edu/~thrun/papers/montemerlo.fastslam-tr.pdf) takes the scalability problem of the earlier EKF approach as its starting point.

## 3. A one-dimensional correction example

To simplify, assume a reference point on the map is known. The robot predicts it is at 2.0 m with variance 0.25, then a sensor measures 2.4 m with variance 0.04. Two different answers — which do we believe?

The gain $K$ decides, and it is just a ratio of how uncertain each side is:

$$
K=\frac{0.25}{0.25+0.04}\approx0.862,\qquad
\hat x=2.0+K(2.4-2.0)\approx2.345
$$

Read $K \approx 0.86$ as “move 86% of the way from the prediction to the measurement”. The gap is $2.4-2.0=0.4$, so we move $0.86 \times 0.4 \approx 0.35$ and land at 2.345 — much closer to the measurement than to the prediction.

That is the right behaviour, because variance 0.04 (σ = 0.2 m) is a far tighter claim than variance 0.25 (σ = 0.5 m). Change the numbers and the filter changes its mind accordingly:

```text
 prediction var   measurement var      K       result    who wins
      0.25             0.04           0.86      2.345    measurement
      0.25             0.25           0.50      2.200    a tie, split evenly
      0.25             4.00           0.06      2.024    prediction
```

Nobody tuned those weights by hand. They fall out of the two variances. In real EKF-SLAM the observation may be a range and bearing rather than the position itself, so Jacobians and correlations have to be included.

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
