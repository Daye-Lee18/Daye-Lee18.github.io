---
layout: study-chapter
title: "Chapter 10. From IMU fusion to LIO"
description: "Why fast inertial prediction and external observations need each other."
importance: 10
category: SLAM
series: slam_history
permalink: /study/slam/history/10-inertial-lio/
---

> **Goal:** Explain IMU drift, deskew, and the relationship between filtering and smoothing.  
> **Workload:** 15 minutes. This chapter is the doorway to the LIO part.

## 1. Why does having an IMU help?

A LiDAR fires at 10 Hz. Between two scans, 100 ms pass — and a robot that lands from a step can rotate 20° in that time. The LiDAR simply has no opinion about what happened in between.

An IMU runs at 200–1000 Hz and fills that gap:

```text
  LiDAR    ●─────────────────────────────●          2 samples in 200 ms
  IMU      ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●        ~200 samples
           └──────── the IMU knows what happened here ────────┘
```

But an accelerometer does not report "acceleration". It reports **specific force**, which is not the same thing, and the difference bites immediately. A robot sitting perfectly still on a table reads:

```text
  accelerometer output   (0, 0, +9.81)  m/s²
  actual acceleration    (0, 0,  0)     m/s²   ← it is not moving at all
```

Integrate the raw reading twice and after 10 seconds the "position" has climbed 490 m into the air. Gravity has to be subtracted first, and to subtract it you must know which way is down — which means knowing your orientation. So attitude, gravity and bias all have to be handled before an accelerometer says anything useful about motion.

And even then, integration only accumulates error (§4), which is why aided navigation with external sensor observations is needed. The [KRoC IMU lecture](https://drive.google.com/file/d/1byqGAKCCsnv8rZbko4RBG9KiQQD_4h8x/view) explains the problems of INS and the role of aiding observations.

## 2. A single scan is not captured at a single instant

The word "scan" makes it sound like a photograph. It is not. A spinning LiDAR sweeps around over 100 ms, and each point is measured at a different moment along the way.

If the robot is moving at 1 m/s, the first and last points of one scan were taken from positions 10 cm apart:

```text
  point index      when measured      robot was at
       0              t + 0 ms           x = 0.00
    1000              t + 25 ms          x = 0.025
    2000              t + 50 ms          x = 0.050
    3000              t + 75 ms          x = 0.075
    4000              t + 100 ms         x = 0.100
```

Treat all of them as if they came from one place and a straight wall comes out bent:

```text
   what the wall is        what the raw scan looks like

        ║                          ╲
        ║                           ╲     smeared by 10 cm
        ║                            ╲    across the sweep
        ║                             ╲
```

Deskew undoes this: using the IMU-derived motion, each point is moved to where it would have been had it been measured at one common reference time. It is not an optional polish step — without it, ICP is being handed a wall that genuinely is bent, and it will faithfully estimate a wrong pose from it.

Note what deskew needs: an accurate per-point timestamp and an accurate extrinsic between LiDAR and IMU. If those are wrong, adding more sensors will not fix it.

## 3. Comparing the design of representative systems

| System                    | Representative estimation approach            | What to look for while reading                                 |
| ------------------------- | --------------------------------------------- | -------------------------------------------------------------- |
| LIO-SAM (2020)            | Tightly coupled LIO built on a factor graph   | IMU preintegration, LiDAR odometry, loop/GPS factors           |
| FAST-LIO2 (2021 preprint) | Direct LIO based on an iterated Kalman filter | Raw-point scan-to-map registration, incremental map management |

[LIO-SAM](https://arxiv.org/abs/2007.00258) uses IMU information for deskew and for the initial guess, and combines measurements in a factor graph. [FAST-LIO2](https://arxiv.org/abs/2107.06829) combines direct point cloud registration with an efficient map data structure. This comparison is a difference of design, not an accuracy ranking. Do not assume that FAST-LIO2's local mapping automatically includes global loop closure.

## 4. A hand calculation

Bias is a small constant offset the accelerometer always adds. Suppose it is $0.01\,m/s^2$ — one thousandth of gravity, an amount you would never notice on a datasheet.

Position error from a constant bias is $\frac12bt^2$, and the $t^2$ is what hurts:

```text
   time      velocity error      position error
   ─────     ──────────────      ──────────────
    1 s          0.01 m/s            0.005 m      harmless
   10 s          0.10 m/s            0.5 m        noticeable
   60 s          0.60 m/s            18 m         useless
  300 s          3.00 m/s            450 m        absurd
```

Nothing changed except the clock. The bias stayed at 0.01 the whole time.

Now compare it against a 1° attitude error, which projects gravity sideways at $9.81\sin(1°) \approx 0.171\,m/s^2$ — seventeen times worse than the bias above. After 10 seconds that alone is 8.6 m.

Two lessons follow. Free-running IMU integration is measured in _seconds_ of usefulness, not minutes. And **attitude error dominates**: getting the orientation slightly wrong is far more damaging than a slightly biased accelerometer, which is why gravity alignment and attitude correction get so much attention.

## Check questions

### Question 1 — Concept

Compare filtering and fixed-lag smoothing in terms of latency, computation, relinearisation and the ability to revise past states. Also explain whether a LIO system has to use only one of the two.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Filtering usually updates the current state and covariance recursively, making low latency and a bounded state size easy to achieve, but it is hard to directly re-optimise marginalised past states with new information. Fixed-lag smoothing keeps several states in a time window together, allowing relinearisation and the handling of delayed measurements, at the cost of more computation and latency. Practical systems can combine them, using a filter for fast propagation and local odometry and smoothing for a keyframe graph or loop closure. The axis of comparison is not old versus new but latency, state horizon, nonlinear models and the need for global correction.

</details>

### Question 2 — Math

Assume attitude is exact and the initial velocity error is zero, with a constant body x-axis accelerometer bias $b_a=0.02m/s^2$. Find the velocity and position error after integrating for 20 seconds with no correction. Also approximate the horizontal-axis acceleration error produced by gravity $9.81m/s^2$ given a roll error of $1^\circ$.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Considering the constant bias alone,

$$
\delta v=b_at=0.02\times20=0.4m/s,
$$

$$
\delta p=\frac12b_at^2
=\frac12\times0.02\times20^2=4m.
$$

Since $1^\circ\approx0.01745rad$, the small-angle gravity projection error is

$$
g\sin(1^\circ)\approx9.81\times0.01745\approx0.171m/s^2,
$$

which is far larger than the bias above. This is why attitude error grows into position error so quickly in IMU integration, and why gravity alignment and attitude correction matter.

</details>

## Original reading

- KRoC IMU: PDF pages 13–24. Local copy: `_resource/slam/kroc2026/05-imu-giseop-kim.pdf`.
- Read only the abstract and system overview of the two LIO papers. Detailed derivations and code analysis are picked up in the LIO part.

<aside class="study-summary" markdown="1">
## What you learned

<dl>
  <dt>IMU propagation</dt><dd>High-rate prediction of orientation, velocity and position by integrating inertial measurements.</dd>
  <dt>Bias</dt><dd>A systematic sensor offset whose integration causes rapidly growing state error.</dd>
  <dt>Deskew</dt><dd>Compensating each LiDAR point for motion during acquisition of a scan.</dd>
  <dt>LIO</dt><dd>Tightly coupled LiDAR–inertial odometry that combines fast inertial prediction with geometric correction.</dd>
  <dt>Filtering vs smoothing</dt><dd>Recursive current-state updates versus joint optimisation over a retained state history.</dd>
</dl>
</aside>
