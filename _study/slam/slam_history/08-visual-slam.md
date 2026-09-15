---
layout: study-chapter
title: "Chapter 8. Visual SLAM and Bundle Adjustment"
description: "Refining poses and 3D points together from camera observations."
importance: 8
category: SLAM
series: slam_history
permalink: /study/slam/history/08-visual-slam/
---

> **Goal:** Understand reprojection error and the roles of tracking and mapping.  
> **Workload:** 15 minutes. Read Chapters 4 and 6 first.

## 1. A photograph does not show depth directly

A LiDAR returns a distance. A camera does not — it returns a _direction_ and nothing else.

Suppose a point lands on pixel (320, 240) of a camera with $f_x = f_y = 500$ and principal point (320, 240). Work backwards through the projection $u = f_xX/Z + c_x$ and you get:

```text
  Z = 1 m   →   the point is at (0, 0, 1)
  Z = 5 m   →   the point is at (0, 0, 5)
  Z = 50 m  →   the point is at (0, 0, 50)

  every one of these lands on exactly pixel (320, 240)
```

One pixel, infinitely many possible 3D points, all on one ray leaving the camera. A single image simply does not contain the answer.

Move the camera 1 m sideways and take a second picture, though, and the ray from the _new_ position is different for each candidate. The two rays cross at exactly one place:

```text
        cam 1                 cam 2
          ●───────────────────────► ray 1
           ╲                    ╱
            ╲                  ╱
             ╲                ╱
              ●──────────────●  ← the two rays meet here: that is the point
```

That is triangulation, and it is why structure is estimated from correspondences across viewpoints rather than from one frame. For how the camera model connects to pose estimation, see the camera projection and relative pose sections of the [KRoC 3D Vision lecture](https://drive.google.com/file/d/1mL52klpHEYU6e-yZk3guMaJocLthSAA7/view).

## 2. What does Bundle Adjustment actually fit?

It fits everything against one simple test: **if my guesses were right, where would this point have landed in the picture?**

Take the guessed 3D point, push it through the guessed camera pose, project it into the image, and compare against the pixel where the feature actually was.

$$
r_{ij}=u_{ij}-\pi(T_{C_iW}P_j)
$$

With numbers:

```text
  guessed point      P = (2.0, 0.0, 10.0)   in the camera frame
  projection         u = 500·(2.0/10.0) + 320 = 420
  actually observed  u = 423
                     ─────────────────────────
  reprojection error r = 423 - 420 = 3 pixels
```

Three pixels of disagreement. Something in the inputs is wrong — the point may be slightly misplaced, or the camera pose slightly off, and from one residual you cannot tell which.

That ambiguity is the point of the word _bundle_. Bundle Adjustment does not fix the point and then the pose; it nudges **all** the points and **all** the poses at once, searching for the arrangement in which every reprojection error across every frame is as small as possible. A point seen in 20 frames is constrained by 20 such tests, and a pose that sees 200 points is constrained by 200. Robust cost, gauge freedom and sparsity are the key implementation elements. BA has roots in photogrammetry and visual reconstruction that predate SLAM; [the original paper by Triggs et al.](https://lear.inrialpes.fr/people/triggs/pubs/Triggs-va99.pdf) covers that background and the numerical optimisation.

## 3. Do we keep optimising every frame?

[ORB-SLAM (2015)](https://arxiv.org/abs/1502.00956) separates tracking, local mapping and loop closing, and is a well-known case of reusing ORB features across several tasks. Thinking about why key scenes are retained as keyframes connects computational cost to map management.

There are also direct methods that use the photometric error of the image. “Feature versus direct” is a distinction about how the observation error is built, while “sparse versus dense” concerns how much information is used. Do not treat them as the same distinction. The [KRoC 3D World lecture](https://drive.google.com/file/d/1OTZjzUGls3fjSQed7LU-xzjS_78e7BIW/view) compares these families.

## 4. A scale thought experiment

Here is the one thing a single camera can never recover, no matter how good the algorithm is.

Take a scene and double everything — the camera's movement _and_ the distances to every point:

```text
  version A                          version B (everything ×2)
  camera moves    1 m                camera moves    2 m
  point at        Z = 10 m           point at        Z = 20 m
  projection u = 500·(2/10)+320      projection u = 500·(4/20)+320
             = 420                              = 420
                                                  ↑
                            identical pixel. identical image. every frame.
```

A dollhouse filmed up close and a real house filmed from far away produce the same video. So a monocular system can report a trajectory that is perfectly shaped and uniformly, say, 1.7× too large, with zero reprojection error to warn you.

Scale has to come from somewhere outside the images: a stereo baseline (a known distance between two cameras), an IMU (gravity is a known 9.81 m/s²), wheel odometry, or a known object size. If learned depth supplies it instead, that is a prior baked in from training data, and its generalisation has to be examined separately.

One evaluation trap follows directly. Aligning a trajectory with Sim(3) lets the alignment rescale it, which silently erases exactly this error; SE(3) alignment does not. Papers reporting the two are not comparable. If learned depth is used, the prior obtained from the data and the conditions for generalisation have to be examined separately.

## Check questions

### Question 1 — Concept

Explain, from an observability standpoint, why the reprojection error of monocular Visual SLAM can be very small while the trajectory scale is wrong. What information can fix the scale?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

In pure monocular projection, scaling all 3D points and the camera translation by the same factor $s$ leaves the normalised image coordinates unchanged. Global metric scale is therefore an unobservable gauge freedom under image observations alone. A known stereo baseline, calibrated depth, a correctly modelled IMU with gravity and dynamics, wheel odometry, a known object size or a metric prior can supply scale. Note also that Sim(3) alignment in evaluation removes scale error, so those results must be distinguished from SE(3)-aligned ones.

</details>

### Question 2 — Math

For a pinhole camera, $u=f_xX/Z+c_x$ and $v=f_yY/Z+c_y$. Derive the projection Jacobian $\partial[u,v]/\partial[X,Y,Z]$ for a 3D camera point $[X,Y,Z]^T$. What numerical problem arises as $Z$ approaches zero?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

The Jacobian is

$$
J_\pi=
\begin{bmatrix}
f_x/Z & 0 & -f_xX/Z^2\\
0 & f_y/Z & -f_yY/Z^2
\end{bmatrix}.
$$

As $Z$ approaches zero each term becomes very large, so a small 3D change appears as a large pixel change and the linearisation becomes unstable. Points with $Z\le0$ are not valid observations in front of the camera, so cheirality and a minimum depth have to be checked before they enter the optimisation.

</details>

## Original reading

- ORB-SLAM: the system overview figure and the description of the three threads. Local copy: `_resource/slam/papers/orb-slam2015.pdf`.
- KRoC 3D Vision: PDF pages 31–38. For the BA text, start with the Introduction and the cost function.

<aside class="study-summary" markdown="1">
## What you learned

<dl>
  <dt>Reprojection error</dt><dd>The pixel difference between an observed feature and the projection predicted by the current pose and map.</dd>
  <dt>Bundle adjustment</dt><dd>Joint nonlinear optimisation of camera poses and 3D landmarks.</dd>
  <dt>Keyframe</dt><dd>A selected camera frame retained for mapping, optimisation and place recognition.</dd>
  <dt>Monocular scale</dt><dd>A gauge freedom: image geometry alone cannot determine absolute metric scale.</dd>
  <dt>Tracking and mapping</dt><dd>Estimating the current camera motion while maintaining and refining the scene representation.</dd>
</dl>
</aside>
