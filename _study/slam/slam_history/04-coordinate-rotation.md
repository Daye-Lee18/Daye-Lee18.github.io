---
layout: study-chapter
title: "Chapter 4. Frames and rotation"
description: "The minimum geometry needed to read 3D SLAM equations."
importance: 4
category: SLAM
series: slam_history
permalink: /study/slam/history/04-coordinate-rotation/
---

> **Goal:** State transform directions explicitly and understand why rotations are not updated by plain addition.  
> **Workload:** 15 minutes. Matrix-vector products are assumed.

## 1. The same point has several sets of coordinates

A LiDAR does not report “the pillar is at (13, 0, 0) in the building”. It reports “the pillar is 3 m in front of _me_”. Same pillar, different numbers, depending on who is describing it.

```text
  the same pillar, two ways of saying where it is

  in the sensor's frame   p_S = (3, 0, 0)     "3 m ahead of me"
  in the world frame      p_W = (13, 0, 0)    "13 m from the building corner"

  neither is wrong — they answer different questions
```

Converting between them is what this chapter is about. In these notes $T_{AB}$ is **the transform that takes B coordinates into A coordinates** — read the subscripts right-to-left, "from B, to A":

$$
p_W=R_{WS}p_S+t_{WS}
$$

$R$ is the rotation and $t$ the translation; $t_{WS}$ is the world coordinate of the sensor origin. In the example above the sensor sat at $t_{WS}=(10,0,0)$ with no rotation, so $p_W = (3,0,0) + (10,0,0) = (13,0,0)$.

When chaining frames, the inner subscripts have to match, as in $T_{WC}=T_{WB}T_{BC}$ — the B's touch and cancel. If you find yourself writing $T_{WB}T_{CB}$, the B's are on the wrong sides and one of the two needs inverting. Frame transforms and observation models are covered in the foundational part of the [Course on SLAM](https://gisbi-kim.github.io/materials/study/soal17courseslam.pdf).

## 2. What if you just add rotation matrices?

With positions this works: if the estimate is off by 0.1 m, add 0.1 m. So why not do the same to a rotation matrix?

Try it in 2D. Take a 30° rotation and a 10° rotation and add the matrices entry by entry:

$$
R_{30}+R_{10}=
\begin{bmatrix}0.866 & -0.500\\ 0.500 & 0.866\end{bmatrix}
+
\begin{bmatrix}0.985 & -0.174\\ 0.174 & 0.985\end{bmatrix}
=
\begin{bmatrix}1.851 & -0.674\\ 0.674 & 1.851\end{bmatrix}
$$

That result is not a 40° rotation. It is not a rotation at all. Feed it the unit vector $(1,0)$ and you get $(1.851, 0.674)$, a vector of length 1.97 — the "rotation" nearly doubled the object. A rotation matrix has to satisfy $R^TR=I$ and $\det R=1$, and adding threw both away. (Here $\det = 1.851^2+0.674^2 \approx 3.88$, not 1.)

The correct composition is multiplication, not addition: $R_{30}R_{10}=R_{40}$. The set of matrices that stay legal under this operation is called $SO(3)$, and the practical consequence is that **an update has to be multiplied on, never added on**.

Having computed a small increment $\delta\theta$, you compose it like this:

$$
R_{\text{new}}=R\operatorname{Exp}([\delta\theta]_\times)
$$

This uses a right perturbation. Some references use a left update, so you must not mix Jacobians across conventions. Quaternions also need their conventions checked, including unit norm and component ordering. [Solà's Quaternion kinematics](https://arxiv.org/abs/1711.02508) covers the relations between representations and the perturbations in detail.

## 3. Checking direction by hand

Most frame bugs are caught by one example small enough to check in your head. Sensor at 2 m along x, no rotation, seeing a point 1 m ahead of itself:

```text
   world origin        sensor            point
        │                │                 │
        0                2                 3        (x axis)
                    t_WS = 2          p_S = 1 ahead
                                      p_W = 2 + 1 = 3   ✓
```

So $p_S=(1,0,0)$ gives $p_W=(3,0,0)$. To go the other way, from a world point to sensor coordinates:

$$
p_S=R_{WS}^{T}(p_W-t_{WS})
$$

Check it: $(3,0,0)-(2,0,0)=(1,0,0)$, back where we started. ✓

Now the failure modes, which are the reason to run the example at all:

```text
  computing p_S from p_W = 3, correct answer is 1

  you got     what went wrong
  ─────────   ──────────────────────────────────────────
  (1,0,0)     correct ✓
  (5,0,0)     added the translation instead of subtracting
              (3 + 2 = 5) — sign flipped
  (3,0,0)     forgot the translation entirely — you returned p_W
  (-1,0,0)    subtracted in the wrong order (2 - 3 = -1)
```

Make an example like this pass before you write complicated code. Make simple examples like this pass before you write complicated code.

## Check questions

### Question 1 — Concept

The robot's translational trajectory is correct, but the point cloud rotates the wrong way around the robot origin. List, in priority order, what you would check before touching any optimiser parameters.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

First check which direction of extrinsic the code expects, $T_{BS}$ or $T_{SB}$. Next check active versus passive rotation, quaternion component order `(w,x,y,z)` or `(x,y,z,w)`, degrees versus radians, axis handedness and the ROS optical-frame convention. A timestamp mismatch produces a similar symptom during rotational motion, so check time synchronisation too. It is more efficient to make unit conversions and a simple known-pose example pass first, and only then investigate residuals and the optimiser.

</details>

### Question 2 — Math

With $T_{AB}=(R_{AB},t_{AB})$, $T_{BC}=(R_{BC},t_{BC})$ and the definition $p_A=R_{AB}p_B+t_{AB}$, derive the rotation and translation of $T_{AC}$, and find $T_{AB}^{-1}$.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Substituting $p_B=R_{BC}p_C+t_{BC}$ into the first equation gives

$$
p_A=R_{AB}R_{BC}p_C+R_{AB}t_{BC}+t_{AB}.
$$

Therefore

$$
R_{AC}=R_{AB}R_{BC},\qquad
t_{AC}=R_{AB}t_{BC}+t_{AB}.
$$

The inverse transform is $p_B=R_{AB}^T(p_A-t_{AB})$, so

$$
T_{AB}^{-1}=(R_{AB}^T,-R_{AB}^Tt_{AB}).
$$

You do not simply subtract the translation vector; the inverse rotation has to be applied to it as well.

</details>

## Original reading

- Quaternion kinematics: whichever parts you need of §2 on rotation representations, §3 on conventions and §4 on perturbations. Local copy: `_resource/slam/foundations/sola2017-quaternion-eskf.pdf`.
- You do not need to derive every Jacobian on a first pass.
