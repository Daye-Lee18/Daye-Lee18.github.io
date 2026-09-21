---
layout: study-chapter
title: "4.2 Why rotations do not add"
description: "Rotation matrices compose by multiplication."
importance: 4.2
category: SLAM
series: slam_history
permalink: /study/slam/history/04-2-rotations/
---

> **Goal:** show why adding rotation matrices is wrong and state how a rotation is actually updated.

---

### Problem 1 — what a rotation keeps

Rotate the vector $(1, 0)$ by 30°. What is its length before and after?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$(0.866, 0.5)$, length 1 both times. **A rotation never changes lengths.** That is what $R^{\top}R = I$ and $\det R = 1$ say.

</details>

### Problem 2 — try adding

Add the 30° and 10° rotation matrices entry by entry. Apply the result to $(1,0)$. What length do you get?

<details class="study-answer" markdown="1">
<summary>Hint</summary>

The first column of $R_\theta$ is $(\cos\theta, \sin\theta)$.

</details>

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

First column: $(0.866 + 0.985,\ 0.5 + 0.174) = (1.851, 0.674)$, length $\approx 1.97$.
**The “rotation” nearly doubled the object.** The sum is not a rotation at all.

</details>

### Problem 3 — the right operation

What do you get from $R_{30}R_{10}$ (multiplication)?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$R_{40}$ exactly. **Rotations compose by multiplication.** The set of matrices closed under this is $SO(3)$ (in 3D).

</details>

### Problem 4 — updating an estimate

An optimiser computes a small correction $\delta\theta$. For a position you would write $p + \delta p$. What do you do for a rotation?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Multiply it on: $R_{\text{new}} = R\,\operatorname{Exp}([\delta\theta]_\times)$. Never $R + \delta R$.
This is a **right** perturbation; some references use a left one ($\operatorname{Exp}(\cdot)R$), so **never mix Jacobians across conventions.**

</details>

### Problem 5 — challenge — debugging order

The robot's translation is right but its point cloud rotates the wrong way around the robot. List what you check before touching optimiser parameters.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

1. Which direction the code expects, $T_{BS}$ or $T_{SB}$.
2. Quaternion order: `(w,x,y,z)` vs `(x,y,z,w)`; degrees vs radians; handedness; ROS optical-frame convention.
3. Timestamps — a time offset during rotation looks similar.
4. Only then residuals and the optimiser. Make a known-pose example pass first.

</details>

---

**You can now:** explain why rotations must be multiplied, not added, and name the conventions that silently break rotation code.
