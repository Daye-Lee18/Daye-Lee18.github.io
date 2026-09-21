---
layout: study-chapter
title: "4.1 Frames and transforms"
description: "Converting a point between sensor and world coordinates."
importance: 4.1
category: SLAM
series: slam_history
permalink: /study/slam/history/04-1-frames/
---

> **Goal:** convert points between frames, invert and chain transforms, and catch sign errors.

In these notes $T_{AB}$ takes **B coordinates into A coordinates** — read the subscripts right to left, “from B, to A”.

---

### Problem 1 — same pillar, two numbers

A LiDAR sits at $x = 10$ in the world, no rotation. It sees a pillar 3 m ahead. Where is the pillar in the world frame?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$10 + 3 = 13$. The sensor says $(3,0,0)$, the world says $(13,0,0)$ — **neither is wrong**, they answer different questions. In symbols: $p_W = R_{WS}\,p_S + t_{WS}$.

</details>

### Problem 2 — the other way

Sensor at $x = 2$, no rotation. A world point is at $x = 3$. Where is it in the sensor frame?

<details class="study-answer" markdown="1">
<summary>Hint</summary>

Undo the translation first.

</details>

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$p_S = R_{WS}^{\top}(p_W - t_{WS}) = 3 - 2 = 1$.

</details>

### Problem 3 — diagnose the bug

Four people computed Problem 2 and got $5$, $3$, $-1$ and $1$. What mistake did each make?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

```text
  1    correct
  5    added the translation (3 + 2) — sign flipped
  3    forgot the translation — returned p_W unchanged
 -1    subtracted in the wrong order (2 − 3)
```

**Most frame bugs are caught by one example small enough to check in your head.**

</details>

### Problem 4 — with a rotation

Sensor at $(2, 0)$, rotated 90° (its forward axis points along world $+y$). It sees a point 1 m ahead. World coordinates?

<details class="study-answer" markdown="1">
<summary>Hint</summary>

“1 m ahead” for this sensor is 1 m in world $+y$.

</details>

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$(2, 1)$. Rotate $(1,0)$ by 90° to get $(0,1)$, then add $(2,0)$. **Rotate, then translate.**

</details>

### Problem 5 — chaining

Which chain is valid: $T_{WB}\,T_{BC}$ or $T_{WB}\,T_{CB}$?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$T_{WB}\,T_{BC}$ — **the inner subscripts touch and cancel**, giving $T_{WC}$. In the other, the B's are on the wrong sides; one of the two needs inverting.

</details>

### Problem 6 — challenge — compose and invert

With $p_A = R_{AB}p_B + t_{AB}$ and $p_B = R_{BC}p_C + t_{BC}$, derive $T_{AC}$ and $T_{AB}^{-1}$.

<details class="study-answer" markdown="1">
<summary>Hint</summary>

Substitute the second into the first.

</details>

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$p_A = R_{AB}R_{BC}p_C + R_{AB}t_{BC} + t_{AB}$, so $R_{AC} = R_{AB}R_{BC}$, $t_{AC} = R_{AB}t_{BC} + t_{AB}$.
Inverse: $p_B = R_{AB}^{\top}(p_A - t_{AB})$, so $T_{AB}^{-1} = (R_{AB}^{\top},\ -R_{AB}^{\top}t_{AB})$ — the translation is rotated too, not just negated.

</details>

---

**You can now:** convert points between frames in both directions, chain transforms with matching subscripts, and diagnose the common sign errors.
