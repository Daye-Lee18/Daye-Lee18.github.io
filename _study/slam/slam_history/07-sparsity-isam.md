---
layout: study-chapter
title: "Chapter 7. Sparsity and iSAM"
description: "Many variables does not mean every variable is connected to every other."
importance: 7
category: SLAM
series: slam_history
permalink: /study/slam/history/07-sparsity-isam/
---

> **Goal:** Distinguish sparsity, fill-in and incremental update.  
> **Workload:** 15 minutes. Read after Chapter 6.

## 1. You do not solve a big matrix by inverting it

Chapter 6 left us minimising a sum over every constraint. With 10,000 poses that is a system of 10,000 unknowns, and solving it by inverting a $10{,}000 \times 10{,}000$ matrix would be hopeless — inversion costs roughly $n^3$, so about $10^{12}$ operations.

It is tractable anyway, because almost all of that matrix is zeros.

Linearising the residual near the current estimate gives $r(X+\delta)\approx r(X)+J\delta$. Writing the weight-absorbed Jacobian as $A$ and the right-hand side as $b$, one iteration takes the form

$$
\min_\delta\|A\delta-b\|^2.
$$

Why the zeros? Because an odometry measurement between $x_5$ and $x_6$ says nothing whatsoever about $x_{900}$. Its row of $A$ touches two variables and is blank everywhere else:

```text
              x0  x1  x2  x3  x4  x5  x6  ...  x900
  odom 5→6     0   0   0   0   0   ■   ■         0
  odom 6→7     0   0   0   0   0   0   ■         0
  loop 0↔900   ■   0   0   0   0   0   0         ■
                                        ↑
                    each row has 2 non-zero blocks, not 10,000
```

With one row per measurement and only two blocks filled in each, the matrix is over 99.9% zeros. Real implementations exploit that with a QR or Cholesky factorisation, which skips the zeros entirely, rather than forming a large inverse that would have to store them.

## 2. A sparse input can still fill in during the computation

Here is the catch. Solving proceeds by **eliminating** variables one at a time — substituting one out and rewriting the rest in terms of what remains. And eliminating a variable makes all of its neighbours talk to each other.

Think of it socially. A landmark $L$ is seen by poses A, B and C. Remove $L$, and whatever it was telling each of them has to be re-expressed as relations directly among A, B and C:

```text
  before eliminating L          after eliminating L

      A ── L ── B                   A ─────── B
           │                         ╲       ╱
           C                          ╲     ╱
                                        C
      3 edges                       3 edges, but now
                                    pose-to-pose (a clique)
```

New edges appeared that no sensor ever measured. That is **fill-in**: the matrix started sparse and got denser as we solved it. Fill-in costs both memory and time, and it is the reason elimination _order_ matters — eliminate a variable with 3 neighbours and you add 3 edges; eliminate one with 50 neighbours and you add 1,225.

[Factor Graphs for Robot Perception](https://www.cs.cmu.edu/~kaess/pub/Dellaert17fnt.html) explains this structure from both the graph and the linear-algebra side. In the line running through Square Root SAM, iSAM and iSAM2, sparse factorisation and the reuse of previous computation are central. Note, though, that a large loop closure can force recomputation over a wide region, so “incremental” does not always mean constant time.

## 3. Draw elimination on paper

Order matters enormously, and you can see it on one tiny graph. Poses A, B, C are linked by odometry; landmark L is seen by all three; and D is a stray pose that only ever saw L.

```text
      A ─── B ─── C        odometry chain: A-B, B-C already exist
       ╲    │    ╱
        ╲   │   ╱
          ╲ │ ╱
            L ─── D        D's only neighbour is L
```

**Ordering 1 — eliminate L first.** Its neighbours are A, B, C, D, so all four must become mutually connected:

```text
  needed: A-B ✓ exists   A-C ✗ NEW   A-D ✗ NEW
          B-C ✓ exists   B-D ✗ NEW   C-D ✗ NEW
                                      → 4 edges of fill-in
```

**Ordering 2 — eliminate D first, then L.** D has a single neighbour, so removing it connects nothing to anything. Then L is left with three neighbours:

```text
  eliminating D:  0 new edges
  eliminating L:  A-B ✓, B-C ✓, A-C ✗ NEW
                                      → 1 edge of fill-in
```

Same graph, same final answer, four times less fill-in — purely from the order. On a real graph this is the difference between a solve that runs at sensor rate and one that does not, which is why libraries spend real effort choosing an ordering (COLAMD and similar heuristics) before touching the numbers.

The rule of thumb the example illustrates: **eliminate cheap, low-degree variables first**, and put the heavily connected ones off until the end.

## Check questions

### Question 1 — Concept

Explain why iSAM2 can be faster than full batch optimisation when a new factor arrives, and also why it does not guarantee constant computation time.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

iSAM2 reuses the existing factorisation stored in the Bayes tree and recomputes mainly the cliques affected by the new factor. It does not relinearise every variable each time; it can selectively relinearise only those variables judged to need it. However, a loop closure connecting far-apart poses, a large state change, or a poor variable ordering can make the affected subtree and the fill-in very large. So incremental update is a reuse structure, not a claim that every update is $O(1)$.

</details>

### Question 2 — Math and graph reasoning

The variable order is $[A,B,C,L]$ and landmark $L$ is connected to poses $A,B,C$. Write down all the fill-in edges created in the pose graph when $L$ is eliminated first, and explain what new edges appear if instead the leaf pose $A$ is eliminated first.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

The neighbours of $L$ are $A,B,C$. Eliminating $L$ makes those neighbours a clique, creating the edges $A-B$, $A-C$ and $B-C$. By contrast, if $A$ is a leaf connected only to $L$, eliminating $A$ first leaves just the single neighbour $L$, so no new edge appears. The example shows why elimination ordering changes fill-in and factor sizes. In real SLAM the ordering is chosen with numerical stability, the relinearisation scope and the graph structure all taken into account.

</details>

## Original reading

- Dellaert & Kaess (2017): read the elimination/sparsity discussion first, then move on to the iSAM part. Local copy: `_resource/slam/foundations/dellaert-kaess2017-factor-graphs.pdf`.
- Before following every equation, draw the factor graph and the post-elimination graph yourself.
