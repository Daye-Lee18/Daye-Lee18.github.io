---
layout: study-chapter
title: "Chapter 1. Why do we need SLAM?"
description: "The relationship between pose and map, odometry versus SLAM, and the broad arc of the field."
importance: 1
category: SLAM
series: slam_history
permalink: /study/slam/history/01-why-slam/
---

> **Goal:** Explain why pose and map have to be estimated together.  
> **Workload:** 15 minutes of reading + 5 minutes of check questions. No prior knowledge needed.

## 1. You need a map to know where you are

Suppose a robot enters a building for the first time and its sensor reports “there is a wall 3 m ahead of me”. That single number does not tell it where it is. The robot could be anywhere in the building that happens to have a wall 3 m in front of it.

```text
"a wall 3 m ahead"  →  which of these is true?

   room A          corridor B        room C
  ┌──────┐        ┌──────────┐      ┌──────┐
  │  🤖 ─┤ 3m     │  🤖 ─────┤ 3m   │ 🤖 ─ │ 3m
  └──────┘        └──────────┘      └──────┘

  all three are consistent with the same measurement
```

Now turn it around. To draw that wall onto a map, you have to write down _where the wall is_, and for that you need to know where the robot was standing when it measured. If the robot was at (0, 0) facing +x, the wall goes at x = 3. If it was at (10, 0), the same measurement puts the wall at x = 13. Same sensor reading, two completely different maps.

So each of the two problems needs the answer to the other one first.

SLAM estimates these two unknowns — **the robot's position and orientation, and the surrounding map** — jointly from observations. The map does not have to be a picture a human would look at; the coordinates of feature points to be re-observed are also a map. The KRoC [History lecture](https://drive.google.com/file/d/1tmWxcQFD0lGZPO3L6wjxT1k6am4EXyMK/view) introduces the problem by separating the requirements of localisation from those of mapping.

## 2. Odometry measures a position _relative to the previous one_

Odometry does not answer “where am I?”. From two consecutive observations it answers a much smaller question: **“how far have I moved since a moment ago?”** What it produces is a position relative to the previous pose, never an absolute position in the building.

Take a concrete case. The robot starts at the origin and drives straight along x, and odometry reports once per second.

```text
t = 0   pose (x, y, z) = (0.00, 0.00, 0.00)   ← the starting point we declared

t = 0 → 1   odometry says: Δ = (+1.00, 0.00, 0.00)     "I moved 1 m forward"
t = 1   pose = (0.00 + 1.00, 0, 0) = (1.00, 0.00, 0.00)

t = 1 → 2   odometry says: Δ = (+1.00, 0.00, 0.00)
t = 2   pose = (1.00 + 1.00, 0, 0) = (2.00, 0.00, 0.00)

t = 2 → 3   odometry says: Δ = (+1.00, 0.00, 0.00)
t = 3   pose = (3.00, 0.00, 0.00)
```

Read the middle column carefully. The sensor never once said “you are at (2.00, 0, 0)”. It only ever said “+1.00 in x”. The absolute pose in the left column is something _we_ computed, by adding up relative measurements starting from a position we simply assumed. In reality each step also carries a rotation, so the quantity being added is a relative pose (translation together with rotation) rather than three numbers.

## 3. Why adding up relative measurements drifts

Because the pose is a running sum, every error in a step stays in the sum forever.

Suppose the robot truly moves 1.00 m per step, but the wheels slip slightly and odometry reports 1.02 m each time.

```text
step      odometry sum      true position      error
  1           1.02              1.00            0.02
 10          10.20             10.00            0.20
100         102.00            100.00            2.00
```

A 2 cm error nobody would notice in a single step has become 2 m after a hundred steps. Nothing went wrong at step 100; the error was accumulated, not created.

Rotation error is worse still, because it turns the _direction_ of every step that follows. Get the heading wrong by 1° early on and 100 m later the trajectory is off sideways by roughly 1.7 m, even if every distance was measured perfectly.

This is what it looks like when the robot walks a 10 m square and comes back to its starting point.

```text
        estimated trajectory                 what actually happened

   (0,10) ┌──────────┐ (10,10)              the robot is standing
          │          │                       exactly where it started
          │          │
          │          │                       but the accumulated numbers say
    start └──────────┘  (0.4, 0.3)           it is 0.5 m away from the start
    (0,0)               ← should be (0,0)
```

The robot can see with its own eyes that it is back at the front door. Its arithmetic disagrees. That gap of 0.5 m is exactly the drift the sum has collected.

## 4. Loop closure turns “I have been here before” into a constraint

Loop closure confirms that the robot has revisited a past place, and adds a constraint between two distant points in time: _the pose at t = 400 and the pose at t = 0 are the same place_.

That constraint is powerful because it is the first piece of information in the whole chain that is **not** relative to the immediately preceding moment. It lets the accumulated 0.5 m be pushed back and redistributed over the entire trajectory, rather than being dumped on the last pose. (Chapter 6 works through that redistribution with actual numbers.)

```text
relative measurements → running sum → accumulated trajectory
                                            ↑
place recognition → geometric verification → revisit constraint
```

Note the middle step. Two places _looking_ similar and being _geometrically verified_ as the same place are different things — a first-floor corridor and a second-floor corridor can look identical.

Something is not SLAM only if loop closure always succeeds; some systems maintain only a local map. When you read a paper, check **which states it estimates and how far its map extends** rather than the name it uses. The [SLAM survey by Cadena et al.](https://arxiv.org/abs/1606.05830) lays out the problem definition, the structure of a full system, and the open challenges.

## 5. Four questions for reading the history

| Question                                     | The answer you will meet in these notes          |
| -------------------------------------------- | ------------------------------------------------ |
| How should error be represented?             | Probability distributions, covariance, filtering |
| How can computation be reduced on a big map? | Conditional independence, sparse optimisation    |
| Which sensors and maps should be used?       | Cameras, LiDAR, IMU, dense maps                  |
| Will it keep working in the real world?      | Revisits, robustness, learning, evaluation       |

This ordering is for study. Real research developed with heavy overlap, and a new method appearing does not mean earlier approaches disappear.

## 6. Check questions

### Question 1 — Concept

“If you just bolt loop closure onto odometry, can you always call it SLAM?” Answer from the perspective of the system components and of what is being estimated.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

The presence of loop closure alone is not enough to decide. First check the scope over which the system estimates the robot state and the environment representation. Odometry mainly accumulates relative motion between consecutive instants, while SLAM also handles the consistency between observations and the state and map together. SLAM without loop closure, or maintaining only a local map, is possible. Conversely, a system that only corrects past poses without producing a reusable environment representation is closer to pose-graph localisation or trajectory optimisation. So describe the input sensors, the state variables, the map representation and which variables the revisit constraints actually update, and only then settle on the term.

</details>

### Question 2 — Math and reasoning

Suppose each of the four sides of a square path is estimated at 10 m, with an independent standard deviation of $0.1m$ on each side's displacement. Ignoring rotation error for simplicity, what is the standard deviation of the position error remaining on one axis after two opposite-direction motions cancel? Also explain the limitation of simply snapping the last pose back to the origin.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

On one axis the variances of two independent motion errors add, so

$$
\sigma_x=\sqrt{0.1^2+0.1^2}\approx0.141m
$$

and the other axis is the same under the same assumption. In practice rotation error also affects the direction of subsequent motion, so the error grows beyond this simple independent sum and correlations appear between axes. Snapping only the last pose to the origin makes just the loop's endpoint agree; the error distributed over the intermediate trajectory and the map remains. You need to adjust the whole trajectory using the uncertainty of all poses and observation constraints.

</details>

## Original reading

- KRoC History slides: the `What is SLAM?` and `History of SLAM` sections. Local copy: `_resource/slam/kroc2026/01-history-ayoung-kim.pdf`.
- Cadena et al. (2016): read only the Introduction first. Local copy: `_resource/slam/papers/cadena2016-slam-survey.pdf`.

<aside class="study-summary" markdown="1">
## What you learned

<dl>
  <dt>Loop closure</dt><dd>Recognising a previously visited place and adding a constraint that can correct accumulated drift.</dd>
  <dt>Local map</dt><dd>A bounded, nearby representation used for fast and stable short-range estimation.</dd>
  <dt>SLAM</dt><dd>Estimating the robot state and an environment representation together because each depends on the other.</dd>
  <dt>Global map</dt><dd>A persistent representation that connects observations across the full trajectory or operating area.</dd>
  <dt>Odometry</dt><dd>Estimating relative motion over time; errors accumulate unless another source provides correction.</dd>
</dl>
</aside>
