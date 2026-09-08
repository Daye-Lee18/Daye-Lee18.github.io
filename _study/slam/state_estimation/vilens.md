---
layout: study-chapter
title: "VILENS — paper review"
description: "A quadruped state estimator that fuses visual, inertial, LiDAR and leg odometry in a graph and estimates a leg velocity bias."
category: SLAM
series: state_estimation
importance: 6
permalink: /study/slam/state-estimation/vilens/
---

[← State estimation paper comparison]({{ '/study/slam/state-estimation/' | relative_url }})

> **One-sentence summary:** A quadruped state estimator that fuses visual, inertial, LiDAR and leg odometry in a graph and estimates a leg velocity bias.

| Item           | Detail                                                                                     |
| :------------- | :----------------------------------------------------------------------------------------- |
| Paper          | VILENS: Visual, Inertial, Lidar, and Leg Odometry for All-Terrain Legged Robots            |
| Venue          | T-RO · online 2022 / issue 2023                                                            |
| Source         | [Paper / author material](https://robots.ox.ac.uk/~mfallon/publications/2022TRO_wisth.pdf) |
| Status         | Introductory review draft · personal close-reading and reproduction notes added below      |
| Source checked | 2026-09-07                                                                                 |

## 1. The problem it addresses

Foot slip, terrain deformation and leg compliance introduce errors into kinematics-based velocity. External sensors can also degenerate in darkness, dust or feature-poor scenes.

## 2. Three key points to present

1. **Tight coupling of four modalities:** visual, inertial, LiDAR and leg constraints are combined in a factor graph.
2. **Leg velocity preintegration:** the velocity from leg odometry is formed into a constraint over a time interval.
3. **Velocity bias estimation:** a linear velocity bias is added to the state and estimated through fusion with the other modalities.

Basis for this technical summary: [paper / author description](https://robots.ox.ac.uk/~mfallon/publications/2022TRO_wisth.pdf).

## 3. How it works

The diagram below is a conceptual flow for understanding; it does not show every thread and update rate in the implementation.

```text
joint and leg velocity → velocity preintegration factor + visual, LiDAR and IMU factors → state and velocity bias estimate
```

The bias is a state introduced to explain the systematic error in leg velocity. Do not equate it with a contact classifier that perfectly detects each foot's slip. The focus of this paper is multi-sensor odometry.

## 4. Experimental results and interpretation

The authors report a total of 2 hours and 1.8 km of experiments on several ANYmal robots, including loose stones, slopes, mud and dark, dusty environments. [Source](https://robots.ox.ac.uk/~mfallon/publications/2022TRO_wisth.pdf)

Coupling with external sensors is essential for the bias to be observable. Behaviour when all sensors weaken at once, and the cost of porting to a different robot, have to be assessed separately.

When reading closely, record not only the names of the compared methods but also the sensor configuration, ground truth, trajectory alignment method, the compute platform, and whether failed segments are included.

## 5. Vision60 application questions

The following are **project application hypotheses and review questions**, kept separate from the paper's validated results.

1. Can Vision60 provide timestamped joint angles, joint velocities and contact information?
2. Over what timescale can slip be explained as a velocity bias?
3. In the sensor ablation, are the effects of the leg factor and of the bias state separated?

**Proposed experiment:** using logs annotated with slipping and normal-contact segments, compare leg velocity against an external reference velocity. Record velocity and height error per segment before and after adding the bias.

## 6. Check questions

Answer each question out loud first, then open the toggle. Practise stating the assumptions and equations together with the failure conditions and how you would verify them.

### Q1. Concept and structure

Why does VILENS estimate a linear velocity bias as a state instead of simply giving leg odometry a fixed covariance?

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

Because terrain deformation, leg compliance and slip can produce persistent, directional velocity errors. A fixed covariance only lowers the confidence; it does not explicitly correct a systematic error. Constraints from other sensors are needed for the bias to become observable and distinguishable from the actual motion.

</details>

### Q2. Math and reasoning

With only the measurement model \(z_v=v+b_v+n\) and both \(v\) and \(b_v\) unknown, why is it hard to separate the two? What extra information is needed?

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

A single measurement only constrains the sum \(v+b_v\), so \(v\leftarrow v+\Delta, b_v\leftarrow b_v-\Delta\) produces the same measurement, leaving the system rank deficient. Temporal and external constraints are needed, such as IMU dynamics, LiDAR and visual pose changes, and a bias random-walk prior. A good answer explains the null space of the observation Jacobian rather than simply saying there are many sensors.

</details>

### Q3. Systems and debugging

What goes wrong if you average the velocity across all legs while one foot in contact is slipping? Propose a way to make it robust.

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

The outlier foot contaminates the body velocity estimate. Maintain per-foot contact probability and innovation, and reduce the influence with gating, a robust loss, per-foot covariance adjustment or consistency checks. Without a real contact sensor, the latency and false detections of torque- and kinematics-based contact estimation also have to be evaluated.

</details>

## 7. Close-reading and presentation record

Use the summary above as a starting point, check the equations, figures and experiment tables in the original, then fill this in yourself. Keep reproduction results you have not yet run separate from the paper's results.

| Item to record           | Personal review notes                                                                |
| :----------------------- | :----------------------------------------------------------------------------------- |
| States, inputs, outputs  | Not written — record frames, units and sensor rates                                  |
| Key equations            | Not written — explain equation numbers, variable meanings, assumptions and residuals |
| Key figures              | Not written — explain the figure number and the data flow in your own words          |
| Experimental evidence    | Not written — table/figure numbers, dataset, baselines, metrics and conditions       |
| Ablation                 | Not written — which element was removed and what changed                             |
| Failure cases and limits | Not written — separate what the authors report from your own inference               |
| Code and reproduction    | Not written — version, configuration, logs, hardware, measurements                   |
| Final judgement          | Not written — reasons to adopt or defer for Vision60                                 |

- [ ] I can explain the three key contributions with evidence from the original paper.
- [ ] I can explain how the states and observations are connected.
- [ ] I have separated the paper's results from the Vision60 application hypotheses.

**Read next:** [LIJO review]({{ '/study/slam/state-estimation/lijo/' | relative_url }}) · [Full comparison table]({{ '/study/slam/state-estimation/' | relative_url }})
