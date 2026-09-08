---
layout: study-chapter
title: "LIO-SAM — paper review"
description: "Builds LiDAR-inertial estimation as a factor graph so GPS and loop constraints can be handled together; the smoothing-based control group."
category: SLAM
series: state_estimation
importance: 3
permalink: /study/slam/state-estimation/lio-sam/
---

[← State estimation paper comparison]({{ '/study/slam/state-estimation/' | relative_url }})

> **One-sentence summary:** Builds LiDAR-inertial estimation as a factor graph so GPS and loop constraints can be handled together; the smoothing-based control group.

| Item           | Detail                                                                                                                         |
| :------------- | :----------------------------------------------------------------------------------------------------------------------------- |
| Paper          | LIO-SAM: Tightly-coupled Lidar Inertial Odometry via Smoothing and Mapping                                                     |
| Venue          | IROS 2020                                                                                                                      |
| Source         | [Paper / author material](https://arxiv.org/abs/2007.00258) · [Official implementation](https://github.com/TixiaoShan/LIO-SAM) |
| Status         | Introductory review draft · personal close-reading and reproduction notes added below                                          |
| Source checked | 2026-09-07                                                                                                                     |

## 1. The problem it addresses

While connecting inertial prediction to LiDAR registration, the system also has to accept relative and absolute constraints that arrive on past states.

## 2. Three key points to present

1. **IMU preintegration:** inertial information is used for scan motion correction and for the initial guess of LiDAR registration.
2. **Local keyframe registration:** a new scan is registered against a limited set of past keyframes to keep computation manageable.
3. **Graph-based fusion:** LiDAR odometry, GPS and loop closure are expressed as factors that correct the trajectory.

Basis for this technical summary: [paper / author description](https://arxiv.org/abs/2007.00258).

## 3. How it works

The diagram below is a conceptual flow for understanding; it does not show every thread and update rate in the implementation.

```text
LiDAR + IMU → preintegration and deskew → feature-based local registration → graph update ← GPS and loop constraints
```

The LiDAR registration result is also used for IMU bias estimation. The official implementation keeps the IMU graph separate from the mapping graph. Do not read it as a structure that puts every raw sensor residual into a single, indefinitely growing graph.

## 4. Experimental results and interpretation

The paper evaluates data at several scales and environments across three platforms. In the review, only compare experiments that use GPS and loops in the same way. The loop implementation in the official repository is described as a proof of concept. [Source](https://arxiv.org/abs/2007.00258) · [Implementation notes](https://github.com/TixiaoShan/LIO-SAM)

The fact that loop constraints can be added does not mean wrong place matches are automatically resolved. Implementation requirements such as per-point time and ring information in the sensor messages also have to be checked.

When reading closely, record not only the names of the compared methods but also the sensor configuration, ground truth, trajectory alignment method, the compute platform, and whether failed segments are included.

## 5. Vision60 application questions

The following are **project application hypotheses and review questions**, kept separate from the paper's validated results.

1. Compared with FAST-LIO2, have you separated local error with loops and GPS off from global error with them on?
2. How does initialisation and resetting of the IMU graph affect the continuity of the output pose?
3. Can Vision60's LiDAR messages be converted into the input format of the official implementation?

**Proposed experiment:** compare results with loops disabled and enabled on the same revisit log. Record global error and pose jumps before and after the loop, separating the continuous state used for control from the corrected state used for mapping.

## 6. Check questions

Answer each question out loud first, then open the toggle. Practise stating the assumptions and equations together with the failure conditions and how you would verify them.

### Q1. Concept and structure

Why can you not conclude that loop closure is robust simply because LIO-SAM uses a factor graph?

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

The graph is a representation that optimises whatever constraints come in. False positives in place recognition, relative pose verification, robust losses and the removal of wrong constraints are separate problems. If a falsely detected constraint is strong, the graph can actually deform the whole trajectory. Candidate generation, geometric verification and graph optimisation have to be kept distinct.

</details>

### Q2. Math and reasoning

Given that the residual of an IMU preintegration factor is roughly \(r=[r_R,r_v,r_p,r_b]\) with covariance \(\Sigma\), write the factor's cost function and explain why the bias has to be relinearised.

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

The cost is \(J\_{imu}=r^\top\Sigma^{-1}r\). The preintegrated rotation, velocity and position depend on the gyro/accel bias assumed over the integration interval. When the bias estimate changes, the residual and Jacobian change too, so a first-order bias correction or reintegration and relinearisation is needed. It is good to add that the Mahalanobis weighting reflects the sensor uncertainty.

</details>

### Q3. Systems and debugging

After a loop closure the map is correct, but the controller became momentarily unstable. How would you design the state interface?

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

Separate the continuous local odometry frame from the globally corrected map frame. The controller uses the odom→base state, which is continuous over short intervals, while SLAM updates the map→odom transform. Also decide per consumer whether the global correction is applied immediately or blended in gradually. Timestamps and a single owner of the transform tree have to be made explicit.

</details>

### Q4. Code review and implementation

**GitHub:** [Official or author-linked repository](https://github.com/TixiaoShan/LIO-SAM)

Describe the data flow by which one LiDAR scan becomes a final pose across `imageProjection.cpp`, `featureExtraction.cpp`, `mapOptimization.cpp` and `imuPreintegration.cpp` in the official implementation, and decide where a bad `time` field should first be detected.

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

`imageProjection.cpp` performs cloud deskew and range image projection, `featureExtraction.cpp` produces edge and surface features. `mapOptimization.cpp` manages keyframes, the factor graph and map registration, and `imuPreintegration.cpp` updates the IMU-rate state and bias. Per-point `time` should be checked for valid range and monotonicity before deskew. For a 10 Hz LiDAR, check that the relative times lie roughly within one scan period, and fail loudly on a violation so it does not hide as an anomalous pose in a later module.

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

**Read next:** [FAST-LIO2 review]({{ '/study/slam/state-estimation/fast-lio2/' | relative_url }}) · [Full comparison table]({{ '/study/slam/state-estimation/' | relative_url }})
