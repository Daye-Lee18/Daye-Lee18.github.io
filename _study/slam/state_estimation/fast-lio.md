---
layout: study-chapter
title: "FAST-LIO — paper review"
description: "Combines LiDAR feature points and IMU in an iterated EKF, handling many observations efficiently; the basis of FAST-LIO2."
category: SLAM
series: state_estimation
importance: 10
permalink: /study/slam/state-estimation/fast-lio/
---

[← State estimation paper comparison]({{ '/study/slam/state-estimation/' | relative_url }})

> **One-sentence summary:** Combines LiDAR feature points and IMU in an iterated EKF, handling many observations efficiently; the basis of FAST-LIO2.

| Item           | Detail                                                                                             |
| :------------- | :------------------------------------------------------------------------------------------------- |
| Paper          | FAST-LIO: A Fast, Robust LiDAR-inertial Odometry Package by Tightly-Coupled Iterated Kalman Filter |
| Venue          | arXiv 2020                                                                                         |
| Source         | [Paper / author material](https://arxiv.org/abs/2010.08196)                                        |
| Status         | Introductory review draft · personal close-reading and reproduction notes added below              |
| Source checked | 2026-09-07                                                                                         |

## 1. The problem it addresses

Processing many LiDAR observations in a tightly coupled filter can make the Kalman gain computation expensive.

## 2. Three key points to present

1. **Tightly coupled fusion:** LiDAR feature points and inertial information are used in the same estimation process.
2. **Iterated EKF:** iterative updates handle the nonlinear observations.
3. **Efficient gain computation:** a formulation based on the state dimension rather than the observation dimension reduces the cost.

Basis for this technical summary: [paper / author description](https://arxiv.org/abs/2010.08196).

## 3. How it works

The diagram below is a conceptual flow for understanding; it does not show every thread and update rate in the implementation.

```text
LiDAR feature points + IMU → motion prediction → iterated filter update → state estimate
```

Understand the filter's state, observations and update in this paper first, then read what FAST-LIO2 changes in feature extraction and map management. A fast filter and a good map data structure are separate design elements.

## 4. Experimental results and interpretation

The authors report a case of fusing more than 1,200 effective feature points from a single scan and completing the whole iterated update within 25 ms in a UAV onboard setting. [Source](https://arxiv.org/abs/2010.08196)

That figure is the result of that implementation and hardware. It does not mean the whole pipeline's cost stays constant as observations grow.

When reading closely, record not only the names of the compared methods but also the sensor configuration, ground truth, trajectory alignment method, the compute platform, and whether failed segments are included.

## 5. Vision60 application questions

The following are **project application hypotheses and review questions**, kept separate from the paper's validated results.

1. What exactly does each of the state dimension and the observation dimension count?
2. Which terms are relinearised during the iterations and which are held fixed?
3. Compared with FAST-LIO2, can the filter improvement be separated from the map management improvement?

**Proposed experiment:** break down the runtime of the current code into IMU processing, correspondence search, filter update and map update, and record each. Check whether the bottleneck really is the gain computation.

## 6. Check questions

Answer each question out loud first, then open the toggle. Practise stating the assumptions and equations together with the failure conditions and how you would verify them.

### Q1. Concept and structure

Compare the contributions of FAST-LIO and FAST-LIO2 along three axes: the filter, observation selection, and the map data structure.

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

FAST-LIO provides the basis for tightly coupling feature points and IMU in an efficient iterated Kalman filter. FAST-LIO2 adds raw-point direct registration and ikd-Tree map management as its core contributions on top of that filter lineage. Explaining every FAST-LIO2 improvement as a change in the filter equations would be inaccurate.

</details>

### Q2. Math and reasoning

Explain the computational reason for using the information form instead of inverting an \(m\times m\) matrix when the number of observations \(m\) is much larger than the state dimension \(n\).

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

The standard gain expression involves inverting \(HPH^\top+R\) in observation space, which is burdensome for large \(m\). Using the Woodbury identity or the information form lets you instead solve \(P^{-1}+H^\top R^{-1}H\) in the \(n\times n\) state space. The complexity advantage also depends on the structure, the sparsity and how \(R\) is handled.

</details>

### Q3. Systems and debugging

The filter update takes 25 ms but the whole node takes 50 ms. Where should you profile?

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

Separate preprocessing and feature extraction, timestamp alignment, deskew, nearest-neighbour search, residual construction, filter iterations, map insertion and deletion, and publish/copy. Look at wall time alongside CPU time, allocation and queue waiting. The paper's filter timing must not be compared against whole-pipeline latency.

</details>

### Q4. Code review and implementation

**GitHub:** [Official or author-linked repository](https://github.com/hku-mars/FAST_LIO)

One cycle of `src/laserMapping.cpp` is taking too long. Which stages would you instrument, and how would you confirm that the optimisation did not change the estimation results?

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

Instrument synchronisation, IMU propagation and undistortion, downsampling, nearest-neighbour search, residual and Jacobian construction, the iterated update, ikd-Tree insertion and deletion, and publishing. Record not just the mean but p95/p99 and the number of points processed. On the same bag before and after the change, compare pose per timestamp, the number of valid residuals and the number of map points, and state the tolerance explicitly. If you parallelise, check for races on the shared map and correspondence buffer with a thread sanitizer or by running deterministically and repeatedly.

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
