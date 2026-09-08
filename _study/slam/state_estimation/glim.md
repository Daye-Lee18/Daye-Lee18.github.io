---
layout: study-chapter
title: "GLIM — paper review"
description: "A mapping framework that optimises states within a time window and inter-submap registration, handling the computation on GPU."
category: SLAM
series: state_estimation
importance: 8
permalink: /study/slam/state-estimation/glim/
---

[← State estimation paper comparison]({{ '/study/slam/state-estimation/' | relative_url }})

> **One-sentence summary:** A mapping framework that optimises states within a time window and inter-submap registration, handling the computation on GPU.

| Item           | Detail                                                                                                                  |
| :------------- | :---------------------------------------------------------------------------------------------------------------------- |
| Paper          | GLIM: 3D Range-Inertial Localization and Mapping with GPU-Accelerated Scan Matching Factors                             |
| Venue          | Robotics and Autonomous Systems 2024                                                                                    |
| Source         | [Paper / author material](https://arxiv.org/abs/2407.10344) · [Official implementation](https://github.com/koide3/glim) |
| Status         | Introductory review draft · personal close-reading and reproduction notes added below                                   |
| Source checked | 2026-09-07                                                                                                              |

## 1. The problem it addresses

When the geometric constraints of the current scan are temporarily weak, past states and registration relations have to be exploited. Optimising a rich set of constraints also means managing the computational cost.

## 2. Three key points to present

1. **Fixed-lag smoothing:** the states within a bounded time window are estimated together.
2. **GPU scan matching factors:** the parallelism of point cloud registration error computation is exploited.
3. **Global registration error minimisation:** the inter-submap registration error of the whole map is optimised directly.

Basis for this technical summary: [paper / author description](https://arxiv.org/abs/2407.10344).

## 3. How it works

The diagram below is a conceptual flow for understanding; it does not show every thread and update rate in the implementation.

```text
range + IMU (+ camera constraints) → local fixed-lag smoothing → submaps → global registration optimisation
```

The paper also describes the tight coupling of multi-camera feature constraints. Read it distinguishing which states are held by the local time window and which by the global submap optimisation.

## 4. Experimental results and interpretation

The authors report the ability to handle situations where range data degenerates completely for several seconds, and real-time processing using a GPU. Such results do not guarantee unbounded degeneracy tolerance or real-time operation on every GPU. [Source](https://arxiv.org/abs/2407.10344) · [Implementation notes](https://github.com/koide3/glim)

The design uses more computation than a filter-based approach, so it can compete with other work on an onboard GPU. The GLIM paper is from 2024 and is distinct from the 2022 work it builds on.

When reading closely, record not only the names of the compared methods but also the sensor configuration, ground truth, trajectory alignment method, the compute platform, and whether failed segments are included.

## 5. Vision60 application questions

The following are **project application hypotheses and review questions**, kept separate from the paper's validated results.

1. How do accuracy, memory and latency change as the time window is lengthened?
2. Does the Vision60 GPU also run perception and control workloads?
3. How much do the camera and the time window each contribute to recovery from geometric degeneracy?

**Proposed experiment:** compare against FAST-LIO2 on the same degeneracy log, recording per-segment error, upper-percentile latency and GPU memory. Match whether the camera is used before comparing.

## 6. Check questions

Answer each question out loud first, then open the toggle. Practise stating the assumptions and equations together with the failure conditions and how you would verify them.

### Q1. Concept and structure

Explain why fixed-lag smoothing can be better than an EKF under temporary LiDAR degeneracy, and what it costs.

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

It relinearises several states and observations in the time window together, so information from before and after the degenerate segment can link the internal states. On the other hand the number of states and factors, the memory, the linear solve cost and the latency all increase. The consistency of the prior produced by marginalising states outside the window also has to be considered.

</details>

### Q2. Math and reasoning

Write the normal equation obtained by first-order linearisation of the nonlinear least squares \(\min*\delta\sum_i\|r_i(x\boxplus\delta)\|*{\Sigma_i^{-1}}^2\), and explain its relation to degeneracy.

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

\(H\delta=-g\) with \(H=\sum_iJ_i^\top\Sigma_i^{-1}J_i\) and \(g=\sum_iJ_i^\top\Sigma_i^{-1}r_i\). Small eigenvalues of \(H\) mean the information constraining that state direction is weak. Damping can give numerical stability, but it does not create new observational information.

</details>

### Q3. Systems and debugging

The mean runtime is real-time, but Vision60's control intermittently misses its deadline. From the perspective of GPU-based SLAM, what would you measure?

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

Measure per-kernel, memory transfer and synchronisation times, p95/p99 and worst-case latency, peak GPU memory, queue depth, contention with perception and learning workloads, and thermal throttling. Looking only at mean FPS misses bursts and blocking. Also check the streams, priorities and resource partitioning between control and SLAM.

</details>

### Q4. Code review and implementation

**GitHub:** [Official or author-linked repository](https://github.com/koide3/glim)

Suppose you add a new velocity factor to GLIM. Beyond implementing the factor, how would you verify registration, config, state timestamps, the Jacobian and real-time performance?

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

First follow the path by which existing factors are created and registered, and the state key and timestamp conventions held by the optimiser. Expose the noise model and robust kernel through the config, and state explicitly which two states the sensor measurement time connects. Match the analytic Jacobian to the manifold perturbation convention and compare it against finite differences. In an A/B test on the same bag with the factor on and off, measure trajectory error, innovation, graph size and solve-time p95/p99, and check there are no dangling keys after marginalisation.

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
