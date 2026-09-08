---
layout: study-chapter
title: "LIJO — paper review"
description: "Combines LiDAR, IMU and joint velocity information in an EKF to reduce high-frequency jitter in quadruped odometry."
category: SLAM
series: state_estimation
importance: 14
permalink: /study/slam/state-estimation/lijo/
---

[← State estimation paper comparison]({{ '/study/slam/state-estimation/' | relative_url }})

> **One-sentence summary:** Combines LiDAR, IMU and joint velocity information in an EKF to reduce high-frequency jitter in quadruped odometry.

| Item           | Detail                                                                                |
| :------------- | :------------------------------------------------------------------------------------ |
| Paper          | Smooth LiDAR–Inertial–Joint Odometry for perception-driven legged locomotion          |
| Venue          | Robot Learning · 2026-08-10                                                           |
| Source         | [Paper / author material](https://www.elspub.com/doi/10.55092/rl20260024)             |
| Status         | Introductory review draft · personal close-reading and reproduction notes added below |
| Source checked | 2026-09-07                                                                            |

## 1. The problem it addresses

If the pose wobbles at rest or low speed, an unstable state can be passed on to perception, planning and control.

## 2. Three key points to present

1. **Joint-based body velocity:** velocity is estimated from joint angles and rates via forward kinematics and used in the EKF prediction.
2. **Dynamic weighting:** the confidence in the joint-based velocity information is adjusted according to the motion speed.
3. **IMU observation model:** the IMU is treated as an observation and the sensors are combined in a manifold EKF.

Basis for this technical summary: [paper / author description](https://www.elspub.com/doi/10.55092/rl20260024).

## 3. How it works

The diagram below is a conceptual flow for understanding; it does not show every thread and update rate in the implementation.

```text
joint angles and rates → kinematic velocity and weighting → EKF prediction → IMU and LiDAR observation update → odometry
```

This flow is a conceptual summary of the publisher's abstract. Contact determination, the weighting formula, the exact state vector and the update schedule need verification against the full text. Do not read the speed-dependent weighting directly as a slip probability.

## 4. Experimental results and interpretation

The publisher's abstract reports that in real quadruped experiments including steep stairs and large loop trajectories, high-frequency jitter was reduced while accuracy was maintained. The quantitative figures and ablations have not yet been verified on this page. [Source](https://www.elspub.com/doi/10.55092/rl20260024)

**A first-pass review based on the publisher's abstract.** Map accuracy and the kind of smoothness that is useful for control are different metrics. Any actual improvement in control performance has to be confirmed by a separate experiment.

When reading closely, record not only the names of the compared methods but also the sensor configuration, ground truth, trajectory alignment method, the compute platform, and whether failed segments are included.

## 5. Vision60 application questions

The following are **project application hypotheses and review questions**, kept separate from the paper's validated results.

1. Can the dynamic weighting distinguish low-speed slip from high-speed normal contact?
2. Does the jitter reduction come with increased filter latency or attenuation of real motion?
3. Compared with VILENS's velocity bias estimation, what is expressed as a state and what as a weight?

**Proposed experiment:** measure pose and velocity variance, relative error and latency together over stationary, low-speed and landing segments. Include a plain low-pass filter as a control group to isolate the effect of the joint fusion itself.

## 6. Check questions

Answer each question out loud first, then open the toggle. Practise stating the assumptions and equations together with the failure conditions and how you would verify them.

### Q1. Concept and structure

How does LIJO's joint-based velocity constraint differ from a plain low-pass filter in the way it reduces pose jitter?

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

A low-pass filter removes high-frequency content from the output without adding any new physical observation, and it introduces latency. Joint-based velocity puts an independent constraint obtained from kinematics into the state estimate. That said, slip and model error can bias this constraint too, so weighting and contact handling are needed.

</details>

### Q2. Math and reasoning

Given a LiDAR velocity observation \(z_L=v+n_L\) and a joint velocity observation \(z_K=v+n_K\) with variances \(\sigma_L^2\) and \(\sigma_K^2\), write the optimal one-dimensional fused value and its variance.

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

For independent Gaussians, \(\hat v=(z_L/\sigma_L^2+z_K/\sigma_K^2)/(1/\sigma_L^2+1/\sigma_K^2)\) and \(\sigma^2=(1/\sigma_L^2+1/\sigma_K^2)^{-1}\). Increasing \(\sigma_K^2\) during slip reduces the influence of the joint observation. If the two noises are correlated or biased, this expression can be overconfident.

</details>

### Q3. Systems and debugging

The stationary jitter RMS is down, but the response on landing on a stair is slower. How would you decide whether to adopt it?

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

Measure accuracy, smoothness and latency as separate metrics. Compare stationary pose and velocity RMS, the phase lag and rise time of landing events, relative pose error, and the actual control tracking and stability. Ablate the dynamic weighting against a plain filter baseline to separate whether the jitter reduction comes from information fusion or from smoothing.

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

**Read next:** [VILENS review]({{ '/study/slam/state-estimation/vilens/' | relative_url }}) · [Full comparison table]({{ '/study/slam/state-estimation/' | relative_url }})
