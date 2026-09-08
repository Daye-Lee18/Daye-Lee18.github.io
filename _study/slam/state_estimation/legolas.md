---
layout: study-chapter
title: "Legolas — paper review"
description: "Learns odometry from leg and inertial sensors alone, addressing situations where external-sensor tracking is difficult."
category: SLAM
series: state_estimation
importance: 13
permalink: /study/slam/state-estimation/legolas/
---

[← State estimation paper comparison]({{ '/study/slam/state-estimation/' | relative_url }})

> **One-sentence summary:** Learns odometry from leg and inertial sensors alone, addressing situations where external-sensor tracking is difficult.

| Item           | Detail                                                                                |
| :------------- | :------------------------------------------------------------------------------------ |
| Paper          | Legolas: Deep Leg-Inertial Odometry                                                   |
| Venue          | 8th CoRL · PMLR 270, 2025                                                             |
| Source         | [Paper / author material](https://proceedings.mlr.press/v270/wasserman25a.html)       |
| Status         | Introductory review draft · personal close-reading and reproduction notes added below |
| Source checked | 2026-09-07                                                                            |

## 1. The problem it addresses

Analytic leg odometry needs a per-robot model and tuning, while existing learning approaches are affected by the need to collect high-quality real-world trajectories and by distribution shift.

## 2. Three key points to present

1. **Learning-based leg-inertial odometry:** motion is estimated from leg and IMU signals.
2. **Training without collecting real-world trajectories:** an approach that reduces the dependence on measured trajectory data.
3. **Evaluation on real robots:** deployment results are presented on two quadruped platforms, indoors and outdoors.

Basis for this technical summary: [paper / author description](https://proceedings.mlr.press/v270/wasserman25a.html).

## 3. How it works

The diagram below is a conceptual flow for understanding; it does not show every thread and update rate in the implementation.

```text
leg sensors + IMU → learned odometry model → relative motion and trajectory estimate
```

Estimating motion from proprioception without registering against the external environment is what distinguishes it from LIO and LIVO. The network architecture, the input time window, the output representation and the training loss should be diagrammed separately during a close reading.

## 4. Experimental results and interpretation

The official abstract reports relative pose error in indoor scenes that is 73% lower than an analytic filter baseline and 87.5% lower than a real-world behavioural cloning baseline. [Source](https://proceedings.mlr.press/v270/wasserman25a.html)

Those relative improvements must not be reinterpreted as a comparison against FAST-LIO2. Generalisation to new robots, friction conditions and gait distributions is a separate problem.

When reading closely, record not only the names of the compared methods but also the sensor configuration, ground truth, trajectory alignment method, the compute platform, and whether failed segments are included.

## 5. Vision60 application questions

The following are **project application hypotheses and review questions**, kept separate from the paper's validated results.

1. Do the joint ordering, units and rate of the training inputs match Vision60's sensors?
2. To what extent does the training environment cover real slip, impact and sensor latency?
3. If it is fused into LIO without an uncertainty estimate, is there a risk of overconfidence in its errors?

**Proposed experiment:** first measure the relative error of the learned model's standalone odometry over normal gait and slip segments. Leave LIO fusion as a later step and consider how the covariance of the new factor would be set.

## 6. Check questions

Answer each question out loud first, then open the toggle. Practise stating the assumptions and equations together with the failure conditions and how you would verify them.

### Q1. Concept and structure

Explain, in terms of input representation and dynamics, why learning-based leg-inertial odometry does not generalise directly to a new robot.

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

The number, ordering and axes of the joints, link lengths, actuator dynamics, gait, friction, and sensor bias and rate all differ, so the same input pattern has a different physical meaning. Normalisation and augmentation alone do not guarantee invariance. Morphology-aware representations, adaptation and uncertainty evaluation are needed.

</details>

### Q2. Math and reasoning

When the model predicts a relative transform \(\hat T*{t,t+1}\) from a time window \(x*{t-k:t}\), write one SE(3) loss and explain the translation–rotation scale problem.

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

For example \(\xi=\mathrm{Log}(T\_{gt}^{-1}\hat T)\) with \(L=\xi^\top W\xi\). In \(\xi=[\rho,\phi]\) the scales of metres and radians differ, so \(W\) or a learned uncertainty is needed. Discontinuities in the rotation representation and the correlated error that accumulates over long rollouts also have to be evaluated separately.

</details>

### Q3. Systems and debugging

Adding the learned odometry as a LIO factor made the overall accuracy worse. What are the possible causes and how would you combine them safely?

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

The bias may have grown out of distribution while a small covariance made the system overconfident. Temporally correlated outputs may also have been reused repeatedly as if they were independent factors. Use innovation gating, OOD and uncertainty estimation, covariance calibration, a low initial weight and a failure fallback, and isolate the cause with a sensor ablation.

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
