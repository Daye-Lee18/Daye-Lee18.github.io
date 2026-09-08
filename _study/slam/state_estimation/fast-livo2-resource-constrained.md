---
layout: study-chapter
title: "Lightweight FAST-LIVO2 — paper review"
description: "Reduces the memory and computation cost of FAST-LIVO2 by controlling observation usefulness and how much of the map is retained."
category: SLAM
series: state_estimation
importance: 12
permalink: /study/slam/state-estimation/fast-livo2-resource-constrained/
---

[← State estimation paper comparison]({{ '/study/slam/state-estimation/' | relative_url }})

> **One-sentence summary:** Reduces the memory and computation cost of FAST-LIVO2 by controlling observation usefulness and how much of the map is retained.

| Item           | Detail                                                                                                             |
| :------------- | :----------------------------------------------------------------------------------------------------------------- |
| Paper          | FAST-LIVO2 on Resource-Constrained Platforms: LiDAR-Inertial-Visual Odometry with Efficient Memory and Computation |
| Venue          | 2025 · arXiv:2501.13876                                                                                            |
| Source         | [Paper / author material](https://arxiv.org/abs/2501.13876)                                                        |
| Status         | Introductory review draft · personal close-reading and reproduction notes added below                              |
| Source checked | 2026-09-07                                                                                                         |

## 1. The problem it addresses

However accurate a LIVO is, it cannot be operated continuously if it exceeds the compute and memory budget of the onboard platform.

## 2. Three key points to present

1. **Adaptive image selection:** the image frames to use are selected with degeneracy in mind.
2. **Map memory management:** a local unified LiDAR-visual map is used together with a long-term visual map.
3. **Efficiency–accuracy trade-off:** resource cost is reduced within the sequential ESIKF update structure.

Basis for this technical summary: [paper / author description](https://arxiv.org/abs/2501.13876).

## 3. How it works

The diagram below is a conceptual flow for understanding; it does not show every thread and update rate in the implementation.

```text
sensor input → degeneracy-based image selection → sequential state update → local unified map and long-term visual map management
```

The core idea is judging when visual constraints are needed, rather than unconditionally cutting frames. Maintaining a long-term visual map does not by itself mean global loop optimisation is performed.

## 4. Experimental results and interpretation

The abstract reports a 33% reduction in per-frame runtime and a 47% reduction in memory against FAST-LIVO2 on Hilti, with a 3 cm increase in RMSE, and includes evaluation on x86 and ARM platforms. [Source](https://arxiv.org/abs/2501.13876)

Mean reduction rates on a particular dataset do not guarantee Vision60's worst-case latency or long-term memory. The trade-off of slightly lower accuracy also has to be taken into account.

When reading closely, record not only the names of the compared methods but also the sensor configuration, ground truth, trajectory alignment method, the compute platform, and whether failed segments are included.

## 5. Vision60 application questions

The following are **project application hypotheses and review questions**, kept separate from the paper's validated results.

1. Are there situations where the image selection criterion misses geometric degeneracy?
2. Which maps, buffers and libraries are included in the memory measurement?
3. Does the computational saving over FAST-LIVO2 translate into meeting the control cycle's deadline?

**Proposed experiment:** on the same long-distance log, record peak memory, the processing latency distribution and per-segment relative error for both methods. On thermally and power-limited hardware, measure again after sustained operation.

## 6. Check questions

Answer each question out loud first, then open the toggle. Practise stating the assumptions and equations together with the failure conditions and how you would verify them.

### Q1. Concept and structure

Why is degeneracy-aware visual frame selection better than simple fixed frame skipping?

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

Always discarding at the same rate can drop images that are needed precisely when the LiDAR constraint is weak. Selecting images according to the information deficit in the state and environment reduces computation while keeping the complementary observations that are actually needed. The computational cost of the selection criterion itself, and its misjudgements, have to be included in the evaluation.

</details>

### Q2. Math and reasoning

If a candidate image's information contribution is taken as \(\Delta H=J_V^\top R_V^{-1}J_V\), propose a selection criterion and explain its limitations.

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

For example, \(\log\det(H+\Delta H)-\log\det(H)\), the increase in the smallest eigenvalue, or a trace-based value can be weighed against the cost. Log-det looks at the overall uncertainty volume, the smallest eigenvalue at the weakest direction. These are sensitive to the linearisation point, scale and model error, and they cost computation, so an approximate indicator may be needed.

</details>

### Q3. Systems and debugging

Mean memory is down, but an OOM occurs during long operation. What should you record?

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

Record the peak and the time trend of RSS and GPU memory, the number of voxels, patches and keyframes, allocator fragmentation, queues and caches, and map pruning events. Check whether a steady state exists, and separate repeated paths from exploration of new areas. A reduction in mean memory is not enough to claim bounded memory.

</details>

### Q4. Code review and implementation

**GitHub:** [Official or author-linked repository](https://github.com/hku-mars/FAST-LIVO2)

This repository also links the lightweight paper, but how would you determine whether the paper's degeneration-aware frame selector and long-term visual map are actually implemented on the current branch?

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

Do not conclude it is included on the strength of the README alone. Search the repository code for the distinctive variables and metrics of the paper's algorithm, and trace the path where the frame selection condition is called and how the selection result affects buffer and map lifetimes. Check tags, branches and commit history against the paper's publication date, and do not mistake the base FAST-LIVO2 `img_en` or plain frame skipping for the paper's selector. If you cannot find it, mark the repository as an upstream/reference implementation only and record the paper-specific code as unreleased or unverified.

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

**Read next:** [FAST-LIVO2 review]({{ '/study/slam/state-estimation/fast-livo2/' | relative_url }}) · [Full comparison table]({{ '/study/slam/state-estimation/' | relative_url }})
