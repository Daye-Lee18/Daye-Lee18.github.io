---
layout: study-chapter
title: "LOAM — paper review"
description: "The starting point of the lineage: separating high-rate odometry from low-rate precise mapping made real-time LiDAR estimation possible."
category: SLAM
series: state_estimation
importance: 9
permalink: /study/slam/state-estimation/loam/
---

[← State estimation paper comparison]({{ '/study/slam/state-estimation/' | relative_url }})

> **One-sentence summary:** The starting point of the lineage: separating high-rate odometry from low-rate precise mapping made real-time LiDAR estimation possible.

| Item           | Detail                                                                                                  |
| :------------- | :------------------------------------------------------------------------------------------------------ |
| Paper          | LOAM: Lidar Odometry and Mapping in Real-time                                                           |
| Venue          | RSS 2014                                                                                                |
| Source         | [Paper / author material](https://publications.ri.cmu.edu/loam-lidar-odometry-and-mapping-in-real-time) |
| Status         | Introductory review draft · personal close-reading and reproduction notes added below                   |
| Source checked | 2026-09-07                                                                                              |

## 1. The problem it addresses

If the sensor moves while a scan is being acquired, every point is measured from a different pose. Both motion estimation and precise mapping have to be handled fast.

## 2. Three key points to present

1. **Two-stage processing:** fast motion estimation and more precise map registration run at different rates.
2. **Geometric feature registration:** edge and planar feature points are used to estimate the motion.
3. **Separated compute budget:** while odometry keeps up with fast changes, mapping performs the more precise registration.

Basis for this technical summary: [paper / author description](https://publications.ri.cmu.edu/loam-lidar-odometry-and-mapping-in-real-time).

## 3. How it works

The diagram below is a conceptual flow for understanding; it does not show every thread and update rate in the implementation.

```text
LiDAR scan → feature extraction → fast odometry and motion correction → precise mapping
```

Read this connected to why the problem of points being collected at different times later made IMU and deskew so important in LIO. The original paper also describes IMU aiding, so do not assume LiDAR-only is the only configuration.

## 4. Experimental results and interpretation

The authors evaluate on several experiments and on KITTI, reporting real-time operation and low drift. Read the comparison methods and sensors of the time as distinct from today's onboard setting. [Source](https://publications.ri.cmu.edu/loam-lidar-odometry-and-mapping-in-real-time)

LiDAR-centred geometric registration depends on sufficient structure and a reasonable motion estimate. Do not equate the two-stage mapping of this paper with global loop closure.

When reading closely, record not only the names of the compared methods but also the sensor configuration, ground truth, trajectory alignment method, the compute platform, and whether failed segments are included.

## 5. Vision60 application questions

The following are **project application hypotheses and review questions**, kept separate from the paper's validated results.

1. What transform does each of odometry and mapping estimate?
2. Where does the velocity needed for intra-scan motion correction come from?
3. Which assumptions weaken when the motion changes rapidly, as at a Vision60 landing?

**Proposed experiment:** first visualise the temporal distribution of points within a scan and the amount of rotation while walking. Before implementing anything, compare how FAST-LIO2 uses the IMU on the same problem.

## 6. Check questions

Answer each question out loud first, then open the toggle. Practise stating the assumptions and equations together with the failure conditions and how you would verify them.

### Q1. Concept and structure

Explain why LOAM separates the rates of odometry and mapping, in terms of accuracy and computation.

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

Odometry keeps up with velocity and motion quickly using little computation, while mapping registers slowly but precisely using more points and more iterations. The two results complement each other, trading off real-time operation against low drift. This mapping must not be equated with global loop closure.

</details>

### Q2. Math and reasoning

Explain the difference between the point-to-plane residual \(r=n^\top(Rp+t-q)\) and the point-to-line distance, and which features each is used for.

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

The plane residual is a scalar: the difference between the transformed point and a plane reference point \(q\), projected onto the normal \(n\). The line residual is the perpendicular component between the point and the line direction \(d\), which can be written as \(\|(I-dd^\top)(Rp+t-q)\|\). The former is used for planar features, the latter for edge features.

</details>

### Q3. Systems and debugging

If the angular rate is not constant within a scan while walking, explain the map errors produced by constant-velocity deskew.

<details class="study-answer" markdown="1">
<summary>Show answer and key points</summary>

The real pose interpolation differs from the assumed one, so points are systematically mis-transformed as a function of time. Planes bend or appear doubled, and the resulting wrong correspondences bias the next pose too. Use per-point timestamps and a high-rate IMU-based trajectory, and visualise the landing segment before and after deskew.

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

**Read next:** [LIO-SAM review]({{ '/study/slam/state-estimation/lio-sam/' | relative_url }}) · [Full comparison table]({{ '/study/slam/state-estimation/' | relative_url }})
