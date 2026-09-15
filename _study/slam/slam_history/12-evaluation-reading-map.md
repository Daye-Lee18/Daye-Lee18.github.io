---
layout: study-chapter
title: "Chapter 12. Evaluation and where to study next"
description: "How to connect the history to performance tables and code reading."
importance: 12
category: SLAM
series: slam_history
permalink: /study/slam/history/12-evaluation-reading-map/
---

> **Goal:** Match the experimental conditions before comparing SLAM results.  
> **Workload:** 15 minutes. The last chapter of History.

## 1. The same number does not mean the same experiment

Two papers report "ATE 0.15 m" on the same dataset. That does not make them comparable, and here is the same estimated trajectory scored four ways to show why:

```text
  how it was evaluated                                   reported ATE
  ─────────────────────────────────────────────────      ────────────
  SE(3) alignment, all frames counted                        0.62 m
  Sim(3) alignment (rescaling allowed, see Ch. 8 §4)         0.15 m
  SE(3), but the 12 s where tracking was lost dropped        0.11 m
  SE(3), first 100 m only                                    0.04 m
```

One trajectory, four numbers, a 15× spread — and every one of them is honestly computed. The differences are entirely in the evaluation protocol.

The traps are worth naming individually:

```text
  Sim(3) alignment        rescales the estimate to fit, hiding exactly
                          the scale error monocular systems suffer from
  dropped failures        a method that gives up when confused scores
                          better than one that keeps trying and drifts
  timestamp association   loose matching lets the aligner pick the
                          most flattering ground-truth pose
```

So when comparing trajectory error, check the alignment type, the scale handling, the timestamp association and what happened to failed segments — before looking at the number.

The [KITTI odometry evaluation](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) sidesteps some of this by scoring sub-trajectories of several lengths (100 m, 200 m, … 800 m) and reporting error as a percentage of distance travelled. That answers "how fast does it drift?", which is a different question from "how far off is the whole trajectory?". Do not treat the two as the same metric.

Recording latency, memory and failure rate alongside accuracy makes it easier to judge suitability for a real robot. Recording latency, memory and failure rate alongside accuracy makes it easier to judge suitability for a real robot.

## 2. A timeline is a tool for orientation

| Period / representative case               | The question learned in these notes                      | Review  |
| ------------------------------------------ | -------------------------------------------------------- | ------- |
| Early probabilistic SLAM                   | How are pose error and map error linked?                 | Ch. 1–2 |
| FastSLAM, 2002                             | Can the problem be split by conditioning on the path?    | Ch. 3   |
| The graph/smoothing family                 | How are past states and large problems updated?          | Ch. 6–7 |
| KinectFusion, 2011 / ORB-SLAM, 2015        | How do sensors and map representation change the design? | Ch. 8–9 |
| LIO-SAM, 2020 / FAST-LIO2, 2021 preprint   | How is fast inertial prediction corrected?               | Ch. 10  |
| DROID-SLAM, 2021 and the neural-map family | What is learned and what is optimised?                   | Ch. 11  |

This is a study timeline linking the original sources of the previous chapters, not a list of first inventions or a complete historical table. Do not read techniques that developed in parallel as a single sequence of replacements.

## 3. How to use the recommended articles and materials

[Giseop Kim's recommended reading on SLAM back-ends](https://gisbi-kim.github.io/post/slam-textbooks/) connects materials for studying rotation, least squares and sparsity. To avoid reading a long textbook in one sitting, these notes split it up as follows.

1. If frames and rotation are confusing, go back to Chapter 4 and Solà's rotation material.
2. If residuals and Jacobians are confusing, work through the small ICP problem in Chapter 5.
3. If you want the structure of a large solver, read Chapters 6–7 and the factor graph text.
4. If you want the sensor pipeline, read Chapters 8–10 and then open the overview and code of a system you care about.

Use the [official SLAM Handbook page](https://asrl.utias.utoronto.ca/~tdb/slam/) as an index for finding broader topics. Pick the chapters you need, and check the edition and the update date of the public repository.

## 4. Check questions

### Question 1 — Concept

Of two SLAM papers, A has the smaller ATE while B has the smaller RPE and failure rate. How would you answer if asked which one to deploy on a long-duration autonomous robot?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Do not pick on a single number; start by matching the operating conditions. ATE mainly reflects global trajectory agreement and RPE local drift over fixed intervals, and both are sensitive to the alignment method. For a long-duration robot, catastrophic failure, relocalisation time, false loop closures, memory growth, latency tails and recovery behaviour matter. Re-evaluate under the same sensors, compute platform and data segments, with the same SE(3)/Sim(3) alignment; and for a product where failure is expensive, weight failure rate and recoverability more heavily than mean accuracy. The final choice is settled by the requirements and risk tolerance over the actual operating distribution.

</details>

### Question 2 — Math and evaluation

A 2D trajectory has estimates $(0,0),(1.1,0),(2.2,0)$ against ground truth $(0,0),(1,0),(2,0)$. With the start point aligned, find the translation ATE RMSE and the translation RPE RMSE over the two consecutive segments. Explain what each metric shows.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

The position error magnitudes at each pose are $0,0.1,0.2$, so

$$
\mathrm{ATE}_{RMSE}
=\sqrt{\frac{0^2+0.1^2+0.2^2}{3}}
\approx0.129m.
$$

For each 1 m ground-truth segment the estimated motion is 1.1 m, so the relative translation error is 0.1 m on both segments:

$$
\mathrm{RPE}_{RMSE}=\sqrt{\frac{0.1^2+0.1^2}{2}}=0.1m.
$$

RPE shows the 10 cm of drift per segment, while ATE also reflects the accumulation of that drift into 20 cm at the final pose. A real evaluation has to state the rotation handling, timestamp association, trajectory alignment and the definition of the segment length.

</details>

## Where to go next

Continue in [LIO & State Estimation]({{ '/study/slam/state-estimation/' | relative_url }}) in the order IMU model → deskew → error-state → LIO-SAM/FAST-LIO2. That page is the comparison table and index that lays out the scope of the study.

<aside class="study-summary" markdown="1">
## What you learned

<dl>
  <dt>ATE</dt><dd>Global trajectory disagreement after a stated alignment procedure.</dd>
  <dt>RPE</dt><dd>Relative motion error over a stated time or distance interval, useful for measuring local drift.</dd>
  <dt>SE(3) vs Sim(3)</dt><dd>Rigid alignment preserves scale, while similarity alignment can also remove a global scale error.</dd>
  <dt>Fair comparison</dt><dd>Matching sensors, data segments, compute, alignment and metric definitions before ranking systems.</dd>
  <dt>Operational robustness</dt><dd>Evaluating failures, recovery, latency and resource growth alongside average accuracy.</dd>
</dl>
</aside>
