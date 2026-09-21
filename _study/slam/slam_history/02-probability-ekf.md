---
layout: study-chapter
title: "Chapter 2. Probability and EKF-SLAM"
description: "Four short problem sets: fusing two estimates, how errors spread, linearisation, the covariance matrix."
importance: 2
category: SLAM
series: slam_history
permalink: /study/slam/history/02-probability-ekf/
---

This chapter is four short problem sets. **Solve first, read after** —
each problem teaches one idea, and the explanation lives in its answer.
Open the hint only if you are stuck.

| Part                                                                         | You will be able to | Problems                                                               |
| :--------------------------------------------------------------------------- | :------------------ | :--------------------------------------------------------------------- | --- |
| [2.1 Fusing two estimates]({{ '/study/slam/history/02-1-fusing-estimates/'   | relative_url }})    | compute a Kalman gain and say why it has that shape                    | 8   |
| [2.2 Errors spread]({{ '/study/slam/history/02-2-errors-spread/'             | relative_url }})    | explain why landmarks become correlated                                | 6   |
| [2.3 Linearisation]({{ '/study/slam/history/02-3-linearisation/'             | relative_url }})    | turn a curved measurement into a straight line, and say when it breaks | 8   |
| [2.4 The covariance matrix]({{ '/study/slam/history/02-4-covariance-matrix/' | relative_url }})    | fill in the matrix and watch a correction travel through it            | 7   |

The only prerequisite is knowing that variance $\sigma^2$ measures spread.
Part 2.3 uses a derivative of $x^2$ — nothing harder.

**Original reading:** Solà, _Course on SLAM_ — motion/observation model sections
(`_resource/slam/foundations/sola2017-course-on-slam.pdf`).
