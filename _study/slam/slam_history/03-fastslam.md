---
layout: study-chapter
title: "Chapter 3. FastSLAM and conditional independence"
description: "Splitting one large joint estimation problem into a path problem and landmark problems."
importance: 3
category: SLAM
series: slam_history
permalink: /study/slam/history/03-fastslam/
---

This chapter is two short problem sets. Read after Chapter 2.

**Solve first, read after** — each problem teaches one idea and the explanation lives in its answer.
Open a hint only if you are stuck.

| Part                                                                             | You will be able to | Problems                            |
| :------------------------------------------------------------------------------- | :------------------ | :---------------------------------- | --- |
| [3.1 Why condition on the path]({{ '/study/slam/history/03-1-condition-on-path/' | relative_url }})    | explain FastSLAM's core trick       | 5   |
| [3.2 Particles, weights and resampling]({{ '/study/slam/history/03-2-particles/' | relative_url }})    | score, resample and judge particles | 6   |

## Original reading

- Montemerlo et al. (2002), FastSLAM: read around Figure 1 and the posterior factorisation. [Original PDF](https://www.cs.cmu.edu/~thrun/papers/montemerlo.fastslam-tr.pdf).
- From the next chapter we pause to set up the geometry needed by both filtering and optimisation.

<aside class="study-summary" markdown="1">
## What you learned

<dl>
  <dt>FastSLAM</dt><dd>A Rao–Blackwellised particle filter that samples robot paths and estimates landmarks inside each path hypothesis.</dd>
  <dt>Conditional independence</dt><dd>Landmark estimates separate only after the robot path is given.</dd>
  <dt>Particle weight</dt><dd>The relative likelihood of one path hypothesis given the observations.</dd>
  <dt>Resampling</dt><dd>Replicating likely particles and discarding unlikely ones to focus computation.</dd>
  <dt>Particle impoverishment</dt><dd>Loss of hypothesis diversity caused by repeated or premature resampling.</dd>
</dl>
</aside>
