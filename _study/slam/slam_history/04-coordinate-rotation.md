---
layout: study-chapter
title: "Chapter 4. Frames and rotation"
description: "The minimum geometry needed to read 3D SLAM equations."
importance: 4
category: SLAM
series: slam_history
permalink: /study/slam/history/04-coordinate-rotation/
---

This chapter is two short problem sets. Matrix-vector products are assumed.

**Solve first, read after** — each problem teaches one idea and the explanation lives in its answer.
Open a hint only if you are stuck.

| Part                                                                    | You will be able to | Problems                                     |
| :---------------------------------------------------------------------- | :------------------ | :------------------------------------------- | --- |
| [4.1 Frames and transforms]({{ '/study/slam/history/04-1-frames/'       | relative_url }})    | convert between frames and catch sign errors | 6   |
| [4.2 Why rotations do not add]({{ '/study/slam/history/04-2-rotations/' | relative_url }})    | update a rotation correctly                  | 5   |

## Original reading

- Quaternion kinematics: whichever parts you need of §2 on rotation representations, §3 on conventions and §4 on perturbations. Local copy: `_resource/slam/foundations/sola2017-quaternion-eskf.pdf`.
- You do not need to derive every Jacobian on a first pass.

<aside class="study-summary" markdown="1">
## What you learned

<dl>
  <dt>Coordinate frame</dt><dd>The reference system in which a point, pose or measurement is expressed.</dd>
  <dt>Rigid transform</dt><dd>A rotation and translation that convert coordinates between two frames.</dd>
  <dt>Transform composition</dt><dd>Chaining frame changes in the correct order to obtain a new transform.</dd>
  <dt>SO(3)</dt><dd>The nonlinear space of valid 3D rotation matrices.</dd>
  <dt>Perturbation</dt><dd>A small local rotation or pose increment used to linearise and optimise on a manifold.</dd>
</dl>
</aside>
