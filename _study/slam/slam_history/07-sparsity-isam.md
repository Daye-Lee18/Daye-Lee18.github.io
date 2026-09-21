---
layout: study-chapter
title: "Chapter 7. Sparsity and iSAM"
description: "Many variables does not mean every variable is connected to every other."
importance: 7
category: SLAM
series: slam_history
permalink: /study/slam/history/07-sparsity-isam/
---

This chapter is two short problem sets. Read after Chapter 6.

**Solve first, read after** — each problem teaches one idea and the explanation lives in its answer.
Open a hint only if you are stuck.

| Part                                                                             | You will be able to | Problems                              |
| :------------------------------------------------------------------------------- | :------------------ | :------------------------------------ | --- |
| [7.1 Why the matrix is mostly zeros]({{ '/study/slam/history/07-1-mostly-zeros/' | relative_url }})    | explain why large graphs are solvable | 4   |
| [7.2 Fill-in and ordering]({{ '/study/slam/history/07-2-fill-in/'                | relative_url }})    | count fill-in and pick a good order   | 4   |

## Original reading

- Dellaert & Kaess (2017): read the elimination/sparsity discussion first, then move on to the iSAM part. Local copy: `_resource/slam/foundations/dellaert-kaess2017-factor-graphs.pdf`.
- Before following every equation, draw the factor graph and the post-elimination graph yourself.

<aside class="study-summary" markdown="1">
## What you learned

<dl>
  <dt>Sparsity</dt><dd>The property that each factor touches only a small subset of all variables.</dd>
  <dt>Variable elimination</dt><dd>Removing variables in sequence to factorise and solve the sparse system.</dd>
  <dt>Fill-in</dt><dd>New connections created between a variable's neighbours during elimination.</dd>
  <dt>Bayes tree</dt><dd>A tree representation of the factorisation that exposes which parts an update affects.</dd>
  <dt>iSAM2</dt><dd>An incremental smoother that reuses previous factorisation and selectively relinearises affected variables.</dd>
</dl>
</aside>
