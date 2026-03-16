---
layout: page
title: Personalization via Few-shot Learning
description: Personalized a music-conditioned 3D motion diffusion model via few-shot learning by selectively fine-tuning cross-attention, audio encoder, and FiLM layers of the EDGE framework to adapt to new dance styles.
img: assets/img/personalization_thumnail.png
importance: 2
category: work
tags:
  - Diffusion
  - Few-Shot Learning
  - 3D Motion Generation
  - Motion Style Transfer
toc:
  sidebar: left
---

<!-- 1108 x 608 -->

### Task

Our team aimed to personalize **a pre-trained diffusion-based 3D motion generation model conditioned on music**. Specifically, the task was to adapt the model to **learn a specific concept or style from only a few samples**. This approach is essential in scenarios where there is no large-scale training dataset available for a new concept, but a general pre-trained model already exists. We focused on developing **a cost-effective fine-tuning method** in terms of both time and data requirements.

Previous research on this problem includes [Textual Inversion][textual_inversion], [DreamBooth][DreamBooth], and [Custom Diffusion][customDiffusion]. Instead of fine-tuning the entire model with few-shot samples, these three methods selectively fine-tune specific parts of the model. Each method targets different network layers for fine-tuning (Fig. 1).

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/personalization/1.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fig 1. Three representative approaches for fine-tuning diffusion models.
</div>

### Problem Statement

The existing dance generation method, [EDGE][edge], can only generate generic dance sequences. Since EDGE is trained on [AIST++][aist++], which mainly consists of general dance motions, it primarily focuses on beat alignment and physical plausibility, without capturing distinctive styles. Our goal was to extend the model’s capability to generate style-specific dances. For example, we aimed to generate dances in the unique K-pop style of NewJeans.

### Data Collection

The [AIST++][aist++] dataset, used to pre-train the EDGE model, contains 1,408 sequences of 3D human dance motion across 10 different dance genres. It provides motion sequences in SMPL format along with audio features extracted using [JukeBox][jukebox].

To construct a personalized dataset for fine-tuning, we first crawled dance videos with corresponding music from YouTube (outputs: .mp4, .wav). We then extracted [SMPL (Skinned Multi-Person Linear Model)][smpl] parameters for each frame using [ROMP][romp]. Motion and audio were segmented into 5-second sub-sequences (60 FPS → 300 frames per sequence). Since EDGE accepts 5-second (300-frame) inputs and generates longer motions via interpolation, this preprocessing ensured compatibility. Finally, we split the dataset into training and testing sets (80/20 split) before starting fine-tuning.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid path="assets/video/60_chillax_fixed.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay=false muted=true%}
    </div>
</div>
<div class="caption">
    Vid 1. One of the personalized dataset result extracted by ROMP. 
</div>

### Model Architecture

#### Baseline model

The baseline model, EDGE (Editable Dance Generation from Music), is a diffusion-based framework trained to generate sequences of SMPL pose parameters. It employs cross-attention mechanisms to condition motion generation on Jukebox audio features.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/personalization/2.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fig 2. Overview of the EDGE model.
</div>

#### Selective Module Fine-Tuning

Inspired by diverse customization approaches, we explored several fine-tuning strategies:

- (A) No Fine-Tuning
- (B) Full Fine-Tuning (similar to DreamBooth)
- (C) Fine-Tuning Audio Transformer Encoder & Cross-Attention Layers (similar to Custom Diffusion)
- (D) Fine-Tuning Audio Transformer Encoder, Cross-Attention Layers, and FiLM Layers (Ours)

Our method extends (C) by adding FiLM layers, enabling more expressive adaptation to personalized dance styles.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/personalization/3.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fig 3. Selective fine-tuning strategies. 
</div>

### Evaluation

We adopted evaluation metrics from the EDGE paper to assess:

1. Motion Quality - measured using $FID_{g}$ and $FID_{k}$
   - $FID_{g}$ (Geometric Features): captures spatial relations defined between body joints, producing boolean vectors.
   - $FID_{k}$ (Kinematic Features): captures dynamic properties such as velocity and acceleration.
2. Physical Plausibility – measured using PFC (Physical Foot Contact) score, which enforces real-world constraints: acceleration can occur only when one foot remains in contact with the ground.
3. Music–Motion Correlation – measured using Beat Alignment, which evaluates synchronization between beats in the music and motion trajectories.

### Results

#### Quantitative

<div class="table-wrap">
  <table class="perf-table">
    <thead>
      <tr>
        <th>Method</th>
        <th>PFC $\downarrow$</th>
        <th>Beat Align $\uparrow$</th>
        <th>$FID_{k} \downarrow$</th>
        <th>$FID_{g} \downarrow$ </th>
        <th>Human evaluation $\downarrow$ (Average Rank)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th scope="row">A (No fine-tuning)</th>
        <td>1.64</td>
        <td>0.207</td>
        <td>117</td>
        <td>2.877</td>
        <td>4</td>
      </tr>
      <tr>
        <th scope="row">B (Full fine-tuning)</th>
        <td>0.88</td>
        <td><b>0.281</b></td>
        <td>84</td>
        <td>0.781</td>
        <td>1.83</td>
      </tr>
      <tr class="highlight">
        <th scope="row">C (Audio Transformer Encoder + Cross Attn Layer)</th>
        <td><b>0.66</b></td>
        <td>0.250</td>
        <td>38</td>
        <td><b>0.445</b></td>
        <td>2.83</td>
      </tr>
      <tr>
        <th scope="row">D (Audio Transformer Encoder + Cross Attn Layer + FiLM)</th>
        <td>0.77</td>
        <td>0.255</td>
        <td><b>36</b></td>
        <td>0.458</td>
        <td><b>1.33</b></td>
      </tr>
    </tbody>
  </table>
</div>
<div class="caption">
    Tab 1. Quantitative results
</div>

The quantitative resluts is shown at Tab 1. The values are averaged by three models fine-tuned each on "New Jeans - Super Shy", "ITZY-Cake", "Chillax-bboy music" dataset. The test dataset is unseen dataset collected from YouTube, collected in the same method as the fine-tuning dataset. Human evaluation were conducted among 6 different students. The question used for human evalution was "which video looks most like Next Jeans / IVE / Chillex? Rank them in order".

#### Qualitative

### Conclusion

Our contributions are as follows.

1. Propose a new task of dance customization for the first time
2. Show reasonable results considering the challenging task of 3D dance generatoin
3. Explore diverse personalization strategies motivated from prior works

Our methods has some limitations in that it requires well-curated data but it can be solved by considering camera paramters. Also our model is less generalizable with musics that are far from training domain which can be solved by adding some regularizatoin tricks.

### Reference

[textual_inversion]: https://arxiv.org/abs/2208.01618
[DreamBooth]: https://openaccess.thecvf.com/content/CVPR2023/html/Ruiz_DreamBooth_Fine_Tuning_Text-to-Image_Diffusion_Models_for_Subject-Driven_Generation_CVPR_2023_paper.html
[customDiffusion]: https://arxiv.org/abs/2305.15779
[edge]: https://openaccess.thecvf.com/content/CVPR2023/html/Tseng_EDGE_Editable_Dance_Generation_From_Music_CVPR_2023_paper.html
[aist++]: https://google.github.io/aistplusplus_dataset/factsfigures.html
[jukebox]: https://assets.pubpub.org/2gnzbcnd/11608661311181.pdf
[romp]: https://arxiv.org/abs/2008.12272
[smpl]: https://dl.acm.org/doi/abs/10.1145/3596711.3596800
