---
layout: page
title: Emotion-Specialized Text-to-Video Retrieval
description: A tool for users to easily access emotional content in videos
img: assets/img/video_retrieval_thumnail.png
importance: 1
category: work
tags: formatting toc
toc:
  sidebar: left
---

### Problem 

We aim to create an text-to-video retrieval model that can better find videos taking emotions into account in their queries. In the age of big data, most people now have countless photos and vidoes in their possession, saved. With the immensely large amount of data that they now have, it is becoming increasingly difficult to efficiently find the videos or photos that someone is looking for. In this regard, the field of video-text retrieval has been receiving more attention and their models' capabilities have improved tremendously over the years. However, we believe that performance of these models can be further improved by focusing on the emotional aspects inherent in both the text and the videos. This is especially more important considering growing demand for personalized and emotionally resonant experiences in digital media. **Thus, we hope to develop a tool for users to easily access emotional content in videos, by modifying video-text retrieval models to incorporate emotion data.**


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/text2video/1.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fig 1. The overview of our task, emotion-specialized text-to-video retrieval task. A user inputs a description of the video they are looking for. For example, they could give as input: 'Could you find a video of me having a great time at my birthday party with my family?'. Our pipeline analyzes the text, understanding the emotional cues and content  context. It then sifts through a sea of video data, pinpointing those precious moments that align most accurately with the query. The output is a collection of video clips ranked by relevance, ready to transport the user back to those times. 
</div>

The Emotion-Specialized Text-to-Video Retrieval Task:

1. The user inputs a video description text as an input to extract the desired video. In the inference stage, to create a cosine similarity matrix, our emotion-specialized text-to-video retrieval model calculates the similarity between the relevant query and the videos in the user's directory. 
2. During inference, the task is to extract the top-n most relevant videos by listing the video vectors that have close cosine similarity to the embedding features based on the text query in the data, in order of the highest similarity.
3. We utilized the CLIP-ViP model, a text-to-video / video-to-text retrieval model that has learned the relationship between the two modalities of text and video well.
Here, our additional goal was to create a model that performs the retrieval task better for emotion-related queries.
4. The reason for this is that we anticipated that if a person needs to find a specific video, they would input a query describing emotions, as people live in their memories.
5. Therefore, we thought that after extracting 8 emotions from the text and performing embedding, we could develop the model so that the text and video cluster well for each emotion in the embedding space.


### Dataset and Data preprocessing  

We performed our experiments on the dataset most widely used for Video-Text Retrieval Tasks: MSR-VTT. Most implementations of models for this task usually use the entire dataset(10K YouTube videos with 200K descriptions) but here, **we filter the dataset (9K train/1K test) to use only the videos that contain emotions, leaving only 6,000 videos from the entire MSR-VTT dataset for training.** Each video is approximately 10-20 seconds long and consists of 20 corresponding captions. We explain the process in detail below.

To enhance the speed of the training, we downscale the frames per seconds to 6. Next, **to extract the emotions in our dataset** for our main task, Emotion-Specialized Text-to-Video Retrieval, we perform a two-step preprocessing process as follows.

Step (1) Video selection and filtration process : We conduct **sentiment analysis on each video caption** to determine the presence of sentimental information. The sentiment analysis library calculates neutral, positive, and negative scores for each video caption and decides whether it is positive or negative based on the compound score, which is an overall score derived from these individual scores. For example, if the composite score is above 0.6, we classify the video as positive, and if it is below -0.6, we classify it as negative. **We abitrarily set -0.6 and 0.6 as the thresholds for classifying the compound scores representing videos with emotion.** We filtered out any videos with compound scores in between -0.6 and 0.6 as videos without sufficient positive or negative sentiments, representative of the existence of emotions. **This leaves us with about 6K videos in the training set.** We utilized [nltk.sentiment.SentimentIntensityAnalyzer][nltk] for this process. 

다만, 위의 Sentimenty Analyzer 모델을 사용했을 때, 특정 감정을 나타내는 단어 예를 들어, "happy", "sad" 및 "fear"와 같은 단어가 들어있어도 compound score가 유의미하지 않은 것이 filter되었기 때문에, 이러한 caption은 maually하게 추출하는 과정을 거쳤다. 

```python
def extract_emotion_manually(df, text=None):
    # assign the default text
    if text is None:
        text = "happy|sad|afraid|fear|surprise|joy|disgust|annoy| anger|angry|" \
                "excite|excited|exciting|scare|scared|scary|fright|frighten|frightened|frightening" \
                "|fearful|fearless|fearfully"
    # extract the data that contains the text
    manual_df = df[df['caption'].str.contains(text)]
    print("Number of manually selecting data:", len(manual_df))
    return manual_df
```

Step (2): After completing the video selection and filtration process in the first step, we determine the emotional information present in each caption. For this purpose, **we use NRCLex, which uses a scoring method based on a predefined lexicon dictionary to calculate eight emotions present in a sentence: joy, trust, fear, surprise, sadness, disgust, anger, and anticipation.** We also include the calculation of positive, negative and neutral emotions from the previous step. **Consequently, the resulting data for each caption contains information on eight emotions along with positive, negative, and neutral sentiments.** We made a use of [NRC Lexicon library][nrclex] for this task. This emotion extraction process is applied to the three data splits we use in our experminets: **the refined 6K training data from the first step, the entire 1K test data, and a combination of 34 sentimental and 34 non-sentimental data used to analyze our model's results.** 즉, step (1)을 거친 training 데이터셋, 기존 1K test set은 validation set, 그리고 기존 1K test set에 step (1)을 거친 34 + non-sentimental 34 (총 68개)는 최종 test data로 사용되었다. 

More specifically, using the lexicon provided by the NRC Lexicon Library, we assigned emotion scores to each caption for the eight emotion types defined by Robert Plutchik: joy, trust, fear, surprise, sadness, disgust, anger, and anticipation. This results in a total of 11 columns in the original dataset. Remember, we apply this process to three datasets in the second step of assigning emotional information: the refined 6K training data from the first step, a combination of 34 sentimental and 34 non-sentimental data for a total of 68 test data, and the entire 1K test data.

### Proposed Model 

#### CLIP-ViP: Baseline Representation Learning 
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/text2video/2.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fig 2. A Baseline model was used for our emotion-specialized text-to-retrieval model
</div>


For our baseline, we use **CLIP-ViP**, a state-of-the-art model for video-text retrieval. It adapts a CLIP image-text backbone for **post-pretraining** on videos by adding a **Video Proxy token** (proxy-guided video attention) and training with **Omnisource Cross-Modal Learning**.

##### Representation Learning Objective
- Learn a **shared embedding space** for video and text.
- **Positive pairs** (e.g., matching Video–Subtitle or Frame–Caption) are pulled **closer**;  
  **Negative pairs** (non-matching pairs) are pushed **farther** using contrastive learning.
- To mitigate the domain gap between *subtitles (S)* in the training data and real-world *captions (C)*, an auxiliary caption is generated for the **middle frame (F)** of each **video (V)**.
- We therefore train on **(V, S)** and the corresponding **(F, C)** pairs jointly.

#### Loss Function (InfoNCE)

We denote video/frame embeddings by \($$v_{i}$$\) and text (subtitle/caption) embeddings by \($$t_{i}$$\), with temperature \($$\tau$$\).

$$
\mathcal{L}_{v2t}
= -\frac{1}{B} \sum_{i=1}^{B}
\log
\frac{\exp\left(v_i^{\top} t_i / \tau\right)}
{\sum_{j=1}^{B} \exp\left(v_i^{\top} t_j / \tau\right)}
$$


$$
\mathcal{L}_{t2v}
= -\frac{1}{B} \sum_{i=1}^{B}
\log
\frac{\exp\left(t_i^{\top} v_i / \tau\right)}
{\sum_{j=1}^{B} \exp\left(t_i^{\top} v_j / \tau\right)}
$$

We use **source-wise** InfoNCE over video sources \(\{V, F\}\) and text sources \(\{S, C\}\). Practical variants:

- (a) \( $$\mathcal{L}_{V\leftrightarrow S} + \mathcal{L}_{F\leftrightarrow C}$$\)
- (b) \( $$\mathcal{L}_{V\leftrightarrow S} + \mathcal{L}_{V\leftrightarrow C}$$ \)
- (c) \( $$\mathcal{L}_{V\leftrightarrow S} + \mathcal{L}_{V\leftrightarrow C} + \mathcal{L}_{F\leftrightarrow C}$$ \)
- (d) \( $$\mathcal{L}_{V\leftrightarrow S,C} + \mathcal{L}_{F\leftrightarrow C}$$ \)  (video paired with both subtitle and auxiliary caption)

A common joint form for (d) expands the negative pools across both \(S\) and \(C\):
$$
\mathcal{L}_{v2t} = -\frac{1}{2B}\sum_{i=1}^{B}\left[\log\frac{\exp\left(v_i^{\top} s_i/\tau\right)}{\sum_{j=1}^{B} \exp\left(v_i^{\top} s_j/\tau\right)+\sum_{j\neq i} \exp\left(v_i^{\top} c_j/\tau\right)} + \log\frac{\exp\left(v_i^{\top} c_i/\tau\right)}{\sum_{j=1}^{B} \exp\left(v_i^{\top} c_j/\tau\right)+\sum_{j\neq i} \exp\left(v_i^{\top} s_j/\tau\right)}\right]
$$


where \($$s_{i}$$ in S\) and \($$c_{i}$$ in C\). The corresponding \( $$ \mathcal{L}_{t2v}$$\) term is defined analogously.

---

CLIP-ViP’s training objective is to **learn generalizable multimodal representations** via contrastive learning, bridging the gap between video content and textual descriptions. These learned representations are then directly used for **retrieval tasks** (text-to-video, video-to-text).

#### Our Model 
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/text2video/3.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fig 2. A framework of our project including modified emotion-specialized text-to-retrieval model.
</div>

We modify this model to additionally take the emotion data we extracted earlier as input. For each of the eight emotions, we initialize an embedding of the same dimensions as the token and positional embeddings. Then for each input sequence and its corresponding emotion scores, we aggregate the embeddings for each emotion by adding them together to create the final emotion embedding. This is then added to each token embedding alongside the positional encodings. This will be shown in the code later on.

### Evaluation 

### Demonstration 

### Discussion and Possible Future Directions 

### Final Report 

Final report is [here][report] and the code is [here][git]

### Reference 
[^1]: Xue, Hongwei, et al. "Clip-vip: Adapting pre-trained image-text model to video-language representation alignment." arXiv preprint arXiv:2209.06430 (2022).

[report]: /assets/html/MIE1517_final_report.html
[git]: https://github.com/Hyejin3194/MIE1517_Project_Emotion-Text-to-Video-Retrieval.git 
[nltk]: https://www.nltk.org/api/nltk.sentiment.SentimentIntensityAnalyzer.html?highlight=sentimentintensity
[nrclex]: https://pypi.org/project/NRCLex/
