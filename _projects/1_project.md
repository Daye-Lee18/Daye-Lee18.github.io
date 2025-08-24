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
# pretty_table: true
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
3. We utilized the CLIP-ViP model[^1], a text-to-video / video-to-text retrieval model that has learned the relationship between the two modalities of text and video well.
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

#### Text Embedding 
The CLIP model is complicated and consists of a hierarchy of many classes. Largely, it consists of two transformers, each for learning the video and text data together. These transformers are further divided into smaller components like encoder and embedding classes. We start with the `CLIPTextEmbeddings class`, which we modify to incorporate the emotion data in the creation of the text embeddings.

We do this by initializing an embedding for each emotion, resulting in a total of 8 emotions. Here, the dimensions of the embeddings are the same as the token and positional embeddings. For each sequence, which has a set of corresponding emotions, we call the embeddings for each emotion and then **average them to create a single aggregated emotion embedding. This embedding is then added to each token embedding in the sequence alongside the positional embeddings.** This allows the model to incorporate the emotion information extracted from each sequence (caption) when learning their representations. Consequenstly, we updated the CLIPTextTransformer class to accept the emotion data. 

```python
class CLIPTextEmbeddings(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)
        self.emotion_embedding = nn.Embedding(8, embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        emotions: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]

        if emotions is not None:
            # Change non-zero values to 1, effectively binarizing the input
            emotions = torch.where(emotions > 0, torch.ones_like(emotions), torch.zeros_like(emotions))
        
        # Retrieve all emotion embeddings
        all_emotion_embeds = self.emotion_embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, 8, embed_dim]

        if emotions is not None:
            emotion_mask = emotions.unsqueeze(-1).type_as(all_emotion_embeds)  # [batch_size, 8, 1]
            selected_emotion_embeds = all_emotion_embeds * emotion_mask  # [batch_size, 8, embed_dim]
            emotion_embeds = selected_emotion_embeds.sum(1) / (emotion_mask.sum(1) + 1e-8)  # [batch_size, embed_dim]
        else:
            emotion_embeds = torch.zeros(batch_size, self.token_embedding.embedding_dim, device=input_ids.device if input_ids is not None else inputs_embeds.device)

        emotion_embeds = emotion_embeds.unsqueeze(1).expand(-1, seq_length, -1)  # [batch_size, seq_length, embed_dim]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings + emotion_embeds

        return embeddings
```

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/text2video/4.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/text2video/5.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fig 3. A Base model's training curves (left) and our model's training curves (right)
</div>

### Evaluation 
#### Evaluation Metric 
We use a Recall@k ($$ \frac{TP}{TP+FN}$$ of Top k samples) which means among the top k samples, the number of samples that are actuallly similar to the query TP is divided by the total number of query sapmles TP + FN. 

For the quantitative evaluation metric, we used the Recall@k metric, which is commonly used for the recommendation task. Recall@k is to compute the recall between the Top k samples based on the ranking. Therefore, among the top k samples, the number of samples that actually correspond to the query(or video), so called True Positive, is divided by the total number of query(or video) samples. For example of Recall@5, you can see in the figure [Fig 4](#Recall@5) of similarity matrix which is ranked by cosine similarity, that consists of 8 text and video pairs. If you look at the first row of the matrix, as we evaluate top 5 ranking, there is the corresponding video at the rank 3 which means true positive of that row becomes 1. Likewise compute R@5 for all the rows in the matrix, we can get the 6 TP divided by 8 of total samples, results in 75% of R@5 for the example matrix.

<div id="Recall@5" class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/text2video/6.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fig 4. Recall@5 
</div>

Likewise the [Recall@1](#Recall@1), you can compute the score with the top 1 ranking between all samples. There is only one True positive sample among 8 samples, so we can get 1 over 8 R@1 score that is 12.5%.

<div id="Recall@1" class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/text2video/7.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fig 4. Recall@1
</div>

In other words, Recall@k represents the probability that the correct item is included within the top-k results across all test queries. `Recall Median` refers to the median of the ranks at which the correct item is first retrieved for each query. It is essentially the median rank of the ground truth across queries, and since correct items should ideally appear earlier in the list, a lower value is better. Similarly, `Recall Mean` denotes the average of the ranks at which the correct items are retrieved. While the median is less sensitive to outliers, the mean is strongly affected by extreme values.

#### Quantitative Results 
To first evaluate the effects of training on different size data splits, we trained the baseline model on the conventional 7k and 9k training sets and validated on the 1k validation set. In addition, we also conducted training on the 6k-emotion dataset we previously created. To evaluate the overall performance of the models, **we first evaluate their performance on the entire test set, labeled `"Emotion+Neutral"` in the tables.** On the other hand, to evaluate the performance of these models on the retrieval of the caption-video pairs with emotions, which we previously identified as 34 videos out of the 1000 videos in the test dataset, we calculate the recall values for only these 34 queries, instead of the total 1000. These results are listed in the columns labeled `"Emotion"` in the tables.


<div class="table-wrap">
  <table class="perf-table">
    <thead>
      <tr>
        <th rowspan="2">Test Data</th>
        <th colspan="2">Emotion+Neutral</th>
        <th colspan="2">Emotion</th>
      </tr>
      <tr>
        <th>T2V</th>
        <th>V2T</th>
        <th>T2V</th>
        <th>V2T</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th scope="row">Recall@1</th>
        <td>49.4000%</td>
        <td>47.8044%</td>
        <td>32.3529%</td>
        <td>41.1765%</td>
      </tr>
      <tr>
        <th scope="row">Recall@5</th>
        <td>73.0000%</td>
        <td>74.9501%</td>
        <td>61.7647%</td>
        <td>67.6471%</td>
      </tr>
      <tr class="highlight">
        <th scope="row">Recall@10</th>
        <td>83.4000%</td>
        <td>84.4311%</td>
        <td>73.5294%</td>
        <td>82.3529%</td>
      </tr>
      <tr>
        <th scope="row">Recall Median</th>
        <td>2.0</td>
        <td>2.0</td>
        <td>3.5</td>
        <td>3.0</td>
      </tr>
      <tr>
        <th scope="row">Recall Mean</th>
        <td>14.5</td>
        <td>10.3</td>
        <td>18.1</td>
        <td>14.6</td>
      </tr>
    </tbody>
  </table>
</div>
<div class="caption">
    Tab 1. MSR-VTT 7k: Baseline Model
</div>

<div class="table-wrap">
  <table class="perf-table">
    <thead>
      <tr>
        <th rowspan="2">Test Data</th>
        <th colspan="2">Emotion+Neutral</th>
        <th colspan="2">Emotion</th>
      </tr>
      <tr>
        <th>T2V</th>
        <th>V2T</th>
        <th>T2V</th>
        <th>V2T</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th scope="row">Recall@1</th>
        <td>49.5000%</td>
        <td>49.3028%</td>
        <td>35.2941%</td>
        <td>35.2941%</td>
      </tr>
      <tr>
        <th scope="row">Recall@5</th>
        <td>74.7000%</td>
        <td>76.6932%</td>
        <td>61.7647%</td>
        <td>67.6471%</td>
      </tr>
      <tr class="highlight">
        <th scope="row">Recall@10</th>
        <td>84.8000%</td>
        <td>85.3586%</td>
        <td>73.5294%</td>
        <td>82.3529%</td>
      </tr>
      <tr>
        <th scope="row">Recall Median</th>
        <td>2.0</td>
        <td>2.0</td>
        <td>2.5</td>
        <td>3.0</td>
      </tr>
      <tr>
        <th scope="row">Recall Mean</th>
        <td>13.4</td>
        <td>9.5</td>
        <td>15.9</td>
        <td>13.1</td>
      </tr>
    </tbody>
  </table>
</div>
<div class="caption">
    Tab 2. MSR-VTT 9k: Baseline Model
</div>

<div class="table-wrap">
  <table class="perf-table">
    <thead>
      <tr>
        <th rowspan="2">Test Data</th>
        <th colspan="2">Emotion+Neutral</th>
        <th colspan="2">Emotion</th>
      </tr>
      <tr>
        <th>T2V</th>
        <th>V2T</th>
        <th>T2V</th>
        <th>V2T</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th scope="row">Recall@1</th>
        <td>49.0000%</td>
        <td>48.4032%</td>
        <td>29.4118%</td>
        <td>32.3529%</td>
      </tr>
      <tr>
        <th scope="row">Recall@5</th>
        <td>73.2000%</td>
        <td>75.6487%</td>
        <td>58.8235%</td>
        <td>70.5882%</td>
      </tr>
      <tr class="highlight">
        <th scope="row">Recall@10</th>
        <td>84.8000%</td>
        <td>84.7305%</td>
        <td>79.4118%%</td>
        <td>79.4118%</td>
      </tr>
      <tr>
        <th scope="row">Recall Median</th>
        <td>2.0</td>
        <td>2.0</td>
        <td>3.5</td>
        <td>2.0</td>
      </tr>
      <tr>
        <th scope="row">Recall Mean</th>
        <td>13.6</td>
        <td>9.9</td>
        <td>16.8</td>
        <td>14.7</td>
      </tr>
    </tbody>
  </table>
</div>
<div class="caption">
    Tab 3. MSR-VTT 6k (Emotion): Baseline Model
</div>


<div class="table-wrap">
  <table class="perf-table">
    <thead>
      <tr>
        <th rowspan="2">Test Data</th>
        <th colspan="2">Emotion+Neutral</th>
        <th colspan="2">Emotion</th>
      </tr>
      <tr>
        <th>T2V</th>
        <th>V2T</th>
        <th>T2V</th>
        <th>V2T</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th scope="row">Recall@1</th>
        <td>23.9000%</td>
        <td>44.8104%</td>
        <td>5.8824%</td>
        <td>8.8235%</td>
      </tr>
      <tr>
        <th scope="row">Recall@5</th>
        <td>41.5000%</td>
        <td>28.1437%</td>
        <td>14.7059%</td>
        <td>20.5882%</td>
      </tr>
      <tr class="highlight">
        <th scope="row">Recall@10</th>
        <td>49.00%</td>
        <td>50.9980%</td>
        <td>23.5294%%</td>
        <td>20.5882%</td>
      </tr>
      <tr>
        <th scope="row">Recall Median</th>
        <td>11.0</td>
        <td>9.0</td>
        <td>83.0</td>
        <td>66.5</td>
      </tr>
      <tr>
        <th scope="row">Recall Mean</th>
        <td>124.4</td>
        <td>72.6</td>
        <td>206.6</td>
        <td>115.5</td>
      </tr>
    </tbody>
  </table>
</div>
<div class="caption">
    Tab 4. MSR-VTT 6k (Emotion): Our Model (Emotion Embeddings)
</div>

As expected, in regard to the model's performance on emotion-containing queries in the "Emotion" columns, the performance degrades on the 6k(emotion) dataset in comparison to the other two datasets with both emotional and neutral data, as the model sees less data during training. From this, **we conclude that training exclusively on data containing only emotion does not translate to improved performance on emotion-containing queries**. In addition, when comparing the results for the entire test set with only the emotion-containing queries, **we find that for all training set sizes, the recall values for "Emotion" are all lower than for "Emotion+Neutral"**. These findings suggest that models trained on emotion-only data may fail to generalize, possibly due to overfitting or noisy emotion annotations. Thus, rather than focusing solely on emotion information, a more balanced approach that integrates both emotion and neutral cues may lead to more robust retrieval. Additionally, We find these results indicate that our current method of extracting emotion information may be insufficient. Therefore, instead of merely focusing on emotion-only signals, future work should refine or disentangle emotion representations in a way that avoids overfitting and captures complementary neutral cues.

The experiment's findings indicate a decline in overall performance when contrasting the 'Baseline' with 'Ours'. Specifically, for 'Text to Video' (T2V), the R@1 metric fell sharply from 29.41% to 5.89%, and for 'Video to Text' (V2T), it decreased from 32.35% to 8.83%. A similar downward trend was observed in the R@5 metric, which dropped from 58.82% to 14.71% for T2V, and from 70.59% to 15.63% for V2T. The R@10 metric also saw a significant reduction, declining from 79.41% to 23.53% for T2V and from 79.41% to 20.59% for V2T.

#### Qualitative Results 

To evaluate the proficiency of our emotion embedding model in learning about emotions, we visualized the embeddings for emotions using `t-SNE`. If our model has effectively learned emotion representations, we would expect embeddings associated with similar emotions to be mapped closer together in the space. Remarkably, the t-SNE visualization revealed that, compared to a baseline model, our model's text features classified as emotional are more distinctly clustered. This suggests that our emotion embedding approach successfully captures the nuances of emotional content of the text.

<div id="t-SNE" class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/text2video/8.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/text2video/9.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fig 5. t-SNE visualization of Joy vs. Non-Joy (Left) and of Trust vs. Non-Trust (Right)
</div>

For example, the embeddings of the joy text, in green dots, in our model are slightly more clustered than the baseline model’s (See in [Fig 5 (Left)](#t-SNE)). However, **the embeddings of the joy videos, in blue dots, are not as closely clustered as the joy text embeddings.**

Likewise, the embeddings of the trust text and trust video show the similar trend (See in [Fig 5 (Right)](#t-SNE)). Since our embedding model only incorporated emotion embeddings for text, we observed a clustering effect for text-related emotions but not for emotions related to videos. **Therefore, to enhance the performance of our model, we suggest the inclusion of a module that learns corresponding video embeddings in addition to text embeddings.** For instance, this can be achieved by refining our existing InfoNCE-based contrastive loss to more explicitly align emotion-specific text and video embeddings, pulling together positive pairs while separating unrelated ones. This enhancement is expected to yield improved performance in tasks involving emotion recognition across both text and video content.

### Demonstration 

### Discussion and Possible Future Directions 

#### What we wish we had known in advance 

1. The quality of emotional data that can be extracted. Emotion is quite a subjective concept, and is difficult to define and classify for. The methods we decided to use in the end were lexicon-based methods, which are not the most advanced technique there is. Thus, our method which utilizes this suboptimal data can have trouble leveraging this data for improving performance on the video retrieval task. The incorporation of this kind of data can actually confuse the model instead, in turn causing a drop in performance.
   
2. The difficulty in incorporating the emotion data. We were only able to use a very simple implementation in the form of “emotion embeddings” applied directly to the sequence data in the form of addition. Many sophisticated methods exist, and we leave this to future work. We had a list of methods we wanted to try, but were not able to implement due to time constraints, such as the attention mechanism or emotion-specific positional embeddings.

<div id="model_improvement" class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/text2video/10.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fig 6. Possible advanced model architecture 
</div>

1. The amount of available data and its relation to our task. (The availability of emotional videos) -> Our main objective was to create a service for users to more easily access emotional videos in the vast amount of data that they have access to. However, most of the datasets that are readily available for research are collected from a wide range of media like YouTube, or other consumer content, rather than for videos that would be commonly taken by regular people on their phones. In this sense, there is a discrepancy in the type of data we would expect our model to be used on, especially in that the datasets contain videos that do not really contain emotions. This negatively affected our models performance.
   
#### Future Direction 

One aspect that we were not able to touch on was the video data. We extracted emotional information from the video/frame captions, but we think it would be possible to extract similar information from the videos themselves. One method we considered was using Facial Expression Recognition(FER) models to extract the expressions of the faces in the videos and convert them into emotions. We think this has potential to improve the performance of our model.

<div id="FER" class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/text2video/11.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fig 7. Facial Expression Recognition (FER) model result example
</div>
### Code 

The final code is [here][git]

### Reference 
[^1]: Xue, Hongwei, et al. "Clip-vip: Adapting pre-trained image-text model to video-language representation alignment." arXiv preprint arXiv:2209.06430 (2022).

[report]: /assets/html/MIE1517_final_report.html
[git]: https://github.com/Hyejin3194/MIE1517_Project_Emotion-Text-to-Video-Retrieval.git 
[nltk]: https://www.nltk.org/api/nltk.sentiment.SentimentIntensityAnalyzer.html?highlight=sentimentintensity
[nrclex]: https://pypi.org/project/NRCLex/
