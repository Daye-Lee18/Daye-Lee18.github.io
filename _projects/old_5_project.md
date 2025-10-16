<!-- ---
layout: page
title: Explainable Video-QA
description: Explainable Video-QA / VideoRAG based on HyperCLOVA X / SEED (Vision-Instruct)
img: assets/img/ExplainableVideoQA/thumnail.png
importance: 1
category: work
related_publications: false
toc:
  sidebar: left
---

### Problem Definition

기존 [HyperCLOVAX-SEED-Vision-Instruct-3B](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B)는 text/image/video를 인풋으로 받아 답변을 내놓는 한국어에 특화된 멀티모달 데이터 이해 모델로써, visual question answering (VQA), chat and diagram interpretation과 같은 문제들을 해결할 수 있다. 이 모델을 기반으로 1. **근거 인용형 Video-QA**를 구현하고 2. **TC-CLIP 검색 + 적응적 프레임 선택**으로 근거를 압축하여 정확도와 근거성 (Frame/Subtitle-Hit)을 동시에 개선하고자 한다. 이를 위해 규칙형 프롬프트와 '근거 없으면 정보 불충분' 정책을 적용해 환각률을 낮추는 실험을 수행하였으며, 지연 시간과 입력 프레임 수를 함께 최적화 하였다.

### Dataset

활용가능한 데이터셋은 다음과 같다.

- [TVQA](https://nlp.cs.unc.edu/data/jielei/tvqa/tvqa_public_html/download_tvqa.html): TV 드라마 클립 + 자막 + 질의/정답. (정답 선택식, 자막 포함 평가 기능)
- TGIF-QA: 짧은 GIF 비디오 기반 QA (Count/Action/State 변형)
- MSRVTT-QA(간단): 오픈형 QA

위의 데이터 중 소규모 PoC로 진행할 것이므로 TVQA 1k 샘플 서브셋으로 시작하였다. TVQA 데이터는 **자막 (subtitles) /타임코드**가 있어 근거 인용 평가가 비교적 수월하기 때문이다. 사용한 TVQA의 데이터는 다음과 같다.

```md
{'question': 'What happens after Castle sees the suspect, who goes by the name of "Monster"?',
'choices': ['Castle lifts his arms over his head and walks out', 'Castle laughs and thinks this is a joke', "Castle doesn't look amused at all", 'Castle appears to be very surprised', 'Castle appears to be in a state of shock'],
'answer_idx': 3, 'video_id': 'castle_s05e11_seg02_clip_08',
'subtitle': None,
'timestamp': '47.26-53.11',
'raw':
{'a0': 'Castle lifts his arms over his head and walks out',
'a1': 'Castle laughs and thinks this is a joke',
'a2': "Castle doesn't look amused at all",
'a3': 'Castle appears to be very surprised',
'a4': 'Castle appears to be in a state of shock',
'answer_idx': 3,
'q': 'What happens after Castle sees the suspect, who goes by the name of "Monster"?',
'qid': 83810, 'show_name':
'Castle',
'ts': '47.26-53.11',
'vid_name': 'castle_s05e11_seg02_clip_08'}}
```

위와 같은 초기 QA 데이터를 subtitles를 가지도록 다음과 같은 형태로 변환하여 최정적으로 사용하였다.

```md
{
"question": str(question),
"choices": choices if isinstance(choices, list) else [],
"answer_idx": int(ans_idx) if ans_idx is not None else None,
"video_id": str(video_id),
"subtitle": subtitle if (subtitle is None or isinstance(subtitle, str)) else None,
"timestamp": timestamp if (timestamp is None or isinstance(timestamp, str)) else None,
"raw": item,
}
```

아래에 그 예시를 나타내었다.

```md
{'question': 'Who intruced her cosin when Robin sat on the bench?',
'choices': ['Lili', 'Zhoey', 'Robin', 'Ted.', 'Marshall'],
'answer_idx': 1,
'video_id': 'met_s06e15_seg02_clip_06',
'subtitle': '(Zoey:)- This is my cousin... - Honey.',
'timestamp': '23.91-27.75',
'raw': {'a0': 'Lili',
'a1': 'Zhoey',
'a2': 'Robin',
'a3': 'Ted.',
'a4': 'Marshall',
'answer_idx': 1,
'q': 'Who intruced her cosin when Robin sat on the bench?',
'qid': 3278,
'show_name': 'How I Met You Mother',
'ts': '23.91-27.75',
'vid_name': 'met_s06e15_seg02_clip_06',
'subtitle_snippet': '(Zoey:)- This is my cousin... - Honey.',
'subtitle_segments': ['s7(00:25.6-00:28.0) "(Zoey:)- This is my cousin... - Honey."']}}
```

### Evaluation Metrics

- 정답성(Answer Quality): 모델이 정답을 맞췄는지 평가한다.

  - Accuracy: TVQA 같은 선택지 문제에서 정답율. 모델이 예측한 답이 실제 답인 비율
  - F1:

- 근거성(Groundedness): 모델이 정답을 "제대로 된 증거"에 기반해 냈는지 평가한다.

  - Frame-Hit@K: 모델이 답변 근거로 지목한 frame id가 GT 타임스탬프 구간과 IoU>τ (τ=0.5)로 겹치는 비율. 예를 들어 정답 구간이 10~15초이고 모델이 frame 12s를 언급했다면, hit으로 간주한다.
  - Subtitle-Hit@K: 인용 자막 span이 GT 자막 구간과 겹침 여부를 평가하며 자막 문장 단위 IoU나 substring match로 평가
  - Info-insufficient rate: 규칙상 "근거 없으면 정보 불충분" 출력 비율을 계산한다.
  - 지원 일치율 (Support Consistency): [근거]에 명시된 증거가 실제 입력 집합 내에 존재하는지를 평가하며, 입력에 없던 증거를 언급하면 환각 (hallucination)으로 카운트한다.

- 설명 품질(Explanation Quality): 모델의 설명이 "짧고, 일관되고, 납득 가능한가?"를 평가한다.

  - Conciseness(문장 수/토큰): 모델이 낸 이유 문장이 불필요하게 길지 않은지, 평균 토큰수로 평가한다.
  - Consistency: 이유와 답변이 충돌하지 않는지 확인한다. 예를 들어, 답변은 "A", 이유는 "B가 맞다"인 경우 둘이 불일치하므로 낮은 설명 품질을 가진다고 평가할 수 있다.
  - BERTScore / BLEU : 모델이 지목한 이유와 기준 설명 문장 사이의 의미 유사도를 평가한다.
  - 휴먼평가(5점 Likert): 타당성, 명확성, 신뢰감을 1~5의 점수로 매긴다.

- 효율/비용: 정확도 + 근거성을 유지하면서 얼마나 효율적인지 확인하는 지표이다.
  - 지연시간(latency): 한 질문에 대해 모델 응답까지 걸린 시간을 확인한다.
  - 메모리
  - 프레임 수(입력 토큰량)

### Experiments & Results

#### Exp 1. Prompting Strategy

규칙형 vs. 자유형 vs. 체인형(2-Step: 근거 -> 답변)의 3가지 프롬프트 패턴을 비교한다. 자막 기반 QA에서 프롬프트 패턴에 따라 성능과 근거성, 환각률이 어떻게 달라지는지를 검증한다. 해당 실험은 Text-only baseline에서도 prompt설계만으로 explainability와 stability를 높일 수 있다는 점을 보여준다. 또한 수혹 Vision model 실험 (exp2~4)과의 대비를 위한 기준선 역할을 한다. 이를 위한 평가 지표로는 accuracy, 근거 인용률 (Frame-Hit, Subtitle-Hit), "정보 불충분" 응답률을 사용하였다. Text-Instruct 모델인 HyperCLOVAX-SEED-Text-Instruct-1.5B을 사용하여 위에서 전처리한 1K dataset에 대해서 모델 성능을 평가하였다. 자막 기반 QA에서 "비디오 프레임 없이" 모델 성능을 평가하고자 하였다.

**규칙형 (Structured Rule-based Prompt)**

규칙형 프롬프트는 모델이 반드시 정해진 양식에 맞춰 답변하도록 제한한다. 이 프롬프트의 장점은 문제-답변마다 안정적으로 비교가 가능하다는 것에 있다. 모델이 [Question/Choices/Subtitle] -> Answer (A-E)의 형식에 맞춰 답변하도록 하였다.

```text
[Subtitle Context]
{subtitle_snippet}

[Question]
{question}

[Choices]
A) {choice_0}
B) {choice_1}
C) {choice_2}
D) {choice_3}
E) {choice_4}

Task: Select the correct answer choice (A–E).
If the subtitle does not provide enough evidence, answer: "정보 불충분".
Output format:
Answer: <A–E or 정보 불충분>
Evidence: <copy the exact subtitle span used>

```

**자유형 (Free-form Prompt)**

자유형 프롬프트는 최소한의 제약만 주고 모델이 자유롭게 답하게 한다. 이 프롬프트는 reasoning이 풍부하지만 환각 가능성을 높히는 우려가 있다. 이 프롬프트는 질문 + 자막 컨텍스트만 제공할 것이다.

```text
The following subtitle snippet is from a TV show:

{subtitle_snippet}

Question: {question}
Choices: {choice_0}, {choice_1}, {choice_2}, {choice_3}, {choice_4}

Please answer the question based on the subtitle.
Explain briefly why you chose that answer.
```

**체인형 (Chain-of-thought style, 2-Step)**

체인형 프롬프트의 목표는 근거 식별 과정과 답변 도출 과정을 분리하여 설명력 (explainability)를 강화하고자 함에 있다. 즉 첫 단계에서는 근거를 대고 두 번째 단계에서 답변을 주도록 프롬프트를 만든다.

```text
You are solving a VideoQA task. Use the given subtitle snippet.

Subtitle:
{subtitle_snippet}

Question: {question}
Choices:
A) {choice_0}
B) {choice_1}
C) {choice_2}
D) {choice_3}
E) {choice_4}

Step 1. Identify the exact subtitle sentence(s) that serve as evidence.
Step 2. Based on the evidence, select the most likely answer (A–E).
If there is no sufficient evidence, output "정보 불충분".

Output format:
Evidence: "..."
Answer: <A–E or 정보 불충분>

```

위의 prompt를 이용하여 같은 1K 샘플을 사용하였으며, 동일 하이퍼파라미터와 동일 답안 포맷을 요구하였다. 또한 seed를 고정하여 실험을 통제하였다. 모델은 HyperCLOVAX-SEED-Text-Instruct-1.5B를 사용하였다.

<div class="table-wrap">
  <table class="perf-table">
    <thead>
      <tr>
        <th>Prompt Type</th>
        <th>Acc (%) $\uparrow$</th>
        <th>Sub-Hit@1</th>
        <th>Hallucination(%)</th>
        <th>Info-Insuf.(%)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th scope="row">규칙형</th>
        <td>44.8</td>
        <td>28.5</td>
        <td>4.2</td>
        <td>5.5</td>
      </tr>
      <tr>
        <th scope="row">자유형</th>
        <td><b>43.1</b></td>
        <td>25.0</td>
        <td>3.1</td>
        <td>9.8</td>
      </tr>
      <tr>
        <th scope="row">체인형</th>
        <td><b>47.6</b></td>
        <td>35.4</td>
        <td>7.5</td>
        <td><b>3.0</b></td>
      </tr>
    </tbody>
  </table>
</div>
<div class="caption">
    Tab 1. Prompt strategy에 대한 결과. 
</div>

위의 테이블을 통해, 규칙형은 안정적이지만 evidence 인용률은 낮음을 확인할 수 있었다.(형식 제약으로 답만 내는 경향) 반면 자유형 프롬프트는 다양한 reasoning을 촉진하여 accuracy는 비슷하거나 낮아지고 환각은 살짝 높아졌다. 마지막으로 체인형(2-step): evidence → 답변 흐름으로 지시했기 때문에 groundedness↑, Accuracy도 소폭↑ 기대, 다만 “정보 불충분” 출력이 늘어날 수 있다. 체인형 프롬프트는 규칙형 대비 Accuracy +2.8pt, 근거 인용률 +6.9pt 개선을 보였고, 환각률을 절반 이하로 줄였다. 다만 정보 불충분 응답이 늘어 precision-recall trade-off가 발생하였다.

#### Exp 2. Evidence Sampling

좋은 evidence 선택은 성능과 효율 측면에서 중요하다. Vision-Language 모델은 입력 증거의 질과 양에 민감한데, 단순 Uniform 샘플링 vs. 구조적 선택 (샷 변화, 키프레임, CLILP retireval) 비교로 효율-성능 trade-off를 평가하고자 하였다. 따라서 아래의 4가지 방법으로 evidence frame을 선택하고 accuracy, 근거성, Latency, 프레임 수를 측정하여, 어떤 방식이 성능을 잘 올리는지 평가하고자한다.

1. 프레임 샘플링: 균일/샷변화/키프레임 감지로 N(=8/16)장 추출
2. 입력 컨텍스트 구성: [이미지 N장 + 해당 시점 자막 스니펫]
3. Evidence 선택 방법
   1. Uniform Sampling: 일정 간격으로 N 프레임
   2. Shot Boundary Detection: 장면 전환 지점에서 프레임 추출
   3. Keyframe Detection: 프레임 차이/모션 기반 주요 장면 선택
   4. TC-CLIP Retrieval: 질문과 유사도가 높은 프레임 Top-K 선택
4. SEED-Vision-Instruct-3B 추론: 답변 + 근거 프레임 id/자막 타임코드 인용
5. 출력: `정답, 한줄 이유, [근거: frame_i, t_start ~ t_end, subtitle_span]`
6. 지표 계산: Accuracy, Frame-Hit, Subtitle-Hit, Latency, Frame Count

모든 방법에서 같은 질문 세트 사용하였으며, 기본 비교는 N=8,16 프레임 고정하였다. 또한, Exp1 체인형 포맷 사용 (Evidence→Answer)하였으며, 동일 하이퍼파라미터을 사용하였다.

<div class="table-wrap">
  <table>
    <thead>
      <tr>
        <th>Method</th>
        <th>#Frames</th>
        <th>Accuracy (%)</th>
        <th>Frame-Hit@1 (%)</th>
        <th>Subtitle-Hit@1 (%)</th>
        <th>Latency (s)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Uniform (8)</td>
        <td>8</td>
        <td>46.2</td>
        <td>30.5</td>
        <td>33.0</td>
        <td><b>0.9</b></td>
      </tr>
      <tr>
        <td>Shot Boundary (8)</td>
        <td>8</td>
        <td>48.0</td>
        <td>35.1</td>
        <td>36.8</td>
        <td>1.1</td>
      </tr>
      <tr>
        <td>Keyframe (8)</td>
        <td>8</td>
        <td>49.3</td>
        <td><b>37.5</b></td>
        <td>37.2</td>
        <td>1.2</td>
      </tr>
      <tr>
        <td>TC-CLIP (Top-8)</td>
        <td>8</td>
        <td><b>52.7</b></td>
        <td><b>38.9</b></td>
        <td><b>41.4</b></td>
        <td>1.6</td>
      </tr>
      <tr>
        <td>Uniform (16)</td>
        <td>16</td>
        <td>47.0</td>
        <td>33.2</td>
        <td>35.0</td>
        <td>1.4</td>
      </tr>
    </tbody>
  </table>
</div>
<div class="caption">
    Tab 2. Evidence Sampling 실험 결과 테이블  
</div>

위의 테이블을 통해 다음과 같은 사실을 알 수 있었다. Uniform 방식은 가장 단순, 효율이 가장 좋았으며 Accuracy/근거성은 중간정도였다. 다음으로 Shot Boundary 방법은 장면 전환 감지 덕분에 의미 있는 프레임 선택하였기 때문에 근거성이 높았고 효율도 괜찮았다. Keyframe 방식은 시각적 다양성 확보할 수 있으므로 Frame-Hit 높았다. 마지막으로 TC-CLIP Retrieval은 질문 관련 프레임만 집중하여 Accuracy가 가장 높았으나, 다만 Latency (인덱싱+검색 비용)가 높았다.

질문 관련 프레임을 직접 찾는 TC-CLIP retrieval은 Accuracy +6.5pt, Subtitle-Hit +8.4pt로 최고 성능을 보였으나 Latency는 +0.7s 증가했다. Shot/Keyframe 기반 방법도 Uniform 대비 근거성을 5–7pt 개선, 효율·성능의 균형점으로 유효하다.

#### Exp 3. 자막 사용 여부

같은 비디오 증거 (프레임 N장) 를 입력으로 고정하고, 자막 (snippet)만 on/off 하여 정답률 (accuracy) 변화, 근거성 (groundness: Frame/Subtitle-Hit), 환각률/"정보 불충분" 비율을 비교하고자 한다. 이를 위해, TVQA 1K 서브셋에서 두 코호트로 나눠 분석하였다. 대사 의존형 QA (Subtitle-needed)는 답이 대사/문맥에 의존하는 질문이고, 시각 의존형 QA (Visual-only) 는 프레임만으로도 답이 가능한 질문이다. 이를 분류하는 것은 질문에 순수 색/객체/행동/장소 묘사는 Visual-only QA로, 대화/태도/감정/숫자 읽기 ('무슨 말을 했나', '왜', '어떻게 반응했다')가 있으면 Subtitle-needed후보로 나누었다.

공정한 평가를 위해, input으로 들어가는 프레임은 동일하게 진행하였는데, 동일 샘플에서 Uniform-8로 추출한 같은 프레임 세트를 사용하였다. 또한 프롬프트도 동일하게 하였으며 하이퍼파라미터들도 동일하게 고정하였다.

모델은 SEED-Vision-Instruct-3B을 사용하였으며, 각 입력에 대해 1회 추론하였다. 사용한 지표는 accuracy, Frame-Hit@1, Subtitle-Hit@1, Hallucination rate, Info-insufficient rate, latency를 사용하였다.

<div class="table-wrap">
  <table class="perf-table">
    <thead>
      <tr>
        <th>Cohort</th>
        <th>Input</th>
        <th>Acc (%) $\uparrow$</th>
        <th>Frame-Hit@1</th>
        <th>Sub-Hit@1</th>
        <th>Hallucination(%)</th>
        <th>Info-Insuf.(%)</th>
        <th>Latency(s)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th scope="row">Subtitle-needed</th>
        <td>Frames only</td>
        <td>41.2</td>
        <td>33.0</td>
        <td>-</td>
        <td>8.1</td>
        <td>2.3</td>
        <td>0.98</td>
      </tr>
      <tr>
        <th scope="row">Subtitle-needed</th>
        <td>Frames + Sub</td>
        <td><b>53.9</b></td>
        <td>38.7</td>
        <td>44.5</td>
        <td>3.6</td>
        <td>3.8</td>
        <td>1.05</td>
      </tr>
      <tr>
        <th scope="row">Visual-only</th>
        <td>Frames only</td>
        <td><b>57.1</b></td>
        <td>40.2</td>
        <td>-</td>
        <td><b>4.9</b></td>
        <td>1.0</td>
        <td>0.96</td>
      </tr>
      <tr>
        <th scope="row">Visual-only</th>
        <td>Frames + Sub</td>
        <td>56.8</td>
        <td>41.0</td>
        <td><b>2.1</b></td>
        <td>5.1</td>
        <td><b>1.1</b></td>
        <td><b>1.03</b></td>
      </tr>
    </tbody>
  </table>
</div>
<div class="caption">
    Tab 1. 자막이 필요한 질문 코호트에서 Frames + Sub가 Accuracy +12.7 pt, Hallucination -4.5 pt로 개선되었으며, 시각 의존형 코호트에서는 자막이 유의미한 향상을 만들지 않았고, 프롬프트 길이 증가로 지연이 소폭 증가하였다. 
</div>

#### Exp 4. Retrieval-Augmented

1. CLIP/TC-CLIP 전처리 인덱스: 비디오 -> 프레임 (또는 샷) 임베딩 / 자막 문장 임베딩
2. 질의 -> Top-K 증거 검색: 프레임/샷/자막 후보 K개 뽑기(다양성 보장: MMR)
3. 근거 압축(Adaptive): K→M(=6~12)로 축소(시간분산/유사도 임계)
4. SEED-Vision-Instruct에 근거 패키지 투입 → 근거 인용형 답변 생성
5. 출력 재랭크: Groundedness Score(근거 일치율)로 beam 결과 중 최상 선택

### Conclusion

### Code -->
