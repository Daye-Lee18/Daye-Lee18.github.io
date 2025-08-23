---
layout: page
title: Dialog Inpainting for Legal Dialogue Systems
description: Conversational Question Answering Systems for SNU laws, 2023 Fall
img: assets/img/Legal_thumnail.png
tags: formatting toc
importance: 3
category: work
toc:
  sidebar: left
---
<!-- thumnail size: 1205 x 690 -->

<!-- * Table of contents
{:toc} -->

### Problem Definition 

대화형 질의응답 시스템을 학습 시키기 위한 training dataset을 수집하기 위해서는 매우 많은 비용이 발생합니다.적용하고자 하는 domain에  대해 전문 지식을 갖춘 사람이 해당 domain의 지식을 반영한 질문과 답변의 쌍을 만들어야 하며, 그 양이 무수히 많아야 하기 때문에 많은 시간과 노동력이 필요하게 됩니다.

Dialog inpainting 방법은 이러한 문제들을 해결하고자 제안된 방법입니다. 아래의 그림과 같이 문서의 텍스트를 작가와 상상속의 독자간의 이야기로 변환하여 dialog를 만들게 됩니다. 문서 내에 문장은 작가가 한 말로 사용되어지고, 그럼 다음 inpainter를 사용하여 작가의 발언 간에 상상속 독자가 무엇을 물었거나 말했을 지를 예측하게 됩니다.

이번 프로젝트는 이러한 방법론을 서울대 학칙을 대상으로 하여 적용하고자 하였습니다.


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Dialogue/1.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Dialogue/2_1.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fig 1. 왼쪽 그림은 이전 Inpainter 논문에서 위키피디아 문서를 기준으로 QA 데이터셋을 만든 방법을 도식화한 결과를 보여주고, 오른쪽 그림에서 위키피디아의 문서가 대화의 가상 Writer의 답변으로 사용됨을 확인할 수 있다.  
</div>

하지만, 기존 방법 그대로를 법률 문서인 서울대 학칙에 적용하기에 2가지 문제가 있었습니다.

기존 Inpainter 논문[^1][^2][^3][^4]에서 대상으로 한 문서는 위키피디아입니다. 하지만 저희가 적용하고자 하는 문서인 서울대 학칙 (See Fig 2.)은 위키피디아와 달리 계층 구조를 가지고 있으며, 내용들 간 참조 관계가 있다는 것입니다. Inpainter 방식을 그대로 적용할 경우 앞서 말한 서울대 학칙 문서의 특징들을 dialog set에 반영할 수 없다는 문제가 발생합니다.

또한, 문서 내 문장을 그대로 한 답변은 실제 사람의 대화스럽지 않으며, 딱딱한 문서 형식의 문장이기 때문에 이러한 방식으로 만들어진 데이터를 통해  챗봇과 같은 모델을 학습시키게 된다면 기존에 챗봇이 가지고 있었던 커뮤니케이션 능력을 상실할 가능성이 높아지며, 질문에 적합하지 않은 형식의 답변을 생성하게 됩니다.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Dialogue/3.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fig 2. 서울대 학칙으로 데이터셋을 만들고 싶은 경우, 법률 문서가 가진 3가지 특징, 계층 구조 (hierarchical structure), 서로 참조 관계 (reference structure) 및 의미적으로 비슷한 문장 구조 (sementical similarity)에 의해 다른 Inpainting 방법이 필요하다.
</div>

### Framework

이제 저희 연구의 전체적인 framework에 대해서 간략히 설명드리겠습니다 (Fig 3).

1. 먼저 data preprocessing을 통해 법률 문서가 가지는 특징을 고려하여, 데이터를 총 ＂4가지 방법＂(Baseline, Hierarchy, Reference, Similarity)으로 전처리 했습니다. 이렇게 해서 저희는 총 4개의 데이터셋을 구성하였습니다.

2. 두번째로, dialogue inpainter를 이용해 context 단위로 나열된 이 개별 데이터셋을, Question and Context Pair로 이루어진 legal dialogue로 재구성했습니다. 

3. 다음으로, 저희의 downstream task인 text generation를 고려하여 context를 자연스러운 답변의 형식으로 바꾸는 context restyling 과정을 거쳤습니다.

4. 마지막으로 Llama-2-7B chat model을 파인튜닝 즉, generator instruction tuning을 진행하였고, 이에 따라 저희 데이터셋에 대한 평가를 진행했습니다. 

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Dialogue/4.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fig 3. 본 연구의 framework overview.
</div>

#### Data preprocessing 

데이터 전처리에서 저희가 중요하게 생각한 것은, 법률이라는 domain의 특성이었습니다. 앞서 언급한 바와 같이, 총 4가지 방법을 통해서 데이터 전처리 과정을 진행했는데, 첫째로, Fig 2. 오른쪽에 초록색 부분에서 보이는 바와 같이, 법률 문서는 하나의 "Article"아래에 "여러 개의 paragraph"로 이루어져 있습니다. 이렇게 article에서 paragraph로 이어지는 가장 단순한 법률 구조에 집중해서 “baseline inpainting” 방식을 가장 먼저 생각해보았구요.

또한, 법률 문서의 경우, 내용의 명확한 ＂상하 위계 질서＂ 즉 hierachy 가 있어, 넓은 관점에서 각 항목에 대한 세부 규정으로 이어지는 narrowing down 구조를 갖고 있습니다. 이러한 구조적 특성을 반영해서 저희는＂hierarchical inpainting 방식”도 개발했습니다. 


이외에도 법률 문서에는 특정 항이 다른 항을 참조하는 관계가 매우 빈번한데요, 저희는 이러한 법률 항목 간의 명시적 참조 관계를 반영해서 ＂reference“ inpainting 방식도 생각했습니다.


마지막으로 다른 조항에 위치한 paragraph 라 하더라도 의미적으로 유사항 paragraph 역시 다수 존재합니다. 실제 legal context에서는 이처럼 암묵적으로 관련 있는 항목을 묶어, 종합적으로 논하는 경우가 많고 이러한 항들 간의 유기적 관련성이 있다는 점을 반영해서,"similarity“에 기반한 inpainter 방식 또한 구현하였습니다.

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
  <img src="/assets/img/Dialogue/6.png" alt="img1" width="100%">
  <img src="/assets/img/Dialogue/7.png" alt="img2" width="100%">
  <img src="/assets/img/Dialogue/8.png" alt="img3" width="100%">
  <img src="/assets/img/Dialogue/9.png" alt="img4" width="100%">
</div>
<div class="caption">
    Fig 4. 서울대학교 학칙 문서 방법 4가지 (Baseline, Hierarchy, Reference, Similarity) 방법에 대한 설명 
</div>

Informative한 legal dialogue를 만들기 위해서는, 우선 가장 기본적으로, article에서 paragraph로 이어지는 법률 구조를 따라 Legal document 전체의 내용을 담을 수 있어야 합니다. 이것이 바로 저희가 주목한 baseline 방법론입니다. Fig 4의 왼쪽 상단의 그림과 같이, 각 article마다 각 paragraph 를 하나의 답변으로 보아 dialogue data를 생성하도록 하기 위해, 기본 raw 데이터셋을 전처리하는 작업을 진행했습니다.

다음으로, 법률 구조에 내재되어 있는 위계 구조를 반영하여, 문서 전체를 broad 한 관점에서도 바라보고, narrow 한 관점에서도 바라보게끔 함으로써 Legal context 를 여러 차원에서 유기적으로 이해하는 데이터를 만들고자 하였습니다. 이것이 저희가 앞서 언급한 Hierarchy 방법론의 핵심입니다. 따라서 Topic class → Chapter and Article class → Paragraph class 로 이어지는 context 를 GPT에게 순차적으로 줌으로써 자연스럽게 문서 “전체”에 대한 주제에서 “세부” 항목에 대한주제로 이어지도록 raw data를 전처리하는 과정을 거쳤습니다.

아래 표 (Tab 1)는 저희가 앞선 4가지 방식에 의한 context 구성을 위해 가장 우선적으로 구축한 Raw data의 예시이며, 위계 구조를 반영하고자 title, topic, chapter, article, paragraph 로 이어지는 indexing column 을 추출했고 관련 조항의 내용, contents 와 더불어, 참조하고 있는 항의 id 를 ref_id 로 넣어 참조 관계까지 같이 담아냈습니다.

다음으로, 법률 문서 내의 각 조항 및 항이 reference 라는 구조를 통해 명시적으로 서로 연결되어 있는 경우,문서에 명확히 그 관계가 나와 있는 만큼, 이를 반영하여 dialogue 를 구성하는 것이 legal context에 대한 체계적인 이해에 있어 중요하게 작용합니다. 이 부분이 앞서 언급한 reference inpainting 방식에 해당하는데요, 따라서 저희는 각 항마다 참조하는 모든 관계를 반영하여 이를 순차적인 context로 주어 질문을 생성하게 할 수 있도록, raw 데이터셋을 아래와 같이 가공하는 작업을 거쳤습니다.

마지막으로 similarity 에 기반한 방법론입니다. 의미론적으로 유사한 조항이나 항들을 묶어 문서에 내재한 암묵적인 법률 관계를 반영하기 위함인데요, Semantically 가장 유사항 2개의 조항을 묶어 inpainter 의 context 로 주는 방식으로 본 task에 접근하였습니다. 

저희는 이러한 일련의 과정을 적용함으로써, 서울대학교의 학칙에 대한 문서의 각 paragraph들을 총 "4가지의 방식"으로 다채로운 context를 만들어주었고, 이를 통해 legal document 전반의 idea와 동시에 세부 사항, 나아가 조항간 암묵적/명시적 관계까지 반영한 baseline, hierarchy, similarity, reference 데이터셋을 구축하고자 했습니다.


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Dialogue/5.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Tab 1. 데이터 전처리 전/후 필요한 columns들로 분류 및 전처리 전/후의 최종 데이터 포맷 예시
</div>

#### Legal Dialog Inpainter 

다음은, 저희 프레임워크의 2번째 스텝인 Legal Dialogue Inpainter입니다. [기존 dialogue Inpainter 논문][ref1]와 동일한 방식을 사용하였는데, 저희는ChatGPT 3.5 engine 을 활용하여, 질문 생성에 대한 대화의 맥락을 유지하도록 함으로써 질문 간 Context-sharing이 이뤄지고, 나아가 실제 대화처럼 자연스럽고 암묵적인 질문을 던질 수 있게끔 유도했습니다 그 과정의 예시를 보시면,

저희가 앞서 만든 전처리된 데이터속 각 항들을 대화의 "답"으로 보고 그 답에 해당하는 질문을 생성하도록 하였습니다. 가져온 예시는 서울대학교의 안전 환경 규정 관련 예시입니다. 첫번째 항을 보시면 안전 관리자가 각종 폐기물에 대한 가이드라인을 만들어야 한다는 "법률 context"에 대해 저희의 inpainter는 전 관리자의 폐기물에 대한 "역할"이 무엇이냐? 라는 "질문"을 생성하였음을 알 수 있었습니다. 

결론적으로 전처리한 4개의 데이터셋을 각각 legal inpainter에 넣어 질문과 context 로 이루어져 있는 4개의 QC (Question-Context) 데이터셋을 만들었습니다. 

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Dialogue/10.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fig 5. ChatGPT 3.5 engine을 Dialog Inpainter로 활용하여 legal dialogue를 생성하는 과정을 보여줌
</div>

#### Context Styler 

Framework 세번째로 앞서 만든 QC데이터셋에서 대화로 보기에는 자연스럽지 않은 question – context (QC) 형식 문제를 해결하고자 단순 context를 자연스러운 답변으로 만들도록 context restyler를 도입하였습니다. 즉, QC 형식에서 QA (Question-Answer)의 형식으로 실제 사람이 대화하는 것과 유사하게 데이터를 구축하였습니다. 

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Dialogue/11.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fig 6. Context restyling 을 거쳐 생성한 QA 데이터 예시를 보여준다. 딱딱한 법률적 문서에서 자연스러운 답변 형식으로 만들어져 Dialogue 라는 대화 형식에 잘 부합함을 확인할 수 있다. 
</div>

#### Downstream task 

저희 framework의 마지막으로, Downstream task로는 대화 생성 테스크를 선택했고, 저희는 llama-2-7B chat model를 파인튜닝하였습니다. 저희의 좀 더 공평한 평가를 파인튜닝 데이터의 양을 맞춰야한다고 생각했습니다. 따라서 가장 적은 대화 개수를 가진 reference data의 수에 맞추어 나머지 데이터들은 sampling하여 데이터의 수를 맞추어 준 후 모델을 파인튜닝하였습니다. 

### Evaluation 

평가의 목표는 그럴듯한 dialogue data augmentation을 넘어서, Legal dialog inpainter의 데이터셋이 챗봇과 하는 법률에 관한 대화를 실제로 더 좋게 만들 수 있는지 확인하는 겁니다. 연구에서 만든 4개의 데이터셋들을 라마-2 모델에 파인튜닝해 그 모델이 생성한 대화를 평가했습니다.

평가 방식으로 질문은 같고 답변은 다른 두 대화에 대해 본 연구팀 멤버들이 Human evaluation을 진행하였고, GPT-3.5를 사용해 evaluation을 보조적으로 진행했습니다.

Metrics은 다음과 같이 4가지에 대해 평가하였습니다. 
1. Accuracy는 어느 대화가 hallucination이 없이 정확한지입니다. Human evaluator의 경우, 전체 대화에서 하나라도 hallucination이 있으면 부정확하다고 평가했으며, 양 대화에서 모두 Hallucination이 발견되었다면 그 양에 상관 없이 정확도가 같다고 평가해, GPT보다 더 엄격하게 평가했습니다.
2. Informativeness는 답변에서 제공한 정보가 풍부하고 깊이 있는지 평가합니다. 
3. Well-formedness는 답변의 구조적인 품질이 좋은지 평가합니다. 이 메트릭을 통해 모델에 의해 생성된 답변이 문법적으로 맞는지 논리적으로 정돈되어있는지 확인할 수 있습니다.
4. Overall quality는 전체 대화의 품질을 평가합니다. 


GPT3.5에는 프롬프트로 태스크와 메트릭을 설명하고 다이얼로그 A, B를 주며 어떤 쪽이 우수한지 물었습니다. 정확도 평가를 위해서는 파인튜닝에 쓰인 Ground Truth를 보여주었습니다.

```html
This is a task to evaluate the quality of a conversational question answering chatbot. You will be given [ground truth dialogue, two candidate multi-turn dialogue sets which has same user inputs and different chatbot responses], and your task is to compare the quality of the candidate responses based on four criteria: Accuracy, Informativeness, well-formedness, overall quality. For each criteria, answer which dialogue is better.

1. Accuracy : whether the chatbot responses are accurate compared with the ground truth dialogue
2. Informativeness: whether the chatbot responses are informative.
3. Well-formedness: whether the chatbot responses are well-formed.
4. Overall Quality: overall quality of the dialogue.

- Ground Truth

- Dialogue A

- Dialogue B

-> Choose the dialogue which is more accurate to the given user inputs. options: [Dialogue A, Equal, Dialogue B]
-> Choose the dialogue which is more informative to the given user inputs. options: [Dialogue A, Equal, Dialogue B]
-> Choose the responses which is more well-formed. options: [Dialogue A, Equal, Dialogue B]
-> Choose the dialogue which has better overall-quality. options: [Dialogue A, Equal, Dialogue B]
```

#### Context-to-Answer 

저희는 크게 두 가지 평가를 진행했는데, 첫번째 평가는 Context to Answer입니다. 조항 그대로인 Context에서 자연스러운 답변 형태인 Answer로 바꿔 파인튜닝했을 때, 챗봇 모델은 어떤 답변을 내놓을까요. Baseline을 QC set으로 파인튜닝한 라마-2 모델로 삼아, QA set으로 파인튜닝한 라마-2 모델과 비교했습니다. 테스트 데이터셋으로는 모델을 파인튜닝한 데이터셋에서 대화 10개를 뽑아, 그 질문을 이용해 대화를 생성했습니다. 

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Dialogue/12.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Dialogue/13.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fig 7. Context-to-Answer 즉, Context Styler를 적용하기 전의 QC 데이터와 적용 후인 QA 데이터를 각각 활용한 Llama-2-7B 모델이 생성한 답변에 대한 human evaluation 결과 (왼쪽) 및 실제 QA, QC 모델이 생성한 대화 예시(오른쪽)
</div>

Human evaluation의 결과를 나타낸 Fig 7를 보면, 그래프의 붉은 색은 QA 모델이 잘 했다, 보라 색은 두 모델이 같다, 푸른 색은 QC 모델이 잘했다는 뜻입니다. QA 모델의 답변이 더 informative하고 전반적 퀄리티도 좋다고 평가했습니다. 다만, 정확도는 QC 모델이 높았는데, Context에서 Answer로 변환하는 과정에서 삽입된 노이즈로 인해 정확성이 떨어졌다고 보여집니다.

Fig 7의 오른쪽에는 QA, QC 모델이 생성한 대화 예시를 나타내었습니다. QA 모델의 답변이 더 자연스럽고 사용자 친화적이었습니다. QC의 경우, 조항과 관련된 이야기만 하고 추가적인 설명이나 조언을 하지 않았습니다. 아래 예시를 보면, 같은 질문에 대해 규정에 대해 안내하는 것은 같지만, QA 모델의 답변이 더 길고 추가적인 설명을 덧붙이고 있습니다. QA 어프로치를 챗봇에 적용하는 것이 사용자 만족도를 높일 수 있을 것이라고 기대합니다.

#### Domain-specific Approaches 

다음 평가는 Hierarchy, Reference, Similarity 세 가지 domain-specific approaches의 퍼포먼스를 확인했습니다. 먼저 평가를 위해 같은 방법으로 Baseline1, Baseline2 데이터셋을 각각 만들었습니다. 비교할 베이스라인 모델로는 Baseline1과 Baseline2 데이터셋으로 파인튜닝한 라마-2-7b 모델을 이용했습니다. 그와 비교할 저희의 모델들에는 (Baseline1 + Hierarchy), (Baseline + Reference) 그리고 (Baseline + Similarity)으로 각각 라마-2-7b 모델을 파인튜닝했습니다 (Fig 3 참고). 

테스트 데이터셋은 총 15개의 대화를 평가했습니다. 공정한 비교를 위해서, 비교 모델과 저희의 모델 양쪽 학습에 사용된 Baseline1 셋에서에서 5개, 베이스라인 모델에만 파인튜닝된 Baseline2 셋에서 5개, 우리 모델에만 파인튜닝된 Domain specific datasest에서 5개 즉, Hierarchy 세팅의 경우 Hierarchy set에서 5개를 골랐습니다.

### Results

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
  <img src="/assets/img/Dialogue/14.png" alt="img1" width="100%">
  <img src="/assets/img/Dialogue/15.png" alt="img2" width="100%">
  <img src="/assets/img/Dialogue/16.png" alt="img3" width="100%">
  <img src="/assets/img/Dialogue/17.png" alt="img4" width="100%">
</div>
<div class="caption">
    Fig 8. 결과  
</div>

먼저 Fig 8의 왼쪽 상단 그림의 왼쪽은 Human Evaluation, 오른쪽 그래프는 GPT 3.5 Evaluation입니다.Hierarchy 모델과 베이스라인을 비교했을 때, Hierarchy 모델이 더 정확하고, 더 informative하고, overall quality도 높았습니다. 여러 항을 아우르는 broad한 질문에서 narrow한 질문으로 전개되는 dataset 덕분에, legal context에 대한 더 잘 이해하게 된 것으로 보입니다.

Fig 8의 오른쪽 상단 그림에는 레퍼런스 모델에 대한 평가를 나타내었는데, 베이스라인보다 더 informative하고 overall quality가 높았습니다. 다른 문서와 달리 법률 문서에만 있는 조항 간 레퍼런스 관계로, 연속적인 조항만으로 이해할 수 없는 컨텍스트를 학습한 영향으로 보입니다. 레퍼런스 관계가 있는 조항 자체가 제한되어 있어 scalablility가 아쉽지만, 부가적인 어프로치로 효과가 있을 것 같습니다.

Fig 8의 왼쪽 하단 그림은 similarity model의 퍼포먼스를 보여주며 baseline보다 모든 면에서 부족하게 나왔습니다. 이는 Embedding space에서의 유사성이 실제 법률 조항 내용의 유사성을 의미하지 않을 수 있기 때문이라고 보입니다. 이러한 결과에 대해 추가적으로 분석해본 결과, Fig 8의 오른쪽 하단에서 보시는 바와 같이, similarity group의 예를 들면, 법률 문서에서는 각 조항을 만든 목적에 대한 항이 굉장히 많았는데요. 첫번째 예시처럼 'purpose' 조항을 similarity 기준으로 모은 탓에, 자연스러운 대화를 만들기 어려워졌습니다. 또한 두번째 예시처럼 Similarity를 기준으로 모았음에도, 인간의 시각으로 비슷해 보이지 않는 케이스도 있음을 확인하였습니다. 


### Conclusion 

저희 연구의 컨트리뷰션은 세 가지입니다. 

1. 저희는 general document보다 더 어렵고, 데이터 수집도 비싼 법률 문서에 dialog inpainting 방식을 적용했습니다.
2. Original dialog inpainting 방식의 약점으로 꼽혔던 문서 그대로의 인위적인 답변을 자연스러운 답변으로 restyling해서 chatbot generation task에 데이터셋을 최적화했습니다.
3. 법률 문서의 참조, 계층구조 등을 반영한 인페인팅을 도입했습니다. 기존의 데이터셋을 restyling해도 얻어낼 수 없는 멀티턴 대화의 아이디어를 제안한 저희의 방식은 novelty를 갖는다고 생각합니다.

저희 연구의 리미테이션으로는 코스트 문제로 조항 수에 비해 적은 질문을 파인튜닝에 사용했다는 점, 법률 문서의 엄밀한 서술이 한영 번역-Inpainting-Restyling 과정에서 LLM을 거치며 무뎌졌다는 점이 있습니다. Future work로, 법률 디테일을 살려서 질문을 형성하도록 프롬프트 디자인을 좀 더 섬세하게 최적화하는 방법으로 연구를 발전시켜 보고 싶습니다.


### Final Report 
Click [here][pdf] 😊

[pdf]: /assets/pdf/Dialogue_inpainter_report.pdf
[ref]: https://proceedings.mlr.press/v162/dai22a.html

### Reference 

[^1]: Dai, Zhuyun, et al. "Dialog inpainting: Turning documents into dialogs." International conference on machine learning. PMLR, 2022.
[^2]: Liu, Yongtai, et al. "Data augmentation for low-resource dialogue summarization." Findings of the Association for Computational Linguistics: NAACL 2022. 2022.
[^3]: Dai, Haixing, et al. "Auggpt: Leveraging chatgpt for text data augmentation." IEEE Transactions on Big Data (2025).
[^4]: Yuan, Mingruo, et al. "Bringing legal knowledge to the public by constructing a legal question bank using large-scale pre-trained language model." Artificial Intelligence and Law 32.3 (2024): 769-805.