---
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

위와 같은 초기 QA 데이터를 비디오 구간 및 subtitles를 가지도록 다음과 같은 형태로 변환하여 최정적으로 사용하였다.

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

### Evaluation Metrics

- 정답성(Answer Quality)

  - Accuracy / F1(TVQA: 선택지 기반 정확도, MSRVTT-QA: EM/F1)
  - 하위지표: 길이·회피율(“정보 불충분” 비율)

- 근거성(Groundedness)

  - Frame-Hit@K: 모델이 인용한 frame id가 GT 타임구간과 IoU>τ로 겹치는 비율
  - Subtitle-Hit@K: 인용 자막 span이 GT 자막 구간과 겹침 여부
  - 지원 일치율: [근거]에 명시된 증거가 실제 입력 집합 내에 존재하는지(환각 방지)

- 설명 품질(Explanation)

  - Conciseness(문장 수/토큰), Consistency(답과 이유 충돌률↓)
  - 자동 평가지표: BERTScore / BLEU(옵션, TVQA는 짧아 한계 有)
  - 휴먼평가(5점 Likert): 타당성, 명확성, 신뢰감

- 효율/비용
  - 지연시간(latency), 메모리, 프레임 수(입력 토큰량)

### Experiments & Results

#### Exp 1. Prompting Strategy

규칙형 vs. 자유형 vs. 체인형(2-Step: 근거 -> 답변)의 3가지 프롬프트 패턴을 비교한다. 이를 위한 평가 지표로는 accuracy, 근거 인용률 (Frame-Hit, Subtitle-Hit), "정보 불충분" 응답률을 사용하였다. Text-Instruct 모델인 HyperCLOVAX-SEED-Text-Instruct-1.5B을 사용하여 위에서 전처리한 1K dataset에 대해서 모델 성능을 평가하였다. 자막 기반 QA에서 "비디오 프레임 없이" 모델 성능을 평가하고자 하였다.

#### Exp 2. Evidence Sampling

좋은 evidence 선택은 성능과 효율 측면에서 증유하다. 따라서 아래의 4가지 방법으로 evidence frame을 선택하고 accuracy, 근거성, Latency, 프레임 수를 측정하여, 어떤 방식이 성능을 잘 올리는지 평가하고자한다.

1. 프레임 샘플링: 균일/샷변화/키프레임 감지로 N(=8/16)장 추출
2. 입력 컨텍스트 구성: [이미지 N장 + (옵션) 해당 시점 자막 스니펫]
3. SEED-Vision-Instruct-3B 추론: 답변 + 근거 프레임 id/자막 타임코드 인용
4. 출력: `정답, 한줄 이유, [근거: frame_i, t_start ~ t_end, subtitle_span]`

#### Exp 3. 자막 사용 여부

같은 비디오 증거 (프레임 N장) 를 입력으로 고정하고, 자막 (snippet)만 on/off 하여 정답률 (accuracy) 변화, 근거성 (groundness: Frame/Subtitle-Hit), 환각률/"정보 불충분" 비율을 비교하고자 한다. 이를 위해, TVQA 1K 서브셋에서 두 코호트로 나눠 분석하였다. 대사 의존형 QA (Subtitle-needed)는 답이 대사/문맥에 의존하는 질문이고, 시각 의존형 QA (Visual-only) 는 프레임만으로도 답이 가능한 질문이다. 이를 분류하는 것은 질문에 순수 색/객체/행동/장소 묘사는 Visual-only QA로, 대화/태도/감정/숫자 읽그 ('무슨 말을 했나', '왜', '어떻게 반응했다')가 있으면 Subtitle-needed후보로 나누었다.

공정한 평가를 위해, input으로 들어가는 프레임은 동일하게 진행하였는데, 동일 샘플에서 Uniform-8로 추출한 같은 프레임 세트를 사용하였다. 또한 프롬프트도 동일하게 하였으며 하이퍼파라미터들도 동일하게 고정하였다.

모델은 SEED-Vision-Instruct-3B을 사용하였으며, 각 입력에 대해 1회 추론하였다. 사용한 지표는 a

#### Exp 3. Retrieval-Augmented

1. CLIP/TC-CLIP 전처리 인덱스: 비디오 -> 프레임 (또는 샷) 임베딩 / 자막 문장 임베딩
2. 질의 -> Top-K 증거 검색: 프레임/샷/자막 후보 K개 뽑기(다양성 보장: MMR)
3. 근거 압축(Adaptive): K→M(=6~12)로 축소(시간분산/유사도 임계)
4. SEED-Vision-Instruct에 근거 패키지 투입 → 근거 인용형 답변 생성
5. 출력 재랭크: Groundedness Score(근거 일치율)로 beam 결과 중 최상 선택

### Conclusion
