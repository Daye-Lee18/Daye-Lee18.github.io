---
layout: page
title: Explainable Video-QA
description: Explainable Video-QA / VideoRAG based on HyperCLOVA X / SEED (Vision-Instruct)
img: assets/img/ExplainableVideoQA/thumnail.png
importance: 1
category: work
related_publications: true
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

위의 데이터 중 소규모 PoC로 진행할 것이므로 TVQA 1~3k 샘플 서브셋으로 시작하였다. 그 이유는 **자막/타임코드**가 있어 근거 인용 평가가 비교적 수월하기 때문이다.

### Pipeline

(A) 실험 1 (프레임만)

1. 프레임 샘플링: 균일/샷변화/키프레임 감지로 N(=8/16)장 추출
2. 입력 컨텍스트 구성: [이미지 N장 + (옵션) 해당 시점 자막 스니펫]
3. SEED-Vision-Instruct 추론: 답변 + 근거 프레임 id/자막 타임코드 인용
4. 출력: `정답, 한줄 이유, [근거: frame_i, t_start ~ t_end, subtitle_span]`

(B) 실험 2 (Retrieval-Augmented)

1. CLIP/TC-CLIP 전처리 인덱스: 비디오 -> 프레임 (또는 샷) 임베딩 / 자막 문장 임베딩
2. 질의 -> Top-K 증거 검색: 프레임/샷/자막 후보 K개 뽑기(다양성 보장: MMR)
3. 근거 압축(Adaptive): K→M(=6~12)로 축소(시간분산/유사도 임계)
4. SEED-Vision-Instruct에 근거 패키지 투입 → 근거 인용형 답변 생성
5. 출력 재랭크: Groundedness Score(근거 일치율)로 beam 결과 중 최상 선택

### Prompt

1. 공통 시스템 프롬프트

   ```md
   당신은 영상 질문응답 어시스턴트입니다.
   규칙:

   1. 반드시 답을 한 줄로 먼저 말합니다.
   2. 이어서 '이유'를 1~2문장으로 설명합니다.
   3. 마지막에 [근거] 섹션에 사용한 프레임 ID와 자막 타임코드를 인용합니다.
      형식 예:
      답: ...
      이유: ...
      [근거] frame: f03,f07 | subtitle: s12(00:12.3-00:14.1)
   ```

2. 사용자 프롬프트 (실험 1)

   ```md
   질문: "{QUESTION_KO}"
   이미지 프레임: f01..f{N}
   각 프레임에는 좌측 상단에 ID가 적혀 있습니다.
   (선택) 자막 스니펫: s11(00:10.1-00:12.0): "{SUB1}" / s12(...): "{SUB2}" ...
   정확한 답변과 이유를 말하고, 실제로 참고한 프레임 ID와 자막 ID/타임코드를 [근거]에 표기하세요.
   ```

3. 사용자 프롬프트 (실험 2)
   ```md
   질문: "{QUESTION_KO}"
   검색으로 찾은 근거(요약):

   - frame: f02(00:05.2), f07(00:11.8), f09(00:14.0)
   - subtitle: s10(00:10.3-00:11.5): "{SUB10}", s13(...): "{SUB13}"
     위 근거만 사용하여 답을 생성하세요. 근거가 불충분하면 "정보 불충분"을 답으로 하세요.
     출력 형식은 [규칙]을 따르세요.
   ```

### Experiments

### Evaluation

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

### Results

#### 실험 1

프롬프트 패턴: 규칙형 vs. 자유형 vs. 체인형 ("먼저 근거 찾기 -> 답변" 2-step )

#### 실험 2

근거 선택: 균일 샘플링 vs 샷경계 vs TC-CLIP Top-K vs Adaptive Top-M

#### 실험 3

자막 사용: 사용 vs 미사용 (한국어 질의-자막 정합 영향)

#### 실험 4

근거 강제 규칙: “근거 없으면 정보 불충분” 규칙 on/off

#### K, M

후보(K=24)→압축(M=8/12) 스윕
