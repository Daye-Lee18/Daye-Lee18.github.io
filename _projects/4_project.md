---
layout: page
title: HealthGenie, Personalized Health Advisor
description: Built a personalized health advisory web service for diet and weight management using YOLOv5-based food detection, calorie tracking, and SVD-based food recommendation on user health data.
img: assets/img/HealthGenie/HealthGenie_thumnail.png
importance: 3
category: work
related_publications: false
tags:
  - Computer Vision
  - Recommender System
  - YOLOv5
  - Django
  - Personalized Health
  - PostgreSQL
  - Streamlit
toc:
  sidebar: left
---

<!-- thumnail size: 11?? x 605 -->

### Motivation

최근 건강에 대한 관심이 매우 크게 증가하고 있는데요. 특히, ‘체중 관리’는 건강과 웰빙에 있어서 중요한 역할을 합니다. 비만이나 과체중은 당뇨병, 고혈압 등 다양한 건강 문제와 매우 밀접한 관련이 있을 뿐만 아니라, 체중을 적절히 유지해야 신체의 기능과 균형을 유지할 수 있고, 자아존중감에도 긍정적인 영향을 미치기 때문입니다.

그럼 체중에 큰 영향을 미치는 요소에는 무엇이 있을까요? 바로, ‘음식으로부터 섭취하는 총 칼로리’가 체중에 가장 큰 영향을 미친다고 할 수 있습니다. 따라서 체중 관리를 위해서는 ‘정확한 식단 추적과 관리’가 필요합니다. 최근에는 인스타그램 등 sns을 통해 음식 사진을 올리고 공유하면서 자연스럽게 식단을 기록하는 경향이 많아지고 있습니다.

### Our Goal

기존 서비스에서는, 건강에 대한 관심 증가로 인해 많은 앱들이 개발되어 사용자들이 스스로 건강을 관리할 수 있는 도구로 활용되고 있지만 대부분 단순히 식단 정보와 체중 정보를 기록하고 관리해주거나, 커뮤니티 게시판을 통해 사용자들이 직접 식단을 추천해주는 기능을 제공하고 있습니다.

저희는 그러한 문화적 특성을 잘 반영하여, 사용자들이 일상 속에서 쉽게 이미지를 업로드 함으로써 자신의 식단 정보를 기록할 수 있는 서비스를 개발하고자 하였습니다. 궁극적으로 저희는 AI기술을 통해 ‘데이터 기반의 접근 방식’으로 식사를 추천해주는 개인 맞춤형 가이드 서비스를 구축하였습니다.

### Product Overview

저희 서비스인 ‘HealthGenie’는 사용자에게 크게 ‘weight tracking’, ‘calorie tracking’, ‘food recommendations’ 3가지 서비스를 제공합니다. 먼저, 사용자가 음식 이미지를 업로드하면 그 사진에 대해 image detection을 수행하고 칼로리를 예측하고 제시해줍니다. 그에 따라 하루 총 칼로리와 특정 영양소, 일일 식사량을 확인할 수 있습니다.

그리고 사용자가 자신의 체중을 함께 입력하면 사용자의 data를 기반으로 기간에 따라서 체중 정보를 시각적으로 확인할 수 있습니다.

세 번째는, ‘Food Recommendations’ 기능입니다. 진행 중인 프로젝트가 있는 사용자는 식사 추천을 받을 수 있습니다. 저희 서비스는 몇 가지 지표를 사용하여 건강 관리에 성공한 사용자를 기반으로 하여 식사를 추천합니다. ‘HealthGenie’의 가장 큰 장점은 맞춤형 서비스으로, 각 사용자의 건강 상태, 목표, 선호하는 식단 스타일에 따라 개인화된 식단 추천을 제공합니다.

#### Target Customers

저희 서비스는 식단과 건강 정보를 추적하고 관리해야 하는 모든 사람들을 대상으로 합니다. 혼자서는 다이어트 관리에 어려움을 겪는 사람들이나, 단기/장기적으로 식단을 관리하고 건강 목표를 달성하고자 하는 사람들, 그리고 자신의 식단 패턴을 파악하고 그에 따라 맞춤형 식단 추천을 받고 싶은 사람들을 위한 서비스입니다.

#### Business Value & Model

개인 맞춤형 건강 리포트와 음식을 추천받을 수 있다는 점에서 장점을 가지며, 또한 사진을 업로드함으로써 식단을 기록하는 문화적 경향과 더불어 사용이 쉬운 UI가 있어 사용자 친화적입니다.

더하여 식단 기록과 해당 프로젝트의 성공 여부를 통해 추후에 연구나 추가 서비스에 사용할 수 있을 것입니다.

저희는 개인 맞춤형 건강관리 서비스를 제공하는 것에서 나아가 프로젝트 진행 상황에 따라 사용자에게 응원과 격려 메시지를 보내어 심리적 공감대를 형성합니다. 이에 더하여 사용자의 편리성을 위해 추천 음식에 링크를 걸어두어 쉽게 구입이 가능하도록 하였습니다.

### Scenario

시나리오와 사용 부분 시나리오 영상 제작

<!-- <div class="responsive-video ratio-4x3">
  {% include video.liquid
      path="https://youtube.com/embed/pLpjMMiW7dg"
      class="embed-item"
      caption="시연 영상: HealthGenie 사용 예시 (YouTube)"
  %}
</div> -->
<!-- <iframe width="100%" height="560"
  src="https://www.youtube.com/embed/pLpjMMiW7dg" title="Demo"
  frameborder="0" allow="autoplay; clipboard-write; encrypted-media; picture-in-picture; web-share"
  allowfullscreen></iframe> -->

<!-- <div class="row justify-content-sm-center">
  <div class="col-sm-10 mt-3 mt-md-0">
    <div class="embed-responsive embed-responsive-16by9">
      <iframe class="embed-responsive-item"
              src="https://www.youtube.com/embed/pLpjMMiW7dg"
              allowfullscreen></iframe>
    </div>
  </div>
</div>
<div class="caption">
    시연 영상: HealthGenie 데모 (YouTube)
</div> -->

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid path="assets/video/HealthGenie_Demo.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay=false muted=true%}
    </div>
</div>
<div class="caption">
    Vid 1. Health Genie Demo Video 
</div>

### Architecture

사용자는 `Django` 기반의 웹 서버를 통해 전체 서비스에 접근할 수 있습니다. 이 웹 서버를 통해 사용자는 애플리케이션에서 제공하는 모든 기능에 접근할 수 있으며, 데이터는 관계형 데이터베이스인 `PostgreSQL`에 저장됩니다.

`Streamlit`을 기반으로 한 대시보드를 제공하여 다양한 지표를 사용자에게 시각적으로 보여줍니다. 이 대시보드를 통해 사용자는 서비스의 다양한 통계 및 분석 결과를 쉽게 확인할 수 있습니다.

음식 인식 서비스는 `YOLOv5`라는 object detection 모델을 활용하며 이를 위해 `Flask` 기반의 웹 서버와 통신하여 음식 인식 모델을 제공합니다. 이에 따라 사용자가 음식 사진을 제공하게 되면, 해당 음식을 감지하고 분류합니다.

또한, 음식 추천 서비스는 `latent factor model`과 `content 기반 추천 알고리즘`을 활용합니다. 이 알고리즘은 데이터베이스에 저장된 사용자들의 생성 데이터셋을 기반으로 작동하며, 사용자에게 맞춤형 음식 추천을 제공합니다.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/HealthGenie/HealthGenie_architecture.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The Overall Pipeline of Our Health Genie Service.
</div>

### Data & Data Schema

저희 서비스를 위한 데이터 스키마에 대해 간략히 소개해 드리겠습니다. 총 `6개의 테이블`로 구성되어 있으며, 각 테이블은 `외래 키`를 사용하여 관계를 표현하고 있습니다.

첫 번째로 소개할 테이블은 "사용자 테이블"입니다. 이 테이블은 사용자가 회원 가입 시 생성되는 정보를 담고 있습니다.

다음으로 "프로젝트 테이블"과 “건강 정보 테이블” 은 사용자가 애플리케이션 내에서 진행하는 프로젝트와 관련된 정보를 저장합니다. 특히 “건강 정보 테이블” 에는 식단 추천에 필요한 알러지나 식이제한, 활동 수준 등의 정보가 포함됩니다.

다음으로는 "체중 추적 테이블"입니다. 이 테이블은 사용자가 애플리케이션을 사용하면서 입력하는 현재 체중 정보를 저장합니다.

그 다음은 "식단 테이블"입니다. 이 테이블은 음식 인식 및 추가적인 입력을 통해 사용자가 저장하는 식단 정보를 관리합니다. 특히 각 식단 테이블은 평가 점수 등의 정보가 포함되어 식단 추천에 사용됩니다.

마지막으로, "음식 테이블"은 음식 인식에 따른 단위 함량당 칼로리 및 해당되는 식이제한 정보를 저장합니다. 음식의 칼로리 정보가 이 테이블에 저장되어 있어 칼로리 추적을 가능하게 합니다.

이렇게 6개의 테이블로 구성된 데이터 스키마를 사용하여 서비스를 운영하고 있습니다.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/HealthGenie/3.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Data Schema Structure 
</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/HealthGenie/9.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Data Schema Structure 
</div>

#### Synthetic Data Generation

먼저 csv file을 팀원의 각 로컬 컴퓨터에서 삽입하는데 문제가 있어서, 자동으로 로컬 postgresql에 데이터를 삽입하도록 `psycogp2` library를 이용해 코딩하였습니다.

소비자 선호도 분석 및 음식 추천, 칼로리, 몸무게 추적 등 서비스 구축을 위해 큰 `가상 데이터`가 필요했습니다. 큰 데이터도 필요했지만 무작위적으로만 가상 데이터를 생성한다면, 소비자 선호도 측정 및 개인 음식 추천을 잘 하는지 알 수 없을 것이기 때문입니다. 따라서 아래의 그림과 같이 무작위로 선택한 food_id에 대해 content-based recommendation한 결과 list 오름 정렬에 따라 그룹별로 rating을 주었습니다.

추천 시스템에서 cold start 문제를 피하기 위해 많은 사용자들의 많은 식단 데이터가 필요했으며 따라서 사용자 당 5개의 프로젝트를 구성하고 `1년 6개월이 넘는 식단 데이터를 구축`하였으며 몸무게 데이터도 가입 날짜에 맞추어 데이터를 적재시켰습니다.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/HealthGenie/4.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Synthetic data generation  
</div>

### Models

#### Food Object Detection Model

우리의 Food object Detection 모델에 대해 더 자세히 소개하겠습니다. YOLO 모델을 선택한 결정적인 이유는 그 `빠른 예측 속도`에 있습니다. 고객에게 신속하게 예측된 음식 정보를 제공하는 것은 Health Genie 서비스의 필수적인 기능 중 하나입니다.YOLO는 '1-stage method'로, 이는 기존의 '2-stage method' 모델들보다 더 빠른 inference를 가능하게 합니다.

아래의 그림은 대표적인 2-stage method인 Faster R-CNN과 YOLO의 아키텍처를 비교한 것입니다. Faster R-CNN은 인퍼런스 과정에서 먼저 객체의 위치에 대한 proposal을 진행한 다음, 그 객체의 클래스를 예측하는 두 단계를 거칩니다. 반면, YOLO는 위치와 클래스를 동시에 예측하는 점에서 차이가 있습니다.우리는 YOLO의 다양한 버전 중에서, DSPNet과 Triple Head 기능이 통합된 YOLOv5를 모델로 선택했습니다.

또한 모델 구축 과정에서 우리는 적절한 데이터셋을 구축하는데에 어려움을 겪었습니다. object detection 모델은 classification 모델과 달리, 음식의 이미지에 음식의 클래스 뿐만 아니라 음식의 위치를 나타내는 bounding box 정보도 필요합니다. 이러한 데이터셋을 생성하는 것과 찾는 것 모두 어려운 일이고 음식으로 데이터를 한정 했을 때는 더욱 데이터의 양이 적었습니다.

다행히 저희는 `Open Image Dataset`의 음식 카테고리를 이용하여 데이터셋을 구축할 수 있었습니다. 그러나, 여전히 데이터는 충분치 않고, 모델 확장을 위해 더 많은 데이터셋을 구축하는 방법이 필요하며 이는 저희의 남아있는 과제 중에 하나입니다.

Food Object Detection 모델에 대해 안내드리겠습니다. 저희 모델은 음식에 특화되어 있어야 한다는 요구 조건과 함께 다중 객체 감지가 가능해야 한다는 요구 조건을 가지고 개발되었습니다. 기존에 공개된 오픈 API들은 대부분 보편적인 객체 감지만을 제공하거나 음식에 대한 단일 분류만을 제공하고 있습니다. 그래서 저희 팀은 앱에 특화된 API가 필요하다고 판단하고 직접 개발하기로 결정했습니다. API는 pytroch model을 `flask`를 이용해서 서빙하는 방식으로 구현하였습니다.

아래 그림을 통해 전체 API의 작동 과정을 설명하겠습니다. 사용자는 음식을 Django 웹 어플리케이션을 통해 입력하며, 이 입력은 API로 전달됩니다. 전달된 이미지는 YOLOv5 기반의 모델을 통해 추론되고, 결과는 JSON 형식으로 반환됩니다.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/HealthGenie/10.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Food Objection Detection Model Overview  
</div>

#### Food Recommender Systems - Model Selection & Hyperparameter Setting

저희 서비스의 음식 추천을 위한 food recommender systems에 대해서 설명하겠습니다. 오른쪽 그림과 같이 `surprise` library에서 지원하는 총 `9개의 모델`에 대해 `cross validation 5 folds`로 진행한 `RMSE` 와 `MAE`값을 계산하였으며 그 결과, BaselineOnly model의 결과가 두 평가 지표 모두에서 성적이 좋았습니다. Model selection에서 뽑힌 BaselineOnly와 SVD 모델의 hyperparameter을 gridy search 방법을 통해 평가하였고 그 결과 오히려 `SVD`의 결과가 좋았습니다. 따라서, 그에 해당하는 best parameter sets들을 이용해 저희의 서비스에 사용하였습니다.

성능이 전반적으로 1이 넘게 나왔는데 이 이유는 가상 데이터에 있는 어쩔 수 없는 무작위성인 것 같았고, 이는 서비스가 상용화되면 해결될 문제라 생각합니다.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/HealthGenie/5.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Food Recommender systems - Model & Hyperparameter selection   
</div>

#### Food Recommender Systems - Algorithm Flow

저희는 `Singular Value Decomposition` 를 사용한 `Latent Factor Model (LFM)`을 사용하여 유사한 사용자 정보를 이용해 목표 사용자에게 음식을 추천해주었습니다.

그 전에 LFM에 들어갈 사용자 매트릭스를 정제할 필요가 있었습니다. 따라서 사용자들의 프로젝트 goal_type과 goal_bmi, 그리고 활동량이 맞는 사용자로만 구성된 사용자 매트릭스를 만들어 LFM에 사용하였습니다.

또한 LFM의 결과인 음식 리스트에는 특정 사용자에게 `알러지를 유발`하는 재료가 포함되어 있거나 식이 제한이 되는 음식이 포함되어 있을 수 있으므로 이러한 음식들을 `제거하는 과정도 포함`하여 개인 맞춤 음식을 추천하도록 Food recommender system을 구성하였습니다.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/HealthGenie/6.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Food Recommender systems - Algorithm Flow   
</div>

저희는 `SurPRISE`라는 추천 시스템 엔진을 위한 library를 활용하였습니다. 간단히 말하자면, Latent Factor Model에서 사용하는 SVD는 기존 rating matrix를 분해하는 것이 아닌 item과 user의 factor에 대한 matrix를 통해 역으로 `rating matrix를 예측`하는 것입니다.

최종 예측 값은 먼저 각 사용자와 아이템의 평균 값과의 차이 bias를 이용해 결과값의 해석을 용이하게 하였습니다. 후에 SVD 값을 더해줌으로써 최종 값을 얻었습니다. 저희는 stochastic gradient descent optimization 방법과, factor의 개수는 5개, learning rate는 0.002로 앞서 hyperparameter selection을 통해 얻은 값을 알고리즘에 사용하였습니다.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/HealthGenie/7.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Food Recommender systems - SurPRISE 
</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/HealthGenie/8.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Food Recommender systems - SVD factorization
</div>

### Takeaways

프로젝트에 대한 challenges와 future works에 대해서는 최종발표 때 다루었기 때문에, 이번 녹화영상의 마지막 부분에서는 저희가 프로젝트를 통해 느꼈던 점들과 배운 점들에 대해서 언급하면서 발표를 마무리하겠습니다.

먼저, 프로젝트를 통해 서비스 작동을 위한 전체적인 architecture를 공부할 수 있는 시간이었으며 꾸준한 작업과 작은 디테일들의 중요성들을 배웠습니다. 각 task가 연결되어야 하는 방법들을 연구하며 각 구성요소들의 interdependencies 을 이해하면서 해결되어야 하는 workflow를 설계하고 개발 과정을 효율적으로 조율하는 법도 배웠습니다. 그리고 프로젝트를 하면서 팀원들과 지속적으로 소통하고 함께 서로 피드백을 주고 받은 과정을 통해 소통의 중요성에 대해서 배울 수 있었습니다.

그리고 사용자의 입장에서 서비스 UI와 모델 구성 방법을 고민하는 시간을 가질 수 있었습니다. 서비스의 목적과 예상 사용자를 고려하여 서비스를 개발하는 것이 중요하고, 사용자의 입장에서 서비스의 유용성을 고려하는 것이 보다 중요하다는 점을 알 수 있었습니다.

계속 발전하는 모델과 개인화된 서비스를 위한 data-driven approaches를 위한 방법을 공부할 수 있었습니다. 또한 새로운 기능을 추가하거나 기존의 서비스를 더욱 개선하기 위해서 지속적으로 아키텍쳐와 데이터베이스를 관리하는 것의 중요함을 알 수 있었습니다.
