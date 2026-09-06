# SLAM 학습 자료

이 폴더에는 SLAM History와 LIO 학습 노트에서 참고하는 공개 자료의 로컬 사본을 둔다. PDF는 개인 학습용이며 Git에는 포함하지 않는다. `manifest.json`에는 원문 URL, 저자, 페이지 수, 체크섬과 다운로드 상태를 기록한다.

## 폴더 구성

- `kroc2026/`: [KRoC 2026 Spatial AI Tutorial](https://sites.google.com/view/kroc26-spatial-ai-tutorial/home) 공개 슬라이드 7개
- `foundations/`: 회전, SLAM course, factor graph, bundle adjustment 기초 자료
- `papers/`: History 노트에서 직접 읽는 대표 논문

## 주요 학습 자료

| 자료                                                 | 원문                                                                                         |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| History of SLAM and the SLAM Handbook Project        | [KRoC 공개 슬라이드](https://drive.google.com/file/d/1tmWxcQFD0lGZPO3L6wjxT1k6am4EXyMK/view) |
| A Short Journey from 3D Vision to 3D Representations | [KRoC 공개 슬라이드](https://drive.google.com/file/d/1mL52klpHEYU6e-yZk3guMaJocLthSAA7/view) |
| Representations for 3D Visual World                  | [KRoC 공개 슬라이드](https://drive.google.com/file/d/1OTZjzUGls3fjSQed7LU-xzjS_78e7BIW/view) |
| SLAM Back-end                                        | [KRoC 공개 슬라이드](https://drive.google.com/file/d/1FGnya__7ZQYsgE7CRhRjggeU2fIQ3izH/view) |
| IMU Basics and Inertial Aided Navigation             | [KRoC 공개 슬라이드](https://drive.google.com/file/d/1byqGAKCCsnv8rZbko4RBG9KiQQD_4h8x/view) |
| Point Cloud Registration                             | [KRoC 공개 슬라이드](https://drive.google.com/file/d/1fwaHF77iwTxcDVl1s1h8BcCZE0EvPmUC/view) |
| AI Fundamentals for Monocular Visual SLAM            | [KRoC 공개 슬라이드](https://drive.google.com/file/d/1-FZ207zXWqZiEudDnd5EEAaybWdIJzNe/view) |
| Quaternion kinematics for the ESKF                   | [arXiv](https://arxiv.org/abs/1711.02508)                                                    |
| Course on SLAM                                       | [UPC repository](https://upcommons.upc.edu/handle/2117/337287)                               |
| Factor Graphs for Robot Perception                   | [저자 페이지](https://www.cs.cmu.edu/~kaess/pub/Dellaert17fnt.html)                          |
| Bundle Adjustment - A Modern Synthesis               | [저자 공개본](https://lear.inrialpes.fr/people/triggs/pubs/Triggs-va99.pdf)                  |
| 추천 학습 순서와 추가 교재                           | [Giseop Kim의 SLAM Back-end 공부자료](https://gisbi-kim.github.io/post/slam-textbooks/)      |

`foundations/grisetti2016-least-squares-icp.pdf`와 `foundations/stachniss2016-graph-slam.pdf`는 기존 직접 링크가 404를 반환해 내려받지 못했다. 같은 주제는 KRoC back-end 및 registration 슬라이드로 먼저 공부하고, `manifest.json`의 원문 링크가 복구되면 추가한다.
