---
layout: page
permalink: /samsung-cote/
title: Samsung COTE Prep
description: 삼성 SW역량테스트 단기 준비 자료 (Python)
nav: false
---

# 삼성 SW역량테스트 단기 준비 가이드 (Python)

> 핵심 패턴 숙지 + 기출 답 코드 분석으로 실전 감각 확보

---

## 시험 특징 요약

| 항목 | 내용 |
|------|------|
| 문제 수 | 2문제 (3~4시간) |
| 난이도 | 백준 골드 3~5 수준 |
| 핵심 유형 | **시뮬레이션** + **BFS/DFS** (거의 매번) |
| 특징 | 조건이 매우 복잡하고 구현량이 많음 |
| 입력 | `sys.stdin.readline` 사용 권장 |

---

## 출제 유형별 빈도 (최신 기출 기준)

```
1순위 ★★★  시뮬레이션 + BFS/DFS 혼합
2순위 ★★★  격자 이동 + 특정 조건 반복 수행
3순위 ★★   완전탐색 (순열/조합) + 시뮬레이션
4순위 ★    우선순위큐 / 다익스트라 (간혹)
```

---

## 공부 방법 (3단계)

### Step 1 — 패턴 파일 읽기

각 패턴을 **읽으면서 다음 질문에 답하기:**
- 이 패턴은 어떤 상황에서 쓰나?
- visited를 어떻게 선언하나?
- 범위 체크 조건은?

### Step 2 — 답 코드 분석법 (핵심)

코드트리 기출 답을 볼 때 **절대 그냥 읽지 말 것.**

```
① 문제 읽기 (5분)
   → 격자? 그래프? 반복 시뮬레이션?
   → 입력 형태와 출력 형태 파악

② 알고리즘 분류 (2분)
   → BFS? DFS? 완전탐색? 시뮬레이션?
   → 복합이면 어느 순서로 쓰이나?

③ 답 코드 구조 파악 (10분)
   → 함수를 어떻게 나눴나?
   → 핵심 로직이 어디에 있나?
   → visited, queue, 방향벡터를 어떻게 정의했나?

④ 핵심 로직 손으로 써보기 (15분)
   → 답 닫고 핵심 부분만 직접 작성
```

### Step 3 — 자가 체크리스트

- [ ] 방향벡터 4방향 / 8방향 선언
- [ ] BFS 기본 틀 (queue, visited, 거리 추적)
- [ ] DFS 기본 틀 (재귀, 백트래킹 옵션)
- [ ] 2D 배열 90도 회전
- [ ] `permutations`, `combinations`, `product` 사용
- [ ] 낙하(중력) 시뮬레이션
- [ ] 범위 체크 `0 <= nr < N and 0 <= nc < M`
- [ ] `sys.stdin.readline` + 입력 파싱

---

## 핵심 패턴 모음

### 시작 템플릿 (항상 맨 위에)

```python
import sys
from collections import deque
from itertools import permutations, combinations, product
input = sys.stdin.readline
sys.setrecursionlimit(10**6)
```

---

### Pattern 1 — 방향 벡터

```python
# 상하좌우 4방향
dr4 = [-1, 1, 0, 0]
dc4 = [0, 0, -1, 1]

# 8방향 (대각선 포함)
dr8 = [-1, -1, -1, 0, 0, 1, 1, 1]
dc8 = [-1, 0, 1, -1, 1, -1, 0, 1]

# 범위 체크 항상!
for i in range(4):
    nr, nc = r + dr4[i], c + dc4[i]
    if 0 <= nr < N and 0 <= nc < M:
        pass
```

---

### Pattern 2 — BFS (최단거리, 영역 탐색)

```python
def bfs(start_r, start_c, board, N, M):
    visited = [[False] * M for _ in range(N)]
    queue = deque()
    queue.append((start_r, start_c, 0))   # (행, 열, 거리)
    visited[start_r][start_c] = True

    while queue:
        r, c, dist = queue.popleft()
        for i in range(4):
            nr, nc = r + dr4[i], c + dc4[i]
            if 0 <= nr < N and 0 <= nc < M and not visited[nr][nc] and board[nr][nc] != 0:
                visited[nr][nc] = True
                queue.append((nr, nc, dist + 1))
    return dist
```

---

### Pattern 3 — DFS (경로 탐색, 백트래킹)

```python
def dfs(r, c, visited, board, N, M):
    visited[r][c] = True
    for i in range(4):
        nr, nc = r + dr4[i], c + dc4[i]
        if 0 <= nr < N and 0 <= nc < M and not visited[nr][nc]:
            dfs(nr, nc, visited, board, N, M)
    # visited[r][c] = False  ← 백트래킹 필요할 때 언주석
```

---

### Pattern 4 — 완전탐색 (순열/조합)

```python
# 순열: 순서 있음, 중복 없음
for perm in permutations([1, 2, 3, 4, 5], 3):
    pass

# 조합: 순서 없음, 중복 없음
for comb in combinations([1, 2, 3, 4, 5], 3):
    pass

# 중복순열: 순서 있음, 중복 허용
for prod in product([0, 1, 2], repeat=3):
    pass
```

---

### Pattern 5 — 회전 (삼성 단골)

```python
def rotate_90_clockwise(matrix):
    """2D 배열 시계방향 90도 회전"""
    N = len(matrix)
    return [[matrix[N - 1 - j][i] for j in range(N)] for i in range(N)]

# zip 방식 (시계방향)
def rotate_zip(matrix):
    return list(zip(*matrix[::-1]))
```

---

### Pattern 6 — 범위 내 영향 (폭발, 확산)

```python
def apply_effect(r, c, power, board, N, M):
    """중심(r,c)에서 상하좌우 power 칸 영향"""
    affected = [(r, c)]
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        for p in range(1, power + 1):
            nr, nc = r + dr * p, c + dc * p
            if 0 <= nr < N and 0 <= nc < M:
                affected.append((nr, nc))
    return affected
```

---

### Pattern 7 — 낙하/중력 시뮬레이션

```python
def apply_gravity(board, N, M):
    for c in range(M):
        col = [board[r][c] for r in range(N) if board[r][c] != 0]
        for r in range(N - 1, -1, -1):
            board[r][c] = col.pop() if col else 0
    return board
```

---

### Pattern 8 — 연결 영역 카운트 (섬 세기)

```python
def count_islands(board, N, M):
    visited = [[False] * M for _ in range(N)]
    count = 0

    def dfs(r, c):
        visited[r][c] = True
        for i in range(4):
            nr, nc = r + dr4[i], c + dc4[i]
            if 0 <= nr < N and 0 <= nc < M and not visited[nr][nc] and board[nr][nc] == 1:
                dfs(nr, nc)

    for r in range(N):
        for c in range(M):
            if board[r][c] == 1 and not visited[r][c]:
                dfs(r, c)
                count += 1
    return count
```

---

### Pattern 9 — 상태를 튜플로 visited 관리

```python
# (r, c, direction) 상태 관리 — 방향·아이템 등 추가 상태가 있을 때
visited = {}
queue = deque()
start_state = (0, 0, 0)  # r, c, dir
queue.append(start_state)
visited[start_state] = 0  # 거리
```

---

## 시험 당일 전략

### 구현 순서

```
① 입력 파싱 먼저 작성
② 핵심 함수 뼈대 작성 (빈 함수라도)
③ 로직 채우기
④ 예제로 테스트
⑤ 엣지케이스 체크 (빈 배열, 1x1 격자 등)
```

### Python 실수 방지

```python
# 1. 리스트 복사
board_copy = [row[:] for row in board]  # O (깊은 복사)
board_copy = board[:]                    # X (얕은 복사)

# 2. 재귀 깊이 — 맨 위에 선언
sys.setrecursionlimit(10**6)

# 3. 정수 나눗셈
5 // 2  # = 2 (몫)
5 / 2   # = 2.5 (실수)
```

### 자주 나오는 키워드 → 유형

| 키워드 | 유형 |
|--------|------|
| 시뮬레이션, 이동, 회전 | 시뮬레이션 패턴 |
| 최단거리, 최소 이동 | BFS |
| 모든 경우, 선택 | 완전탐색 |
| 낙하, 중력, 폭발 | 낙하 시뮬레이션 |
| 범위 내, 십자 | 영향 범위 패턴 |
| 연결된, 영역 | DFS/BFS 영역 카운트 |
