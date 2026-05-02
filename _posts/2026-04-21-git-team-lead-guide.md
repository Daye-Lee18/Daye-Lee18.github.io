---
layout: post
title: Git 팀 사용 가이드 - 팀장
date: 2026-04-21
description: 팀장이 관리하는 Git 흐름
tags: git team
categories: guide
toc:
  sidebar: left
---

## 이 문서에서 보는 것

- 프로젝트 시작 설정
- PR 리뷰
- squash merge
- 버전 태깅과 제출

> 팀원용 기본 흐름은 [팀원 문서]({% post_url 2026-04-21-git-team-member-guide %})에서 먼저 보고 오면 좋습니다.
{: .block-tip }

---

## PHASE 1 - 프로젝트 시작 (팀장, 1회만)

### Step 1. 두 remote 연결

```bash
git remote add origin [우리팀 레포 URL]
git remote add submit [학원 레포 URL]
```

### Step 2. Branch Protection 설정

GitHub 레포 → **Settings** → **Rules → Rulesets** → **New branch ruleset**

#### Ruleset Name

```text
main-protection
```

#### Enforcement status

`Active` 유지

#### Bypass list

상황에 따라 선택합니다.

- admin 권한을 회수하는 경우: 비워두기
- admin 권한을 유지하는 경우: 본인 계정 또는 Repository admin 추가

#### Target branches

**Add target** 클릭 → **Include by pattern** 선택 → `main` 입력 후 Add

#### Branch rules

| 항목 | 설정 | 이유 |
| --- | --- | --- |
| Restrict deletions | ✅ 체크 | main 브랜치 실수로 삭제 방지 |
| Require a pull request before merging | ✅ 체크 | main 직접 push 금지, PR 강제 |
| Required approvals | `1` | 팀장 1명 승인 필수 |
| Dismiss stale pull request approvals | ✅ 권장 | 코드 수정 시 재승인 요구 |
| Block force pushes | ✅ 체크 | `git push --force` 방지 |
{: .table .table-sm .table-striped}

### Step 3. PR 템플릿 추가

`.github/PULL_REQUEST_TEMPLATE.md` 파일을 만들어 main에 push합니다.

---

## PHASE 3 - 리뷰 & 병합 (팀장)

### Step 1. 코드 리뷰

- PR → Files changed 탭 → 라인별 코멘트
- 수정 요청 시: `Request changes`
- 승인 시: `Approve`

#### PR 변경사항이 너무 많을 때

```bash
git fetch origin
git branch -a
```

원격 브랜치가 보이면 로컬에서 확인용 브랜치를 만들어 볼 수 있습니다.

```bash
git checkout -b review/casting-factory-from-kim origin/import/casting-factory-from-kim
git checkout main
```

특정 파일만 확인할 때:

```bash
git diff main...review/casting-factory-from-kim -- path/to/file.py
```

### Step 2. Squash merge

여러 커밋을 하나로 압축해서 main에 넣는 방식입니다.

```text
feat/login 브랜치: A -> B -> C -> D -> E
main:               ... -> F
```

| 방식 | main 히스토리 | 언제 쓰나 |
| --- | --- | --- |
| Create a merge commit | 브랜치 커밋 전부 + merge 커밋 추가 | 히스토리 전부 보존할 때 |
| Squash and merge | 커밋 1개로 압축 | 팀 작업, main 히스토리 깔끔하게 |
| Rebase and merge | 브랜치 커밋 전부 | 선형 히스토리 유지할 때 |
{: .table .table-sm .table-striped}

### Step 3. 브랜치 삭제

```bash
git push origin --delete feat/기능명
git branch -d feat/기능명
git branch -D feat/기능명
```

> squash merge 후에는 feat 브랜치 역할이 끝납니다.
{: .block-warning }

---

## PHASE 4 - 버전 태깅 (팀장, 마일스톤마다)

```bash
git checkout main
git pull origin main
git tag -a v0.1 -m "v0.1"
git push origin v0.1
```

| 태그 | 기준 |
| --- | --- |
| v0.1 | 핵심 기능 첫 동작 |
| v0.2 | 기능 추가 완료 |
| v1.0 | 최종 제출 |
{: .table .table-sm .table-striped}

---

## PHASE 5 - 학원 레포 최종 제출 (팀장만, 제출 시 1회)

```bash
git checkout main
git pull origin main
git push submit main
```

> `submit` remote는 제출 목적 전용입니다.
{: .block-warning }

---

## 핵심 원칙 요약

| 원칙 | 이유 |
| --- | --- |
| main 직접 push 금지 | branch protection으로 강제됨 |
| 작업 시작 전 항상 `git fetch` | 충돌 방지 |
| PR 단위를 작게 유지 | 리뷰 부담 감소 |
| Squash merge 사용 | main 히스토리 가독성 유지 |
| submit은 제출용만 | origin과 용도 분리 |
{: .table .table-sm .table-striped}

---

## GitHub Actions 탭 용어

**Workflow** - 자동화 작업 정의 파일

**Event** - workflow를 트리거한 원인

```yaml
on:
  push:
    branches: [main]
  pull_request:
  workflow_dispatch:
```

**Status** - 실행 결과

**Branch** - 어느 브랜치에서 트리거됐는지

**Actor** - 누가 트리거했는지
