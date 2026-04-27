---
layout: post
title: Pydantic naming convention guideline
date: 2026-04-21
description: Class naming protocols for collaborating on code 
tags: team pydantic pep8
featured: true
categories: guide
toc:
  sidebar: left
mermaid:
  enabled: true
---

## 1. Pydantic 일반 설명

Pydantic은 Python에서 **데이터의 타입과 구조를 검증**하는 라이브러리다.
클래스를 정의하면 그 클래스의 객체(object)로 데이터를 주고받고, 타입이 맞지 않으면 자동으로 에러를 낸다.

```python
from pydantic import BaseModel

class CreateOrdInput(BaseModel):
    user_id: int
    quantity: int

data = CreateOrdInput(user_id=1, quantity=3)      # OK
data = CreateOrdInput(user_id="abc", quantity=3)  # ValidationError 자동 발생
```

### 1-1. Naming Style 비교

| 이름 | 규칙 | 예시 |
| --- | --- | --- |
| **PascalCase** | 모든 단어 첫 글자 대문자 | `CreateTaskInput` |
| **camelCase** | 첫 단어만 소문자, 나머지 대문자 | `createTaskInput` |
| **snake_case** | 소문자 + 언더바 | `create_task_input` |
| **kebab-case** | 소문자 + 하이픈 | `create-task-input` |
{: .table .table-sm}

```
create task input  →  (PascalCase) CreateTaskInput
                   →  (camelCase)  createTaskInput
                   →  (snake_case) create_task_input
                   →  (kebab-case) create-task-input
```

Python [PEP 8](https://peps.python.org/pep-0008/) 표준:
- **클래스명** → PascalCase (`class CreateTaskInput`)
- **함수/변수명** → snake_case (`def create_task()`)

### 1-2. 모델 클래스 Suffix 규칙

데이터의 역할에 따라 접미사(Suffix)를 붙여 구분한다.

| Suffix | 용도 | 패턴 | 예시 |
| --- | --- | --- | --- |
| `Input` | 함수에 넣는 입력값 | `[동사][명사]Input` | `CreateOrdInput` |
| `Record` | DB 행(row) 1:1 표현 | `[테이블명]Record` | `OrdRecord` |
| `Result` | 연산 결과 (DB 아님) | `[동사][명사]Result` | `AllocateTaskResult` |
| `Event` | 이벤트/로그 페이로드 | `[명사]Event` | `TransErrorEvent` |
{: .table .table-sm}

### 1-3. 필드명 — `snake_case`

```python
class CreateOrdInput(BaseModel):
    user_id: int           # snake_case
    prod_opt_id: int
    quantity: int
```

### 1-4. Validator — `validate_<field_name>`

```python
class CreateOrdInput(BaseModel):
    quantity: int

    @field_validator("quantity")
    @classmethod
    def validate_quantity(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("quantity must be positive")
        return v
```

<details>
<summary><code>@field_validator</code> 와 <code>@classmethod</code> 가 왜 같이 쓰이나?</summary>

<br>

둘의 역할이 다르다.

<ul>
  <li><strong><code>@field_validator("quantity")</code></strong> — Pydantic에게: <code>quantity</code> 값이 들어올 때 이 함수를 먼저 실행해라</li>
  <li><strong><code>@classmethod</code></strong> — Python에게: 객체 생성 전이라 <code>self</code>가 없으니, 클래스 자체(<code>cls</code>)로 호출할 수 있다</li>
</ul>

{% highlight python %}
@field_validator("quantity")   # Pydantic에게: "quantity 들어오면 이 함수 써라"
@classmethod                   # Python에게:   "객체 없이 클래스로 호출할 수 있다"
def validate_quantity(cls, v):
    if v <= 0:
        raise ValueError("...")
    return v
{% endhighlight %}

validator는 객체가 만들어지기 <strong>전에</strong> 값을 검사해야 하므로, 인스턴스(<code>self</code>) 대신 클래스(<code>cls</code>)를 받는 <code>@classmethod</code>를 쓴다.<br>
단, <code>cls</code>는 함수 안에서 실제로 쓸 일이 거의 없다 — Pydantic v2 스펙이 요구해서 형식상 붙이는 것이고, 실제로 신경 쓸 건 검사할 값인 <code>v</code>뿐이다.

<br><br>

실행 흐름:

{% highlight text %}
CreateOrdInput(quantity=-1)
      ↓
Pydantic: "quantity 필드 들어왔다"
      ↓
@field_validator("quantity") 찾아서 실행
      ↓
validate_quantity(cls, v=-1) 호출
      ↓
raise ValueError → ValidationError 발생
{% endhighlight %}

</details>

### 1-5. model_config (Pydantic v2)

```python
class OrdRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # ORM 객체 → Pydantic 변환 시 필요
```

---

### 1-6. 자주 쓰이는 Pydantic 데코레이터 (`@`)

<details>
<summary><strong>@field_validator</strong> — 특정 필드 하나 검증</summary>

<br>
<strong>언제:</strong> 특정 필드 값이 조건을 만족하는지 검사할 때<br><br>

{% highlight python %}
from pydantic import BaseModel, field_validator

class CreateOrdInput(BaseModel):
    quantity: int

    @field_validator("quantity")
    @classmethod
    def validate_quantity(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("quantity must be positive")
        return v

CreateOrdInput(quantity=-1)  # ValidationError 발생
CreateOrdInput(quantity=3)   # OK
{% endhighlight %}

</details>

---

<details>
<summary><strong>@model_validator</strong> — 필드 간 관계 검증</summary>

<br>
<strong>언제:</strong> 두 필드를 같이 봐야 할 때 (한 필드만으로는 판단 불가)<br><br>

{% highlight python %}
from pydantic import BaseModel, model_validator

class ShipmentOrdInput(BaseModel):
    quantity: int
    available_stock: int

    @model_validator(mode="after")
    def validate_stock(self) -> "ShipmentOrdInput":
        if self.quantity > self.available_stock:
            raise ValueError("수량이 재고보다 많습니다")
        return self

ShipmentOrdInput(quantity=10, available_stock=5)   # ValidationError
ShipmentOrdInput(quantity=3,  available_stock=5)   # OK
{% endhighlight %}

<br>
<code>mode="after"</code> : 객체 생성 <strong>후</strong> 실행 → <code>self</code>로 모든 필드 접근 가능<br>
<code>mode="before"</code> : 객체 생성 <strong>전</strong> 실행 → raw dict 상태

</details>

---

<details>
<summary><strong>Field()</strong> — 함수 없이 간단한 제약</summary>

<br>
<strong>언제:</strong> 범위·길이처럼 단순한 조건은 <code>@field_validator</code> 없이 한 줄로 처리<br><br>

{% highlight python %}
from pydantic import BaseModel, Field

class CreateOrdInput(BaseModel):
    quantity:  int = Field(gt=0)                      # 0 초과
    user_id:   int = Field(ge=1)                      # 1 이상
    prod_nm:   str = Field(min_length=1, max_length=50)
{% endhighlight %}

<br>

<table class="table table-sm">
  <thead><tr><th>옵션</th><th>의미</th></tr></thead>
  <tbody>
    <tr><td><code>gt</code></td><td>초과 (greater than)</td></tr>
    <tr><td><code>ge</code></td><td>이상 (greater or equal)</td></tr>
    <tr><td><code>lt</code></td><td>미만</td></tr>
    <tr><td><code>le</code></td><td>이하</td></tr>
    <tr><td><code>min_length</code> / <code>max_length</code></td><td>문자열 길이</td></tr>
    <tr><td><code>pattern</code></td><td>정규식</td></tr>
  </tbody>
</table>

</details>

---

<details>
<summary><strong>@computed_field</strong> — DB에 없는 계산값 노출</summary>

<br>
<strong>언제:</strong> DB에 저장하진 않지만 응답에 포함시키고 싶은 파생 값이 있을 때<br><br>

{% highlight python %}
from pydantic import BaseModel, computed_field

class OrdRecord(BaseModel):
    unit_price: int
    quantity:   int

    @computed_field
    @property
    def total_price(self) -> int:
        return self.unit_price * self.quantity

record = OrdRecord(unit_price=1000, quantity=3)
print(record.total_price)   # 3000
print(record.model_dump())  # {'unit_price': 1000, 'quantity': 3, 'total_price': 3000}
{% endhighlight %}

</details>

---

<details>
<summary><strong>@field_serializer</strong> — JSON 출력 형식 커스텀</summary>

<br>
<strong>언제:</strong> DB에서 꺼낸 값과 API 응답으로 내보낼 형식이 다를 때 (특히 날짜·enum)<br><br>

{% highlight python %}
from pydantic import BaseModel, field_serializer
from datetime import datetime

class OrdRecord(BaseModel):
    created_at: datetime

    @field_serializer("created_at")
    def serialize_created_at(self, v: datetime) -> str:
        return v.strftime("%Y-%m-%d %H:%M")

record = OrdRecord(created_at=datetime(2026, 4, 28, 9, 0))
print(record.model_dump())
# {'created_at': '2026-04-28 09:00'}   ← datetime 객체가 아닌 문자열로 출력
{% endhighlight %}

</details>

---

<details>
<summary><strong>한눈에 비교</strong></summary>

<br>

<table class="table table-sm">
  <thead><tr><th>데코레이터</th><th>언제 쓰나</th></tr></thead>
  <tbody>
    <tr><td><code>@field_validator</code></td><td>특정 필드 하나 검증</td></tr>
    <tr><td><code>@model_validator</code></td><td>필드 간 관계 검증</td></tr>
    <tr><td><code>Field()</code></td><td>간단한 범위·길이 제약 (함수 불필요)</td></tr>
    <tr><td><code>@computed_field</code></td><td>DB에 없는 계산값 노출</td></tr>
    <tr><td><code>@field_serializer</code></td><td>JSON 출력 형식 커스텀</td></tr>
  </tbody>
</table>

</details>

---

## 2. 우리 테이블 종류 요약

| 테이블 종류 | 언제 쓰는가 | Pydantic Suffix | 해당 테이블 |
| --- | --- | --- | --- |
| **Master** | 기준 데이터. 잘 안 변함 | `Record` | `user_account`, `category`, `product`, `zone`, `equip`, `trans` |
| **Operational (txn)** | 작업 발생/진행 기록 | 생성 입력 → `Input` / DB 조회 → `Record` | `ord`, `equip_task_txn`, `trans_task_txn`, `insp_task_txn` |
| **State (stat)** | 현재 상태 스냅샷. 빠른 조회 | `Record` | `ord_stat`, `equip_stat`, `trans_stat`, `chg_loc_stat` |
| **Log** | 행동/이벤트/에러 기록 | 행동·이벤트 → `Event` / 데이터·에러 → `Record` | `log_action_*`, `log_event`, `log_err_*` |
{: .table .table-sm}

---

## 3. 우리 ERD → Pydantic 클래스명 매핑

### Master tables → `[테이블명]Record`

| DB 테이블 | Pydantic 클래스 |
| --- | --- |
| `user_account` | `UserAccountRecord` |
| `category` | `CategoryRecord` |
| `product` | `ProductRecord` |
| `product_option` | `ProductOptionRecord` |
| `zone` | `ZoneRecord` |
| `equip` | `EquipRecord` |
| `trans` | `TransRecord` |
{: .table .table-sm}

### Operational tables → `Input` / `Record`

| DB 테이블 | 쓰임 | Pydantic 클래스 |
| --- | --- | --- |
| `ord` | 주문 생성 입력 | `CreateOrdInput` |
| `ord` | DB 행 조회 | `OrdRecord` |
| `equip_task_txn` | 태스크 할당 입력 | `AssignEquipTaskInput` |
| `equip_task_txn` | DB 행 조회 | `EquipTaskTxnRecord` |
| `trans_task_txn` | 태스크 할당 입력 | `AssignTransTaskInput` |
| `trans_task_txn` | DB 행 조회 | `TransTaskTxnRecord` |
| `insp_task_txn` | 검사 생성 입력 | `CreateInspTaskInput` |
| `insp_task_txn` | DB 행 조회 | `InspTaskTxnRecord` |
{: .table .table-sm}

### State tables → `[테이블명]Record`

| DB 테이블 | Pydantic 클래스 |
| --- | --- |
| `ord_stat` | `OrdStatRecord` |
| `item_stat` | `ItemStatRecord` |
| `equip_stat` | `EquipStatRecord` |
| `trans_stat` | `TransStatRecord` |
| `chg_loc_stat` | `ChgLocStatRecord` |
| `strg_loc_stat` | `StrgLocStatRecord` |
| `ship_loc_stat` | `ShipLocStatRecord` |
{: .table .table-sm}

### Log tables → `Event` / `Record`

| DB 테이블 | Pydantic 클래스 |
| --- | --- |
| `log_action_user` | `UserActionEvent` |
| `log_action_operator_handoff_acks` | `OperatorHandoffEvent` |
| `log_action_operator_rfid_scan` | `RfidScanEvent` |
| `log_action_admin` | `AdminActionEvent` |
| `log_event` | `SystemEvent` |
| `log_data_equip` | `EquipDataRecord` |
| `log_data_trans` | `TransDataRecord` |
| `log_err_equip` | `EquipErrorRecord` |
| `log_err_trans` | `TransErrorRecord` |
{: .table .table-sm}

### 함수 흐름 예시

Pydantic 클래스는 함수 간 데이터를 주고받는 "타입이 있는 그릇" 역할을 한다.

```
[외부 입력]
    │
    ▼
CreateOrdInput              ← Pydantic이 타입/필드 자동 검증
    │
    ▼
def create_ord()            ← DB INSERT
    │
    ▼
OrdRecord                   ← DB 행 반환
    │
    ▼
def assign_equip_task()     ← AssignEquipTaskInput 생성 후 DB INSERT
    │
    ▼
EquipTaskTxnRecord          ← 완료된 태스크 행 반환
    │
    ▼ (에러 발생 시)
TransErrorEvent  →  def log_trans_error()  →  log_err_trans 저장
```

```python
# 주문 생성
def create_ord(data: CreateOrdInput) -> OrdRecord: ...

# 설비 태스크 할당
def assign_equip_task(data: AssignEquipTaskInput) -> EquipTaskTxnRecord: ...

# 상태 조회
def get_trans_stat(trans_id: int) -> TransStatRecord: ...

# 에러 로그 기록
def log_trans_error(event: TransErrorEvent) -> None: ...
```
