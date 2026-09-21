---
layout: study-chapter
title: "2.1 Fusing two estimates"
description: "Kalman gain from scratch, by problems."
importance: 2.1
category: SLAM
series: slam_history
permalink: /study/slam/history/02-1-fusing-estimates/
---

> **Goal:** compute a Kalman gain by hand and explain why it is shaped the way it is.

The robot predicts its position; a sensor measures it. They disagree. Which do you believe?

---

### Problem 1 — no maths

Two estimates of the same position:

```text
prediction    2.0 m    "give or take 0.5 m"
measurement   2.4 m    "give or take 0.2 m"
```

Pick one number as your best guess. Is it 2.0, 2.4, 2.2, or something else — and closer to which?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

Somewhere between, **closer to 2.4**, because the measurement claims to be more precise.
2.2 (the plain average) would treat both as equally trustworthy, which they are not.

The rest of this page is just making "closer to the more confident one" exact.

</details>

### Problem 2 — how much more confident?

"Give or take" is a standard deviation $\sigma$. Filters work with the **variance** $\sigma^2$.
Write both variances. How many times smaller is the measurement's?

<details class="study-answer" markdown="1">
<summary>Hint</summary>

Square each "give or take".

</details>

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$P = 0.5^2 = 0.25$ (prediction), $R = 0.2^2 = 0.04$ (measurement). $0.25/0.04 = 6.25$ times smaller.

$\sigma$ was only 2.5× smaller, but the variance is 6.25× smaller — **variances square the gap**.
From here on, $P$ = prediction variance, $R$ = measurement variance.

</details>

### Problem 3 — the rule

The filter moves a fraction $K$ of the way from prediction to measurement:

$$
\hat x = 2.0 + K\,(2.4 - 2.0), \qquad K = \frac{P}{P+R}
$$

Compute $K$.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$K = 0.25 / 0.29 \approx 0.862$.

Read it as **"move 86% of the way"**. $K$ is called the **Kalman gain**.
$P+R$ is all the uncertainty in the room; $P$ is the prediction's share of it.
The more of the doubt is yours, the further you should move towards the other opinion.

</details>

### Problem 4 — the answer

Finish the calculation. Does it agree with your guess in Problem 1?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$\hat x = 2.0 + 0.862 \times 0.4 \approx 2.345$. Between the two, much nearer 2.4 — exactly the intuition, now with a number.

$(2.4 - 2.0)$ has a name: the **innovation**, the disagreement between what you expected and what you saw.

</details>

### Problem 5 — equal trust

Keep $P = 0.25$ and change the sensor to $R = 0.25$. Compute $K$ and $\hat x$.

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$K = 0.25/0.50 = 0.5$, $\hat x = 2.2$ — the plain average.

**The plain average is the special case $P = R$.** Problem 1's "2.2" was secretly assuming equal trust.

</details>

### Problem 6 — a bad sensor

Now $R = 4.0$ (give or take 2 m). $K$ and $\hat x$?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$K = 0.25/4.25 \approx 0.059$, $\hat x \approx 2.024$. The filter barely moves.

```text
    R       K       x̂       who wins
  0.04    0.86    2.345    measurement
  0.25    0.50    2.200    tie
  4.00    0.06    2.024    prediction
```

Nobody tuned these by hand. **State how much you trust each source, and the blend follows.**

</details>

### Problem 7 — uncertainty afterwards

The new variance is $P^{+} = (1-K)\,P$. Compute it for $R = 0.04$.
Is it bigger or smaller than $P$? Than $R$?

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$P^{+} = (1 - 0.862) \times 0.25 \approx 0.034$ — **smaller than both** 0.25 and 0.04.

Combining two independent opinions gives more certainty than either alone.
That is the entire reason to fuse sensors.

</details>

### Problem 8 — challenge

Show that $P^{+} = \dfrac{PR}{P+R}$, and use it to argue that $P^{+}$ is always smaller than both $P$ and $R$.

<details class="study-answer" markdown="1">
<summary>Hint</summary>

Substitute $K = P/(P+R)$ into $(1-K)P$. Then compare $PR/(P+R)$ with $PR/P$.

</details>

<details class="study-answer" markdown="1">
<summary>Show answer</summary>

$1 - K = R/(P+R)$, so $P^{+} = PR/(P+R)$.
Since $P + R > P$, $\;PR/(P+R) < PR/P = R$. By symmetry it is also $< P$.

</details>

---

**You can now:** compute $K = P/(P+R)$, the update $\hat x = x + K(z - x)$, and $P^{+} = (1-K)P$ — and explain each without the formula.
