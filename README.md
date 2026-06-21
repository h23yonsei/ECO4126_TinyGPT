# TinyGPT: Transformer 언어 모델 구현 (ECO4126)

학부생 이승현 (2020142189)

---

## 목차

1. [개요](#1-개요)
2. [프로젝트 구조](#2-프로젝트-구조)
3. [Makemore와 학습 프레임워크](#3-makemore와-학습-프레임워크)
4. [단계별 학습 내용](#4-단계별-학습-내용)
   - [Notebook 01: Bigram](#41-notebook-01-bigram--가장-단순한-언어-모델)
   - [Notebook 02: MLP + Embedding](#42-notebook-02-mlp--embedding--분산-표현의-도입)
   - [Notebook 03: Domain Shift](#43-notebook-03-domain-shift--모델의-일반화-능력-검증)
   - [Notebook 04: Positional Embedding](#44-notebook-04-positional-embedding--병렬-sequence-처리)
   - [Notebook 05: Self-Attention](#45-notebook-05-self-attention--동적-가중치-메커니즘)
   - [Notebook 06: Transformer Block](#46-notebook-06-transformer-block--완성된-구조)
5. [TinyGPT 최종 구현 — 개인 프로젝트](#5-tinygpt-최종-구현--개인-프로젝트)
6. [전체 학습 결과 비교](#6-전체-학습-결과-비교)
7. [핵심 개념 정리](#7-핵심-개념-정리)

---

## 1. 개요

이 프로젝트는 Andrej Karpathy의 `makemore` 프레임워크를 기반으로, Bigram 통계 모델에서 출발해 Multi-Head Self-Attention을 갖춘 Transformer 언어 모델까지 단계적으로 구현한 기록입니다. 각 Notebook은 이전 단계의 한계를 하나씩 극복하며 아키텍처를 발전시키는 구조로 설계되어 있습니다.

모든 Notebook은 `device = "cuda" if torch.cuda.is_available() else "cpu"` 로 GPU/CPU를 자동 선택합니다. GPU 없이 CPU 환경에서도 실행 가능하지만 학습 속도는 크게 느려집니다.

| 단계 | 핵심 개념 | 모델 |
| --- | --- | --- |
| 01 | 토큰화 + 통계적 생성 | Bigram |
| 02 | 분산 표현 (Dense Vector) | MLP + Embedding |
| 03 | 도메인 일반화 | MLP (다른 데이터셋) |
| 04 | 병렬 Sequence 처리 | Positional Embedding LM |
| 05 | 동적 Context 가중치 | Single-Head Self-Attention |
| 06 | 규제화 + 깊이 적층 | Multi-Head Transformer |

---

## 2. 프로젝트 구조

```text
ECO4126_TinyGPT/
├── notebook_01.ipynb              # 1단계: Bigram 언어 모델
├── notebook_02.ipynb              # 2단계: MLP + Embedding
├── notebook_03.ipynb              # 3단계: Domain Shift 실험
├── notebook_04.ipynb              # 4단계: Positional Embedding + Sequence 처리
├── notebook_05.ipynb              # 5단계: Single-Head Self-Attention
├── notebook_06.ipynb              # 6단계: Multi-Head Transformer Block
├── notebook_06_h23yonsei.ipynb    # 최종: 내가 직접 구현한 TinyGPT (Oz 데이터셋)
├── input.txt                      # 학습 데이터: The Wonderful Wizard of Oz
└── README.md
```

---

## 3. Makemore와 학습 프레임워크

`makemore`는 Karpathy가 Character-level 언어 모델을 가르치기 위해 설계한 교육용 프레임워크입니다. 이 프로젝트에서 `makemore`는 다음 세 가지 핵심 역할을 담당했습니다.

### 3.1. Character-level Tokenizer 및 Vocabulary 구축

텍스트의 고유 문자를 정수 Index로 변환하는 인코딩/디코딩 시스템:

```python
chars = sorted(list(set(text)))
stoi = { ch: i for i, ch in enumerate(chars) }   # string → int
itos = { i: ch for ch, i in stoi.items() }        # int → string
```

### 3.2. Autoregressive 생성 루프

학습된 모델에서 현재 Context를 받아 다음 Token의 확률 분포를 도출하고, 그로부터 샘플링:

```python
logits = model(context)          # (B, T, vocab_size) — NB04 이후 Sequence 모델
logits = logits[:, -1, :]        # 마지막 위치의 예측만 사용 → (B, vocab_size)
probs = F.softmax(logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
context = torch.cat([context[:, 1:], next_token], dim=1)  # 슬라이딩 윈도우
```

### 3.3. Block Size 기반 Context 실험

`block_size` 하이퍼파라미터 하나로 모델이 참조하는 과거 Context 길이를 제어. 이 변수를 1 → 3 → 16 → 32 → 64로 점진적으로 키우며 표현 능력의 변화를 관찰했습니다.

---

## 4. 단계별 학습 내용

---

### 4.1. Notebook 01: Bigram — 가장 단순한 언어 모델

| 항목 | 값 |
| --- | --- |
| block_size | 1 (이전 Token 1개만 참조) |
| 입력 → 출력 | 문자 1개 → 다음 문자 1개 |
| Embedding | One-Hot Encoding |
| Loss | Cross Entropy |

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x):
        x = x.view(-1)                             # (B, 1) → (B,)
        logits = self.token_embedding_table(x)     # (B, vocab_size)
        return logits
```

학습 포인트: One-Hot Encoding은 각 Token을 독립된 이진 벡터로 표현하기 때문에 Token 간의 유사성을 전혀 학습할 수 없다는 것을 확인했습니다. Bigram은 이전 Token 하나만 보기 때문에 "the big" 이후에 무엇이 와야 하는지는 알 수 없으며, 이 단계에서 장거리 의존성(Long-range Dependency) 문제를 처음으로 실감했습니다.

---

### 4.2. Notebook 02: MLP + Embedding — 분산 표현의 도입

| 항목 | 값 |
| --- | --- |
| block_size | 3 (이전 Token 3개 참조) |
| Embedding 차원 | 10 |
| 은닉층 차원 | 200 |
| Activation | Tanh |

```python
class MLPCharacterModel(nn.Module):
    def __init__(self, vocab_size, block_size, emb_dim=10, hidden_dim=200):
        self.net = nn.Sequential(
            nn.Embedding(vocab_size, emb_dim),
            nn.Flatten(),
            nn.Linear(block_size * emb_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, vocab_size),
        )
```

학습 포인트: `nn.Embedding`은 Token을 연속적인 Dense Vector로 매핑하며, 유사한 Token이 벡터 공간에서 가까이 위치하도록 학습된다는 것을 이해했습니다. 다만 MLP는 입력 크기가 `block_size × emb_dim`으로 고정되어 있어 Context 길이를 유연하게 늘릴 수 없다는 구조적 한계가 있음을 직접 확인했습니다.

---

### 4.3. Notebook 03: Domain Shift — 모델의 일반화 능력 검증

| 데이터셋 | 어휘 크기 | block_size | Epoch | 최종 Loss |
| --- | --- | --- | --- | --- |
| 이름(Names) | 27 | 3 | 30 | ~2.26 |
| Shakespeare | 65 | 16 | 10 | 1.8584 |

학습 포인트: 동일한 MLP 클래스 구조를 유지하면서 데이터셋과 하이퍼파라미터(emb_dim 10→64, hidden 200→256)를 교체했을 때 모델이 새로운 분포를 학습할 수 있다는 것을 직접 확인했습니다. 모델 구조와 데이터는 별개로 분리될 수 있으며, Vocabulary와 Tokenizer는 데이터셋마다 새로 구축해야 한다는 점을 이해했습니다.

---

### 4.4. Notebook 04: Positional Embedding — 병렬 Sequence 처리

| 항목 | 이전 (Notebook 02~03) | 이후 (Notebook 04~) |
| --- | --- | --- |
| 입력 shape | (B, T) | (B, T) |
| 레이블(y) shape | (B,) — 다음 Token 1개 | (B, T) — 1칸씩 밀린 시퀀스 |
| 출력 shape | (B, vocab_size) | (B, T, vocab_size) |
| 학습 방식 | 다음 Token 1개 예측 | 모든 위치의 다음 Token 동시 예측 |
| block_size | 3~16 | 32 (NB06: 64) |

```python
class NextTokenDataset(Dataset):
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]  # 1칸씩 밀린 시퀀스
        return x, y

class TinySequenceLM(nn.Module):
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        tok = self.token_embedding(x)                    # (B, T, emb_dim)
        pos = self.position_embedding(pos)[None]         # (1, T, emb_dim)
        h = tok + pos                                    # 위치 정보 주입
        logits = self.lm_head(h)                         # (B, T, vocab_size)
```

학습 포인트: 이 단계가 시리즈 전체에서 가장 중요한 구조적 전환점이었습니다. Forward pass 한 번으로 T개의 위치에서 T개의 예측을 동시에 계산할 수 있는 이유는, 각 위치가 `tok[t] + pos[t]`만으로 독립적으로 처리되기 때문입니다 — 위치 간 정보 교환이 없으므로 한 번의 행렬 연산으로 T개의 예측을 병렬 계산할 수 있습니다. Causal Mask(미래 Token 차단)는 다음 단계인 NB05의 Self-Attention에서 비로소 도입됩니다. Positional Embedding 없이는 모델이 입력의 순서 정보를 알 수 없어 Bag-of-Words와 다를 바가 없다는 점도 확인했습니다.

---

### 4.5. Notebook 05: Self-Attention — 동적 가중치 메커니즘

| 항목 | 설명 |
| --- | --- |
| 메커니즘 | Scaled Dot-Product Attention |
| 입력 → Q, K, V | 각각 Linear Projection으로 생성 |
| Causal Mask | `torch.tril()`로 미래 Token 차단 |
| 스케일 인자 | $1/\sqrt{C}$ — Softmax 포화 방지 |

```python
class SingleHeadSelfAttention(nn.Module):
    def forward(self, x):
        B, T, C = x.shape                              # C = emb_dim
        k = self.key(x)                                # (B, T, C)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * (C ** -0.5)   # Scaled Dot-Product
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # Causal Mask
        wei = F.softmax(wei, dim=-1)

        out = wei @ v                                  # (B, T, C)
        return out
```

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

학습 포인트: MLP의 가중치는 학습 후 고정되지만, Self-Attention의 가중치 $QK^T$는 입력 자체에 의해 매 Forward마다 동적으로 결정된다는 것을 이해했습니다. Causal Mask가 없으면 모델이 미래 Token을 참조해 정답을 맞혀버리기 때문에, Autoregressive 생성을 위해 반드시 필요하다는 점도 확인했습니다.

---

### 4.6. Notebook 06: Transformer Block — 완성된 구조

| 컴포넌트 | 역할 |
| --- | --- |
| Multi-Head Attention | 4개의 Head가 서로 다른 종류의 관계를 병렬 학습 |
| Feed-Forward Network | $d_{model} \to 4d_{model} \to d_{model}$ (비선형 변환) |
| Residual Connection | `x = x + sublayer(x)` — 기울기 흐름 안정화 |
| LayerNorm (Pre-LN) | Attention/FFN 입력 전 정규화 |
| Dropout | 과적합 방지 |

```python
class Block(nn.Module):
    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # Pre-LN → Multi-Head Attention → Residual
        x = x + self.ffwd(self.ln2(x))  # Pre-LN → Feed-Forward → Residual
        return x

class TinyGPT(nn.Module):
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        tok = self.token_embedding(x)
        pos = self.position_embedding(pos)[None]
        h = tok + pos
        h = self.blocks(h)    # 4개의 Block을 순차 통과
        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits
```

학습 포인트: Multi-Head Attention은 각 Head가 서로 다른 종류의 관계(문법적 의존성, 의미적 유사성 등)를 동시에 독립적으로 학습할 수 있어 Single-Head보다 표현력이 풍부하다는 것을 이해했습니다. Residual Connection 없이 Block을 여러 개 쌓으면 기울기 소실이 발생하고, Pre-LN은 Post-LN보다 학습 초기 안정성이 높다는 점도 확인했습니다.

---

## 5. TinyGPT 최종 구현 — 개인 프로젝트

`notebook_06_h23yonsei.ipynb`는 Notebook 06의 Transformer 구조를 기반으로, 학습 데이터를 Shakespeare에서 **The Wonderful Wizard of Oz**로 교체하여 직접 구현한 버전입니다.

### 5.1. 학습 데이터: `input.txt` 준비

| 항목 | 내용 |
| --- | --- |
| 출처 | [Project Gutenberg — *The Wonderful Wizard of Oz* by L. Frank Baum](https://www.gutenberg.org/ebooks/55) |
| 다운로드 형식 | Plain Text UTF-8 (`.txt`) |
| 최종 크기 | 4,582줄 / 약 210 KB |

Project Gutenberg에서 받은 원본 `.txt` 파일에는 본문 앞뒤로 법적 고지, 라이선스 텍스트, 메타데이터 블록이 포함되어 있습니다. 이를 그대로 학습에 사용하면 모델이 저작권 고지문 패턴까지 학습하므로, 다음 두 가지를 수동으로 제거했습니다:

1. **머리말 제거** — 파일 첫 줄부터 `*** START OF THE PROJECT GUTENBERG EBOOK ***` 구분선까지의 Gutenberg 고지문 전체
2. **꼬리말 제거** — `*** END OF THE PROJECT GUTENBERG EBOOK ***` 구분선 이후의 라이선스 및 저작권 블록 전체

편집 결과, `input.txt`는 제목 "The Wonderful Wizard of Oz"로 시작해 Chapter XXIV 마지막 줄("And oh, Aunt Em! I'm so glad to be at home again!")로 끝나는 순수 소설 본문만 담고 있습니다.

### 5.2. 아키텍처 하이퍼파라미터

| 파라미터 | 값 | 설명 |
| --- | --- | --- |
| `vocab_size` | 68 | Oz 텍스트의 고유 문자 수 |
| `emb_dim` | 128 | Token + Position Embedding 차원 |
| `block_size` | 64 | Context 윈도우 길이 |
| `n_heads` | 4 | Attention Head 수 |
| `head_size` | 32 | 각 Head의 차원 (= 128 / 4) |
| FFN 은닉층 | 512 | Feed-Forward 중간 차원 (= 4 × 128) |
| `n_blocks` | 4 | Transformer Block 적층 수 |
| `dropout` | 0.1 | Regularization 비율 |
| Optimizer | AdamW | lr=3e-4, weight_decay=0.01 (PyTorch 기본값) |
| Batch size | 64 | |
| `max_steps` | 300 | Epoch당 최대 학습 스텝 수 |
| Epochs | 100 | |

### 5.3. 아키텍처 흐름

```text
Input (B, T)
    ↓
Token Embedding (68 → 128)  +  Position Embedding (64 → 128)
    ↓
[Transformer Block × 4]
  각 Block:
    Pre-LayerNorm → Multi-Head Attention (4 heads) → Residual
    Pre-LayerNorm → Feed-Forward (128→512→128) → Residual
    ↓
Final LayerNorm
    ↓
LM Head Linear (128 → 68)
    ↓
Output logits (B, T, 68)
```

### 5.4. 개인 프로젝트 커스터마이징

Notebook 06 기본 구조에서 다음 6가지를 직접 추가·수정했습니다:

| # | 변경 사항 | 위치 | 이유 |
| --- | --- | --- | --- |
| 1 | `ReLU` → `GELU` | `FeedForward` | GPT-2 이후 표준 Activation. 음수 입력을 부드럽게 처리해 Gradient flow 개선 |
| 2 | Weight Tying | `TinyGPT.__init__` | Token Embedding과 LM Head 가중치 공유 — 파라미터 절약 + 성능 향상 |
| 3 | Gradient Clipping | `train_one_epoch` | `clip_grad_norm_(max_norm=1.0)`으로 Gradient 폭발 방지 |
| 4 | Train / Val Split | Dataset 설정 | 80/20 분리로 Overfitting 여부를 Val Loss로 모니터링 **(이 버전부터 도입)** |
| 5 | Cosine LR Scheduler | Training loop | `CosineAnnealingLR`로 학습 후반 수렴 안정화 |
| 6 | Temperature + Top-k Sampling | `sample_gpt` | 생성 다양성과 품질을 파라미터로 조절 가능하게 개선 |

### 5.5. 학습 수렴

| Epoch | Train Loss | Val Loss | 비고 |
| --- | --- | --- | --- |
| 0 | 6.4455 | 2.4399 | 무작위 초기화 기준선 (Epoch 내 300 스텝 평균) |
| 50 | 1.1249 | 0.9833 | 안정적 하강 |
| 99 | **1.0445** | **0.8920** | 최종 수렴 |

### 5.6. 생성 결과 ("Dorothy" 프롬프트, temperature=0.8, top_k=40)

```text
Dorothy sitting the Wicked Witch looked up to kill the Witch of the
East, and when you came out of Dorothy dress the end of the road long
her like a place was of tin, and very and made of my fierce was ready of tin.

Dorothy looked up the road rooms silver shoes. And asked Dorothy ask he Scarecrow and said:

"I don't know," said the Scarecrow. "But I shall never get to to the
others," said the Scarecrow, "but you can I certainly fearing and
the Tin Woodman had at the straw days and said, "let us at it, and called
the Winkies were all lay down and pulled through a balloon. And lived in the
wolves who made the little in a heart, for her grews so fierce and that in her they
would be done it, and woman shall have down not let until you wear around. The road
for all the Winged Monkeys was grew sharp away a good-bye. What came to chase show
and flew air with a small short, and I could get back on with the cyclone road them."

"What was shall go to the Emerald City?" asked the Scarecrow.

They had s
```

Oz 텍스트의 반복적인 서사 구조 덕분에 Shakespeare(Loss 1.3316)보다 훨씬 낮은 Loss로 수렴했으며, 대화 구조와 캐릭터 이름이 자연스럽게 나타나는 것을 확인할 수 있었습니다.

### 5.7. 개정 이력 (v1 → v2)

Colab에서 하이퍼파라미터를 조정하고 재학습한 결과입니다.

| 항목 | v1 (초기) | v2 (현재) | 변경 이유 |
| --- | --- | --- | --- |
| Learning rate | 1e-3 | **3e-4** | AdamW 표준 권장값으로 조정 — 초기 lr이 높아 초반 손실이 불안정했음 |
| Batch size | 32 | **64** | GPU 활용률 향상 및 gradient 노이즈 감소 |
| max_steps/epoch | 없음 (전체) | **300** | 긴 epoch에서 과도한 학습 시간 방지 |
| 초기 Train Loss | 2.5435 | 6.4455 | max_steps 평균 방식 변경으로 초반 높은 값 반영 |
| 최종 Train Loss | 0.8174 | **1.0445** | Val loss 기준으로 비교 시 동등 수준 |
| 최종 Val Loss | (미측정) | **0.8920** | Train/Val 분리 모니터링 추가 |

---

## 6. 전체 학습 결과 비교

| Notebook | 모델 | 데이터셋 | 초기 Loss | 최종 Loss (Train) | 최종 Loss (Val) | Epochs |
| --- | --- | --- | --- | --- | --- | --- |
| 01 | Bigram | Names | ~2.5 | ~2.5 | — | 20 |
| 02 | MLP | Names | 2.3573 | 2.2642 | — | 30 |
| 03 | MLP | Shakespeare | 2.5999 | 1.8584 | — | 10 |
| 04 | Positional LM | Shakespeare | ~3.06 | ~2.46 | — | 100 |
| 05 | Single-Head Attention | Shakespeare | 2.9403 | 2.2439 | — | 100 |
| 06 | TinyGPT | Shakespeare | 2.6609 | 1.3316 | — | 100 |
| **06_h23yonsei** | **TinyGPT (개인 프로젝트)** | **Oz** | **6.4455** | **1.0445** | **0.8920** | **100** |

Notebook 04가 다른 단계보다 Loss가 높게 정체된 이유는, Positional Embedding만 추가했을 뿐 Attention 메커니즘이 없어서 각 위치가 독립적으로 처리되었기 때문입니다. Self-Attention이 도입된 Notebook 05부터 Token 간 상호작용이 시작되고 Loss가 의미 있게 개선됩니다.

---

## 7. 핵심 개념 정리

### 7.1. 필수 PyTorch 함수

| 함수 | 목적 |
| --- | --- |
| `F.one_hot()` | 정수 → One-Hot 벡터 |
| `nn.Embedding()` | 정수 → Dense Vector 매핑 (학습 가능) |
| `F.cross_entropy()` | 분류 Loss 계산 |
| `torch.multinomial()` | 확률 분포에서 샘플링 |
| `torch.tril()` | 하삼각행렬 생성 (Causal Mask용) |
| `masked_fill(-inf)` | Attention Mask 적용 |
| `nn.LayerNorm()` | 채널 방향 정규화 |
| `nn.Dropout()` | 학습 시 임의 뉴런 비활성화 — 과적합 방지 |
| `.transpose(-2, -1)` | Attention Score 계산 시 행렬 전치 |
