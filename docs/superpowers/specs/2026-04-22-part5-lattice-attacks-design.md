# Part 5 — Breaking it for Real: Lattice Attacks from Scratch

- **작성일**: 2026-04-22
- **작성자**: dkkang
- **상태**: Draft (사용자 리뷰 대기)
- **맥락**: 기존 책 `ML-KEM from Scratch` (Parts 1–4 완료, 19 notebooks) 뒤에 덧붙이는 **Part 5**.

## 1. 목적

순수 Python/numpy만으로 **LLL 기저 축소**, **BKZ**, **Kannan embedding 기반 primal 공격**을 밑바닥부터 구현한다. toy-LWE 인스턴스에 실제로 공격을 수행하고, "작은 파라미터는 뚫리지만 ML-KEM은 안 뚫리는" 현상을 직접 체험시킨다. 마지막 노트북에서 lattice-estimator의 수치와 비교해 실제 ML-KEM 파라미터 안전성의 근거를 닫는다.

**학습 성공 기준**: 독자가 "왜 LLL/BKZ는 작은 LWE는 뚫지만 ML-KEM은 못 뚫나"를 Hermite factor·cost estimate 관점에서 한 문단으로 설명할 수 있어야 한다.

## 2. 스코프

### In-Scope

- **LLL 기저 축소** (Lenstra-Lenstra-Lovász, 1982) 직접 구현
- **BKZ** (Schnorr-Euchner block reduction) — block size 2–8 범위
- **SVP enumeration** (BKZ 내부 사용)
- **Kannan embedding** — LWE 인스턴스를 uSVP 격자 문제로 변환
- **Primal attack** — embedding + BKZ + 복원까지 end-to-end
- toy-LWE (Part 2에서 구축한 `pqc_edu.lwe` 재활용)에 대한 공격 실험 (n = 10–40)
- 실제 ML-KEM 파라미터 안전성은 **lattice-estimator 공개 수치 인용**으로 다룸

### Non-Goals

- ❌ fpylll 또는 타 외부 격자 라이브러리 의존
- ❌ Dual attack / hybrid attack 구현 (19번 노트북에서 개관만)
- ❌ Sieve 알고리즘 (GaussSieve, BDGL, NV 등) — BKZ 내부 enumeration에 한정
- ❌ 실제 ML-KEM 파라미터에 primal 공격 실행 (작동 안 함 — 의미 없음)
- ❌ 대용량 block size (9 이상) — 순수 Python enumeration 시간 제약
- ❌ toy-MLWE 인스턴스 구축 (스코프 초과)

## 3. 의사결정 요약

| 항목 | 결정 | 근거 |
|------|------|------|
| 공격 범위 | LLL + BKZ + primal | "축소만"은 LWE와 연결 약함, "전부"는 범위 초과 |
| 라이브러리 정책 | 순수 numpy/Python | 책의 "밑바닥부터" 철학 유지 |
| 노트북 수 | 4 | LLL / BKZ / primal / ML-KEM 연결 마무리 |
| 실제 ML-KEM 연결 | 외부 estimator 수치 인용 | toy-MLWE 구축 부담 회피, 공식 숫자 사용 신뢰성 |
| block size 최대 | 8 | Python enumeration 실용 한계 |
| toy-LWE 최대 차원 | 40 | 2–3분 실행 예산 내 |

## 4. 프로젝트 구조

```
pqc/
├── pqc_edu/
│   ├── ... (기존)
│   └── attacks_advanced/          # 신규 서브패키지
│       ├── __init__.py
│       ├── lll.py                 # Gram-Schmidt + LLL reduction
│       ├── bkz.py                 # BKZ block reduction + SVP enumeration
│       ├── embedding.py           # Kannan embedding lattice builder
│       └── primal.py              # full primal attack loop
├── notebooks/
│   ├── ... (기존 19개)
│   ├── 16_lll_basis_reduction.ipynb
│   ├── 17_bkz_and_scaling.ipynb
│   ├── 18_primal_attack_on_lwe.ipynb
│   └── 19_ml_kem_parameters_and_estimator.ipynb
├── ko/notebooks/                  # 한글판은 영어 Part 5 완료 후 일괄 번역
└── tests/
    ├── ... (기존)
    ├── test_lll.py
    ├── test_bkz.py
    └── test_primal_attack.py
```

### 원칙

- `pqc_edu.attacks_advanced`는 `pqc_edu.lwe`의 `ToyPublicKey` / `ToySecretKey`를 읽음. 기존 toy LWE 인프라 재활용.
- 노트북 번호 16–19 (Part 4가 15까지). URL 안정성 유지.
- `_toc.yml`에 **Part 5 — Breaking It for Real** 추가.

## 5. 학습 흐름 (노트북별)

| # | 노트북 | 핵심 질문 | 주요 산출 | 도입 모듈 |
|---|--------|-----------|-----------|-----------|
| 16 | lll_basis_reduction | "좋은 기저"란 무엇이고 LLL은 어떻게 만드나? | Gram-Schmidt 직접 구현, LLL 구현, 2D 기저 축소 before/after 시각화, Hermite factor 설명 | `lll.py` |
| 17 | bkz_and_scaling | LLL의 한계는? BKZ가 왜 더 강한가? | BKZ 구현, block size 2/4/6/8 품질 비교, 실행 시간 vs block size 그래프 | `bkz.py` |
| 18 | primal_attack_on_lwe | BKZ로 어떻게 실제 LWE를 푸는가? | Kannan embedding 유도, primal 공격 end-to-end, n=10~40 성공률 그래프 | `embedding.py`, `primal.py` |
| 19 | ml_kem_parameters_and_estimator | ML-KEM 파라미터는 왜 안전한가? | toy 수치 vs Albrecht estimator 공식 결과, ML-KEM-512/768/1024 cost estimate 표, Part 5 요약 | (개념만) |

**난이도 곡선**: 16 수학(기저 축소 직관) → 17 알고리즘 개선 → 18 실제 공격 → 19 현실 연결.

## 6. 핵심 인터페이스

### 의존 그래프

```
lll ──┬─→ bkz ────────┐
      │               ├─→ primal
      │   embedding ──┘
lwe ──┴────────────────┘  (ToyPublicKey/SecretKey 제공)
```

### 주요 시그니처

```python
# pqc_edu/attacks_advanced/lll.py
def gram_schmidt(B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """B: n×m basis (rows are vectors).
    Returns (B_star, mu): GSO vectors and coefficient matrix."""

def lll_reduce(B: np.ndarray, delta: float = 0.75) -> np.ndarray:
    """LLL reduction. Returns the reduced basis (integer ndarray, rows)."""


# pqc_edu/attacks_advanced/bkz.py
def svp_enumerate(B: np.ndarray) -> np.ndarray:
    """Exact SVP via Schnorr-Euchner enumeration.
    Input basis (rows). Returns the shortest nonzero lattice vector."""

def bkz_reduce(
    B: np.ndarray,
    block_size: int,
    delta: float = 0.99,
    max_tours: int = 10,
) -> np.ndarray:
    """BKZ reduction with the given block size. Returns reduced basis."""


# pqc_edu/attacks_advanced/embedding.py
from pqc_edu.lwe import ToyPublicKey

def kannan_embedding(pk: ToyPublicKey, M: int | None = None) -> np.ndarray:
    """Build the primal-attack lattice. Returns an (m+n+1) × (m+n+1)
    integer basis whose shortest vector encodes (e, 1) from the LWE instance.
    M is the embedding factor; defaults to ceil(sigma * sqrt(n))."""


# pqc_edu/attacks_advanced/primal.py
from dataclasses import dataclass

@dataclass
class PrimalResult:
    secret: np.ndarray | None         # recovered s, or None on failure
    recovered_error: np.ndarray | None  # recovered e, if any
    reduction_time: float
    block_size: int
    status: str  # "success" | "reduction_failed" | "short_vector_not_secret"

def primal_attack(
    pk: ToyPublicKey,
    block_size: int = 4,
    time_budget_s: float = 120.0,
) -> PrimalResult:
    """Full primal attack: embedding → BKZ → extract (s, e)."""
```

### 규약

- **기저 표현**: numpy 정수 배열 (`dtype=np.int64`). Rows = lattice vectors.
- **불변 패턴**: 축소 함수는 새로운 배열 반환 (입력 mutate 안 함).
- **내부 GSO**: float64 사용 (정밀도 이슈 발생 시 fractions.Fraction fallback 가능하나 toy 스케일에서는 불필요).

## 7. 테스트 & 검증 전략

### Level 1 — 단위 테스트 (pytest)

| 파일 | 검증 내용 |
|------|-----------|
| `test_lll.py` | LLL 조건 만족 (size-reduced `\|μ_ij\| ≤ 1/2` + Lovász `\|\|b*_{i+1}\|\|² ≥ (δ - μ²) \|\|b*_i\|\|²`), 2D 수작업 케이스 `[[1,1],[1,-1]]` 대각화, 축소 전후 determinant 불변 |
| `test_bkz.py` | `bkz_reduce(B, block_size=2)` == `lll_reduce(B)`, block 커질수록 첫 기저 벡터 길이 단조 감소, 3D 사전 계산된 결과 일치 |
| `test_primal_attack.py` | n=10, 15, 20 각각에서 toy LWE 복원 성공 (fixed seed, 최대 10회 시도) |

### Level 2 — 노트북 내 실험

- **노트북 16**: Hermite factor 측정 — LLL 후 첫 기저 벡터 길이 vs determinant^(1/n) 비율이 (4/3)^((n-1)/4) 이론치 근처
- **노트북 17**: 고정된 랜덤 기저에 대해 block size = 2, 4, 6, 8 BKZ 실행 후 첫 벡터 길이 비교
- **노트북 18**: n = 10, 15, 20, 25, 30, 35, 40 각각 10개 인스턴스 공격, 성공률 + 평균 시간 그래프

### 명시적 제외

- ❌ fpylll 또는 SageMath 대조
- ❌ Dual attack, hybrid attack 실제 실행
- ❌ Sieve 알고리즘
- ❌ n ≥ 50 LWE 공격 (Python LLL의 시간 예산 초과)

## 8. 리스크 & 완화

| 리스크 | 완화 |
|--------|------|
| 순수 Python LLL이 n=40 근처에서 너무 느림 | 정수 연산 위주 + numpy 벡터화. n ≤ 40 상한 명시. 노트북은 각 ≤ 3분. |
| BKZ enumeration이 block size ≥ 10에서 폭주 | `block_size` 상한 8. 노트북 17에서 "block을 더 키우려면 fpylll" 안내. |
| Kannan embedding 잘못 구현 시 short vector가 secret과 무관 | `test_primal_attack.py`에서 복원 직접 검증. embedding 행렬 구조를 노트북 18에 명시 |
| GSO 부동소수점 누적 오차로 LLL 조건이 경계에서 실패 | `delta = 0.75` 기본값 사용 (0.99가 아닌). 테스트는 수작업 소형 케이스로 고정. |
| 노트북 19가 "외부 숫자만 인용"이라 밋밋해질 우려 | toy 데이터(n=30 Python 초) → estimator log-cost → ML-KEM 실효 차원 순서의 **단일 통합 표**를 중심에 배치 |

## 9. 성공 기준

1. 4개 노트북 (16~19)이 `jupyter nbconvert --execute`로 에러 없이 실행 (각 ≤ 3분)
2. `pytest tests/test_lll.py tests/test_bkz.py tests/test_primal_attack.py` 전부 통과
3. 노트북 18에서 n별 성공률 그래프: n ≤ 20 성공률 ~100%, n ≥ 35 실패 급증
4. 노트북 19에서 toy vs ML-KEM 비교 표 한 자리에 출력
5. **학습 성공**: 독자가 "왜 LLL/BKZ는 작은 LWE는 뚫지만 ML-KEM은 못 뚫나"를 Hermite factor·cost estimate 관점에서 한 문단 설명 가능

## 10. 다음 단계

본 설계 승인 후 `writing-plans` 스킬로 구현 계획서 작성.

Part 5 영어판 완성 후:
1. 한글 번역 (Part 4 + Part 5 한 번에 일괄)
2. Part 6 = `pq-messenger` 사이블링 저장소 착수
