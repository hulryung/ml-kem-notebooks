# Part 4 — Quantum Reckoning: Shor's Algorithm from Scratch

- **작성일**: 2026-04-22
- **작성자**: dkkang
- **상태**: Draft (사용자 리뷰 대기)
- **맥락**: 기존 책 `ML-KEM from Scratch` (Part 1–3 완료, 14 notebooks) 뒤에 덧붙이는 **Part 4**.

## 1. 목적

교육용 양자 회로 시뮬레이터를 Python + numpy만으로 밑바닥부터 구현하고, 그 위에서 **Shor의 알고리즘**을 실행해 RSA-크기의 작은 정수(15, 21, 35)를 인수분해한다. 마지막에 "왜 Shor는 ML-KEM(격자)을 못 깨는가"로 책 전체의 서사를 종결한다.

**학습 성공 기준**: 독자가 "왜 Shor가 RSA를 깨고 ML-KEM은 못 깨는가"를 한 문단으로 설명할 수 있어야 한다.

## 2. 스코프

### In-Scope

- 상태 벡터 양자 시뮬레이터 (`2^n` 차원 복소 벡터로 `n`-qubit 상태 표현)
- 표준 게이트: H, X, Z, CNOT, phase gate `R_k`
- Quantum Fourier Transform (QFT / inverse QFT)
- Shor 알고리즘 전체 루프: 랜덤 `a` 선택 → 양자 order finding → continued fraction 후처리 → 재시도
- 인수분해 대상: **N = 15, 21, 35**
- 실패 시나리오 시연 (bad `a` 선택, 홀수 period 등)
- Shor vs Lattice 종결 논의 (Hidden Subgroup Problem, 격자에 대한 양자 알고리즘 현황)

### Non-Goals

- ❌ 프로덕션 양자 시뮬레이터 (성능 최적화, 노이즈 모델링, 텐서 네트워크 등)
- ❌ Quantum reversible arithmetic 회로 — `modular_exp_circuit`은 고전 lookup으로 시뮬레이션 (교육적 정직성 명시)
- ❌ Shor의 변형판 (Kitaev phase estimation, Beauregard 등)
- ❌ Qiskit 또는 타 양자 프레임워크 호환
- ❌ N ≥ 77 — 시뮬레이션 시간 제약
- ❌ Grover, Deutsch-Jozsa 등 다른 양자 알고리즘
- ❌ 노이즈·에러 정정 논의

## 3. 의사결정 요약

| 항목 | 결정 | 근거 |
|------|------|------|
| 시뮬레이터 기술 | 순수 numpy | 책의 "밑바닥부터" 정체성 유지, 외부 의존성 최소화 |
| 양자 primer 깊이 | 단계적 (qubit → QFT → Shor) | Deutsch/Grover까지 가면 스코프 초과, Shor 집중 |
| 인수분해 대상 | N=15, 21, 35 + 실패 케이스 | 점진적 확장성 + 확률적 본질 시연 |
| ML-KEM 연결 | 전용 마지막 노트북 | 책 전체 서사 완결 |
| 노트북 수 | 4 | 12 기초 / 13 QFT / 14 Shor / 15 종결 |
| 노트북 구조 | 선형 교과서형 | 기존 Part 1–3과 톤 일치 |

## 4. 프로젝트 구조

```
pqc/
├── pqc_edu/
│   ├── ... (기존 모듈 유지)
│   └── quantum/                    # 신규 서브패키지
│       ├── __init__.py
│       ├── simulator.py            # QuantumState, 표준 게이트
│       ├── circuits.py             # QFT, inverse QFT, modular_exp_circuit
│       ├── shor.py                 # Shor 메인 루프 + ShorResult / ShorAttempt
│       └── classical_utils.py      # continued fraction, gcd
├── notebooks/
│   ├── ... (기존 14개 유지)
│   ├── 12_quantum_basics.ipynb
│   ├── 13_qft_and_period.ipynb
│   ├── 14_shor_breaks_rsa.ipynb
│   └── 15_shor_vs_lattice.ipynb
├── ko/notebooks/                   # 한글판은 영어 완료 후 일괄 번역
└── tests/
    ├── ... (기존)
    ├── test_quantum_simulator.py
    ├── test_qft.py
    └── test_shor.py
```

### 원칙

- `pqc_edu.quantum`은 기존 `polyring`, `ml_kem` 등과 **의존성 독립**. Shor 전용 서브패키지.
- 노트북 번호는 12–15 (기존 11 다음). URL 안정성 유지.
- `_toc.yml`에 **Part 4 — Quantum Reckoning** 추가. Part 3 "Beyond ML-KEM" 뒤.
- 기획 단계에서 "Part 5"로 불렀던 것은 실제 책 TOC 상 **Part 4**가 된다. 본격 격자 공격(LLL/BKZ)이 나중에 Part 5로 붙는다.

## 5. 학습 흐름 (노트북별)

| # | 노트북 | 핵심 질문 | 주요 산출 | 도입 모듈 |
|---|--------|-----------|-----------|-----------|
| 12 | quantum_basics | qubit이란 무엇이고 왜 "2^n 복소 벡터"인가? | `QuantumState` 클래스 직접 작성, H·X·CNOT 게이트, Bell state 측정 상관 실험 | `simulator.py` |
| 13 | qft_and_period | QFT는 무엇이고 주기 찾기는 왜 인수분해와 연결되는가? | QFT 직접 구현, 주기 4 수열의 스펙트럼 시각화, N=15·a=7에서 order=4 발견 | `circuits.py` |
| 14 | shor_breaks_rsa | Shor가 실제로 RSA를 깨는 모습 | Shor 전체 회로 조립, 15/21/35 인수분해, 실패 재시도 로그, continued fraction 후처리 | `shor.py`, `classical_utils.py` |
| 15 | shor_vs_lattice | 왜 Shor는 ML-KEM을 못 깨는가? | Hidden Subgroup Problem 관점 설명, abelian vs non-abelian, 격자 문제 현황 요약 | (개념만) |

**난이도 곡선**: 12 입문 → 13 밀도 정점 → 14 극적 실행 → 15 정리. 독자가 양자 배경이 있으면 12·13을 스킵하고 14부터 읽어도 작동.

## 6. 핵심 인터페이스

### 의존 그래프

```
simulator ──┐
            ├─→ circuits ─→ shor
classical_utils ───────────┘
```

### 주요 시그니처

```python
# pqc_edu/quantum/simulator.py
class QuantumState:
    """n-qubit state. Amplitudes stored as a length-2^n complex vector."""
    def __init__(self, n: int): ...
    def apply_1q(self, gate: np.ndarray, target: int) -> None
    def apply_2q(self, gate: np.ndarray, ctrl: int, target: int) -> None
    def measure(self, qubits: list[int], rng) -> int

H: np.ndarray       # Hadamard
X: np.ndarray       # Pauli-X
Z: np.ndarray       # Pauli-Z
CNOT: np.ndarray    # controlled-NOT (4x4 for apply_2q)
def Rk(k: int) -> np.ndarray   # phase exp(2πi/2^k)

# pqc_edu/quantum/circuits.py
def qft(state: QuantumState, qubits: list[int]) -> None            # in-place
def inverse_qft(state: QuantumState, qubits: list[int]) -> None    # in-place
def modular_exp_circuit(state, a: int, N: int,
                        x_qubits: list[int], y_qubits: list[int]) -> None
    """|x>|y> -> |x>|y XOR (a^x mod N)>. Educational: uses classical lookup
    over the superposition basis states — not a quantum-gate reversible
    arithmetic circuit. Noted explicitly in notebook 14."""

# pqc_edu/quantum/shor.py
@dataclass
class ShorAttempt:
    a: int
    measured: int
    inferred_period: int | None
    status: str                 # "success" | "a_not_coprime" | "odd_period" | "trivial_factor" | ...

@dataclass
class ShorResult:
    N: int
    factors: tuple[int, int] | None
    attempts: list[ShorAttempt]
    total_quantum_runs: int

def factor(N: int, rng, max_attempts: int = 10) -> ShorResult

# pqc_edu/quantum/classical_utils.py
def continued_fraction_denominator(measured: int, counting_qubits: int, N: int) -> int | None
def gcd(a: int, b: int) -> int
```

### 규약

- **State mutation**: `QuantumState`는 in-place. 양자 회로가 본질적으로 직렬 상태 업데이트.
- **Qubit indexing**: little-endian. Qubit 0이 최하위 비트.
- **Randomness**: 항상 인자 — 결정론적 테스트 가능.
- **Unitarity**: 모든 게이트는 유닛터리. `measure`만 비-unitary.

### 교육적 정직성

`modular_exp_circuit`은 고전 lookup으로 구현된다. 실제 양자 컴퓨터에서는 quantum reversible arithmetic 회로가 필요하다. 이 단축은 **알고리즘적 동작**(`|x⟩|0⟩ → |x⟩|a^x mod N⟩`의 모든 `x`에 대한 superposition 생성)을 정확히 재현하며, 시뮬레이션 단축이지 알고리즘 단축이 아니다. 노트북 14에 이 구분을 명시하는 전용 markdown 셀을 배치한다.

## 7. 테스트 & 검증 전략

### Level 1 — 단위 테스트 (`pytest`)

| 파일 | 검증 내용 |
|------|-----------|
| `test_quantum_simulator.py` | `\|0⟩` → H → `(\|0⟩ + \|1⟩)/√2`, CNOT의 `\|10⟩ → \|11⟩`, Bell state 측정 상관 1.0, 게이트 unitarity `U·U† = I` |
| `test_qft.py` | 1-qubit QFT = Hadamard, 2-qubit QFT 수작업 계산 일치, `inverse_qft(qft(x)) == x` (3-qubit 이상 10회 랜덤), QFT 행렬이 표준 DFT 행렬과 일치 |
| `test_shor.py` | `factor(15, rng)`가 `{3, 5}` 반환 (고정 seed), `factor(21)` → `{3, 7}`, `factor(35)` → `{5, 7}`, `continued_fraction_denominator` 단위 검증 |

### Level 2 — 속성 검증 (노트북 14 내부)

- **성공률**: N=15에 대해 100회 실행, 최대 10회 시도 안 성공률. 이론값과 비교.
- **실패 분포**: 시도 실패 유형별 카운트 (`a_not_coprime`, `odd_period`, `trivial_factor`).
- **규모 vs 시간**: N=15/21/35 실행 시간 막대그래프 — O(4ⁿ) 이상 성장 관찰.

### Level 3 — 서사적 검증 (노트북 13)

- 주기 4인 수열을 QFT한 후 크기 스펙트럼 plot — 피크가 `{0, 2, 4, 6}/8`에 형성됨을 시각적으로 확인.
- N=15, a=7 케이스에서 여러 seed의 측정 히스토그램을 누적.

### 실행 편의

- 관련 테스트 전체: `pytest tests/test_quantum_simulator.py tests/test_qft.py tests/test_shor.py`
- 시뮬레이터는 최대 12 qubit (Shor-35)까지 지원. 그 이상은 명시적 오류.
- 노트북 14의 N=35 케이스는 한 번만 실행 (수 초 소요).

### 명시적 제외

- ❌ 양자 노이즈 모델 (depolarizing, amplitude damping)
- ❌ Qiskit 교차 검증
- ❌ N ≥ 77 케이스
- ❌ Quantum arithmetic 회로 완전 구현

## 8. 리스크 & 완화

| 리스크 | 완화 |
|--------|------|
| QFT가 교재 정의와 어긋남 (bit-ordering 등 convention 차이) | Level 1에서 `inverse_qft(qft(x)) == x` + 2-qubit 수작업 비교 엄격 검증. 통과 전엔 14/15 진행 금지 |
| Shor 성공률이 낮아 노트북 14가 지루 | seed 고정한 "빠른 성공 데모" 먼저 보여주고, 이후 "진짜 랜덤"으로 실패 케이스 시연 |
| 노트북 12의 양자 기초가 너무 급함 | Bell state 같은 작은 시각적 데모를 primer 중간에 삽입. "벡터 곱일 뿐"임을 체감 |
| `modular_exp_circuit` 고전 lookup이 교육적 오해 유발 | 노트북 14에 전용 markdown 셀로 "시뮬레이션 단축 vs 알고리즘 단축" 명시 |
| 15번 노트북이 밋밋하게 끝남 | Hidden Subgroup Problem의 abelian/non-abelian 구분과 dihedral HSP / quantum sieve 현황 짧게 짚음. 독자가 후속 학습 방향을 얻도록 |

## 9. 성공 기준

1. 4개 노트북(12~15)이 `jupyter nbconvert --execute`로 에러 없이 순차 실행됨
2. `pytest tests/test_quantum_simulator.py tests/test_qft.py tests/test_shor.py` 전부 통과
3. N=15, 21, 35 모두 Shor로 인수분해 성공 (최대 10회 시도 내)
4. 노트북 13에서 QFT 스펙트럼이 주기 4 패턴을 피크로 보여줌
5. 노트북 14에서 적어도 한 번의 "실패 후 재시도" 로그 출력 (Shor의 확률적 본질 시연)
6. **학습 성공 기준**: 독자가 "왜 Shor가 RSA를 깨고 ML-KEM은 못 깨는가"를 한 문단으로 설명 가능

## 10. 다음 단계

본 설계 승인 후 `writing-plans` 스킬로 구현 계획서를 작성한다. 계획은 태스크별 TDD 사이클, 파일/코드 상세, 노트북 셀 구조를 담는다.

Part 4 영어판 완성 후:
1. 한글 번역 (`ko/notebooks/12-15`) 및 `ko/_toc.yml` 갱신
2. Part 5 (LLL/BKZ 공격) 브레인스토밍 시작
3. 이후 `pq-messenger` 사이블링 저장소 착수
