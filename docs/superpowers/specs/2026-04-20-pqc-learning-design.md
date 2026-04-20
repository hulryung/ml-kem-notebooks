# PQC 학습 프로젝트 설계

- **작성일**: 2026-04-20
- **작성자**: dkkang
- **상태**: Draft (사용자 리뷰 대기)

## 1. 목적

격자 기반 Post-Quantum Cryptography (PQC)의 핵심 알고리즘인 **ML-KEM (Kyber, FIPS 203)** 을 Python으로 직접 구현하며 원리를 학습한다. 최종 산출물은 "읽는 책이자 실행 가능한 코드"인 Jupyter Notebook 커리큘럼이다.

**학습 성공 기준**: 커리큘럼을 완주한 사람이 "LWE가 왜 어렵고, ML-KEM이 그걸 어떻게 쓰는가"를 한 문단으로 설명할 수 있어야 한다.

## 2. 스코프

### In-Scope

- 격자 기본 개념 → toy LWE → Ring-LWE/MLWE → ML-KEM (FIPS 203) 순차적 구현
- ML-KEM 세 파라미터셋 모두: **ML-KEM-512 / 768 / 1024**
- NTT(Number Theoretic Transform) 직접 구현
- 결정론적 샘플링 (CBD, SHAKE 기반)
- Toy 파라미터에서의 실제 공격 데모 (brute force, 가우시안 제거)
- X25519 + ML-KEM 하이브리드 키 교환 응용
- 단위 테스트 및 자체 왕복 검증

### Non-Goals

- ❌ 실사용 프로덕션 암호화 (사이드채널 방어, 상수 시간 연산 미구현)
- ❌ NIST KAT (Known Answer Test) 통과 — 비트 단위 인코딩·시드 체계 불일치 가능
- ❌ ML-DSA (Dilithium) 등 서명 스킴
- ❌ 성능 최적화 — 가독성 우선
- ❌ 기존 PQC 라이브러리(`liboqs`, `pqcrypto`) 호출

## 3. 의사결정 요약

| 항목 | 결정 | 근거 |
|------|------|------|
| 목적 | 학습/교육 | 사용자 선택 |
| 대상 알고리즘 | 격자 기반, ML-KEM 중심 | 실용 중요도 + 난이도 균형 |
| 결과물 형태 | Jupyter Notebook 중심 | 설명·수식·시각화·코드 통합 흐름 |
| 외부 의존성 | numpy까지 허용, PQC 완제품 금지 | 가독성과 학습 투명성 균형 |
| 정확도 수준 | 스펙 준수 + 자체 테스트 | KAT 통과는 디버깅 부담 과다 |
| 부가 기능 | 시각화 + toy 공격 데모 + 하이브리드 응용 | 원리 체감에 가장 효과적 |
| 파라미터셋 | 512/768/1024 전부 | 코드 부담 낮고 비교 실험 가능 |
| 노트북 구성 | 선형 교과서형 (01 → 09) | 학습 순서 = 커리큘럼 |

## 4. 프로젝트 구조

```
pqc/
├── README.md                   # 시작 가이드, 읽는 순서, 환경 설정
├── pyproject.toml              # 의존성: numpy, jupyter, matplotlib
├── notebooks/
│   ├── 01_lattice_intro.ipynb
│   ├── 02_toy_lwe.ipynb
│   ├── 03_attacking_toy_lwe.ipynb
│   ├── 04_polynomial_rings.ipynb
│   ├── 05_ring_lwe.ipynb
│   ├── 06_ml_kem_spec.ipynb
│   ├── 07_ml_kem_tests.ipynb
│   ├── 08_hybrid_kem.ipynb
│   └── 09_wrap_up.ipynb
├── pqc_edu/                    # 노트북 간 공유되는 재사용 코드
│   ├── __init__.py
│   ├── lattice.py              # 격자 기본 연산, 시각화 헬퍼
│   ├── lwe.py                  # toy LWE keygen/encrypt/decrypt
│   ├── attacks.py              # toy 공격 구현
│   ├── polyring.py             # Z_q[x]/(x^n+1) 다항식 환
│   ├── ntt.py                  # NTT / INTT
│   ├── sampling.py             # CBD, uniform, SHAKE 기반 PRF
│   ├── ml_kem.py               # FIPS 203 K-PKE + ML-KEM
│   └── params.py               # 세 파라미터셋 테이블
├── tests/
│   ├── test_lwe.py
│   ├── test_polyring.py
│   ├── test_ntt.py
│   └── test_ml_kem.py
└── docs/superpowers/specs/
    └── 2026-04-20-pqc-learning-design.md  (이 문서)
```

### 원칙

- 노트북 = "설명과 실행의 흐름", `pqc_edu/` = "재사용 로직"
- 각 기능이 처음 도입되는 노트북은 인라인으로 직접 구현 → 이후 노트북은 `pqc_edu`에서 import
- 각 모듈은 단일 책임

## 5. 학습 흐름 (노트북별)

| # | 노트북 | 핵심 질문 | 주요 산출 | 도입 모듈 |
|---|--------|-----------|-----------|-----------|
| 01 | lattice_intro | 격자가 뭔가? 왜 어려운가? | 2D 격자 시각화, SVP/CVP 직관 | `lattice.py` |
| 02 | toy_lwe | LWE 문제는 어떻게 생겼나? | n=4 예제 수동 실행 | `lwe.py` |
| 03 | attacking_toy_lwe | 진짜 안 풀리나? | n=10~20 공격, 시간 폭증 그래프 | `attacks.py` |
| 04 | polynomial_rings | 왜 다항식 환인가? | naive vs NTT 곱셈 비교 | `polyring.py`, `ntt.py` |
| 05 | ring_lwe | LWE → Ring-LWE → MLWE | 다항식 벡터로 LWE 재구성 | (위 모듈 재사용) |
| 06 | ml_kem_spec | FIPS 203은 어떻게 생겼나? | K-PKE + ML-KEM 전체 구현 | `sampling.py`, `ml_kem.py`, `params.py` |
| 07 | ml_kem_tests | 내가 만든 게 맞나? | 왕복 테스트, 크기·속도 표 | (전 모듈) |
| 08 | hybrid_kem | 실제 배포는? | X25519 + ML-KEM-768 하이브리드 | (응용) |
| 09 | wrap_up | 종합 정리 | 보안 근거, 배포 이슈, 더 읽을거리 | — |

**난이도 곡선**: 01~03 수학 직관 → 04~07 구현 → 08~09 응용·정리. 앞 3개만 읽어도 "왜 어려운가"는 얻어 감.

## 6. 핵심 인터페이스

### 의존 그래프

```
params ──┐
         ├─→ sampling ─→ ml_kem
polyring ─┴─→ ntt ──────┘
lattice ─→ lwe ─→ attacks
```

### 주요 시그니처

```python
# pqc_edu/params.py
@dataclass(frozen=True)
class MLKEMParams:
    name: str          # "ML-KEM-512" | "768" | "1024"
    k: int             # 모듈 차원 (2, 3, 4)
    eta1: int; eta2: int
    du: int; dv: int
    n: int = 256
    q: int = 3329

ML_KEM_512, ML_KEM_768, ML_KEM_1024  # 상수

# pqc_edu/lwe.py  (toy, 학습용)
def toy_keygen(n: int, q: int, sigma: float, rng) -> tuple[PublicKey, SecretKey]
def toy_encrypt(pk, bit: int, rng) -> Ciphertext
def toy_decrypt(sk, ct) -> int

# pqc_edu/polyring.py
class Poly:                      # Z_q[x]/(x^n+1)
    coeffs: np.ndarray           # shape (n,)
    def __add__, __mul__, ...    # 모듈러 연산
def poly_mul_naive(a, b) -> Poly
def poly_mul_ntt(a, b) -> Poly

# pqc_edu/ntt.py
def ntt(poly: Poly) -> Poly
def intt(poly: Poly) -> Poly

# pqc_edu/sampling.py
def cbd(seed: bytes, eta: int, n: int) -> Poly
def sample_uniform(seed: bytes, n: int, q: int) -> Poly
def prf(seed: bytes, nonce: int, length: int) -> bytes  # SHAKE256

# pqc_edu/ml_kem.py
def keygen(params: MLKEMParams, rng) -> tuple[EncapsKey, DecapsKey]
def encaps(params, ek, rng) -> tuple[SharedSecret, Ciphertext]
def decaps(params, dk, ct) -> SharedSecret
```

### 규약

- `ml_kem.py`의 각 함수에는 "Algorithm N, FIPS 203" 주석으로 스펙 매핑
- 랜덤성은 항상 인자로 받음 → 결정론적 테스트 가능
- `Poly`는 불변; 연산은 새 객체 반환

## 7. 테스트 & 검증 전략

### Level 1 — 단위 테스트 (`pytest`)

| 파일 | 검증 내용 |
|------|-----------|
| `test_polyring.py` | 교환·결합·분배 법칙, `x^n = -1` 환 규칙 |
| `test_ntt.py` | `intt(ntt(a)) == a`, `poly_mul_naive == poly_mul_ntt` |
| `test_lwe.py` | toy LWE 왕복 100회 |
| `test_ml_kem.py` | 세 파라미터셋 `decaps(encaps())` 왕복 100회 |

### Level 2 — 속성 기반 검증 (노트북 07 안에서)

- 공개키·암호문 크기가 FIPS 203 Table 3과 일치:
  - ML-KEM-512: ek=800B, ct=768B
  - ML-KEM-768: ek=1184B, ct=1088B
  - ML-KEM-1024: ek=1568B, ct=1568B
- 결정론성: 같은 시드 → 같은 키
- 실패율: 10,000회 왕복 시 decryption failure 0회

### Level 3 — 공격 역검증 (노트북 03)

- toy LWE의 n을 키워가며 공격 시간 측정
- "작은 n은 뚫리고 n이 커지면 폭증"하는 그래프로 어려움 체감

### 명시적 제외: NIST KAT

- 비트 단위 인코딩·시드 체계를 스펙과 정확히 맞추지 않음
- README에 "자체 테스트 수준. 실제 암호화에 사용 금지" 경고 명시

### 실행 편의

- `pytest tests/` 한 방으로 Level 1 전부
- `jupyter nbconvert --execute` 로 노트북 통과 확인
- 시간 오래 걸리는 06/07은 축소 파라미터 옵션 제공

## 8. 리스크 & 완화

| 리스크 | 완화 |
|--------|------|
| FIPS 203 스펙 오독 | 함수마다 Algorithm 번호 주석, 노트북 06에서 의사코드-Python 좌우 비교 |
| NTT 버그가 뒷 노트북 전체로 전파 | `test_ntt.py`에서 naive vs NTT 강하게 검증, 노트북 06은 그 이후 진행 |
| toy 공격 데모 실행 시간 폭증 | n을 10→15→20으로 단계적 증가, 20 이상은 "포기" 처리 — 포기 자체가 교육 포인트 |
| 노트북-패키지 코드 중복 | "첫 도입은 인라인, 이후는 import" 원칙 준수 |

## 9. 성공 기준

1. 9개 노트북이 에러 없이 순차 실행됨
2. `pytest tests/` 전부 통과
3. ML-KEM 세 파라미터셋 왕복 테스트 통과 + 키/암호문 크기 스펙 일치
4. 노트북 03에서 공격 시간이 n에 따라 지수적으로 증가하는 그래프 생성
5. 노트북 08에서 X25519 + ML-KEM 하이브리드로 양측이 같은 공유 비밀 도출
6. **학습 성공 기준**: 완주자가 "LWE가 왜 어렵고 ML-KEM이 그걸 어떻게 쓰는가"를 한 문단으로 설명 가능

## 10. 다음 단계

본 설계가 승인되면 `writing-plans` 스킬로 전환하여 구현 계획서를 작성한다. 구현 계획은 모듈별 작업 단위, 순서, 각 단계의 검증 방법을 담는다.
