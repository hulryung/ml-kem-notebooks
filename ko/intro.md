# ML-KEM 밑바닥부터

격자 기반 post-quantum cryptography를 Python으로 밑바닥부터 **ML-KEM (Kyber / FIPS 203)** 을 구현하며 학습하는 14개 노트북 커리큘럼입니다. 초보자를 위한 사전 지식 프렐류드와 signature·실전 배포까지 다루는 확장편을 포함합니다.

```{warning}
본 구현은 **교육 목적 전용**입니다. Constant-time 연산이 아니며 사이드채널에도 취약하고, NIST Known-Answer Test로 검증하지 않았습니다. **실제 암호화에 사용하지 마십시오.**
```

## 무엇을 만드는가

완주하면 다음을 수행하는 Python 코드를 직접 작성해 보게 됩니다.

- ML-KEM-512 / 768 / 1024 키 쌍 생성 — FIPS 203 Table 3의 바이트 크기와 정확히 일치
- 32바이트 공유 비밀의 encapsulation / decapsulation
- 각 파라미터셋당 200회 왕복 테스트 0회 실패 통과
- X25519와 결합한 하이브리드 KEM — TLS 1.3이 실제 배포하고 있는 패턴

그리고 toy 크기의 LWE를 **직접 공격해서 깨 봄**으로써 실제 파라미터가 왜 안 풀리는지 체감하게 됩니다.

## 세 파트, 하나의 흐름

**Part 1 — 사전 지식 (선택).** 격자 수학에 들어가기 전 맥락이 필요한 독자용. 양자 위협 모델, 10분짜리 고전 암호 복습, 책 전체에서 쓰이는 수학 어휘를 다룹니다. RSA가 무엇을 하는지 알고 모듈러 연산이 익숙하다면 건너뛰어도 됩니다.

**Part 2 — 본편: ML-KEM 만들기.** 메인. 2차원 격자 → toy LWE → toy LWE 공격 → 다항식 환과 NTT → Ring/Module-LWE → FIPS 203 ML-KEM 구현 → 테스트와 벤치마크 → 하이브리드 X25519 + ML-KEM → 정리로 이어지는 9개 노트북.

**Part 3 — ML-KEM 너머.** 격자 기반·해시 기반 signature (ML-DSA, SLH-DSA) 개관과 실전 배포 현황·FAQ·용어집을 담은 참고용 두 노트북.

## 읽는 순서

파트를 순서대로 따라가거나, 기본기가 이미 있다면 Part 2부터 읽으셔도 됩니다.

## 소스

코드, 테스트, 노트북: [github.com/hulryung/ml-kem-notebooks](https://github.com/hulryung/ml-kem-notebooks)
