# ML-KEM 밑바닥부터

격자 기반 Post-Quantum Cryptography를 Python으로 밑바닥부터 **ML-KEM (Kyber / FIPS 203)** 을 구현하며 학습하는 9개 노트북 커리큘럼입니다.

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

## 읽는 순서

노트북은 서로 이어져 있습니다 — 순서대로 읽으세요. Ring-LWE와 NTT를 이미 알고 있다면 04, 05는 훑어봐도 됩니다.

1. 격자 소개 — 격자란 무엇이고 왜 어려운가?
2. Toy LWE — 작은 수로 본 LWE 문제
3. Toy LWE 공격 — 작은 LWE를 직접 깨 보기
4. 다항식 환과 NTT — ML-KEM이 빠른 이유
5. Ring-LWE와 Module-LWE — LWE를 다항식 환으로 끌어올리기
6. ML-KEM 스펙 구현 — FIPS 203 따라가기
7. 테스트와 벤치마크 — 구현 검증
8. 하이브리드 X25519 + ML-KEM — 실제 배포 패턴
9. 정리 — 우리가 만든 것, 다음 단계

## 소스

코드, 테스트, 노트북: [github.com/hulryung/ml-kem-notebooks](https://github.com/hulryung/ml-kem-notebooks)
