# WAVES stem SED validation

WAVES가 생성한 stem을 frozen AudioSet-Strong SED 모델로 검사하는 독립적인 후처리 프로젝트입니다.
WAVES 생성 파이프라인을 수정하거나 모델을 학습하지 않습니다.

현재 작업 범위는 **Phase 1 분석과 Phase 2 단일 WAV 추론 prototype**입니다.
WAVES의 planned support와 SED 출력 비교는 내부 consistency 검사이며,
실제 영상과의 동기화를 입증하지 않습니다.

구조 분석, 확인된 metadata 누락, 단계별 계획은 [분석 문서](docs/phase1-analysis.md)에 기록합니다.

구현 순서:

1. WAVES 산출물과 의존성 조사
2. ATST-F Strong checkpoint 로딩, frozen inference, raw frame probability 저장
3. WAVES metadata adapter, ontology 및 명시적 source mapping
4. 역할별 temporal metric과 JSON/CSV report
5. 시각화와 batch 처리
6. controlled corruption 평가
7. calibration 후 설정에 따른 PASS / REVIEW / FAIL 판단

모델·오디오·추론 출력은 Git에 넣지 않습니다. 주요 작업 단위마다 로컬 커밋을 남깁니다.
