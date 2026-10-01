# Phase 1: WAVES 산출물 조사와 평가 도구 설계

조사 기준일: **2026-09-25**.

이 문서는 구현에 앞서 WAVES의 산출물, 사용 가능한 자료, 의존성 및 모듈 경계를
조사한 설계 기록이다. 조사 대상은 `C:\WAVES`, 독립 평가 도구의 작업 경로는
`C:\multitrack-audio-filtering`이었다. 조사에서 확인한 사실과 당시 결정한 설계를
구분하여 기록한다. 검증 계획은 실제 실행 결과를 의미하지 않는다.

현재 Phase 1~7의 구현은 완료되어 있다. 현재 설치·실행 방법은
[README](../README.md), 각 단계의 실행 근거는 [Phase 2](phase2-validation.md)부터
[Phase 7](phase7-validation.md)까지의 검증 문서를 기준으로 한다. 실제 생성 음원의
확보와 운영 threshold 보정에 필요한 조건은 [실데이터 준비 현황](real-data-readiness.md)에 있다.

## 1. 실제 WAVES 산출물과 연결 지점

WAVES의 생성 파이프라인은 유지한다. 별도 도구에서 최종 stem WAV를 읽고, 독립적인
frozen SED 모델로 프레임별 이벤트 확률을 산출한다. CLAP, 에너지 게이트, TF containment를
다시 구현하거나 AudioSet의 다른 WAV와 유사도를 비교하는 접근은 사용하지 않는다.

조사한 WAVES 코드의 출력 구조는 다음과 같다. `<run>`은 materializer의 `--output`이다.

```text
<run>/
  prepared_manifest.json
  pass1/<key>.json
  sam_manifest.json
  sam_attempts/<key>/cNN_aM.wav
  sam_attempts/<key>/metadata.json
  dsp_reports/<key>.json
  pass2/<key>.json
  final/<key>/stem_NN.wav
  final/<key>/metadata.json
```

근거: `C:\WAVES\docs\REPRODUCIBILITY.md:61`의 실행 디렉터리 설명과
`scripts/pipeline/materialize_stems.py:28`의 실제 저장 코드.

| 필요한 정보 | 실제 저장 위치와 필드 | 해석 |
| --- | --- | --- |
| 최종 WAV | `final/<key>/metadata.json` → `stems[].file` | metadata 파일이 있는 디렉터리에 대한 상대 파일명 `stem_01.wav` 등 |
| stem 식별자 | `stems[].candidate_id` | 선택된 원본 candidate의 안정적인 ID. `stem_NN`은 최종 순번이므로 식별자로 우선하지 않음 |
| clip 식별자 | 최종 metadata의 `key` | 영상 ID와 동일하다고 추정하지 않음 |
| 최종 의미/역할 | `stems[].final_label`, `stems[].final_role` | Pass 2 relabel 결과를 반영하며 역할은 `onset/span/ambience` |
| 원래 source description | `sam_manifest.json`의 `clips[].candidates[].description` 또는 DSP report의 `candidates[].description` | 최종 caption과 의미가 다를 수 있으므로 별도로 보존 |
| 계획된 시간 구간 | 위 candidate의 `activity_intervals` | 초 단위 구간 목록. 최종 metadata에는 없음 |
| 선택 이력 | `candidate_id`, `selected_attempt`, `merged_candidate_ids` | 후보/시도와 semantic merge 이력 |
| WAV provenance | `file`, `sha256`, `source_attempt_file`, `source_attempt_sha256` | 선택된 SAM 파일을 그대로 복사하였다는 추적 근거 |
| 영상 경로/기타 ID | `prepared_manifest.json` 또는 `sam_manifest.json`의 clip 정보 | 없으면 optional/missing으로 처리 |

정확한 필드 근거:

- `schemas/pass1_schema.json:21`: `label`, `description`, `role`, `activity_intervals`를
  candidate 필수 필드로 정의한다.
- `scripts/pipeline/build_sam_manifest.py:23`: Pass 1 candidate를 보존하며
  `candidate_id = <key>__<두 자리 후보 번호>`를 추가한다.
- `scripts/pipeline/run_sam_attempts.py:80`: description, role, activity_intervals를 SAM
  metadata로 넘기고, 81행에서 양성 temporal anchor로 변환한다.
- `scripts/pipeline/score_and_build_dsp_evidence.py:116`: 위 candidate 필드를 DSP report에
  보존한다.
- `schemas/final_pass2_schema.json:31`: final stem 필드는 candidate ID, 선택 시도,
  최종 label/role, merged IDs뿐이다. 30행의 `additionalProperties: false` 때문에
  final schema 자체에 temporal support나 description을 추가할 수 없다.
- `scripts/pipeline/materialize_stems.py:31`: 후보/시도 쌍으로 실제 WAV를 찾는다.
  32~36행에서 `stem_NN.wav`로 복사하고 파일명, 해시, source 경로를 기록한다.
  38~40행에서 clip key, title, mixed_audio, stems, clip_summary를 저장한다.

### 최종 metadata와 계획 metadata의 결합

최종 waveform 경로, source description, temporal support는 단일 최종 JSON에
모두 들어 있지 않다. 따라서 **최종 metadata와 중간 candidate metadata를 candidate ID로
조인하는 방식**을 채택하였다. 파일명으로 시간 구간이나 원본 candidate를 추정하지 않는다.

Pass 2는 의미를 바꾸거나 여러 후보를 묶을 수 있지만 최종 시간 구간을 새로 작성하지
않는다. `README.md:15`와 `docs/PIPELINE_SPEC.md`의 merge 설명처럼, semantic merge는
대표 WAV 하나를 남기는 동작이며 음원을 합산하지 않는다. 따라서 merged child들의
계획 구간을 기계적으로 합집합 처리하면 실제 대표 WAV에 없는 이벤트를 기대할 수 있다.

Frozen 자료에서는 다음 사례를 확인하였다.

- `fb5k_3156__03`은 `fb5k_3156__02`를 merge한다. 최종 label은 반복 chopping이지만
  child는 원래 다른 시점의 후보였다. `frozen_finals.json:754`의 이유와 772행의
  최종 caption을 함께 읽어야 한다.
- `legacy_16__02`는 원래 `span`이지만 최종 `onset`으로 바뀌었고,
  `legacy_16__01`을 merge한다. 원래 span 구간의 시작을 개별 충격 onset으로 간주하면
  평가가 왜곡된다. 근거: `data/frozen_pass2/frozen_reports.json:13569`,
  `data/frozen_pass2/frozen_finals.json:1126` 및 1150행.

이 조사에 따라 final label/role과 원래 description/role/support를 모두 보존하고,
`reference_origin = waves_pass1_planned`를 기록하는 adapter를 설계하였다. Merge, role 변경,
relabel로 시간 기준이 불명확하면 provenance와 상태로 남기며 최종 역할에 맞는 timing을
임의로 생성하지 않는 원칙을 정하였다. 선택 parent의 planned support는 merge 전체의
정답 시간을 의미하지 않는다. 실제 정규화 계약과 상태는 [Phase 3](phase3-validation.md)에 있다.

## 2. 조사 시점의 자료와 입력 한계

`C:\WAVES`에는 코드, 문서, frozen JSON, 평가 표가 있었다. 조사 시점에 디렉터리를
재귀 확인한 결과 **WAV 파일은 0개이며, materialized `final/*/metadata.json`도 없었다.**
이는 `README.md:37`과 `docs/REPRODUCIBILITY.md`가 대용량 음원/영상을 제외한다고
설명하는 내용과 일치한다.

`data/frozen_pass2/frozen_reports.json`과 `frozen_finals.json`의 직접 집계 결과는 다음과 같다.

| 항목 | 확인 값 |
| --- | ---: |
| clip 수 | 29 |
| 계획 candidate 수 | 108 |
| 기록된 SAM attempt 수 | 307 |
| final stem 수 | 62 |
| final role | onset 18 / span 37 / ambience 7 |
| parent label과 final label이 다른 stem | 41 |
| parent role과 final role이 다른 stem | 2 |
| merged children을 가진 final stem | 10 |

이 값은 JSON 직접 집계 결과이며, 모델 추론이나 품질 평가 결과는 아니다. 같은 cohort의
예상 keep/merge/drop 수는 `scripts/pipeline/validate_frozen_artifacts.py:24`에
62/11/35로 명시되어 있다.

Frozen report의 모든 108개 candidate에는 description, role, activity_intervals가 있다.
하지만 307개 attempt에는 `file`이나 SHA-256 대신 `audio_id`만 있다.
`fb5k_1229/01/a1`이 그 예이다(`frozen_reports.json:33`). Frozen final 62개에도 WAV 파일 경로는 없다.
따라서 frozen JSON만으로 metadata adapter를 검증할 수는 있지만 실제 SED 추론을
실행할 수는 없다.

`data/system_eval/evaluation_manifest66.json:16`의 `waves_file` 등에는
`${WAVES_ROOT}/listening_test/.../sam_attempts/...wav` 형태의 과거 평가 경로가 있다.
조사 시점에 해당 경로의 파일은 checkout에 존재하지 않았다. `audio_id`를 이 경로 패턴으로
임의 변환하지 않는 원칙을 정하였다. Frozen adapter에는 명시적 `audio_id → path` 매핑이
필요하며, materialized 자료는 실제 run 디렉터리의 metadata와 WAV를 사용한다.

실제 stem이 없는 조건에서 Phase 2의 검증 입력으로 출처가 확인되는 공개 WAV를
선택하였다. 이 검증의 범위는 checkpoint loading과 raw frame 출력이며 WAVES stem의
품질 평가와 구분한다.

## 3. 환경과 의존성 선택

초기 환경 조사값은 Windows, Python 3.11.9, PyTorch 2.10.0+cpu, NumPy 2.4.3,
Intel i3-1315U/16 GB RAM이었으며 CUDA는 없었다. CPU 단일 파일 추론을 우선 검증하고,
WAVES 환경과 분리한 `.venv`를 사용하는 구성을 선택하였다. 이 초기 조사값과 실제
평가 도구의 설치 버전은 구분한다. Phase 2 검증 환경의 NumPy는 `1.26.4`이다.

WAVES는 `requirements-pipeline.txt:2`에서 torch/torchaudio 2.10.0을 고정하고,
`requirements-eval.txt:1`에서 NumPy 2.0.2를 고정한다.
`pyproject.toml:11`도 `numpy>=2.0,<2.1`을 요구한다.

조사한 PretrainedSED 기준 revision은
`1aa47e482f7e89904cba2338999345025d8b4e36`이다. upstream은 일반적인 설치형
Python 패키지 구조가 아니며 `requirements.txt:1`은 `numpy<2`를 요구한다. 전체
requirements에는 학습, 여러 다른 backbone, dataset/evaluation용 의존성이 함께 있다.
WAVES의 전체 환경에 그대로 합치는 방식은 NumPy 제약부터 충돌한다.

이에 따라 ATST-F 추론에 필요한 MIT 코드만 revision을 고정하여 vendor하고, 공개
`ATST-F_strong_1.pt` checkpoint와 strong label 447개 metadata를 사용하는 방식을
선택하였다. 모델 본문은 가능한 한 유지하고 package import 경로만 조정하는 원칙이다.
라이선스, 원본 revision, 원본 경로와 변경 범위는
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)에 기록한다. Checkpoint loading,
오디오 입력, raw 출력 형식은 독립 adapter에서 관리한다.

이 선택의 근거가 된 upstream 코드는 다음과 같다.

- `models/atstframe/ATSTF_wrapper.py`는 16 kHz 입력, 64 mel bin,
  hop 160, FFT/window 1024의 전처리를 정의한다.
- `models/prediction_wrapper.py:37`의 strong head 기본 class 수는 **447**이며,
  40행의 sequence 길이는 250이다. 10초 입력에 대해 약 40 ms 간격이다.
- `inference.py`는 10초 단위 chunk, 마지막 chunk zero padding, strong logits에
  sigmoid를 적용하는 추론 경로를 보여 준다.
- `hf_dataset_gen/metadata/class_labels_indices_strong.csv`에는 실제 class ID/name이
  들어 있다. 전체 AudioSet 527개 class 순서를 strong head 순서로 가정하면 안 된다.
- upstream `load_checkpoint`는 head 크기가 다르면 해당 weight를 제거할 수 있다.
  본 도구는 frozen 추론용이므로 학습 head의 누락/크기 불일치를 허용하지 않는다.
  재구성 가능한 mel buffer처럼 의도된 예외만 이름으로 제한하여 검증한다.

모델은 `eval()`, `requires_grad_(False)`, `torch.inference_mode()`로 실행한다.
학습 코드, optimizer, fine-tuning은 구현 범위에서 제외하였다. CPU 실행 결과는
[Phase 2 검증 기록](phase2-validation.md)에 정리되어 있다.

## 4. 모듈 분리 원칙과 초기 설계

초기 추론 경로는 모델 코드, 오디오 입력, 확률 저장 형식을 분리하도록 설계하였다.
Phase 1~2의 주요 모듈 구성은 다음과 같다.

```text
pyproject.toml
requirements*.txt
README.md
THIRD_PARTY_NOTICES.md
docs/
  phase1-analysis.md
src/waves_sed/
  __init__.py
  __main__.py
  audio.py                    # 읽기, mono/16 kHz 변환, 입력 유효성
  prediction.py               # raw frame 결과 타입, NPZ 저장/재사용
  cli.py                      # checkpoint 준비, 단일 WAV 추론
  backends/
    base.py                   # SEDBackend 경계
    atst.py                   # frozen ATST-F Strong adapter
  _vendor/pretrained_sed/      # attribution과 revision이 고정된 모델 부분
  resources/                  # checkpoint에 대응하는 class ID/name metadata
tests/                        # 출력 계약, 입력 경계, chunk/time 처리 검증
```

후속 단계도 역할을 분리하는 원칙을 유지하였다. 현재 구현의 주요 파일과 책임은 다음과 같다.

```text
src/waves_sed/
  adapters/waves.py           # materialized WAVES / frozen metadata 조인
  metadata.py                 # StemMetadata 정규화 계약
  ontology.py                 # AudioSet ontology 검증과 계층 탐색
  mapping.py                  # 명시적 source-family mapping
  temporal_reference.py       # planned reference / external reference 경계
  events.py                   # 확률에서 이벤트 구간 추출
  temporal_metrics.py         # onset, span, ambience 지표와 event matching
  evaluation.py               # 입력·reference·mapping·지표 연결
  reporting.py                # JSON, CSV, 집계
  batch.py                    # 여러 stem의 추론과 cache 재사용
  visualization.py            # 시간축 시각화
  corruption*.py              # controlled corruption과 결과 비교
  decision.py                 # 선택적 판정 정책
configs/                      # mapping, event extraction, corruption, filter 설정
outputs/                      # 로컬 추론·평가·시각화 산출물, Git 제외
```

`StemMetadata`의 핵심 필드는 stem_id, audio_path, source_description, role,
expected_intervals, optional_video_id이다. 조사한 WAVES에는 독립적인 final
description 필드가 없으므로 final_label을 source text로 사용할 경우
`description_origin=final_label`을 명시하고 원래 description을 따로 보존한다.
누락된 audio_path/support/영상 ID는 추정하지 않고 누락 상태로 남긴다.

`SEDBackend.predict(audio_path)`는 특정 WAVES metadata를 몰라도 동작한다.
예측 결과와 temporal reference는 evaluator에서 결합한다. 모델 교체나 외부 영상
annotation 연결 시 지표와 report 형식을 재사용하기 위한 경계이다. BEATs backend나
외부 영상 detector 자체를 구현하였다는 의미는 아니다.

## 5. Raw 출력 계약과 최초 검증 기준

최초 산출물은 threshold를 적용하지 않은 확률 cache다. NPZ에는 float32
`probabilities[T, 447]`, `frame_start_seconds[T]`, `frame_end_seconds[T]`, class ID/name
배열을 담는다. 프레임을 단일 timestamp만으로 표현하지 않아 마지막 부분 frame의
시간 범위를 분명하게 한다. JSON provenance에는 모델 이름, upstream revision,
checkpoint hash, 입력 hash, 원래/모델 sample rate, 입력 duration, chunk/padding 정책을
기록한다. 후속 threshold나 smoothing 변경은 이 cache를 다시 읽어 처리한다.

처리 정책은 실제 입력 길이를 보존한다. 10초보다 짧은 마지막 chunk는 모델 입력을 위해
pad하되, 결과에서는 실제 오디오 길이 밖의 frame을 제거하고 마지막 frame 끝을 실제
duration으로 제한한다. 임의 파일 길이를 10초 또는 가장 가까운 정수 초로 간주하지 않는다.
빈 파일, 잘못된 sample rate, 비유한 sample은 명시적으로 실패시킨다.

단일 WAV 추론을 확인하기 위한 최초 검증 기준은 다음과 같이 정의하였다.

1. 공개 strong checkpoint의 학습 weight가 모두 로드되고 head가 447개 class와
   정확히 대응하는지 확인한다. 임의 weight로 대체한 결과를 성공으로 간주하지 않는다.
2. 실제 WAV 한 개에 대해 CPU 추론을 완료하고 확률이 유한한 `[0,1]` 값인지, class
   순서와 출력 shape가 일치하는지 확인한다.
3. NPZ 저장 후 재로드한 확률/시간/라벨이 보존되는지 검증한다.
4. chunk 경계, 마지막 padding 제거, stereo/다른 sample rate 입력 처리를 작은
   합성 입력으로 검증한다. 이 검증은 음향 인식 정확도에 대한 주장이 아니다.
5. 소스 출처, checkpoint hash, 실행 명령, 실제 경과 시간과 테스트 결과를 README
   또는 별도 검증 기록에 남긴다. 실제 실행 결과와 이 계획을 구분한다.

## 6. 단계별 책임과 평가의 범위

다음 표는 설계 단계에서 정의한 책임 분리를 요약한다. 각 단계의 현재 구현과 실제
검증 결과는 개별 Phase 문서에 기록되어 있다.

| Phase | 구현 범위 | 확인 기준 |
| --- | --- | --- |
| 1 | 저장소/metadata/환경 조사, 구현 경계 확정 | 현재 문서와 실제 코드/JSON의 필드가 대응함 |
| 2 | frozen ATST-F Strong loading, 단일 WAV 추론, raw NPZ/provenance | 실제 checkpoint로 실행 완료, 프레임/class/시간 검증, cache 재로드 |
| 3 | WAVES adapter, ontology loader, 명시적 manual mapping | 존재하는 class ID만 허용, 여러 allowed classes 지원, mapping 누락은 `unsupported_mapping`, planned support provenance |
| 4 | event extraction, 역할별 metric, JSON/CSV 집계 | 합성 interval로 IoU와 missing/extra 검증, tolerance 제한의 최적 일대일 onset matching |
| 5 | batch와 waveform/expected/probability/detected 시각화 | 한 시간축 위의 정렬 확인, 실패 stem과 누락 metadata 추적 |
| 6 | target event shift/remove/duplicate 및 duration corruption | 알려진 100/200/500/1000 ms 변형에 대한 metric 민감도 측정 |
| 7 | 선택적 PASS/REVIEW/FAIL/UNSUPPORTED 정책 | config가 지정된 metric에만 판단, null이면 판단 비활성 |

Phase 4에서는 target score를 allowed family 확률의 최대값으로 계산하고 event threshold,
smoothing, minimum duration, onset tolerance를 config로 분리한다. onset은 event 수와
matching error, span은 구간 집합의 IoU/coverage/precision과 boundary error, ambience는
occupancy와 confidence를 사용한다. Ontology의 parent/child 동시 활성화를 foreign event로
즉시 실패 처리하지 않는다. Allowed family 밖의 top-k activation은 우선 보고 자료다.

WAVES 시간 기준은 `expected_support` 또는 `planned_support`로 부른다. 이와 SED의
일치는 내부 timing consistency이며 실제 video synchronization을 입증하지 않는다.
`ExternalVideoReference`는 사람이 만든 annotation 또는 별도 visual detector의 시간
기준을 연결하기 위한 경계이다. 해당 인터페이스는 Phase 4에서 구현하였으며, 실제 외부
annotation이나 visual detector는 포함하지 않는다. Human evaluation 플랫폼도 구현
범위에 포함하지 않았다.
