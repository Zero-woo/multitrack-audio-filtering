# Phase 4: 저장된 확률 기반 이벤트 추출과 역할별 평가

## 개요와 평가 범위

평가 모듈은 정규화한 WAVES stem metadata와 `FramePrediction` NPZ에 저장된 소리 확률을
입력받는다. 목표 소리의 이벤트 구간을 추출하고 역할별 시간 지표와 의미적 근거를 계산하여
JSON/CSV로 기록한다. 추출 설정을 변경할 때는 저장된 확률을 재사용하므로 SED 추론을
반복하지 않는다. 기본 NumPy 환경에서 실행하며 WAVES 생성 코드와 원본 음원을 보존한다.

기본 시간 기준은 WAVES의 생성 계획이다. 따라서 산출 지표는 계획과 생성 음원 사이의
일관성을 나타내며 실제 영상과의 동기화 정확도를 직접 입증하지 않는다. 예제 추출 설정은
보정 전 초깃값이다. 품질 판정은 별도의 [Phase 7 정책](phase7-validation.md)에서 수행하며,
판정 설정을 지정하지 않은 평가 결과의 `decision`은 `null`이다.

이 문서는 현재 평가 계약과 **2026-09-27의 Phase 4 기준 검증 결과**를 구분하여 설명한다.
배치 추론·시각화와 변형 실험은 각각 [Phase 5](phase5-validation.md),
[Phase 6](phase6-validation.md)에서 설명한다.

## 코드 구성

| 모듈 | 책임 |
| --- | --- |
| `evaluation_config.py` | 평가 설정 schema와 값의 유효성 검사 |
| `mapping.py` | 허용 클래스 확률을 목표 확률로 집계 |
| `events.py` | 목표 확률 평활화와 검출 구간 추출 |
| `temporal_reference.py` | 기대 시간 구간의 출처·상태·사용 가능 여부 해석 |
| `temporal_metrics.py` | onset·span·ambience 지표 계산 |
| `evaluation.py` | cache 동일성 검사와 stem 단위 평가 조합 |
| `reporting.py` | stem·clip·dataset 집계, JSON/CSV 출력과 입력 보호 |
| `phase4.py` | `evaluate` CLI 입력과 실행 연결 |

모듈 경로의 기준은 `src/waves_sed/`이다.

## 입력과 실행

`evaluate`는 Phase 3의 `adapt-waves`가 만든 schema 1 manifest를 `--stems`로 읽는다.
`StemMetadata.from_dict()`와 `load_stems()`는 필수 필드, duplicate ID/key, 비유한 수,
잘못된 구간과 알 수 없는 필드를 검사한다. stem의 상대 audio 경로와 provenance 입력 경로는
manifest 디렉터리 기준으로 해석한다. 저장된 Phase 3 mapping 힌트는 사용하지 않고,
현재 mapping 설정과 해당 prediction의 실제 class ID 순서로 다시 해석한다.

실제 frozen metadata를 정규화하는 명령은 다음과 같다.

```powershell
cd C:\multitrack-audio-filtering
.venv/Scripts/python.exe -m waves_sed adapt-waves --frozen-finals C:/WAVES/data/frozen_pass2/frozen_finals.json --frozen-reports C:/WAVES/data/frozen_pass2/frozen_reports.json --mappings configs/source_mappings.example.json --output outputs/phase3/frozen-stems.json
```

Prediction index는 아래 구조만 허용한다. 예시 ID와 경로는 실행 대상 manifest 및 cache에
맞게 지정한다. 상대 NPZ 경로의 기준은 **index JSON이 있는 디렉터리**이다.

```json
{
  "schema_version": 1,
  "predictions": {
    "demo_clip::01": "../predictions/demo-stem-01.npz"
  }
}
```

Index의 최상위 필드는 `schema_version`, `predictions` 두 개이며 version은 정수 `1`이다.
Unknown stem ID, duplicate key, 빈 경로와 잘못된 구조는 명령 오류다. 일부 stem의 index 항목을
생략하는 것은 허용한다. 생략되었거나 지정한 cache 파일이 없으면 해당 행은 `missing_prediction`,
NPZ가 잘못되었으면 `invalid_prediction`으로 남는다. 그 stem의 event 수나 오차를 0으로 만들지 않는다.

아직 cache를 연결하지 않은 상태에서도 실제 manifest 전체의 지원 범위를 보고할 수 있다.
다음 명령은 빈 index를 의도적으로 만들므로 평가 수치는 산출하지 않는 진단용 실행이다.

```powershell
New-Item -ItemType Directory -Force outputs/phase4 | Out-Null
'{"schema_version":1,"predictions":{}}' | Set-Content -Encoding UTF8 outputs/phase4/prediction-index.json
.venv/Scripts/python.exe -m waves_sed evaluate --stems outputs/phase3/frozen-stems.json --predictions outputs/phase4/prediction-index.json --mappings configs/source_mappings.example.json --config configs/evaluation.example.json --output-dir outputs/phase4/frozen-report
```

실제 평가에는 해당 index에 각 stem의 **실제 음원으로 추론한 cache**를 연결한다.
파일 연결은 stem 이름이나 source 설명의 유사성이 아니라 오디오 동일성 검사를 통과해야 한다.
`--ontology path.json`은 선택 사항이며 생략하면 번들 ontology를 사용한다. `--config`와
`--output-dir`은 필수다. 명령은 stdout에 출력 위치와 상태별 개수 요약을 기록한다.

## 설정과 event 추출

`configs/evaluation.example.json`은 다음 기본값을 명시한다.

```json
{
  "schema_version": 1,
  "event": {
    "threshold": 0.5,
    "median_window_frames": 1,
    "min_duration_seconds": 0.0
  },
  "onset_tolerance_seconds": 0.2,
  "allow_ambiguous_reference": false,
  "outside_family_threshold": 0.5,
  "outside_family_top_k": 5
}
```

`EvaluationConfig.load()`는 정수 schema version 1을 요구한다. 나머지 생략 필드는 위 기본값을
사용하며, unknown field, duplicate key, NaN/Infinity와 잘못된 타입은 거부한다. Threshold는
`[0,1]`, duration/tolerance는 0 이상, median window는 양의 홀수 frame 수, top-k는 양의 정수다.
예시값은 실험 출발점이며 실제 자료에 대한 품질 threshold 보정 결과가 아니다.

`aggregate_target()`은 mapping에서 허용한 클래스 중 backend에 실제로 존재하는 ID만 모아
`P_target(t) = max_c P(c,t)`를 계산한다. Mapping이 없거나 허용 ID가 backend에 하나도 없으면
`unsupported_mapping`이다. 이 경우 0 확률 벡터를 대신 만들어 평가하지 않는다.

`smooth_target()`과 `extract_events()`의 순서는 다음과 같다.

1. Raw target 확률에 선택한 median window를 적용한다. `1`은 smoothing을 하지 않는 설정이다.
   각 연속 구간의 양 끝은 가장자리 값으로 padding한다. 창 크기는 초가 아닌 frame 수이므로
   짧은 마지막 frame도 한 frame으로 취급한다.
2. `smoothed_probability >= threshold`인 frame을 활성화한다.
3. 실제 시간 경계가 닿는 활성 frame을 하나의 event로 묶는다. 작은 양의 시간 gap도 별도 구간이며
   smoothing이나 event 병합이 gap을 건너지 않는다.
4. 실제 event 길이가 `min_duration_seconds` 이상인 event만 남긴다. 경계의 부동소수점 뺄셈에는
   절대 `1e-12`초 이내 허용을 사용한다.

모든 event는 실제 frame support의 반개구간 `[start, end)`이며 마지막 부분 frame도 원래
끝 시각을 사용한다. Raw NPZ와 target 원본은 바뀌지 않는다. 아래 confidence 측정에는
smoothing·threshold·최소 길이 제거 전의 raw 확률을 사용한다.

## 시간 기준과 사용 가능 상태

`TemporalReference.resolve(stem, *, allow_ambiguous=False)`는 origin/status/intervals,
사용 가능 여부와 이유, provenance를 포함하는 `TemporalSupport`를 반환한다.
기본 구현은 `WavesPlannedReference`이며 `origin=waves_pass1_planned`를 요구한다.

| 상태 | 기본 평가 정책 |
| --- | --- |
| `planned` + 유효한 비어 있지 않은 구간 | 계획과의 비교 허용 |
| `ambiguous` | 기본 제외. `allow_ambiguous_reference=true`일 때만 명시적으로 허용 |
| `missing`, 구간 `null`, 구간 `[]` | 평가 불가. 무음 정답으로 해석하지 않음 |
| 알 수 없는 origin/status | 평가 불가 |

Relabel, 역할 변경, merge 등으로 `ambiguous`인 기록은 허용 설정을 사용해도 상태가
`planned`로 바뀌지 않는다. 원래 구간과 이유, opt-in 설정을 report에 보존한다. WAVES 계획은
영상의 ground truth가 아니므로 이 측정만으로 실제 영상과의 동기화를 주장할 수 없다.
선택 parent의 계획만 사용하며 merged child의 구간을 임의로 합치지 않는다.

`planned`로 저장되었지만 issues, merge 목록, 명시적인 final/planned label·role에서 모호함이
드러나면 `inconsistent_reference_status`로 거부한다. 이 모순은 opt-in으로 우회하지 않는다.
외부 provider의 명시적 usable 빈 annotation은 인터페이스에서 허용하지만, WAVES 빈 계획은
평가 불가라는 정책을 유지한다.

`ExternalVideoReference`는 독립적인 영상 annotation/detector를 연결하기 위한 Protocol이다.
실제 영상 검출기는 구현되어 있지 않다. Python API `evaluate_stem(..., reference=provider)`로
같은 계약의 provider를 연결할 수 있으며, 현재 CLI는 `WavesPlannedReference`를 사용한다.
외부 구현은 영상 식별자와 annotation/detector provenance를 제공하고, 누락·모호함을 선언해야 한다.

## Cache 동일성과 관측 범위

`validate_cache_identity()`는 파일명이나 audio ID 대신 SHA-256을 비교한다.

- 실제 stem audio 파일이 있으면 현재 bytes의 hash와 cache의 `audio_sha256`을 비교한다.
  성공 상태는 `verified_current_audio`, `source_file_verified=true`다.
- 실제 파일이 없더라도 명시적인 `provenance.raw_final_stem.sha256`이 cache hash와 같으면
  `verified_manifest_hash`, `source_file_verified=false`로 사용할 수 있다. 현재 원본 파일을
  읽어 검증한 경우와 구별되며 manifest의 명시적 hash에 의존한 연결이다.
- 현재 audio와 manifest hash가 다르거나, cache가 다른 audio를 가리키거나, hash가 없거나
  잘못되었거나, 존재하는 원본을 읽을 수 없으면 사용 불가 이유를 남긴다.

이 검사는 source audio를 decode하거나 추론하지 않는다. Cache provenance를 보존하고
실제 source 또는 manifest에 명시된 bytes와 연결하는 검사다.

평가에는 backend 이름과 양의 유한 `duration_seconds`가 있어야 한다. Frame은 0초부터
시작해 빈틈없이 이어져야 하고 마지막 끝 시각이 선언된 duration과 같아야 한다
(끝 시각 비교의 절대 허용은 `1e-12`초). Expected 구간은 cache 전체 범위 안에 있어야 한다.
일부 expected event가 관측된 부분에 들어오더라도 cache 내부 gap은 관측되지 않은 시간이므로
완전한 clip의 extra activity나 precision을 평가하지 않는다.

저수준 event 함수는 gap을 안전하게 분리하지만, 완전한 stem report의 관측 정책은 더 엄격하다.
저수준 `ambience_metrics()`도 expected 구간이 연속된 관측 block 안에 완전히 들어오지 않으면
예외를 발생시켜 관측 누락을 확률 0으로 취급하지 않는다.

## 역할별 metric 계약

아래에서 `E`는 expected 구간의 합집합, `D`는 detected 구간의 합집합이며 `|·|`는 초 단위
길이다. Duration은 겹치거나 닿는 구간을 병합한 뒤 측정한다. Onset matching에서는 각 입력
구간을 개별 event로 보존하므로 같은 시작 시각의 event도 서로 다른 개체다.

### Onset

Expected/detected 시작 시각을 정렬한 동적 계획법으로 일대일 대응을 구한다. 먼저 tolerance
이내의 **대응 수를 최대화**하고, 같은 대응 수에서는 **절대 onset 오차 합을 최소화**한다.
Greedy 최근접 매칭이 아니다. 허용 경계는 포함하며 양의 tolerance 경계에서는 상대 `1e-12`
오차를 허용한다. Tolerance가 0이면 정확히 같은 시작 시각만 대응한다.
Event 수가 각각 `N`, `M`일 때 DP 시간·공간 복잡도는 `O(NM)`이고 정렬 비용이 추가된다.

| 필드 | 의미·단위 |
| --- | --- |
| `expected_event_count`, `detected_event_count` | 입력 expected/detected event 개수 |
| `matched_event_count` | 대응된 쌍 수 |
| `missing_event_count`, `extra_event_count` | expected−matched, detected−matched |
| `event_recall`, `event_precision` | matched/expected, matched/detected |
| `mean_onset_error_ms`, `median_onset_error_ms` | 대응된 쌍의 절대 시작 오차 평균·중앙값, 밀리초 |
| `matches` | 원래 입력의 expected/detected index, signed/absolute onset error, 밀리초 |

Signed error는 `detected_start - expected_start`다. 대응이 없으면 평균·중앙값은 `null`이다.
분모가 0인 recall/precision도 `null`이다. 실제 평가 가능한 reference에서 검출이 0개인 경우
missing count와 recall 0은 유효한 측정이다. Reference 자체가 누락된 경우와 구별한다.

### Span

| 필드 | 계산 |
| --- | --- |
| `temporal_iou` | `|E ∩ D| / |E ∪ D|` |
| `expected_coverage` | `|E ∩ D| / |E|` |
| `detected_precision` | `|E ∩ D| / |D|` |
| `onset_error` | 첫 detected 시작−첫 expected 시작, signed 초 |
| `offset_error` | 마지막 detected 끝−마지막 expected 끝, signed 초 |
| `out_of_window_activation` | `|D \ E|`, 초 |

Expected/detected/intersection/union duration도 `_duration_seconds` 필드로 함께 기록한다.
Onset/offset error는 여러 구간 전체의 가장 바깥 경계 비교이며 구간별 matching 오차가 아니다.
어느 쪽이든 비면 경계 오차는 `null`, 0인 분모의 비율도 `null`이다. 두 집합이 모두 비면 IoU는
`null`, 한쪽만 비면 IoU는 0이다. Empty WAVES reference는 이 저수준 계산에 들어가기 전에 제외된다.

### Ambience

| 필드 | 계산 |
| --- | --- |
| `occupancy_in_expected_span` | `|E ∩ D| / |E|` |
| `mean_target_confidence` | `Σ_frame P_target(frame) × |frame ∩ E| / |E|` |
| `out_of_window_activity` | `|D \ E|`, 초 |

Onset 정확도는 요구하지 않는다. Confidence는 raw 확률이며 부분적으로 겹친 frame은 겹친
길이만큼만 기여한다. Expected 구간끼리 겹쳐도 중복 계산하지 않는다. Expected가 비면 occupancy와
confidence는 `null`이다. Expected/detected/intersection/observed-expected duration을 초로 기록한다.

## 의미적 근거와 평가 보고서

`semantic_evidence`는 허용 클래스 밖에서 raw 최대 확률이 `outside_family_threshold` 이상인
클래스를 peak 내림차순으로 top-k 기록한다. 각 행에는 duration 가중 평균 확률, threshold 이상인
총 길이, 모델 이름과 ontology 이름, 허용 클래스와의 ancestor/descendant 관계가 들어 있다.
Hierarchy를 알 수 없으면 `unknown_hierarchy`로 남긴다. Allowed set 밖이라는 이유만으로 foreign
sound나 실패로 판정하지 않는다. Parent/child의 동시 활성도 관계 정보와 함께 확인할 수 있다.

Mapping이 명시한 `foreign_classes`는 별도 목록에 기록하며 threshold 미만 값도 보존한다.
Backend에 없는 foreign ID도 별도로 보고한다. 이 evidence에는 event smoothing·최소 길이 제거를
적용하지 않는다. 증거와 사용자가 선언한 foreign 목록은 품질 판정과 분리되어 있다.

`evaluate_stem()`의 행에는 source/role/backend, mapping과 reference 상태, expected/detected 구간,
사용 불가 이유, cache 검사, metric, semantic evidence, 설정·입력·모델 provenance가 들어간다.
Cache와 mapping이 유효하면 reference가 모호해 평가하지 않는 행에도 detection을 제공할 수 있다.
따라서 `detection_status=available`과 `evaluation_status=unavailable`이 함께 나오는 것은 정상이다.

`write_reports()`의 출력은 다음과 같다.

- `dataset.json`: 전체 stem 기록, 역할별 dataset/clip 요약, provenance와 현재 출력 파일 index.
- `stems/<SHA256(stem_id)>.json`: stem별 전체 기록. Windows에서 `::`를 파일명으로 사용하지 않는다.
- `clips/<SHA256(clip_key)>.json`: clip별 stem 기록과 요약.
- `stems.csv`, `clips.csv`: 비교용 평면 표. 배열과 객체는 JSON 문자열, null은 빈 셀이다.

집계는 `evaluated` 행의 유한 수치만 역할별 산술평균으로 계산하며 metric마다
`contributing_count`를 남긴다. 미지원·누락·모호한 행을 0으로 끼워 넣지 않는다. 측정이 없으면
mean과 onset pooled count는 `null`, 기여한 행 수는 0이다. Onset micro precision/recall은
평가된 행의 대응·expected·detected 개수를 합쳐 계산하며, 비율의 분모가 0이면 `null`이다.
Onset 평균 오차의 일반 mean은 stem별 평균의 평균이며 모든 쌍을 합친 별도 통계가 아니다.

평가 설정, metric 구현 버전, backend/checkpoint/preprocessing/upstream revision, mapping hash,
reference origin이 다른 evaluated 행은 같은 집계로 섞지 않고 오류를 낸다.
누락된 context는 null로 유지하므로 알려진 context와 같은 것으로 간주하지 않는다.

출력 전 모든 파일 경로를 검사하여 명시적 입력 JSON, NPZ, stem audio, provenance의 원본 입력과
audio/video를 덮어쓰지 못하게 한다. 실제 같은 파일을 가리키는 hard link와 출력 간 alias도 검사한다.
각 파일은 임시 파일을 통한 atomic replace로 쓰고 `dataset.json`을 마지막에 교체한다.
여러 파일 전체를 하나의 transaction으로 쓰는 것은 아니다. 이전 실행에서 남은 파일은 삭제하지
않으며 현재 결과는 `dataset.json`에 등재된 파일만 사용해야 한다.

## 검증 방법과 기준 실행 결과

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/ruff.exe check src tests scripts
.venv/Scripts/ruff.exe format --check src tests scripts
```

Synthetic regression은 greedy matching 실패와 최대 대응 수 우선 정책, 원래 index 보존,
작은 문제에 대한 전수 assignment 비교, 겹친 구간, 부분 frame 가중치, 시간 gap,
미관측 구간, empty/null 정책, cache 동일성, 모호한 계획 opt-in과 source 보호를 확인한다.
실제 모델 품질이나 WAVES 영상 동기화 정확도를 이 테스트 통과만으로 검증한 것은 아니다.

### 회귀 테스트와 의존성 분리 검증: 2026-09-27

- 전체 테스트는 **515 passed, 1 skipped**로 나타났으며 실행 시간은 13.69초였다. Skip은
  환경변수를 지정하지 않은 실제 checkpoint 통합 테스트였다. audioread의 Python 3.11
  deprecation 경고 3개가 발생하였다.
- Ruff lint/format, `git diff --check`, wheel build를 통과하였다. 추론 코드를 변경하지 않았으므로
  모델을 다시 로딩하는 Phase 2 통합 검증은 반복하지 않았다.
- `.cache/phase4-package-smoke`에 wheel과 NumPy 1.26.4만 설치하고 실제 cache API 평가와
  frozen metadata CLI 평가를 실행하였다. torch/torchaudio/librosa/soundfile이 없음을 확인하였다.
- 실제 WAVES frozen 자료 **29 clips / 62 stems**를 평가 명령에 연결하였다. 원본 WAV/cache가
  없어 빈 prediction index를 사용하였으며 **62개 모두 unavailable**로 나타났다. 당시 사용한
  `source_mappings.example.json`의 mapping 결과는 supported 2 / unsupported 60이었고,
  reference는 planned 19 / ambiguous 43으로 보존하였다. 평가 가능한 metric의 기여 수는 0이었으며
  unavailable 행의 missing/extra event 수는 `null`로 기록하였다.

### 실제 cache를 이용한 설정 변경 검증

Phase 2에서 생성한 공개 지하철 음원 cache의 938 frames를 사용하는 재현 명령은 다음과 같다.

```powershell
.venv/Scripts/python.exe scripts/verify_cached_evaluation.py --prediction outputs/predictions/metro.npz --audio .cache/PretrainedSED/test_files/752547__iscence__milan_metro_coming_in_station.wav --class-id /m/0195fx --output-dir outputs/phase4/metro-demo
```

Class는 실제 vocabulary의 `Subway, metro, underground`이다. 검증 스크립트의 reference는
**전체 길이를 쓰는 synthetic fixture**이며, origin을 `synthetic_full_track_demonstration`으로
기록한다. 이 fixture는 WAVES 계획이나 영상 annotation을 대체하지 않으며, 표의 IoU는
fixture와 검출 구간의 일치도이다. 실제 WAVES 인식·동기화 성능으로 해석하지 않는다.

| 추출 threshold | 검출 event 수 | synthetic 전체 길이 대비 IoU |
| --- | ---: | ---: |
| 0.2 | 1 | 0.7978965802497207 |
| 0.5 | 0 | 0.0 |

두 설정 모두 추가 추론 없이 실행되었고 NPZ SHA-256은 유지되었다. 실제 cache를 이용한
계산·보고서 출력과 threshold 변경 후 재평가가 정상 동작함을 확인하였다.
스크립트와 CLI 모두 입력 audio뿐 아니라 cache에 기록된 원본 audio 경로도 보호한다.

로컬 산출물은 Git에서 제외된다.

- `outputs/phase4/frozen-reports/`: frozen 62개 stem 지원 상태 진단.
- `outputs/phase4/metro-demo/threshold-0.2/`, `threshold-0.5/`: synthetic reference를 쓴 실제 cache report.
- `outputs/phase4/wheel-frozen-reports/`, `wheel-metro-demo/`: NumPy만 설치한 wheel 환경의 동일 검증.

## 적용 한계

기준 검증에는 실제 WAVES stem WAV가 포함되지 않았다. 실제 생성 품질 측정에는 물리 파일이
확보된 WAVES 실행 산출물과 stem별 cache가 필요하다. 영상 동기화 검증에는 생성 계획과 독립적인
시간 기준도 필요하다. 자료 연결 상태와 실데이터 검증 절차는
[실데이터 검증 준비 문서](real-data-readiness.md)에서 설명한다.
