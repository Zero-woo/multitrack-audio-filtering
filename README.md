# WAVES stem SED validation

WAVES가 생성한 stem을 frozen AudioSet-Strong SED 모델로 검사하는 독립적인 후처리 프로젝트입니다.
WAVES 생성 파이프라인을 수정하거나 모델을 학습하지 않습니다.

현재 **Phase 1~7: frozen 추론, metadata/mapping, 역할별 평가, batch/시각화, controlled corruption과 설정 기반 판정**을 구현했습니다.
WAVES의 planned support와 SED 출력 비교는 내부 consistency 검사이며,
실제 영상과의 동기화를 입증하지 않습니다.

구조 분석, 확인된 metadata 누락, 단계별 계획은 [분석 문서](docs/phase1-analysis.md),
추론 검증은 [Phase 2 검증 기록](docs/phase2-validation.md),
metadata/mapping 사용법은 [Phase 3 문서](docs/phase3-validation.md),
cache 기반 평가와 측정 단위는 [Phase 4 문서](docs/phase4-validation.md),
batch/시각화 사용법은 [Phase 5 문서](docs/phase5-validation.md),
파형 변형과 조작 전후 비교는 [Phase 6 문서](docs/phase6-validation.md),
선택적인 판정 설정은 [Phase 7 문서](docs/phase7-validation.md)에 있습니다.

## 설정 기반 판정 (Phase 7)

`evaluate`와 `visualize`에 `--filter-config`를 추가하면 계산된 metric에 정책을 적용합니다.
기준이 없으면 판정하지 않습니다. [기본 설정](configs/filter.example.json)은 모두 null이며,
실제 운영 기준은 validation 결과에 따라 별도 파일에 명시합니다.

```powershell
.venv/Scripts/python.exe -m waves_sed evaluate --stems outputs/stems.json --predictions outputs/batch/prediction-index.json --mappings configs/source_mappings.example.json --config configs/evaluation.example.json --filter-config configs/filter.example.json --output-dir outputs/filtered-reports
```

숫자 경계 또는 `{"pass": 값, "fail": 값}`으로 REVIEW 구간을 설정할 수 있습니다.
누락·모호한 근거는 REVIEW, 적용 가능한 역할의 미지원 mapping은 UNSUPPORTED로 남깁니다.
JSON/CSV와 그림에 각 조건의 값·경계·결과를 기록하며, 정책 변경에 새 추론은 필요하지 않습니다.
PASS는 명시한 시간 일관성 조건의 충족이며 전체 음질이나 영상 동기화의 보증은 아닙니다.
자세한 [경계값·판정 순서·실행법](docs/phase7-validation.md)을 확인하세요.

## Controlled corruption (Phase 6)

원본 대조군을 보존하고 시간 이동, 구간 삭제/복제, 길이 단축/반복 연장 WAV를 만듭니다.
각 변형은 독립적으로 생성되며 expected support는 그대로 유지합니다. 예제에는
+100/200/500/1000ms 이동이 포함됩니다. 구간 편집 예제는 대상 음원에 맞게 바꿉니다.

```powershell
uv pip install --python .venv/Scripts/python.exe -e ".[corruption]"
.venv/Scripts/python.exe -m waves_sed corrupt --stems outputs/stems.json --stem-id "실제 stem ID" --config configs/corruption.example.json --output-dir outputs/corruption/experiment
.venv/Scripts/python.exe -m waves_sed batch-infer --stems outputs/corruption/experiment/stems.json --output-dir outputs/corruption/batch
.venv/Scripts/python.exe -m waves_sed evaluate --stems outputs/corruption/experiment/stems.json --predictions outputs/corruption/batch/prediction-index.json --mappings configs/source_mappings.example.json --config configs/evaluation.example.json --output-dir outputs/corruption/reports
.venv/Scripts/python.exe -m waves_sed compare-corruptions --experiment outputs/corruption/experiment/experiment.json --report outputs/corruption/reports/dataset.json --output-dir outputs/corruption/comparison
```

`comparison.json/csv`에 metric 차이와 검출 구간 변화를 남깁니다. 비교에는 추론이나 오디오
라이브러리가 필요하지 않습니다. 기존 실험 파일은 덮어쓰지 않으므로 생성에는 새 출력 경로를
사용합니다. 원본 길이/채널/샘플레이트를 유지하며, 경계 손실과 중첩을 기록합니다.
구간 복제가 실제 extra event 검출을 보장하지는 않습니다. [처리 정책과 실제 검증 결과](docs/phase6-validation.md)를 확인하세요.

## Batch와 시간축 그림 (Phase 5)

여러 stem을 처리할 때는 정규화 manifest를 `batch-infer`에 전달합니다. 하나의 frozen 모델로
차례대로 추론하고, 일치하는 cache는 재사용합니다. 일부 stem이 실패해도 나머지를 계속 처리하며
`batch-status.json`에 사유를 남깁니다. 성공한 cache만 `prediction-index.json`에 연결됩니다.

```powershell
# 그림이 필요할 때 선택 설치. Batch cache 재사용/평가는 NumPy만으로 가능합니다.
uv pip install --python .venv/Scripts/python.exe -e ".[visualization]"

# stems.json과 mapping 경로는 자신의 실제 입력으로 바꿉니다.
.venv/Scripts/python.exe -m waves_sed batch-infer --stems outputs/stems.json --output-dir outputs/batch
.venv/Scripts/python.exe -m waves_sed evaluate --stems outputs/stems.json --predictions outputs/batch/prediction-index.json --mappings configs/source_mappings.example.json --config configs/evaluation.example.json --output-dir outputs/reports
.venv/Scripts/python.exe -m waves_sed visualize --stems outputs/stems.json --predictions outputs/batch/prediction-index.json --mappings configs/source_mappings.example.json --config configs/evaluation.example.json --output-dir outputs/visuals
```

`outputs/visuals/index.html`에서 stem별 그림을 확인합니다. PNG 기본 출력에
`--format svg`를 사용하면 SVG를 저장하고, `--stem-id "실제 ID"`로 하나만 선택할 수 있습니다.
원본 파형의 min/max, expected support, raw/median target 확률과 detected 구간이 같은 시간축에
표시됩니다. 누락되거나 모호한 자료는 그림에도 명시하며 확률 0이나 정상 판정으로 대체하지 않습니다.

Batch는 matching cache만 재사용하고, 기존 cache가 손상되거나 입력/모델이 다르면 충돌로
남깁니다. `--overwrite`를 명시하면 cache를 새로 추론합니다. 실패가 하나라도 있으면 종료 코드는
1이며 나머지 성공 결과는 보존됩니다. 원본 audio와 입력 metadata는 덮어쓰지 않습니다.

## Cache 기반 시간 평가 (Phase 4)

`adapt-waves`로 만든 metadata와 해당 stem에서 추론한 raw NPZ를 연결합니다.
Prediction index는 `{"schema_version":1,"predictions":{"실제 stem_id":"../predictions/stem.npz"}}`
형식이며 상대 경로는 index 파일 기준입니다. 설정을 바꿔도 추론을 다시 실행하지 않습니다.

```powershell
.venv/Scripts/python.exe -m waves_sed evaluate --stems outputs/phase3/stems.json --predictions outputs/phase4/prediction-index.json --mappings configs/source_mappings.example.json --config configs/evaluation.example.json --output-dir outputs/phase4/reports
```

위 metadata/index 경로는 실제 파일로 바꿉니다. 출력은 stem/clip별 JSON, `dataset.json`,
`stems.csv`, `clips.csv`입니다. onset은 최적 일대일 matching, span은 구간 합집합의 IoU/coverage,
ambience는 occupancy와 시간 가중 raw confidence를 계산합니다. NumPy만으로 실행할 수 있습니다.

WAVES 계획이 `ambiguous`이면 기본적으로 시간 평가에서 제외합니다. 누락된 cache/reference와
미지원 mapping은 `unavailable` 사유와 `metrics=null`로 남깁니다. Source/cache SHA-256과 전체
관측 시간축을 확인하며, outside-family evidence를 자동 실패로 판정하지 않습니다.
판정 설정을 생략하면 `decision=null`이며 예제 threshold는 아직 보정되지 않았습니다.

현재 저장된 실제 공개 예제 cache로 threshold 변경·report 출력을 확인하는 명령:

```powershell
.venv/Scripts/python.exe scripts/verify_cached_evaluation.py --prediction outputs/predictions/metro.npz --audio .cache/PretrainedSED/test_files/752547__iscence__milan_metro_coming_in_station.wav --class-id /m/0195fx --output-dir outputs/phase4/metro-demo
```

이 검증의 expected support는 **의도적인 전체 길이 synthetic fixture**이며 WAVES 계획이나
영상 annotation이 아닙니다. 실제 WAVES 음원 품질 검증에는 해당 run의 WAV와 cache가 필요합니다.

## WAVES metadata와 source mapping (Phase 3)

실제 frozen 자료 29개 clip/62개 stem을 모델 없이 정규화할 수 있습니다.

```powershell
.venv/Scripts/python.exe -m waves_sed adapt-waves --frozen-finals C:/WAVES/data/frozen_pass2/frozen_finals.json --frozen-reports C:/WAVES/data/frozen_pass2/frozen_reports.json --mappings configs/source_mappings.example.json --output outputs/phase3/frozen-stems.json
.venv/Scripts/python.exe -m waves_sed map-source --description "dog barking" --mappings configs/source_mappings.example.json
```

실제 WAVES 실행 결과는 `adapt-waves --metadata ... --sam-manifest ...` 또는 `--dsp-report ...`로
연결합니다. Mapping은 최종 label을 기준으로 명시적 alias만 사용합니다.
설정에 없는 설명은 `unsupported_mapping`으로 남기고, planned description·role·시간·merge 이력은
별도로 보존합니다. `supported`는 mapping 가능 여부이며 소리 검출이나 품질 통과를 의미하지 않습니다.

공식 ontology와 ATST-F vocabulary에는 차이가 있습니다. 447개 모델 ID 중 31개는 공식 archive에
없고, 11개는 표시명이 다릅니다. 실제 ID를 보존하여 차이를 보고하며 없는 hierarchy는 추정하지 않습니다.
상세한 누락/시간 기준 정책과 cache의 target max 집계는 [Phase 3 문서](docs/phase3-validation.md)를 참고하세요.

## 설치 및 실행 (PowerShell)

검증 환경: Windows / Python 3.11.9 / CPU. WAVES와 별도 환경을 사용합니다.
NumPy만 설치하면 raw cache를 읽을 수 있고, `[atst]` 추가 의존성은 추론에만 필요합니다.
PyTorch/torchaudio는 모델과 mel 변환, einops는 ATST patch 변환,
librosa/soundfile은 upstream과 동일한 오디오 읽기·리샘플링에 사용합니다.

```powershell
cd C:\multitrack-audio-filtering
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-cpu.txt

# 공식 약 329 MiB 체크포인트 다운로드. 이미 있으면 SHA-256만 검증합니다.
.venv/Scripts/python.exe -m waves_sed download

# 임의의 WAV: mono 변환, 16 kHz 리샘플링, frozen CPU 추론
.venv/Scripts/python.exe -m waves_sed infer C:/path/to/stem.wav --output outputs/predictions/stem.npz

# 모델/체크포인트 없이 raw cache 조회, 선택적으로 CSV 출력
.venv/Scripts/python.exe -m waves_sed inspect outputs/predictions/stem.npz --csv outputs/predictions/stem.csv
```

`uv` 없이도 `py -3.11 -m venv .venv`와
`.venv/Scripts/python.exe -m pip install -r requirements-cpu.txt`를 사용할 수 있습니다.
기존 `.venv`가 있으면 생성 명령을 다시 실행할 필요가 없습니다.
CUDA 환경은 이 작업에서 검증하지 않았습니다. 별도 환경에 해당 플랫폼의 호환되는
torch/torchaudio 2.10.0을 설치한 뒤 `pip install -e ".[atst]"`, `--device cuda`를 사용합니다.
명시적으로 CUDA를 요청했는데 사용 불가능하면 새 추론은 오류를 반환합니다.

같은 입력/모델/provenance에 해당하는 출력이 이미 있으면 `infer`는 모델 로딩 없이
cache를 재사용합니다. 입력 파일의 내용이나 모델 설정이 다르면 기존 cache를 덮어쓰지 않고
오류를 반환합니다. 새 추론을 강제하려면 `--overwrite`를 사용합니다.
재사용은 checkpoint 파일이 없어도 가능하며, 실행 환경(device/dependency) 변경만으로
기존 확률을 다시 계산하지 않습니다. 기존 결과의 실행 환경은 metadata에 남아 있습니다.
기본 CPU thread 수는 4이며 `--threads`로 조절합니다.

## 원시 확률 형식

NPZ는 pickle 없이 읽을 수 있습니다. smoothing, binary event 추출, 판정 threshold는
적용하지 않습니다. CLI의 top classes는 최대 확률 순서의 확인용 요약입니다.

| 필드 | 형식 / 의미 |
| --- | --- |
| `schema_version` | 현재 1 |
| `probabilities` | float32 `[T, 447]`, sigmoid 이후 확률 |
| `frame_start_seconds`, `frame_end_seconds` | float64 `[T]`, 초 단위 출력 bin `[start, end)` |
| `class_ids`, `class_names` | 해당 checkpoint의 정확한 출력 순서, 문자열 447개 |
| `metadata_json` | 입력/모델 hash, upstream revision, 전처리·chunk 정책, sample rate, dependency 버전 |

10초 chunk마다 250개 frame을 계산합니다. 마지막 chunk는 zero pad하지만 padding-only
frame은 저장하지 않고 마지막 bin 끝은 **원본 WAV duration**으로 제한합니다.
리샘플링으로 올림된 sample duration은 `resampled_duration_seconds`에 따로 저장합니다.
40ms bin은 출력 시간 축이며 실제 receptive field나 이벤트 경계의 정확도를 의미하지 않습니다.
Transformer는 chunk 전체 문맥을 사용하고 10초 경계에서 문맥이 끊깁니다.

```python
from waves_sed.prediction import FramePrediction

prediction = FramePrediction.load("outputs/predictions/stem.npz")
print(prediction.probabilities.shape)
print(prediction.class_ids[0], prediction.class_names[0])
# mapping된 class column들의 max로 P_target(t)를 계산합니다.
```

공식 AudioSet ontology와 명시적 WAVES source mapping은 Phase 3에 구현했습니다.
Event 추출과 metric/report는 Phase 4, 선택적인 필터 판정은 Phase 7에 구현했습니다. 447-class vocabulary와 ontology는
서로 다른 metadata이며, class ID로 연결합니다.

## 검증

```powershell
# 모델 없이 실행할 테스트 (real-checkpoint test는 기본 skip)
.venv/Scripts/python.exe -m pytest -q

# 실제 checkpoint를 사용한 stereo / 22.05 kHz / 여러 chunk / 부분 tail 통합 테스트
$env:WAVES_SED_CHECKPOINT = (Resolve-Path .cache/checkpoints/ATST-F_strong_1.pt).Path
.venv/Scripts/python.exe -m pytest -q

.venv/Scripts/ruff.exe check src tests scripts
.venv/Scripts/ruff.exe format --check src tests scripts
```

upstream과 수치 비교를 재현하려면 (일반 추론에는 upstream clone이 필요하지 않습니다):

```powershell
git clone https://github.com/fschmid56/PretrainedSED.git .cache/PretrainedSED
git -C .cache/PretrainedSED checkout 1aa47e482f7e89904cba2338999345025d8b4e36
.venv/Scripts/python.exe scripts/verify_upstream.py --upstream .cache/PretrainedSED --checkpoint .cache/checkpoints/ATST-F_strong_1.pt --audio .cache/PretrainedSED/test_files/752547__iscence__milan_metro_coming_in_station.wav
```

이 예제 WAV의 출처/라이선스는 [third-party 기록](THIRD_PARTY_NOTICES.md)에 있습니다.
소스 WAV·checkpoint·생성 NPZ는 Git에 넣지 않습니다.

## WAVES 연결 시 주의할 실제 차이

`C:\WAVES`의 최종 `final/<key>/metadata.json`에는 label/role과 candidate provenance가 있지만,
description과 `activity_intervals`는 없습니다. adapter는 SAM manifest 또는 DSP report를
candidate ID로 조인합니다. 특히 relabel, role 변경, merge가 있으면 기존 planned support의
의미가 달라질 수 있어 그 상태를 보존해야 합니다. 현재 WAVES checkout에는 실제 WAV가 없어
이번 추론 검증에는 upstream의 공개 예제 WAV를 사용했습니다.

## 단계별 계획

구현 순서:

1. WAVES 산출물과 의존성 조사
2. ATST-F Strong checkpoint 로딩, frozen inference, raw frame probability 저장
3. WAVES metadata adapter, ontology 및 명시적 source mapping
4. 역할별 temporal metric과 JSON/CSV report
5. 시각화와 batch 처리
6. controlled corruption 평가
7. 명시적 설정에 따른 PASS / REVIEW / FAIL / UNSUPPORTED 판단 (여기까지 완료, calibration 별도)

모델·오디오·추론 출력은 Git에 넣지 않습니다. 주요 작업 단위마다 로컬 커밋을 남깁니다.
