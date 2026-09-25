# WAVES stem SED validation

WAVES가 생성한 stem을 frozen AudioSet-Strong SED 모델로 검사하는 독립적인 후처리 프로젝트입니다.
WAVES 생성 파이프라인을 수정하거나 모델을 학습하지 않습니다.

현재 작업 범위는 **Phase 1 분석과 Phase 2 단일 WAV 추론 prototype**입니다.
WAVES의 planned support와 SED 출력 비교는 내부 consistency 검사이며,
실제 영상과의 동기화를 입증하지 않습니다.

구조 분석, 확인된 metadata 누락, 단계별 계획은 [분석 문서](docs/phase1-analysis.md),
실행 결과와 검증 범위는 [Phase 2 검증 기록](docs/phase2-validation.md)에 있습니다.

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
# 추후 mapping된 class column들의 max로 P_target(t)를 계산할 수 있습니다.
```

AudioSet 전체 ontology, WAVES source mapping, event 추출, metric/report와 필터 판정은
후속 단계입니다. 포함된 447-class vocabulary는 실제 metadata에서 가져온 모델 출력 목록이며,
임의 source-to-class mapping을 포함하지 않습니다.

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
description과 `activity_intervals`는 없습니다. 후속 adapter는 SAM manifest 또는 DSP report를
candidate ID로 조인해야 합니다. 특히 relabel, role 변경, merge가 있으면 기존 planned support의
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
7. calibration 후 설정에 따른 PASS / REVIEW / FAIL 판단

모델·오디오·추론 출력은 Git에 넣지 않습니다. 주요 작업 단위마다 로컬 커밋을 남깁니다.
