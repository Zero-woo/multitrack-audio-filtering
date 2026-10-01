# Phase 2: ATST-F 단일 WAV 추론과 원시 확률 저장

검증 기준일: **2026-09-25**. 후속 CLI 회귀 검증은 commit `d01aee7`을 기준으로 별도 기록한다.

Phase 2는 frozen ATST-F Strong 모델로 단일 WAV를 분석하고, 시간별 소리 확률을
재사용 가능한 cache로 저장하는 단계이다. 모델 추론과 이후 이벤트 추출·지표 계산을
분리하여 검출 설정을 변경할 때 모델을 다시 실행하지 않도록 구성하였다.

이 문서의 수치와 테스트 개수는 Phase 2 당시의 검증 결과이다. 현재 전체 기능과
실행 방법은 [README](../README.md)를 기준으로 한다. 추론 실측값의 로컬 근거는
`outputs/reports/metro-inference.json`이다.

## 입출력과 코드 구성

입력은 단일 WAV와 공식 strong checkpoint이다. WAVES metadata 없이도 모델을 실행할 수
있으며, source mapping과 시간 기준의 연결은 후속 단계에서 담당한다.

| 모듈 | 책임 |
| --- | --- |
| `audio.py` | 오디오 읽기, 입력 유효성 확인, mono·16 kHz 변환 |
| `backends/atst.py` | 고정 checkpoint 검증·로드, chunk 단위 추론 |
| `prediction.py` | `FramePrediction` 검증, NPZ 저장·로드 |
| `cli.py` | `infer`, `inspect`, checkpoint 준비 명령 |

출력 NPZ에는 `probabilities[T, 447]`, frame 시작·종료 시간, class ID/name, provenance가
포함된다. 확률은 float32이며 smoothing과 threshold를 적용하지 않는다. Provenance에는
입력 오디오 hash, checkpoint hash, upstream revision, 전처리와 시간축 정보를 기록한다.
`inspect`와 cache 조회에는 모델을 로드할 필요가 없다.

## 환경과 모델

Windows / Python 3.11.9 / Intel i3-1315U / RAM 약 16 GB에서 CPU로 실행하였다.
WAVES 및 시스템 Python과 분리한 `.venv`를 사용하였다. 설치 버전은 torch/torchaudio
`2.10.0+cpu`, NumPy `1.26.4`, librosa `0.11.0`, soundfile `0.13.1`, einops `0.8.1`,
pytest `8.4.2`였다. CUDA 실행은 검증하지 않았다.

- upstream: PretrainedSED revision `1aa47e482f7e89904cba2338999345025d8b4e36`.
- checkpoint: 공식 release의 `ATST-F_strong_1.pt`, 344,480,550 bytes.
- checkpoint SHA-256:
  `fbf2577958e3648d55ee8cea7e0e5260c4505fb0c13964e1ec81bc367cee5eda`.
  공식 다운로드 파일에서 계산한 값이며 publisher-signed digest는 아니다.
- 모든 학습 weight를 검증하여 로드하고, `eval()` 및 parameter freeze를 적용하였다.
  공식 checkpoint에 없는 deterministic mel buffer 두 개만 이름으로 허용한다.
- 모델 출력은 **447개 class**이며 실제 upstream metadata를 classifier 순서에 맞추었다.
  전체 AudioSet ontology와 WAVES source mapping은 [Phase 3](phase3-validation.md)에서 다룬다.

## 실제 WAV 추론 결과

입력은 upstream의 `test_files/752547__iscence__milan_metro_coming_in_station.wav`다.
iscence의 Freesound 752547이며 upstream attribution은 CC BY-NC 4.0이다.
음원은 이 저장소에 재배포하지 않았다. 출처는 [third-party 기록](../THIRD_PARTY_NOTICES.md)에 있다.

| 항목 | 결과 |
| --- | --- |
| 원본 오디오 | 44,100 Hz, mono, 1,653,688 samples |
| 원본 길이 | **37.49859410430839초** |
| 리샘플링 결과 | 16,000 Hz, 599,978 samples, 37.498625초 |
| raw 확률 | **938 × 447, float32**, smoothing/threshold 없음 |
| 확률 범위 | `1.8619566333200055e-7` ~ `0.4613507091999054`, 모두 유한 |
| 마지막 출력 bin의 끝 | 원본 길이와 같은 `37.49859410430839초` |
| NPZ | `outputs/predictions/metro.npz`, 1,573,777 bytes |
| 실행 요약 | `outputs/reports/metro-inference.json` |
| 최종 새 추론 | **8.978초**, CPU `--threads 4`, `cache_hit=false` |

입력 SHA-256은
`13c04eca0967e264d18e03aef53d2fa6aee566d7946bdb0df7f6cd7ac72909d1`이다.
첫 실행은 32.28초, 원본 duration 처리 수정 후 새 추론은 8.978초로 나타났다.
초기 import/JIT cache 등의 영향이 있으므로 일반적인 성능 benchmark로
해석하지 않는다.

10초 chunk마다 250개 frame을 계산하고 마지막 chunk를 zero padding하였다. Padding-only
frame은 제외하였으며, 리샘플링으로 올림된 길이를 원본 duration과 구분하여 기록하였다.
40 ms bin은 출력 시간 축이지 실제 event boundary의 보장된 정확도가 아니다.

## upstream 수치 비교와 테스트

수정하지 않은 pinned upstream을 별도 프로세스에서 실행하여 같은 입력의 **첫 10초**를
비교하였다. 허용 절대 오차는 `1e-6`이었다.

| 비교 대상 | shape | 최대 절대 차이 |
| --- | --- | ---: |
| 전처리 waveform | `[160000]` | **0.0** |
| mel | `[1, 1, 64, 1001]` | **0.0** |
| sigmoid 확률 | `[1, 447, 250]` | **0.0** |

모든 adapter module의 eval 상태와 parameter freeze도 확인하였다. 이 비교는
첫 chunk에 대한 수치 동등성 확인이다. 당시 비교 JSON은 실행 출력으로만 확인하였으며
별도 결과 파일은 저장하지 않았다. 재현 스크립트는
[verify_upstream.py](../scripts/verify_upstream.py)다.

2026-09-25 기준 검증 결과는 다음과 같다.

- 실제 checkpoint 통합 테스트를 포함한 전체 suite: **79 passed**, 11.06초.
  Stereo, 22.05 kHz 리샘플링, 10초 초과 입력, 부분 tail, model freeze와 NPZ 왕복을 포함한다.
- 경고 3개는 audioread가 가져오는 Python 3.11의 `aifc`, `audioop`, `sunau` deprecation이다.
- 실제 checkpoint 환경변수가 없으면 해당 통합 테스트 1개는 skip된다.
  추론 의존성이 없는 최소 설치에서는 audio-loading 테스트도 skip된다.
  Cache/CLI 검증에는 추론 라이브러리 import를 차단하는 테스트가 포함된다.
- `ruff check` 및 `ruff format --check` 통과.
- `uv pip check`와 CPU requirements 설치 dry-run 통과.
- `waves_stem_sed-0.1.0-py3-none-any.whl` build 성공. Build log에서 class metadata,
  vendor 모델과 MIT license의 패키지 포함을 확인하였다.

Commit `d01aee7`에서는 CSV export가 metadata의 원본 audio 경로와 hardlink alias를
덮어쓰지 않도록 보호하고 회귀 테스트 3개를 추가하였다. 이 변경 후 결과는
**81 passed, 1 skipped, 경고 3개**다. 이 실행에서는 checkpoint 환경변수를 지정하지 않아
실제 모델 테스트가 skip되었다. 따라서 실제 checkpoint 포함 **79 passed** 결과와
후속 회귀 검증 결과는 서로 다른 실행 기록이다. 기존 metro cache를 재사용한 실행도
확인하였으며 `cache_hit=true`, 0.067초로 나타났다.

## 재현 명령

아래 PowerShell 명령은 설치된 `.venv`, 공식 checkpoint 및 pinned upstream clone을
사용한다. 환경 생성과 다운로드 명령은 [README](../README.md)에 있다.
일반적인 추론에는 upstream clone이 필요하지 않다.

```powershell
cd C:\multitrack-audio-filtering

# --overwrite는 기존 cache 재사용 대신 새 추론을 실행한다.
.venv/Scripts/python.exe -m waves_sed infer .cache/PretrainedSED/test_files/752547__iscence__milan_metro_coming_in_station.wav --checkpoint .cache/checkpoints/ATST-F_strong_1.pt --output outputs/predictions/metro.npz --device cpu --threads 4 --overwrite

# 저장된 raw 결과 조회: checkpoint나 모델 로딩 없이 동작한다.
.venv/Scripts/python.exe -m waves_sed inspect outputs/predictions/metro.npz

# 실제 checkpoint를 포함한 테스트
$env:WAVES_SED_CHECKPOINT = (Resolve-Path .cache/checkpoints/ATST-F_strong_1.pt).Path
.venv/Scripts/python.exe -m pytest -q

# upstream 첫 10초 비교: 결과 JSON을 콘솔에 출력한다.
.venv/Scripts/python.exe scripts/verify_upstream.py --upstream .cache/PretrainedSED --checkpoint .cache/checkpoints/ATST-F_strong_1.pt --audio .cache/PretrainedSED/test_files/752547__iscence__milan_metro_coming_in_station.wav --seconds 10 --threads 4

.venv/Scripts/ruff.exe check src tests scripts
.venv/Scripts/ruff.exe format --check src tests scripts
uv pip check --python .venv/Scripts/python.exe
uv pip install --python .venv/Scripts/python.exe -r requirements-cpu.txt --dry-run
uv build --wheel
```

테스트 개수는 후속 회귀 테스트가 추가되면 증가할 수 있다. 위 79개는 이 문서의
기준 검증 시점 수치다. Weight, 음원, NPZ, 실행 결과 JSON과 build 산출물은 Git에서 제외한다.

## 검증 범위와 한계

검증 시점의 `C:\WAVES` checkout에는 실제 WAV가 없었다. 따라서 이 결과는 공개 예제
WAV로 checkpoint loading, frozen 추론, 시간축과 raw cache의 동작을 확인한 기록이다.
WAVES stem의 품질, SED 인식 정확도, 영상 동기화 또는 filtering 성능을 검증한 결과는 아니다.

저장한 raw 확률은 [Phase 3](phase3-validation.md)의 source mapping,
[Phase 4](phase4-validation.md)의 시간 지표, [Phase 5](phase5-validation.md)의 batch와
시각화에서 사용한다. [Phase 6](phase6-validation.md)은 변형 음원을 새로 추론하여
지표의 변화를 비교하고, [Phase 7](phase7-validation.md)은 명시적 판정 정책을 적용한다.
각 단계의 구현 여부와 검증 근거는 해당 문서에 기록되어 있다.
