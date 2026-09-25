# Phase 2 검증 기록

검증일: 2026-09-25. 범위는 **Phase 1 저장소 조사와 Phase 2 frozen ATST-F Strong의
단일 WAV 추론·raw probability 저장**이다. 아래 수치는 재개 전 완료된 검증의 기준
기록이며, [인수인계 5절](../HANDOFF.md)과 로컬
`outputs/reports/metro-inference.json`을 근거로 정리했다. 재개 후 CLI 수정의 회귀 테스트
결과는 별도로 기록한다.

## 환경과 모델

Windows / Python 3.11.9 / Intel i3-1315U / RAM 약 16 GB에서 CPU로 실행했다.
WAVES 및 시스템 Python과 분리한 `.venv`를 사용했다. 설치 버전은 torch/torchaudio
`2.10.0+cpu`, NumPy `1.26.4`, librosa `0.11.0`, soundfile `0.13.1`, einops `0.8.1`,
pytest `8.4.2`다. CUDA 실행은 검증하지 않았다.

- upstream: PretrainedSED revision `1aa47e482f7e89904cba2338999345025d8b4e36`.
- checkpoint: 공식 release의 `ATST-F_strong_1.pt`, 344,480,550 bytes.
- checkpoint SHA-256:
  `fbf2577958e3648d55ee8cea7e0e5260c4505fb0c13964e1ec81bc367cee5eda`.
  공식 다운로드 파일에서 계산한 값이며 publisher-signed digest는 아니다.
- 모든 학습 weight를 검증해 로드하고, `eval()` 및 parameter freeze를 적용했다.
  공식 checkpoint에 없는 deterministic mel buffer 두 개만 이름으로 허용한다.
- 모델 출력은 **447개 class**이며 실제 upstream metadata를 classifier 순서에 맞췄다.
  전체 AudioSet ontology나 WAVES source mapping은 아직 구현하지 않았다.

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
처음 실행할 때는 32.28초가 걸렸고, 원본 duration 처리 수정 후 새로 실행한 결과가
8.978초다. 초기 import/JIT cache 등의 영향이 있으므로 일반적인 성능 benchmark로
해석하지 않는다.

10초 chunk마다 250개 frame을 계산하고 마지막 chunk를 zero pad했다. Padding-only
frame은 제외했으며, 리샘플링으로 올림된 길이를 원본 duration과 구분해 기록했다.
40 ms bin은 출력 시간 축이지 실제 event boundary의 보장된 정확도가 아니다.

## upstream 수치 비교와 테스트

수정하지 않은 pinned upstream을 별도 프로세스에서 실행해 같은 입력의 **첫 10초**를
비교했다. 허용 절대 오차는 `1e-6`이었다.

| 비교 대상 | shape | 최대 절대 차이 |
| --- | --- | ---: |
| 전처리 waveform | `[160000]` | **0.0** |
| mel | `[1, 1, 64, 1001]` | **0.0** |
| sigmoid 확률 | `[1, 447, 250]` | **0.0** |

모든 adapter module이 eval 상태이며 parameter가 frozen인지도 확인했다. 이 비교는
첫 chunk에 대한 수치 동등성 확인이다. 당시 비교 JSON은 tool output에만 남았으며
별도 결과 파일은 저장하지 않았다. 재현 스크립트는
[verify_upstream.py](../scripts/verify_upstream.py)다.

기준 검증 결과:

- 실제 checkpoint 통합 테스트를 포함한 전체 suite: **79 passed**, 11.06초.
  Stereo, 22.05 kHz 리샘플링, 10초 초과 입력, 부분 tail, model freeze와 NPZ 왕복을 포함한다.
- 경고 3개는 audioread가 가져오는 Python 3.11의 `aifc`, `audioop`, `sunau` deprecation이다.
- 실제 checkpoint 환경변수가 없으면 해당 통합 테스트 1개는 skip된다.
  추론 의존성이 없는 최소 설치에서는 audio-loading 테스트도 skip된다.
  Cache/CLI 검증에는 추론 라이브러리 import를 차단하는 테스트가 포함된다.
- `ruff check` 및 `ruff format --check` 통과.
- `uv pip check`와 CPU requirements 설치 dry-run 통과.
- `waves_stem_sed-0.1.0-py3-none-any.whl` build 성공. Build log에서 class metadata,
  vendor 모델과 MIT license의 패키지 포함을 확인했다.

재개 후 `d01aee7`에서 CSV export가 metadata의 원본 audio 경로와 hardlink alias를
덮어쓰지 않도록 보호하고 회귀 테스트 3개를 추가했다. 이 변경 후 결과는
**81 passed, 1 skipped, 경고 3개**다. 이 실행에서는 checkpoint 환경변수를 지정하지 않아
실제 모델 테스트가 skip되었다. 위의 실제 checkpoint 포함 **79 passed** 결과와 합쳐서
새 전체 suite를 모두 실제 모델로 재실행한 것으로 해석하지 않는다. 기존 metro cache를
재사용한 실행도 확인했으며 `cache_hit=true`, 0.067초였다.

## 재현 명령

아래 PowerShell 명령은 설치된 `.venv`, 공식 checkpoint 및 pinned upstream clone을
사용한다. 환경 생성과 다운로드 명령은 [README](../README.md)에 있다.
일반적인 추론에는 upstream clone이 필요하지 않다.

```powershell
cd C:\multitrack-audio-filtering

# --overwrite는 기존 cache 재사용 대신 새 추론을 실행합니다.
.venv/Scripts/python.exe -m waves_sed infer .cache/PretrainedSED/test_files/752547__iscence__milan_metro_coming_in_station.wav --checkpoint .cache/checkpoints/ATST-F_strong_1.pt --output outputs/predictions/metro.npz --device cpu --threads 4 --overwrite

# 저장된 raw 결과 조회: checkpoint나 모델 로딩 없이 동작합니다.
.venv/Scripts/python.exe -m waves_sed inspect outputs/predictions/metro.npz

# 실제 checkpoint를 포함한 테스트
$env:WAVES_SED_CHECKPOINT = (Resolve-Path .cache/checkpoints/ATST-F_strong_1.pt).Path
.venv/Scripts/python.exe -m pytest -q

# upstream 첫 10초 비교: 결과 JSON을 콘솔에 출력합니다.
.venv/Scripts/python.exe scripts/verify_upstream.py --upstream .cache/PretrainedSED --checkpoint .cache/checkpoints/ATST-F_strong_1.pt --audio .cache/PretrainedSED/test_files/752547__iscence__milan_metro_coming_in_station.wav --seconds 10 --threads 4

.venv/Scripts/ruff.exe check src tests scripts
.venv/Scripts/ruff.exe format --check src tests scripts
uv pip check --python .venv/Scripts/python.exe
uv pip install --python .venv/Scripts/python.exe -r requirements-cpu.txt --dry-run
uv build --wheel
```

테스트 개수는 후속 회귀 테스트가 추가되면 증가할 수 있다. 위 79개는 이 문서의
기준 검증 시점 수치다. Weight, 음원, NPZ, 실행 결과 JSON과 build 산출물은 Git에서 제외한다.

## 검증의 한계와 다음 단계

현재 `C:\WAVES` checkout에는 실제 WAV가 없다. 따라서 이번 결과는 공개 예제 WAV를
통해 checkpoint loading, frozen 추론, 시간 축과 raw cache가 작동함을 확인한 것이다.
WAVES stem의 품질, SED 인식 정확도, 영상 동기화 또는 filtering 성능을 검증한 결과는 아니다.

다음은 **Phase 3: WAVES metadata adapter, 실제 AudioSet ontology loader와 명시적
source mapping**이다. 최종 WAVES metadata에는 description과 activity_intervals가
없으므로 SAM manifest 또는 DSP report를 candidate ID로 연결하고, merge/relabel/role
변경 이력을 보존해야 한다. 구체적인 연결 근거는 [Phase 1 분석](phase1-analysis.md)에 있다.
역할별 metric, 시각화, batch, controlled corruption, PASS/REVIEW/FAIL 정책은 Phase 4~7
계획에 남아 있다.
