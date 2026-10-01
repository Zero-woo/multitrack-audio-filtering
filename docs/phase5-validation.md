# Phase 5: 배치 추론과 저장된 확률의 시각화

## 개요와 입출력

배치 추론과 시각화는 여러 stem을 처리하고 검출 결과의 시간적 위치를 검토하기 위한 기능이다.
`batch-infer`는 정규화된 manifest를 읽어 stem별 원시 확률 cache를 생성한다. `visualize`는
저장된 cache와 평가 설정을 읽어 파형·기대 구간·확률·검출 구간을 같은 시간축에 표시한다.

추론, 지표 계산, 그림 생성은 독립적으로 실행할 수 있다. `evaluate`는 stem·clip·dataset
JSON/CSV를 출력하며, `visualize`는 평가 보고서를 포함한 index JSON, PNG/SVG와 정적 HTML을
출력한다. WAVES 생성 코드와 원본 음원은 보존한다. WAVES 계획에 대한 비교는 내부 일관성
평가이며 영상 정답과의 비교를 의미하지 않는다.

판정 설정을 지정하지 않으면 `decision`은 `null`이다. 현재 제공되는 `--filter-config`와
판정 근거 표시는 [Phase 7](phase7-validation.md)에서 설명한다. 이 문서의 검증 수치는
**Phase 5 완료 시점의 검증 기록**에 해당하며, 기록 기준은 `03cd970`(2026-09-28)이다.

## 코드 구성

| 모듈 | 책임 |
| --- | --- |
| `batch.py` | 모델 단일 초기화, stem별 추론·cache 재사용, 부분 실패 처리 |
| `visualization.py` | 파형 envelope, 시간축 패널, 보고서 검증과 그림 저장 |
| `phase5.py` | `batch-infer`·`visualize` CLI와 index·gallery 생성 |

모듈 경로의 기준은 `src/waves_sed/`이다. 지표 계산은 Phase 4의 `evaluate_stem()`을 재사용한다.

## 설치와 의존성

기본 패키지 의존성은 NumPy뿐이다. `evaluate`와 모든 stem이 유효한 cache hit인
`batch-infer`에는 PyTorch, torchaudio, librosa, soundfile, Matplotlib이 필요하지 않다.
새 추론에는 기존 `[atst]` 환경과 고정된 공식 checkpoint가 필요하다.

시각화는 선택 의존성 `[visualization]`을 사용한다.

```powershell
cd C:\multitrack-audio-filtering
uv pip install --python .venv/Scripts/python.exe -e ".[visualization]"
```

의존성 범위는 `matplotlib>=3.9,<4`, `soundfile==0.13.1`이다. 기준 검증 환경에서는
Matplotlib 3.11.2와 soundfile 0.13.1을 사용하였다. 새 환경에서 추론까지 실행하려면
프로젝트의 기존 `[atst]` 설치 절차도 적용한다.

Matplotlib의 정적 backend는 GUI 창 없이 파일을 출력할 수 있으므로 이 작업에는
`FigureCanvasAgg`를 직접 연결한다. 전역 pyplot 창이나 GUI event loop를 만들지 않는다.
설계 근거는 [Matplotlib backend 문서](https://matplotlib.org/stable/users/explain/figure/backends.html)에
설명된 비대화형 출력 방식이다. PNG/SVG 저장 시 format을 명시하며,
[savefig 문서](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html)의
파일 출력 인터페이스를 사용한다. Waveform은
[soundfile 문서](https://python-soundfile.readthedocs.io/en/latest/)의 파일 읽기 API로
원본 sample rate와 channel을 유지하면서 읽는다.

## 기본 실행 순서

먼저 Phase 3의 `adapt-waves`로 정규화한 manifest를 준비한다. 아래의
`outputs/stems.json`은 실제 stem audio 경로를 포함하는 실행 대상 manifest 경로로 지정한다.
Frozen metadata의 `audio_id`만으로 파일 경로를 추측하지 않으며,
실제 WAV가 없으면 `batch-infer`는 그 stem을 `missing_audio`로 기록한다.

```powershell
.venv/Scripts/python.exe -m waves_sed batch-infer --stems outputs/stems.json --checkpoint .cache/checkpoints/ATST-F_strong_1.pt --device cpu --threads 4 --output-dir outputs/batch
.venv/Scripts/python.exe -m waves_sed evaluate --stems outputs/stems.json --predictions outputs/batch/prediction-index.json --mappings configs/source_mappings.example.json --config configs/evaluation.example.json --output-dir outputs/evaluation
.venv/Scripts/python.exe -m waves_sed visualize --stems outputs/stems.json --predictions outputs/batch/prediction-index.json --mappings configs/source_mappings.example.json --config configs/evaluation.example.json --output-dir outputs/visualization
```

`outputs/visualization/index.html`을 브라우저에서 열면 stem별 그림과 평가 상태를 볼 수 있다.
HTML, JSON, `plots/`를 함께 복사하면 상대 링크를 유지할 수 있다. 외부 CDN이나 서버는
사용하지 않는다. `visualize`는 저장된 `dataset.json`을 읽는 명령이 아니라, 현재 cache,
mapping, config에서 `evaluate_stem()`을 다시 호출하는 명령이다. 설정을 바꾸면
`evaluate`와 `visualize`를 다시 실행하고 raw cache는 그대로 재사용한다.

## Batch 옵션과 실행 정책

| 옵션 | 의미 |
| --- | --- |
| `--stems` | 필수. 정규화된 schema 1 manifest |
| `--output-dir` | 필수. cache와 batch 보고서를 쓸 디렉터리 |
| `--checkpoint` | 기본 `.cache/checkpoints/ATST-F_strong_1.pt` |
| `--device` | `cpu` 또는 `cuda`, 기본 `cpu` |
| `--threads` | 양의 정수, 기본 `4`; 새 backend의 PyTorch thread 수 |
| `--overwrite` | 기존 cache의 일치 여부와 관계없이 해당 stem을 새로 추론 |

Stem은 manifest 순서로 처리한다. 새 추론이 처음 필요한 시점에 frozen ATST backend를
초기화하고 이후 stem에서 같은 모델을 재사용한다. 순차 실행이며 여러 GPU나 병렬 worker를
구동하지 않는다. 한 stem의 추론 실패는 해당 행에 남기고 다음 stem을 계속 처리한다.
공통 모델 초기화가 실패하면 같은 실행에서 초기화를 반복하지 않으며, 뒤에 나오는 유효한
cache hit은 계속 처리할 수 있다.

Cache 재사용 조건은 기존 단일 `infer` 명령과 같다. 입력 audio의 현재 SHA-256,
고정 checkpoint SHA-256, preprocessing ID, upstream revision, package version,
backend 이름, **447개 class ID와 이름의 순서**가 모두 일치해야 한다. Manifest에
`raw_final_stem.sha256`이 있으면 현재 audio와의 일치도 먼저 확인한다. 파일명이나
stem 설명만 일치하는 cache는 재사용하지 않는다.

Cache hit에는 현재 실행에서 요청한 device를 덮어쓰지 않는다. NPZ의 실제 추론 device와
dependency provenance를 유지한다. `batch-status.json`의 `requested_device`는 해당
명령의 요청값이며, 각 행의 `prediction_provenance.device`는 cache를 만든 당시 값이다.
동일한 cache를 CPU와 GPU 환경 사이에서 옮겨 사용할 수 있다.

**모든 항목이 cache hit이면 checkpoint 파일을 열지 않는다.** 존재하지 않는 checkpoint
경로를 인자로 주어도 유효한 cache 재사용은 가능하다. Batch provenance의
`checkpoint_sha256`은 고정된 기대값이며, `checkpoint_hash_scope`가 이 범위를 설명한다.
이 필드는 현재 `checkpoint_path`에 있는 파일을 읽어 검증하였다는 의미가 아니다.
실제 새 추론에서는 backend가 checkpoint bytes를 검증한다.

기존 cache가 손상되었거나 입력/모델과 다르면 `cache_conflict`로 남기고 파일을 보존한다.
새 output 디렉터리를 선택하거나 `--overwrite`를 명시해야 다시 추론한다. Overwrite 중
추론이 실패해 기존 cache가 남더라도, 해당 실행의 성공 index에는 그 cache를 넣지 않는다.
유효한 새 결과는 임시 파일을 거쳐 교체한다. Raw NPZ에는 threshold나 smoothing을 적용하지
않으며, 이 설정은 이후 평가 단계에만 속한다.

## Batch 출력과 종료 코드

```text
outputs/batch/
  prediction-index.json
  batch-status.json
  predictions/
    <SHA256(stem_id UTF-8)>.npz
```

`clip::candidate` 같은 stem ID를 Windows 파일명에 직접 넣지 않는다. 파일명 hash는
stem ID의 UTF-8 문자열로 계산하며, audio hash와는 다른 용도다.

Prediction index는 기존 `evaluate`와 `visualize`에서 그대로 읽을 수 있다.

```json
{
  "schema_version": 1,
  "predictions": {
    "clip::candidate": "predictions/<stem_id의_SHA256>.npz"
  }
}
```

상대 경로는 index 파일이 있는 디렉터리 기준이다. **해당 실행에서 `cached` 또는
`inferred`로 성공한 stem만** index에 들어간다. 이전 실행의 불필요한 cache는 지우지 않으며,
index에 없는 파일은 현재 결과에 포함되지 않는다. 재실행 시 status와 index는 최신 실행
결과로 교체하므로 이 파일들은 실행 이력 저장소가 아니다.

`batch-status.json`은 다음 정보를 포함한다.

- `summary`: stem 수, `cached_count`, `inferred_count`, `failed_count`, `backend_initializations`.
- `stems`: ID, audio/cache 경로와 hash, 처리 상태, error/reason, 실제 prediction provenance.
- `provenance`: 입력 manifest 경로/hash, 고정 모델·전처리 정보, 요청 device/thread,
  overwrite 여부, 실제 추론 여부와 index 정책.

실패 상태에는 `missing_audio`, `audio_unreadable`, `source_hash_mismatch`,
`cache_conflict`, `inference_failed`, `cache_unreadable`이 있다. 일부 실패가 있어도
나머지를 처리하고 status와 성공 index를 쓴다. Batch 종료 코드는 모든 stem이 성공하면
`0`, 실패 stem이 있으면 `1`이다. 잘못된 manifest나 output 충돌 등 명령 전체의 오류도
`1`이며, 사전 검증에서 걸리면 출력 파일을 쓰지 않는다. 인자 구문 오류는 argparse의
일반적인 종료 코드 `2`를 따른다.

## 시각화 옵션과 그림의 의미

| 옵션 | 의미 |
| --- | --- |
| `--stems`, `--predictions` | 필수. 정규화 manifest와 raw prediction index |
| `--mappings`, `--config` | 필수. 현재 mapping 및 evaluation 설정 |
| `--filter-config` | 선택. Phase 7의 역할별 판정 설정 |
| `--output-dir` | 필수. 그림·gallery·index 출력 디렉터리 |
| `--ontology` | 선택. 생략하면 번들 AudioSet ontology |
| `--format` | `png` 또는 `svg`, 기본 `png` |
| `--max-waveform-points` | 기본 `4000`, CLI에서는 `2` 이상; waveform envelope bin 수 상한 |
| `--stem-id` | 정확히 일치하는 stem 하나만 선택; 생략하면 전체 |

예를 들어 하나의 stem을 SVG로 저장할 수 있다.

```powershell
.venv/Scripts/python.exe -m waves_sed visualize --stems outputs/stems.json --predictions outputs/batch/prediction-index.json --mappings configs/source_mappings.example.json --config configs/evaluation.example.json --stem-id "clip::candidate" --format svg --max-waveform-points 2000 --output-dir outputs/visualization-svg
```

그림은 동일한 초 단위 x축에 네 영역을 배치한다.

1. **Waveform:** 원본 sample rate에서 모든 channel의 최소·최대 envelope를 표시한다.
   정수 sample 경계로 bin을 나누고 모든 sample을 순차 읽는다. 단순 stride sampling으로
   impulse를 건너뛰거나 channel 평균으로 역상 신호를 지우지 않는다. 읽기 block은 제한되며
   waveform 전체 배열을 메모리에 올리지 않는다. `--max-waveform-points`는 이 표시용
   envelope의 해상도만 바꾸며 raw prediction과 metric에는 영향을 주지 않는다.
2. **Expected:** reference origin/status를 표시하고 계획 구간을 그린다. Ambiguous 구간은
   주황색 사선 무늬로 구분하며 metric에서 제외되었는지도 표기한다. 기본 설정에서는
   ambiguous 계획을 평가에 사용하지 않는다. 구간이 없으면 unavailable 이유를 보여준다.
3. **Target probability:** 허용 class의 raw max 확률, 선택된 median smoothing 결과,
   event 추출 threshold를 표시한다. Smoothing window가 `1`이면 raw 곡선만 표시한다.
   각 prediction의 실제 start/end 경계를 사용하며 짧은 마지막 frame도 그대로 유지한다.
   시간 gap은 연결하거나 보간하지 않는다.
4. **Detected:** 평가 보고서의 event 구간을 표시한다. 검출 가능한 상태에서 event가
   없으면 `No detected events`, 근거가 없으면 `Detection unavailable`로 구분한다.

머리말에는 stem 설명/ID, role, backend, reference 상태, 평가 상태와 role별 주요 metric을
표시한다. WAVES 계획과 영상 정답의 차이도 그림에 남긴다. 긴 설명은 renderer의 실제 문자
폭에 맞춰 두 줄 이내로 정리하고 일부를 생략할 수 있으며, 전체 원문은 JSON에 유지한다.
한글/CJK 글꼴은 설치된 Malgun Gothic, Noto Sans CJK KR, Noto Sans KR, AppleGothic 중
사용 가능한 것을 찾아 fallback으로 사용한다. 해당 글꼴이 없는 환경에서 모든 CJK 문자의
표시를 보장하지는 않는다. Metadata 안의 수식처럼 보이는 문자열도 일반 텍스트로 표시한다.

## 누락 자료와 report 결합

Missing/corrupt cache나 unsupported mapping을 0 확률로 대신하지 않는다. Waveform이
있으면 보여주고 target curve와 detection은 unavailable로 설명한다. Cache와 audio identity가
검증되어 검출은 가능하지만 reference가 없거나 ambiguous이면 raw curve와 검출 구간은
보여줄 수 있으며 temporal metric은 계속 `null`이다.

WAV가 없더라도 명시된 manifest hash로 검증된 cache는 Phase 4 정책대로 사용할 수 있다.
이때 waveform은 unavailable이고 `source_file_verified=false`다. 파일 bytes를 직접 읽어
검증한 상태와 구분한다. 읽을 수 없거나 sample이 없는 audio에도 waveform을 만들어내지
않는다. 단순히 waveform이 없다는 이유만으로 렌더링 전체를 실패시키지는 않는다.

`render_stem()` API는 전달된 보고서의 stem, mapping, config, backend와 prediction
provenance를 확인한다. Cache 파일 hash와 전달된 raw 배열도 대조하며, 평가 후 source가
바뀌었다면 거부한다. In-memory prediction만으로 만들고 cache 경로/hash를 묶지 않은
보고서는 신뢰된 raw curve를 표시하지 않는다. API로 만든 별도의 `TemporalReference`
보고서를 그릴 수 있지만, CLI의 reference는 기존 `WavesPlannedReference`다.

## 시각화 출력과 파일 보호

```text
outputs/visualization/
  index.html
  visualization-index.json
  plots/
    <SHA256(stem_id UTF-8)>.png
```

SVG를 선택하면 확장자가 `.svg`가 된다. Index의 `summary`에는 선택된 stem 수,
`rendered_count`, `failed_count`가 있다. 각 `stems[]` 행에는 stem ID, 상대 `path`,
`status` (`rendered` 또는 `failed`), 전체 `evaluation`, `visualization` 상세정보와
`error`가 들어간다. 상세정보에는 waveform 상태, native sample rate/channel/sample 수,
표시용 envelope 방식, target curve/reference 상태 등이 포함된다. 최상위 provenance는
입력 파일 hash, evaluation config, 출력 format/point 설정과 `inference_performed=false`를
남긴다.

`rendered`는 그림 파일이 생성되었다는 뜻이며 metric 사용 가능 여부나 품질 합격을 뜻하지
않는다. Unavailable 평가도 설명이 있는 그림을 정상 생성하였다면 종료 코드는 `0`이다.
실제로 그림 생성에 실패한 stem이 있으면 다른 stem을 계속 처리하고 종료 코드 `1`을
반환한다. 개별 오류는 stderr와 index에 남긴다. `--stem-id`가 manifest에 없거나 index가 잘못된
경우, 의존성이 없거나 출력 사전 검증이 실패한 경우에도 종료 코드 `1`을 반환한다.

현재 결과는 index에 있는 `rendered` 행이다. 이전에 만든 다른 format/다른 stem의 파일은
자동 삭제하지 않는다. 실패 후 남은 이전 그림도 현재 결과로 사용하지 않는다. HTML은
성공한 행의 상대 hash 경로만 연결하고 실패 행에는 설명을 표시한다. Metadata 텍스트는
HTML escaping을 거치며 외부 script나 사용자 문자열로 만든 파일명을 사용하지 않는다.

Batch와 시각화 모두 **전체 출력 경로를 먼저 검증**한다. 각 명령의 입력 metadata와
stem audio, provenance에 기록된 source media/metadata를 output이 덮어쓸 수 없다.
시각화는 읽는 cache와 index/config/mapping/ontology도 보호한다. Batch는 checkpoint 경로와
기존·이전 cache header에 기록된 source audio를 보호하며, 자신의 cache 교체는 위의
`--overwrite` 정책을 따른다. 동일 경로뿐 아니라 hardlink/symlink alias와 output끼리의
충돌도 검사한다. 선택하지 않은 stem의 입력도 보호한다. 손상된 NPZ에서도 읽을 수 있는
audio provenance는 보호하며, 보호를 위해 읽었다고 cache를 유효한 것으로 승격하지 않는다.

## 기준 실행과 검증 결과: Phase 5 완료본

### 실제 CPU 배치 추론과 cache 재사용

2026-09-28의 CPU 실행에서는 기존 공개 metro WAV의 `0–2`초와 `10–12.5`초 구간을
PCM16 WAV 두 개로 저장한 뒤 실제 frozen ATST 추론을 실행하였다. 이는 WAVES 생성 stem이
아니며, manifest에 `synthetic_demo_metadata`, missing reference와 원본 excerpt provenance를
명시하였다. 첫 실행은 `inferred_count=2`, `backend_initializations=1`이었다.
동일 결과 디렉터리로 재실행하면서 의도적으로 존재하지 않는 checkpoint 경로를 전달하였을 때
`cached_count=2`, `inferred_count=0`, `backend_initializations=0`을 확인하였다.

자료는 `outputs/phase5/demo-inputs/`, cache는 `outputs/phase5/demo-batch/`, 평가 보고서는
`outputs/phase5/demo-evaluation/`, gallery는 `outputs/phase5/demo-visuals/index.html`에 있다.
두 stem 모두 cache 기반 detection은 사용 가능하였고 reference가 없으므로 temporal metric은
`null`이었다. PNG 두 개와 정적 gallery를 생성하였다. 결과를 다시 확인하는 명령은 다음과 같다.

```powershell
.venv/Scripts/python.exe -m waves_sed batch-infer --stems outputs/phase5/demo-inputs/stems.json --checkpoint outputs/phase5/intentionally-absent-checkpoint.pt --output-dir outputs/phase5/demo-batch
.venv/Scripts/python.exe -m waves_sed evaluate --stems outputs/phase5/demo-inputs/stems.json --predictions outputs/phase5/demo-batch/prediction-index.json --mappings outputs/phase5/demo-inputs/mapping.json --config outputs/phase5/demo-inputs/evaluation.json --output-dir outputs/phase5/demo-evaluation
.venv/Scripts/python.exe -m waves_sed visualize --stems outputs/phase5/demo-inputs/stems.json --predictions outputs/phase5/demo-batch/prediction-index.json --mappings outputs/phase5/demo-inputs/mapping.json --config outputs/phase5/demo-inputs/evaluation.json --output-dir outputs/phase5/demo-visuals
```

첫 명령은 위 demo cache가 이미 있는 환경의 재사용 확인용이다. Cache가 없다면 유효한
공식 checkpoint 경로로 바꿔야 한다. `outputs/`의 자료와 weights는 Git에 포함하지 않는다.

테스트는 batch의 단일 초기화·부분 실패·재사용·overwrite·출력 보호, waveform의 impulse와
역상 stereo·native 시간 경계·읽기 크기 제한, raw/smoothed curve와 partial frame,
report/cache/config 불일치, missing/ambiguous/unsupported 상태, PNG/SVG, 긴 머리말,
HTML escaping, 선택 실행과 optional dependency 분리를 확인한다. CLI의 fresh process
검사에서는 inference library import를 차단한 cache hit과 시각화도 실행한다.

### 회귀 테스트와 배포 패키지 검증

- 전체 테스트는 **647 passed, 1 skipped**로 나타났으며 실행 시간은 54.62초였다. Skip은
  환경변수를 설정하지 않은 실제 checkpoint 통합 테스트였다. 실제 CPU batch 동작은 별도 실행으로
  확인하였다. audioread의 Python 3.11 deprecation 경고 3개가 발생하였다.
- Ruff lint/format, `git diff --check`, wheel build를 통과하였다.
- 새 `.cache/phase5-package-smoke` 환경에 wheel과 NumPy 1.26.4만 설치하였다. torch,
  torchaudio, librosa, soundfile, Matplotlib이 없는 상태에서 실제 demo cache 두 개를
  재사용하였고 backend 초기화는 0회였다.
- 같은 wheel 환경에 `[visualization]`만 추가한 뒤 SVG 두 개와 gallery를 생성하였다.
  결과는 `outputs/phase5/wheel-visuals/`에 있다. 추론 라이브러리는 설치하지 않았다.
- 실제 frozen metadata 62개를 `batch-infer`에 전달한 결과 전부 `missing_audio`로 나타났으며,
  backend 초기화는 0회였고 성공 index는 비어 있었다. 첫 stem을 선택하여 누락·모호함을
  표시하는 SVG도 생성하였다. 산출물은 `outputs/phase5/frozen-batch/`, `frozen-visual/`에 있다.

### 전체 길이 파형과 확률의 시간축 검증

전체 길이의 기존 metro cache도 Python `render_stem()` API로 PNG/SVG에 연결하였다.
`outputs/phase5/metro-full-demo.png`, `.svg`에는 원본 44,100 Hz / 1,653,688 samples의
파형과 **938개 raw frame**, expected/detected 구간이 같은 시간축에 표시된다.
PNG를 직접 열어 패널·축·범례·머리말이 겹치지 않는 것을 확인하였다.

이 그림은 Phase 4의 `synthetic_full_track_demonstration` reference를 그대로 사용한다.
Expected 구간은 **의도적으로 만든 전체 길이 fixture**이며 WAVES의 계획이나 영상 annotation이
아니다. 이미지에 origin/status를 표시하였으며 해당 IoU는 실제 인식·동기화 성능을 나타내지 않는다.
Raw cache나 원본 WAV를 변경하지 않고 그림을 생성하였다.

## 적용 한계와 관련 기능

기준 실행 당시 `C:\WAVES` checkout에는 실제 WAVES stem WAV가 없었다. 공개 음원과 합성
fixture를 이용한 검증은 실행 경로와 시간 표현을 확인한 것이며, WAVES 생성 품질이나 영상
동기화 성능을 입증하지 않는다. 음원 변형에 대한 모델과 지표의 반응은
[Phase 6의 controlled corruption 실험](phase6-validation.md)에서 별도로 검증하였다.
