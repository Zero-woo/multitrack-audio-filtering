# 다음 Codex 세션 인수인계

최종 갱신: 2026-09-25. 사용자가 일시 중지를 요청한 후 재개를 요청하여,
남은 Phase 2 오류 수정과 검증·문서 정리를 완료했다.
**Phase 1 분석과 Phase 2 단일 WAV frozen ATST-F 추론이 완료된 상태**다.
이 문서는 중지 당시 초안을 대체하며, 다음 세션은 Phase 3부터 이어 갈 수 있다.

## 1. 다음 세션에서 가장 먼저 확인할 것

1. `git status --short`, `git log -6 --oneline`과 이 문서를 읽는다.
2. `docs/phase1-analysis.md`에서 실제 WAVES 필드·merge/role 변경 문제를 확인한다.
3. `docs/phase2-validation.md`에서 완료된 검증 범위와 재현 명령을 확인한다.
4. `C:\WAVES`에 실제 run/WAV가 추가되었는지 확인한다. 현재 checkout에는 WAV가 없다.
5. 새 사용자 지시를 확인한다. Phase 3 작업이라면 adapter/ontology/manual mapping부터 구현한다.
   실제 WAV 경로가 없을 때 frozen JSON의 `audio_id`로 파일명을 만들어 내지 않는다.

## 2. 요청과 작업 범위

- 작업 저장소: `C:\multitrack-audio-filtering`. 처음에는 빈 Git 저장소였다.
- WAVES 저장소: `C:\WAVES`, 조사 기준 HEAD `07af161`. 읽기만 했고 수정하지 않았다.
- 사용자 요구 원문:
  `C:\Users\owq05\.codex\attachments\62b196b9-c141-4c60-a879-514daa2f1960\Pasted text.txt`.
  PowerShell에서는 `Get-Content -Encoding UTF8`로 읽는다.
- 전체 목표는 독립 pretrained SED로 WAVES stem의 의미·시간 일관성을 측정하는 후처리 도구다.
  생성 모델 수정, 학습/fine-tuning, reference WAV 유사도 비교는 하지 않는다.
- 먼저 구조를 분석·설명하고 최소 단일-WAV prototype을 검증하라는 요청에 따라 Phase 1~2를 진행했다.
- WAVES planned support는 영상의 정답 시간이 아니다. 내부 consistency와 영상 동기화는 구분한다.
- 사용자 요청대로 의미 있는 작업 단위마다 로컬 commit을 남겼다. Push는 하지 않았다.

## 3. 완료한 구현

- 설치 가능한 `waves-stem-sed` 패키지, `waves-sed` / `python -m waves_sed` CLI.
- `SEDBackend` Protocol과 frozen `ATSTBackend`.
- 공식 `ATST-F_strong_1.pt` 다운로드·SHA-256 확인·CPU 추론.
- `eval()`, `requires_grad_(False)`, `torch.inference_mode()` 적용.
- checkpoint의 모든 학습 weight를 검증하고, 명시적인 deterministic mel buffer 2개만 누락 허용.
  잘못된 classifier head를 random weight로 대체하지 않는다.
- upstream과 동일한 mono/16 kHz 처리, 10초 chunk/zero padding, sigmoid.
- 실제 metadata로 만든 447개 class ID/name과 정확한 classifier 출력 순서.
- raw NPZ 저장/재로드, 선택적 long-format CSV, JSON console 요약.
- 입력/모델 hash·전처리/패키지 버전·class 순서로 기존 cache 재사용 여부 확인.
- 모델·checkpoint·torch가 없어도 NPZ 조회/CSV 내보내기/cache 재사용 가능.
- padding-only frame 제거, 마지막 bin 끝을 원본 WAV duration으로 제한.
- CLI 입력/출력 덮어쓰기 보호. 리뷰에서 발견한 `inspect --csv` 원본 WAV 보호 누락도 수정했다.
  metadata의 원본 경로와 hardlink alias를 검사하고 regression test를 추가했다.
- 오프라인 테스트, 선택적 실제 checkpoint 통합 테스트, upstream 수치 비교 스크립트.

**미구현 후속 기능:** WAVES adapter, 전체 ontology loader, manual source mapping,
TemporalReference, event extraction, 역할별 metric, 집계 report/시각화/batch,
controlled corruption, PASS/REVIEW/FAIL 정책. 이들을 구현 완료로 취급하지 않는다.

## 4. 파일별 구현/변경 내용

| 파일 | 구현/변경 |
| --- | --- |
| `.gitignore` | venv/cache/weights/outputs/build 및 Python cache 제외 |
| `.gitattributes` | 소스/문서/CSV/TXT/LICENSE LF 설정 |
| `README.md` | 설치, CLI, NPZ 계약, 검증 명령, WAVES 연결 주의사항, 로드맵 |
| `docs/phase1-analysis.md` | 실제 WAVES 필드/경로/누락/merge, 환경과 단계별 설계 분석 |
| `docs/phase2-validation.md` | 실제 checkpoint 추론과 검증 결과/한계/재현 명령 |
| `HANDOFF.md` | 현재 상태와 후속 작업 인수인계 |
| `THIRD_PARTY_NOTICES.md` | MIT/revision/metadata 및 checkpoint·예제 WAV 출처 |
| `pyproject.toml` | src package, NumPy 기본 의존성, atst/dev extras, CLI/data, pytest/Ruff |
| `requirements-cpu.txt` | CPU torch/torchaudio 2.10.0, NumPy 1.26.4, editable atst/dev 설치 |
| `src/waves_sed/__init__.py` | 패키지 버전 0.1.0 |
| `src/waves_sed/__main__.py` | CLI 진입점 |
| `src/waves_sed/backends/__init__.py` | backend package, inference dependency lazy import |
| `src/waves_sed/backends/base.py` | `predict(audio_path) -> FramePrediction` Protocol |
| `src/waves_sed/backends/atst.py` | weight 검증, 동결, chunk inference, 원본 duration clipping |
| `src/waves_sed/audio.py` | 모델 없는 frame grid, upstream 오디오 읽기, 리샘플링/원본 duration 분리, invalid audio 처리 |
| `src/waves_sed/prediction.py` | 결과 dataclass, 값/시간/label/JSON 검증, 무pickle NPZ, 원자적 저장, 오류 정규화 |
| `src/waves_sed/labels.py` | packaged 447-class vocabulary 로딩/index/중복 검증 |
| `src/waves_sed/provenance.py` | pinned revision/URL/hash, download/검증, preprocessing ID |
| `src/waves_sed/cli.py` | download/infer/inspect, cache 확인, CSV/JSON, source 덮어쓰기 보호 |
| `tests/test_audio.py` | 1 sample/부분 tail/10초 경계/invalid frame grid |
| `tests/test_prediction.py` | NPZ 왕복/Unicode/무pickle/invalid 값·시간·label·metadata |
| `tests/test_labels.py` | 실제 447개 ID/name/output 순서 |
| `tests/test_cli.py` | inference import 없는 workflow, cache mismatch, 경로 보호, 원본/hardlink regression |
| `tests/test_audio_loading.py` | stereo mono 평균, 22.05→16 kHz, empty/nonfinite 입력 거부 |
| `tests/test_atst_integration.py` | 공식 모델 stereo/22.05 kHz/여러 chunk/부분 tail/동결/NPZ, 기본 skip |
| `scripts/verify_upstream.py` | pinned clean upstream을 별도 프로세스로 실행, waveform/mel/확률 수치 비교 |

Vendor 루트: `src/waves_sed/_vendor/pretrained_sed/`.

| 파일 | 내용 |
| --- | --- |
| `models/atstframe/ATSTF_wrapper.py` | upstream 복사, import 2개만 package-relative로 변경 |
| `models/atstframe/audio_transformer.py` | upstream ATST 구조 그대로 |
| `models/atstframe/transformer.py` | upstream attention/block 그대로 |
| `models/transformer_wrapper.py` | upstream base wrapper 그대로 |
| `LICENSE` | upstream MIT 원문 |
| `UPSTREAM.md` | revision/출처/변경 범위 |
| `MANIFEST.sha256.json` | 원본/수정본 hash |

`_vendor`, `pretrained_sed`, `models`, `models/atstframe`의 `__init__.py`는 package marker이다.

Resources 루트: `src/waves_sed/resources/`.

| 파일 | 내용 |
| --- | --- |
| `audioset_strong.json` | index/id/name 447행, classifier 순서 |
| `class_labels_indices_strong.csv` | 원본 실제 metadata 456행 |
| `audioset_strong.provenance.json` | 원본 경로/hash/변환 규칙 |
| `README.md` | vocabulary provenance |
| `__init__.py` | resource package marker |

## 5. 검증된 정상 동작

- 실제 공개 WAV: 44,100 Hz mono, 1,653,688 samples, **37.49859410430839초**.
- 전체 CPU 추론 결과: **938 × 447 float32**, 확률 범위 약 `1.8619566e-7`~`0.4613507092`.
- `outputs/predictions/metro.npz`: 1,573,777 bytes.
- `outputs/reports/metro-inference.json`: 실제 최종 추론 summary/provenance.
- 16 kHz 리샘플링 길이는 599,978 samples/37.498625초. 출력 bin 끝은 원본 duration으로 제한.
- 최초 추론 32.28초, 최종 새 추론 **8.978초**, CPU4 threads/cache_hit=false.
  초기 import/JIT cache 차이가 있으므로 benchmark로 일반화하지 않는다.
- 최종 실제 cache 재사용: **cache_hit=true, 0.067초**.
- 변경하지 않은 upstream과 첫10초 비교: waveform/mel/probabilities의 **최대 절대 차이 모두0**.
  허용 오차 1e-6, parameter freeze/module eval 확인. 이 비교 JSON은 당시 tool output에만 있다.
- 실제 checkpoint 통합 테스트 포함 실행: **79 passed**, 11.06초 (CSV regression3개 추가 이전).
- 최종 오프라인 실행: **81 passed, 1 skipped**, 17.09초. skip은 환경변수 없는 실제 모델 test이다.
- audioread의 Python3.11 `aifc/audioop/sunau` deprecation warning3개만 있다.
- Ruff check/format check, dependency check, CPU requirements dry-run, wheel build 통과.
- 최신 CLI 보호 수정 후 wheel을 다시 만들지는 않았다. 배포가 필요하면 `uv build --wheel` 재실행.

이 입력은 WAVES stem이 아니라 upstream 공개 예제다. 이 검증은 추론 경로의 정확성 확인이며
SED의 실제 인식 성능이나 filter 품질을 입증하지 않는다.

## 6. 환경과 Git

Windows / Python3.11.9 / i3-1315U / RAM약16GB / CUDA없음.
프로젝트 `.venv` 설치 완료. 시스템 Python 및 WAVES 환경은 변경하지 않았다.

주요 버전: torch/torchaudio2.10.0+cpu, numpy1.26.4, librosa0.11.0,
soundfile0.13.1, einops0.8.1, pytest8.4.2.

Git commit:
- `b98c4a7`: 초기 범위/단계 문서
- `a7beefd`: WAVES 산출물 분석/통합 계획
- `f958804`: frozen inference와 raw cache
- `d01aee7`: 입력 경계 처리와 CSV source 보호, 추가 검증
- 최종 사용/검증/인수인계 문서는 위 commit 다음의 문서 commit으로 정리한다.
  정확한 최신 hash는 `git log`를 확인한다.

로컬 Git 제외 자산:
- `.cache/PretrainedSED`: 공식 repository clone, revision
  `1aa47e482f7e89904cba2338999345025d8b4e36`.
- `.cache/checkpoints/ATST-F_strong_1.pt`: 344,480,550bytes.
  SHA256 `fbf2577958e3648d55ee8cea7e0e5260c4505fb0c13964e1ec81bc367cee5eda`.
  공식 다운로드에서 계산한 hash이며 publisher-signed digest는 아니다.
- 예제 WAV:
  `.cache/PretrainedSED/test_files/752547__iscence__milan_metro_coming_in_station.wav`.
  SHA256 `13c04eca0967e264d18e03aef53d2fa6aee566d7946bdb0df7f6cd7ac72909d1`.
  iscence/Freesound752547, upstream attribution상 CC BY-NC4.0. Git에 재배포하지 않았다.
- `dist/waves_stem_sed-0.1.0-py3-none-any.whl`: 빌드 검증용 산출물.
- `outputs/`: raw NPZ와 실제 실행 요약.

## 7. 미해결 제약과 다음 작업 순서

Phase1~2에서 발견한 코드 오류는 수정했다. 남은 제약:
- 실제 WAVES WAV 없음. 향후 실제 stem 검증은 materialized run 또는 명시적인 경로 매핑이 필요.
- CUDA/Linux/다른 Python 버전 실행은 미검증.
- Phase3~7 기능은 아직 미구현. 현재 PASS/FAIL을 내리지 않음.
- WAVES merge/relabel/role 변경 시 planned support 해석은 adapter 정책으로 명시해야 함.

후속 순서:
1. 사용자가 새로 요청하는 범위와 현재 Git 상태 확인.
2. WAVES adapter를 구현하기 전 `final/<key>/metadata.json` + SAM manifest/DSP report의
   candidate 조인을 실제 schema와 frozen fixture로 확인.
3. `StemMetadata`에 final 의미/role, 원래 description/role/support와 provenance를 구분해 보존.
   파일 경로/시간 누락을 추정하지 않고 명시적인 상태로 남김.
4. 실제 AudioSet ontology loader와 명시적 수동 mapping config 구현.
   모호한 source는 `unsupported_mapping`; allowed family 여러 class 지원.
5. Phase3 테스트/문서/commit 후 Phase4: event extraction, 최적 onset matching,
   span 구간 집합 IoU/coverage, ambience occupancy/confidence와 JSON/CSV.
   synthetic interval로 missing/extra/timing을 검증하고 모든 threshold를 config로 분리.
6. 이후 Phase5 시각화/batch → Phase6 controlled corruption → Phase7 optional decision 순서.
   자세한 기준은 원문과 `docs/phase1-analysis.md` 참조.

## 8. 중요한 코드 구조와 주의사항

- materializer: `C:\WAVES\scripts\pipeline\materialize_stems.py`.
  최종 `stems[]`는 final_label/final_role/candidate_id/selected_attempt/merged_candidate_ids와
  file/hash/source-attempt provenance를 포함한다.
- **최종 metadata에는 description/activity_intervals가 없다.**
  `sam_manifest.json` 또는 `dsp_reports/<key>.json` candidate로 조인해야 한다.
- semantic merge는 대표 WAV 하나를 선택한다. child 시간 구간의 무조건적인 합집합은 잘못이다.
- frozen 자료: 29clips/108candidates/307attempts/62final stems.
  41 label변경, 2 role변경, 10 stems에 merged children.
  attempts에 실제 path 대신 `audio_id`만 있으며 checkout에 WAV는0개다.
- `legacy_16__02`처럼 span→onset 변경 시 원래 span을 개별 onset 정답으로 사용하지 않는다.
- PretrainedSED 전체는 pip package 구조가 아니고 학습/다른 backbone dependency가 많아
  MIT 최소 모델만 고정했다. 추후 vendor 업데이트는 hash/provenance와 수치 비교를 함께 갱신.
- 모델 출력은447개. 원본CSV456행/일반AudioSet527개 순서와 다르다.
  `as_strong_train_classes` 순서와 실제 CSV 이름을 조인했다.
  실제 ID에는 `/t/`도 있으므로 prefix를 임의 제한하지 않는다.
- 10초250frame/40ms bin은 출력 시간 축이다. centered STFT/global attention 때문에
  receptive field나 onset 정밀도와 동일하지 않고 chunk 경계에서 문맥이 끊긴다.
- FramePrediction은 probabilities[T,C], start/end[T], class IDs/names, JSON metadata.
  pickle없이 NPZ를 읽고 finite JSON만 허용한다.
- cache hit은 기존 device/dependency provenance를 보존하며 환경 변화만으로 재추론하지 않는다.
  모델/전처리 의미를 변경하면 preprocessing ID 또는 package version도 바꿔야 한다.
- CLI top-k는 raw 확률 요약이며 foreign event/품질 판정이 아니다.

## 9. 재개 명령

```powershell
cd C:\multitrack-audio-filtering
git status --short
git log -6 --oneline
.venv/Scripts/python.exe -m waves_sed inspect outputs/predictions/metro.npz
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/ruff.exe check src tests scripts
.venv/Scripts/ruff.exe format --check src tests scripts

# 실제 모델 검증이 필요할 때만:
$env:WAVES_SED_CHECKPOINT = (Resolve-Path .cache/checkpoints/ATST-F_strong_1.pt).Path
.venv/Scripts/python.exe -m pytest -q

# upstream 수치 비교가 필요할 때만:
.venv/Scripts/python.exe scripts/verify_upstream.py --upstream .cache/PretrainedSED --checkpoint .cache/checkpoints/ATST-F_strong_1.pt --audio .cache/PretrainedSED/test_files/752547__iscence__milan_metro_coming_in_station.wav
```

환경/weights는 이미 준비되어 있다. 의미 있는 코드 변경이나 실패가 없으면 무거운 검증을
불필요하게 반복하지 않고 다음 구현 단계로 진행한다.
