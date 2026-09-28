# 다음 Codex 세션 인수인계

최종 갱신: 2026-09-28. **Phase 1~6: frozen 추론, metadata/mapping, 역할별 평가,
batch/시각화와 controlled corruption evaluation을 완료했다.**
토큰 소진 후 저장된 변경을 이어 받아 Phase 6 파형 생성·paired report·실제 9개 추론과 검증을 마무리했다.
다음 단계는 Phase 7의 설정 기반 품질 판정 정책이다. 현재 `decision=null`이다.

## 1. 다음 세션에서 가장 먼저 확인할 것

1. `git status --short`, `git log -6 --oneline`과 이 문서를 읽는다.
2. `docs/phase1-analysis.md`에서 실제 WAVES 필드·merge/role 변경 문제를 확인한다.
3. `docs/phase6-validation.md`에서 corruption 정책, 새 CLI와 실제 검증 한계를 확인한다.
   `docs/phase5-validation.md`에서 batch/plot 계약과 선택 의존성을 확인한다.
   `docs/phase4-validation.md`에서 metric 단위/빈값/집계/해시 검증과 실행 방법을 확인한다.
   `docs/phase3-validation.md`는 metadata/ontology 계약, `docs/phase2-validation.md`는 추론 기록이다.
4. `C:\WAVES`에 실제 run/WAV가 추가되었는지 확인한다. 현재 checkout에는 WAV가 없다.
5. 새 사용자 지시를 확인한다. `reference_status=ambiguous`는 기본 평가 제외이며 설정 opt-in이
   필요하다. 실제 WAV가 없을 때 `audio_id`로 파일명을 만들지 않는다.

## 2. 요청과 작업 범위

- 작업 저장소: `C:\multitrack-audio-filtering`. 처음에는 빈 Git 저장소였다.
- WAVES 저장소: `C:\WAVES`, 조사 기준 HEAD `07af161`. 읽기만 했고 수정하지 않았다.
- 사용자 요구 원문:
  `C:\Users\owq05\.codex\attachments\62b196b9-c141-4c60-a879-514daa2f1960\Pasted text.txt`.
  PowerShell에서는 `Get-Content -Encoding UTF8`로 읽는다.
- 전체 목표는 독립 pretrained SED로 WAVES stem의 의미·시간 일관성을 측정하는 후처리 도구다.
  생성 모델 수정, 학습/fine-tuning, reference WAV 유사도 비교는 하지 않는다.
- 먼저 구조를 분석·설명하고 최소 단일-WAV prototype을 검증하라는 요청에 따라 Phase 1~2를 진행했다.
- 이어서 사용자가 다음 단계를 요청하여 Phase 3~6을 구현했다. Phase 7은 아직 구현하지 않았다.
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
- Materialized final metadata와 frozen finals/reports를 공통 StemMetadata로 변환한다.
  최종 의미/role과 이전 계획을 구분하고 merge/변경/누락 및 원본/hash를 보존한다.
- 선택 parent의 planned support만 사용한다. reference는 planned/ambiguous/missing으로 구분한다.
- 공식 ontology 632개 노드의 DAG 로딩/검증/ancestor·descendant 탐색과 CC BY-SA attribution.
- 모델 447개 ID 중 공식 archive에 없는 31개와 이름이 다른 11개를 명시적으로 보고한다.
- JSON 수동 mapping, 정확한 alias, 선택적 descendant 확장, unsupported 상태와 ID 기반 target max.
- `adapt-waves`와 `map-source` CLI. 모델이나 새 의존성 없이 동작하고 원본 파일을 보호한다.

- 설정 기반 median smoothing/threshold/최소 길이 event extraction. Raw cache는 변경하지 않는다.
- WavesPlannedReference와 ExternalVideoReference Protocol, ambiguous opt-in/누락/모순 상태 처리.
- Onset 최적 matching, span 합집합 IoU/coverage, ambience 시간 가중 raw confidence.
- 실제 audio hash 또는 명시적 final manifest hash로 cache identity 확인. 전체 관측 timeline 검사.
- Outside-family top-k와 ancestor/descendant/unknown hierarchy 표시, 명시적 foreign evidence.
- `evaluate` CLI로 기존 cache 모음의 stem/clip/dataset JSON과 CSV를 저장한다.
  미지원/누락은 null, 기여 수가 있는 역할별 평균과 onset pooled count를 제공한다.
- 입력 및 원본 media/hardlink 보호, 추론 없는 재평가, NumPy만 있는 wheel 환경 검증.

- `batch-infer`: 실제 normalized audio 경로만 사용, 필요할 때 모델을 한 번 로드하고 순차 추론한다.
  입력/모델/vocabulary가 일치하는 cache 재사용, 개별 실패 기록, 성공한 항목만 index에 등록한다.
- `visualize`: native-rate 전 채널 min/max 파형, expected, raw/median target, detected를 같은 축에
  표시하고 PNG/SVG 및 static HTML 목록을 저장한다. 추론 없이 report/cache/source를 다시 연결 검증한다.
- 누락·모호함을 그림에 명시하고 원본·cache·config·report mismatch를 거부한다.
  긴 제목과 한글 font fallback, 전체 출력 preflight, atomic export를 지원한다.

- `corrupt`: 원본 길이/샘플레이트/채널을 유지하는 shift/remove/duplicate/shorten/extend WAV 생성.
  원본 대조군과 각 독립 변형의 manifest/hash/application을 저장하고 expected/reference는 유지한다.
- `compare-corruptions`: 실험 manifest에 연결된 동일 설정/모델/reference 보고서의 metric 차이,
  검출 개수·support IoU·단일 event displacement, 유효 기여 수가 있는 operation별 요약.
- 변형 생성 실패·파형 변화 없음·누락·불일치는 unavailable/null로 보고한다. NumPy만으로 비교 가능.
- 실제 공개 WAV에서 대조군+8변형을 한 모델로 추론하고 paired JSON/CSV와 9개 PNG/HTML을 생성했다.
  Synthetic 전체 길이 reference임을 명시하며 영상 annotation/인식 성능으로 해석하지 않는다.

**미구현 후속 기능:** PASS/REVIEW/FAIL 정책. HTML은 static gallery이며
브라우저에서 threshold를 바꾸는 대화형 편집기는 구현하지 않았다.

## 4. 파일별 구현/변경 내용

| 파일 | 구현/변경 |
| --- | --- |
| `.gitignore` | venv/cache/weights/outputs/build 및 Python cache 제외 |
| `.gitattributes` | 소스/문서/CSV/TXT/LICENSE LF 설정 |
| `README.md` | 설치, CLI, NPZ 계약, 검증 명령, WAVES 연결 주의사항, 로드맵 |
| `docs/phase1-analysis.md` | 실제 WAVES 필드/경로/누락/merge, 환경과 단계별 설계 분석 |
| `docs/phase2-validation.md` | 실제 checkpoint 추론과 검증 결과/한계/재현 명령 |
| `docs/phase3-validation.md` | adapter/mapping 사용법, ontology 차이, 누락/시간 기준 정책, 실제 자료 검증 |
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

Phase 3 추가 파일과 변경 내용:

| 파일 | 구현/변경 |
| --- | --- |
| `src/waves_sed/metadata.py` | StemMetadata와 JSON 직렬화 |
| `src/waves_sed/adapters/__init__.py` | adapter API export |
| `src/waves_sed/adapters/waves.py` | load_materialized/load_frozen, candidate 조인/충돌 검사, 선택 parent support/provenance, 명시적 audio ID mapping |
| `src/waves_sed/ontology.py` | 공식/custom DAG, hash/중복/cycle 검사, 탐색, vocabulary coverage |
| `src/waves_sed/mapping.py` | JSON rule 검증, exact alias, class ID 해석, unsupported와 raw max 집계 |
| `src/waves_sed/phase3.py` | 새 CLI, JSON 출력, 입력 metadata/config/음원/영상 덮어쓰기 방지 |
| `src/waves_sed/cli.py` | adapt-waves/map-source 연결; 기존 추론 경로 유지 |
| `configs/source_mappings.example.json` | 실제 Bark/Chop/Chopping (food)/Walk, footsteps ID의 제한된 예제 |
| `src/waves_sed/resources/audioset_ontology.json` | 공식 archive 원본 632 nodes |
| `src/waves_sed/resources/audioset_ontology.provenance.json` | source/revision/hash/license |
| `src/waves_sed/resources/audioset_ontology.LICENSE.md` | Google/Dan Ellis 및 CC BY-SA 4.0 attribution |
| `tests/test_ontology.py`, `tests/test_mapping.py` | graph/metadata 차이/exact mapping/unsupported/column 정렬 검증 |
| `tests/test_waves_adapter.py`, `tests/test_phase3_cli.py` | 실제 schema/누락/merge/role 변경/CLI/원본 보호 검증 |
| `tests/fixtures/waves/frozen_finals.json`, `frozen_reports.json`, `README.md` | 원본 WAVES 4개 사례의 최소 fixture와 출처 |

README/THIRD_PARTY_NOTICES/HANDOFF도 Phase 3 상태로 갱신했다. 새 dependency는 없다.

Phase 4 추가/수정 파일:

| 파일 | 구현/변경 |
| --- | --- |
| `src/waves_sed/evaluation_config.py`, `configs/evaluation.example.json` | 엄격한 설정 schema, uncalibrated event/tolerance/evidence 기본값 |
| `src/waves_sed/events.py` | 확률 median smoothing, 실제 gap 분리, threshold와 실제 길이 필터 |
| `src/waves_sed/temporal_reference.py` | provider Protocol, Waves 계획 사용 정책, typed support/provenance 검증 |
| `src/waves_sed/temporal_metrics.py` | 구간 합집합/교집합, 최적 onset DP, span/ambience metric |
| `src/waves_sed/evaluation.py` | cache identity와 전체 관측 범위, role dispatch, semantic evidence, null 상태/provenance |
| `src/waves_sed/metadata.py` | `from_dict`/`load_stems`, Phase 3 manifest 검증/상대 경로 해석 |
| `src/waves_sed/phase4.py`, `src/waves_sed/cli.py` | evaluate CLI, 명시적 stem ID→NPZ index, 입력/원본 보호 |
| `src/waves_sed/reporting.py` | 역할별 평균/분모, pooled count, context 혼합 방지, hashed 파일명, atomic JSON/CSV |
| `tests/test_evaluation_config.py`, `test_events.py`, `test_temporal_reference.py`, `test_temporal_metrics.py` | 타입/경계/gap/모호함/최적 matching/부분 frame/empty 정책 회귀 |
| `tests/test_evaluation.py`, `test_phase4_cli.py`, `test_reporting.py` | 해시/누락/재평가/aggregate/source 보호/추론 import 차단 |
| `scripts/verify_cached_evaluation.py` | 실제 cache를 두 threshold로 재평가, synthetic 전체 길이 reference임을 명시 |
| `docs/phase4-validation.md`, `README.md`, `HANDOFF.md` | 실행법, metric 단위/한계/최종 검증/다음 작업 |

Phase 4도 새 dependency를 추가하지 않았다. 기본 NumPy 외에 scipy나 inference library는 필요 없다.

Phase 5 추가/수정 파일:

| 파일 | 구현/변경 |
| --- | --- |
| `src/waves_sed/batch.py` | 단일 lazy backend 순차 추론, exact cache 재사용, 개별 실패/성공 index, 출력·source hash 보호 |
| `src/waves_sed/visualization.py` | streaming min/max native 파형, report/배열/hash 연결 검증, 공통 시간축 PNG/SVG, 글꼴·긴 제목 처리 |
| `src/waves_sed/phase5.py` | batch-infer/visualize orchestration, 모든 입력 preflight, 선택 stem, escaped HTML/JSON index |
| `src/waves_sed/cli.py` | 새 명령 연결, 기존 lazy inference 유지 |
| `pyproject.toml` | visualization extra: matplotlib>=3.9,<4 및 soundfile==0.13.1 |
| `tests/test_batch.py` | 45 tests: 모델 1회 초기화, cache/failure/overwrite/원본 보호 |
| `tests/test_visualization.py` | 37 tests: impulse/stereo/부분 bin, unavailable/ambiguous/identity, PNG/SVG/긴 제목 |
| `tests/test_phase5_cli.py` | 50 tests: offline CLI, 통합 결과, HTML escaping, 선택·오류·원본/hardlink 보호 |
| `docs/phase5-validation.md`, `README.md`, `HANDOFF.md` | 단계별 명령, 실제 검증, 제약, 다음 단계 |

Matplotlib은 그림에 필요하다. soundfile은 그림/파형 편집에 쓰며 기본 dependency는 계속 NumPy 하나다.

Phase 6 추가/수정 파일:

| 파일 | 구현/변경 |
| --- | --- |
| `src/waves_sed/corruption.py` | 엄격한 설정/spec, 정확한 sample 반올림, 5가지 파형 조작, crop/overlap/no-effect 기록 |
| `src/waves_sed/corruption_experiment.py` | 원본 보존 control, 독립 FLOAT WAV, immutable 실험 manifest, 전체 입력/hardlink 보호 |
| `src/waves_sed/corruption_comparison.py` | manifest/report/hash/context 연결 검증, 역할별 delta·검출 변화, unavailable·분모, JSON/CSV |
| `src/waves_sed/phase6.py`, `src/waves_sed/cli.py` | corrupt/compare-corruptions CLI, strict JSON, input 변경 감지, lazy optional dependency |
| `configs/corruption.example.json` | +100/200/500/1000ms 및 구간 삭제/복제/길이 변경 8개 예제 |
| `pyproject.toml` | corruption extra: soundfile==0.13.1 |
| `tests/test_corruption.py` | 108개 sample 정확도/비정상 설정/거대한 이동·길이/원본 보존 검사 |
| `tests/test_corruption_comparison.py` | 38개 synthetic oracle/설정·hash 불일치/빈값·분모/원본 identity 검사 |
| `tests/test_phase6_cli.py` | 49개 생성→평가→비교, 원본·미선택·hardlink 보호, import 격리/입력 변경 검사 |
| `scripts/verify_corruption_pipeline.py` | synthetic reference를 명시하는 실제 모델 공개 음원 검증, report/plot/비교 재현 |
| `docs/phase6-validation.md`, `README.md`, `HANDOFF.md` | 조작/비교 정책, 실행법, 실제 관측 결과와 한계, 다음 작업 |

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
- Phase 2 최종 오프라인 실행: **81 passed, 1 skipped**, 17.09초.
- Phase 3 최종 전체 실행: **226 passed, 1 skipped**, 9.83초. skip은 환경변수 없는 실제 모델 test다.
- Phase 4 최종 전체 실행: **515 passed, 1 skipped**, 13.69초. Ruff lint/format 및 wheel build 통과.
- Phase 5 최종 전체 실행: **647 passed, 1 skipped**, 54.62초. Ruff lint/format 및 wheel build 통과.
- Phase 6 최종 전체 실행: **842 passed, 1 skipped**, 69.67초. Phase 6 신규 195개 통과.
  Ruff lint/format, diff check, wheel build 및 독립 환경 검증 통과.
- audioread의 Python3.11 `aifc/audioop/sunau` deprecation warning3개만 있다.
- Ruff check/format check, dependency check, CPU requirements dry-run, wheel build 통과.
- Phase 3 최종 코드로 wheel을 빌드했다. 별도 `.cache/phase3-package-smoke` 환경에 wheel과
  NumPy만 설치해 632-node ontology/447-class metadata/mapping을 검증했다. torch는 설치되지 않았다.

Phase 3 실제 자료 검증:
- WAVES frozen 29 clips / 62 stems 모두 정규화했다.
- relabel41 / role변경2 / merge10 / 음원경로누락62 / video ID누락62.
- reference ambiguous43 / planned19. 예제 mapping supported2 / unsupported60.
- `outputs/phase3/frozen-stems.json`: 전체 normalized metadata와 mapping/provenance.
- `outputs/phase3/metro-chop-mapping.json`: 기존 실제 NPZ의 두 class target max.
  938 frames를 직접 NumPy max와 비교하여 동일함을 확인했다. 이 소리가 검출되었다는 주장은 아니다.

이 입력은 WAVES stem이 아니라 upstream 공개 예제다. 이 검증은 추론 경로의 정확성 확인이며
SED의 실제 인식 성능이나 filter 품질을 입증하지 않는다.

Phase 4 실제 자료 검증:
- `outputs/phase4/frozen-reports/dataset.json`: 29 clips / 62 stems, 모두 unavailable.
  실제 cache가 없는 빈 index이므로 missing_prediction 62, unsupported_mapping 60,
  ambiguous_reference 43이다. metric/null과 기여 수 0으로 남으며 실패 event를 만들지 않는다.
- `outputs/phase4/metro-demo/threshold-0.2/`, `threshold-0.5/`: 실제 metro NPZ 938 frames를
  /m/0195fx (Subway, metro, underground)에 연결했다. 각 threshold에서 1개 / 0개 event 검출.
  원본 cache hash 불변, inference_performed=false.
- 이 demo의 전체 길이 support는 synthetic fixture이며 WAVES 계획/영상 annotation이 아니다.
  `origin=synthetic_full_track_demonstration`을 명시한다. 여기서의 IoU는 성능 점수가 아니다.
- `.cache/phase4-package-smoke`: wheel과 NumPy 1.26.4만 설치. 추론 라이브러리 부재 확인 후
  같은 cache API와 frozen CLI를 실행했다. 결과는 outputs/phase4/wheel-*에 있다.

Phase 5 실제 자료 검증:
- 공개 metro WAV의 0~2초 / 10~12.5초 구간을 별도 PCM16 WAV 두 개로 저장했다.
  metadata는 명시적인 demo fixture이고 WAVES 생성 stem이나 영상 annotation이 아니다.
- `outputs/phase5/demo-inputs/`: `stems.json`, `mapping.json`, `evaluation.json`, excerpt WAV 2개.
  둘 다 expected support가 없어 시간 metric은 null이며 detection은 available이다.
- `outputs/phase5/demo-batch/`: 실제 CPU 추론 2개, backend_initializations=1 확인.
  같은 batch를 없는 checkpoint 경로로 다시 실행하여 cached=2/inferred=0/initializations=0 확인.
- `outputs/phase5/demo-evaluation/`, `demo-visuals/index.html`: 평가 report와 2개 PNG/HTML.
- `outputs/phase5/metro-full-demo.png`, `.svg`: 기존 938-frame metro cache와 Phase 4의 synthetic
  전체 길이 reference를 Python API에 연결한 그림. 직접 이미지를 확인했고 네 패널이 같은 축으로 보인다.
- `outputs/phase5/frozen-batch/`: 실제 frozen 62개 모두 missing_audio, backend초기화0, 빈 성공index.
  이는 예상된 자료 부재 결과이며 모델 실패가 아니다. `frozen-visual/`은 첫 stem의 누락/모호함 SVG다.
- `.cache/phase5-package-smoke`: wheel+NumPy만으로 batch cache hit 확인 후 visualization extra만
  추가해 `outputs/phase5/wheel-visuals/`의 SVG 2개 생성. torch/torchaudio/librosa는 설치하지 않았다.

Phase 6 실제 자료 검증:
- `outputs/phase6/metro-demo/inputs/`: 명시적인 synthetic 전체 길이 support metadata, Subway mapping,
  threshold0.2/median1/min_duration0. `origin=synthetic_full_track_demonstration`, status=demonstration.
- `experiment/`: 원본 control과 8개 변형의 WAV/manifest. 원본 SHA 불변, 조작 전후 동일 reference.
- `batch/`: 실제 CPU inferred9/cached0/failed0, 모델 초기화1, 45.356초. 모든 cache는938×447.
- `reports/dataset.json`, `comparison/comparison.json`과 CSV: 8쌍 비교 성공.
  100/200/500/1000ms 이동에 대한 관측 onset 변화는80/160/480/1000ms.
  Remove/shorten은 검출을 쪼개 개수+2/+1, duplicate는 중첩되어 구간 변화 없음.
  이는 단일 공개 clip의 반응이며 실제 품질·영상 annotation·일반 민감도 점수가 아니다.
- `visuals/index.html`: 9개 PNG와 JSON index. 500ms shift 그림을 직접 확인했다.
- 일반 evaluate CLI는 WAVES reference만 사용하므로 synthetic origin을 승인하지 않는다.
  이 demo 재평가는 `scripts/verify_corruption_pipeline.py`의 명시적 reference API로 한다.
- `.cache/phase6-package-smoke`: wheel+NumPy만으로 8쌍 재비교, 원래 comparison과 동일 확인.
  이후 corruption extra(soundfile)만 추가하여 8개 WAV 생성. torch/torchaudio/librosa/matplotlib 없음.
  8개 변형의 디코딩된 sample/application 동일 확인. PEAK chunk 일부 bytes는 재생성 시 달라졌으므로
  experiment ID와 WAV hash의 의미를 구분한다. Cache identity는 실제 파일 bytes의 SHA다.
- `outputs/phase6/wheel-comparison`, `wheel-experiment`: 독립 설치 검증 산출물.

## 6. 환경과 Git

Windows / Python3.11.9 / i3-1315U / RAM약16GB / CUDA없음.
프로젝트 `.venv` 설치 완료. 시스템 Python 및 WAVES 환경은 변경하지 않았다.

주요 버전: torch/torchaudio2.10.0+cpu, numpy1.26.4, librosa0.11.0,
soundfile0.13.1, einops0.8.1, pytest8.4.2.
Phase 5 시각화 검증은 matplotlib3.11.2, pillow12.3.0을 사용했다.

Git commit:
- `b98c4a7`: 초기 범위/단계 문서
- `a7beefd`: WAVES 산출물 분석/통합 계획
- `f958804`: frozen inference와 raw cache
- `d01aee7`: 입력 경계 처리와 CSV source 보호, 추가 검증
- `bc10e83`: Phase 2 사용/검증/인수인계 문서
- `f47c66c`: 공식 ontology와 명시적 source mapping
- `fe25584`: WAVES adapter와 offline mapping CLI
- `048c327`: Phase 3 검증/시간 기준 한계 문서
- `3524817`: temporal reference/event extraction/역할별 metric
- `73876f7`: cache 검증 evaluator와 JSON/CSV report/CLI
- `79a315e`: Phase 4 문서/인수인계
- `287277c`: resumable batch inference와 단일 backend
- `f271193`: PNG/SVG evidence와 batch/visualize CLI, HTML gallery
- `03cd970`: Phase 5 문서/인수인계
- `ebde56e`: controlled waveform experiment, paired evaluation, CLI/테스트/실제 검증 스크립트
- Phase 6 문서 commit이 뒤에 이어진다. 정확한 hash는 `git log`를 확인한다.

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
- `.cache/audioset-ontology`: 공식 ontology 조사용 clone. 일반 실행에는 필요 없다.

## 7. 미해결 제약과 다음 작업 순서

Phase 1~6에서 발견한 코드 오류는 수정했다. 남은 제약:
- 실제 WAVES WAV 없음. 향후 실제 stem 검증은 materialized run 또는 명시적인 경로 매핑이 필요.
- CUDA/Linux/다른 Python 버전 실행은 미검증.
- Phase 7 기능은 아직 미구현. 현재 PASS/FAIL을 내리지 않음.
- 공식 ontology에 없는 31개 모델 ID는 hierarchy가 알려져 있지 않음. 직접 mapping은 가능하나 descendant 확장 금지.
- 실제 reference annotation이나 threshold calibration은 없음. WAVES 계획만으로 영상 동기화를 입증하지 못함.
- Report 각 파일은 atomic replace지만 여러 파일 전체를 한 transaction으로 쓰는 것은 아님.
  재실행에서 남은 옛 파일은 지우지 않으므로 현재 dataset.json의 index만 사용해야 함.

후속 순서:
1. 사용자가 새로 요청하는 범위와 현재 Git 상태 확인.
2. 원문 15절 및 Phase 7 요구를 확인한다. `FilterDecision` PASS/REVIEW/FAIL/UNSUPPORTED와
   역할별 threshold config를 설계하되 기본 null이면 판정에 쓰지 않는다.
3. 평가 metric과 optional decision을 분리한다. 미지원 mapping, 누락/모호한 reference,
   cache 오류, 필요한 metric이 null인 경우를 정상 판정으로 오해하지 않도록 상태 정책을 정한다.
4. `onset.min_event_recall/max_mean_onset_error_ms`, `span.min_tiou`,
   `ambience.min_occupancy`처럼 명시한 조건만 사용한다. REVIEW/FAIL 경계와 어떤 조건이
   판정에 기여했는지 기록한다. 근거 없는 calibrated default를 만들지 않는다.
5. 기존 JSON/CSV/시각화와 연결하고, threshold 없는 기존 동작과 raw cache 재사용을 유지한다.
   Synthetic report로 경계값·null·역할·미지원·복수 조건을 검증한다.
6. 실제 WAVES run/WAV가 추가되면 소수 stem의 선택 구간에 Phase 6을 적용한다.
   단일 공개 demo로 실용 threshold나 perceptual quality를 보정하지 않는다.
7. 실행법/테스트/한계를 기록하고 의미 있는 단위로 commit한다. Human evaluation platform이나
   학습은 구현하지 않는다. 사용자 요청에 따라 단계별로 진행한다.

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
- `evaluation.py`는 추론 모듈을 import하지 않는다. config, reference, cache, mapping을 검증하고
  role별 metric 함수를 호출한다. Raw target 확률/NPZ는 event extraction과 분리되어 있다.
- Onset 오차 통계는 절대 ms, span onset/offset은 signed 초이며 여러 구간의 외곽 경계다.
  out_of_window_activation/activity는 초다. Ratio의 0분모와 대응 없는 오차는 null이다.
- WAVES empty/null 계획은 unavailable이다. 외부 provider의 명시적 빈 annotation은 API에서 허용한다.
- `planned`인데 relabel/role 변경/merge 신호가 있으면 inconsistent_reference_status로 거부한다.
- 파일이 없을 때 명시적인 raw_final_stem.sha256으로 cache를 연결할 수 있지만 source_file_verified=false다.
  실제 파일 bytes를 읽어 검증한 verified_current_audio와 구분한다.
- Report 집계는 evaluated 행만 사용하고 설정/모델/mapping hash/reference origin 혼합을 거부한다.
  missing cache의 요청 경로는 provenance.prediction_request에 남긴다.
- Batch는 실제 audio 파일이 없으면 missing_audio로 남긴다. manifest hash만 있는 cache를 평가하는
  Phase 4와 다르다. 모든 cache가 일치하면 모델/decoder/plot library와 checkpoint가 필요 없다.
- `--overwrite`는 정상 matching cache도 새로 추론한다. 기본값은 손상/불일치 cache를 보존한다.
- Batch manifest/전체 cache 출력/report 출력은 처리 전에 함께 검사한다. 이전에 남은 cache의
  metadata.audio_path도 보호한다. 성공 index는 현재 실행의 성공 항목만 포함한다.
- Plot은 report-bound NPZ path/hash와 배열·metadata를 검증하고 현재 source bytes도 확인한다.
  메모리에만 있는 report는 raw array의 출처를 묶을 수 없어 target curve를 unavailable로 표시한다.
- 파형은 native sample rate/전 채널 min/max envelope이다. stereo 평균이나 stride 샘플링이 아니다.
  max_waveform_points는 표시 bin 수이며 모든 sample을 bounded block으로 읽어 impulse를 보존한다.
- HTML은 정적 local gallery이며 외부 CDN/모델 호출 없이 PNG/SVG를 링크한다. 제목은 HTML escape한다.
- Corruption은 원본 native sample frame을 고정한다. 빈 곳은0, clip 밖 삽입은crop, 중첩은덧셈이다.
  Shorten은 prefix만 남기고 extend는 반복한다. Fade/time stretch/normalization/clipping은 없다.
  float32 decode/FLOAT WAV이며 고정확도 원본의 float32 변환 한계와 boundary artifact가 존재한다.
- Expected/reference/description/role은 변형 전후 고정한다. 원본 metadata는 provenance.corruption에
  nested 보존하고 현재 변형 raw_final_stem hash는 새 WAV로 바꾼다. 변형은 누적하지 않는다.
- `audio_changed=false`는 민감도 비교에서 제외한다. 구간 제거/복제는 semantic missing/extra를
  보장하지 않는다. 실제 파형 조작과 관측 SED 반응을 별도 필드로 남긴다.
- Compare는 저장된 report와 manifest hash/context를 비교하며 현재 WAV/NPZ를 다시 읽지 않는다.
  현재 bytes 검증에는 evaluate를 다시 실행한다. 생성 manifest는 기존 파일을 덮어쓰지 않는다.
- mapping의 supported는 alias/class column 연결 성공이다. 검출 여부나 품질 점수가 아니다.
- `reference_status=planned`도 실제 영상의 정답을 뜻하지 않는다. 이 archive의 ambiguous43개를
  그대로 정답으로 간주하지 않는다. merged child 시간은 parent와 합집합으로 만들지 않았다.
- 공식 ontology revision은 `d417d32bf59c711abb5910fd2f76a0eb44697991`, 원본 SHA256은
  `9c685f4403eecc3ca9be37fd7285cf212feaaea6ff7229d3e7ca89e0d1f2d15d`다.
  ID 기준으로 416개가 겹치고 31개는 누락, 표시명11개 차이. 원본 두 자료를 수정하지 않았다.

## 9. 재개 명령

```powershell
cd C:\multitrack-audio-filtering
git status --short
git log -6 --oneline
uv pip install --python .venv/Scripts/python.exe -e ".[visualization,corruption]"
.venv/Scripts/python.exe -m waves_sed compare-corruptions --experiment outputs/phase6/metro-demo/experiment/experiment.json --report outputs/phase6/metro-demo/reports/dataset.json --output-dir outputs/phase6/metro-demo/comparison
# 실제 추론까지 재현할 때만 새 output-dir로 실행:
# .venv/Scripts/python.exe scripts/verify_corruption_pipeline.py --audio .cache/PretrainedSED/test_files/752547__iscence__milan_metro_coming_in_station.wav --class-id /m/0195fx --corruptions configs/corruption.example.json --output-dir outputs/phase6/metro-demo-new
.venv/Scripts/python.exe -m waves_sed batch-infer --stems outputs/phase5/demo-inputs/stems.json --output-dir outputs/phase5/demo-batch
.venv/Scripts/python.exe -m waves_sed visualize --stems outputs/phase5/demo-inputs/stems.json --predictions outputs/phase5/demo-batch/prediction-index.json --mappings outputs/phase5/demo-inputs/mapping.json --config outputs/phase5/demo-inputs/evaluation.json --output-dir outputs/phase5/demo-visuals
.venv/Scripts/python.exe -m waves_sed inspect outputs/predictions/metro.npz
.venv/Scripts/python.exe -m waves_sed adapt-waves --frozen-finals C:/WAVES/data/frozen_pass2/frozen_finals.json --frozen-reports C:/WAVES/data/frozen_pass2/frozen_reports.json --mappings configs/source_mappings.example.json --output outputs/phase3/frozen-stems.json
.venv/Scripts/python.exe -m waves_sed evaluate --stems outputs/phase3/frozen-stems.json --predictions outputs/phase4/frozen-prediction-index.json --mappings configs/source_mappings.example.json --config configs/evaluation.example.json --output-dir outputs/phase4/frozen-reports
.venv/Scripts/python.exe scripts/verify_cached_evaluation.py --prediction outputs/predictions/metro.npz --audio .cache/PretrainedSED/test_files/752547__iscence__milan_metro_coming_in_station.wav --class-id /m/0195fx --output-dir outputs/phase4/metro-demo
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
