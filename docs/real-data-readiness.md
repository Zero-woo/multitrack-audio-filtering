# 실제 WAVES 데이터의 가용성과 검증 절차

조사 기준일: 2026-09-29. 대상: WAVES revision `07af161`의 frozen 29개 clip / 62개 final stem.
이 문서는 생성 음원·metadata·검수 자료의 가용성, 수동 mapping 후보의 적용 범위와 실제 데이터 검증 절차를 정의한다.
음원 위치 기록의 대조와 CLI 누락 처리 검증을 포함하며, **실제 WAVES 음원 평가와 운영 threshold 보정은 미수행 상태다.**
Phase 1~7의 기능 계약과 사용법은 [프로젝트 개요](../README.md) 및 단계별 문서에 기술한다.

## 입력 자료와 가용성

조사 대상 `C:\WAVES`의 숨김·Git 제외 파일까지 검색한 결과 WAV/FLAC/MP3/OGG/M4A/AIFF/AAC는 없었다.
WAVES `README.md`, `docs/REPRODUCIBILITY.md`, `docs/DEVELOPMENT_CHECKPOINT_2026-08-11.md`도
큰 평가 음원과 로컬 데모가 Git 밖에 있음을 명시한다. 다음 자료를 실제 실행 폴더에서 연결해야 한다.

| 자료 | 조사 결과 | 연결에 필요한 입력 |
| --- | --- | --- |
| Frozen finals/reports | 62개 final, 선택 attempt의 `audio_id`만 있음 | 실제 selected WAV 경로 또는 materialized metadata |
| System evaluation manifest | 30개 clip / 66개 target의 과거 `waves_file` 기록 | 해당 실행 폴더와 실제 파일 |
| Listening 18 manifest/UI | A/B/C 54개 음원 경로, 조사 시점에 모두 없음 | 음원과 method key; A/B/C만으로 WAVES를 식별하지 않음 |
| Human alignment JSON | generated_waves 12행, original_waves 13행의 누음 평균 | 대상 WAV의 provenance, 별도 시간/품질 검수 기준 |

가장 직접적인 입력은 WAVES 실행 결과의 `final/<clip_key>/metadata.json`, 같은 폴더의 `stem_NN.wav`,
그리고 `sam_manifest.json` 또는 `dsp_reports/<clip_key>.json`이다. Final metadata의 file/hash와
선택 attempt를 확인한다. Final metadata에는 계획 description/activity_intervals가 없으므로
기존 adapter로 중간 자료를 조인한다. 파일 연결에는 선택된 음원과 해당 metadata가 있는 실제 실행 경로가 필요하다.

## 과거 평가 경로 대조

`data/system_eval/evaluation_manifest66.json`의 `(key, target_id)`를 frozen `(clip_key, candidate_id)`와
대조하였다. **57개 identity가 일치**하고, 기록된 `waves_file`의 candidate 번호와 선택 attempt도 일치한다.
선택 attempt 불일치는 0개, system에만 있는 target은 9개다. 경로의 일치는 파일 bytes 검증을 뜻하지 않는다.

기록 예:

```text
${WAVES_ROOT}/listening_test/foleybench5k_listening21_work/generated_condition/sam_attempts/fb5k_1229/c01_a1.wav
```

이외 일치 경로는 `add12_work/generated`, `add5_work/generated_condition`,
`replace_38_work/generated_condition`, `replace_241_work/generated_condition` 아래에 있다.
`${WAVES_ROOT}`는 과거 manifest의 placeholder이며 현재 존재하는 WAV 경로가 아니다.
`sam_attempt1_file`은 비교용 첫 attempt일 수 있으므로 `waves_file` 대신 쓰지 않는다.

| System manifest에 없는 frozen candidate | 선택 attempt |
| --- | ---: |
| `fb5k_2747__01` | 1 |
| `fb5k_2747__02` | 3 |
| `fb5k_3316__03` | 1 |
| `fb5k_3316__05` | 3 |
| `fb5k_383__03` | 2 |

경로를 ID로 재구성하면 안 되는 사례도 있다. `legacy_38__01/__02`는 기록상 `foley_38` 폴더,
`random_241__02/__03`는 `foley_241` 폴더를 사용한다. Frozen final role과 system role이 다른
`fb5k_671__03`(ambience/span), `legacy_16__02`(onset/span)는 final role을 유지한다.

`results/metric_human_alignment.json`의 generated 12행은 모두 같은 candidate에 대응하지만,
그 행에는 대상 음원의 파일/hash/선택 attempt가 없다. 평가 스크립트
`scripts/evaluation/evaluate_metric_human_alignment.py`는 저장소에 없는 `method_key.json`의
`sources`에서 음원을 찾는다. 결과에 있는 competitor 경로는 대상 WAV 경로가 아니다.
따라서 method key와 실제 파일 없이 청취 점수를 현재 selected stem에 확정 연결하지 않는다.
이 값들은 target가 clear/weak인 응답의 누음 평균이며, 전체 PASS/FAIL이나 onset 정답도 아니다.

## Source mapping 후보와 적용 범위

[`source_mappings.frozen-review.json`](../configs/source_mappings.frozen-review.json)은 최종 설명의
명시적 alias로 구성한 18개 규칙이다. 대소문자/공백 정규화만 사용하며 descendant/foreign class를 추가하지 않는다.
모든 class ID가 실제 모델 447개 출력과 pinned ontology에 있음을 확인하였다.
**문자 설명을 모델 column에 연결하는 후보이지, 실제 음원으로 검증된 인식 성능이 아니다.**
가상 장면의 웃음·엔진·whoosh는 렌더링된 음색을 들은 뒤 재검토한다. 광범위한 Engine class도
차종이나 특정 hum 음색을 확인하는 근거가 아니다. 각 규칙의 `notes`에 해석 범위를 남겼다.

| 최종 description alias | 모델 class / ID | Stem 수 |
| --- | --- | ---: |
| player weapon fire | Gunshot, gunfire `/m/032s66` | 1 |
| Aircraft machine-gun fire | Machine gun `/m/04zjc` | 2 |
| Aircraft explosion | Explosion `/m/014zdl` | 1 |
| Woman landing with a thud; Heavy wooden panel thumps | Thump, thud `/m/07qnq_y` | 2 |
| wizard laugh | Laughter `/m/01j3sz` | 1 |
| basketball bouncing on gym floor | Basketball bounce `/m/018w8` | 1 |
| lubricant spray bursts | Spray `/m/07qlf79` | 1 |
| P-38 aircraft engine roar | Aircraft engine `/m/014yck` | 1 |
| Puppy low growl | Growling `/m/0ghcn6` | 1 |
| repeated cleaver chopping food | Chopping (food) `/m/07pn_8q` | 1 |
| Crowd ambience | Crowd `/m/03qtwd` | 1 |
| Box fan running | Mechanical fan `/m/02x984l` | 1 |
| Dirt bike engine revving through a forest trail; underwater motorcycle engines revving | Accelerating, revving, vroom `/m/07q2z82` | 2 |
| dog puffing exhalation | Puff `/m/07q34h3` | 1 |
| Trailer running-gear rattle | Rattle `/m/07qn4z3` | 1 |
| Soft lightsaber swing whooshes | Whoosh, swoosh, swish `/m/07rqsjt` | 1 |
| Man walking footsteps; running footsteps on grass | Walk, footsteps `/m/07pbtc8` | 2 |
| Vehicle engine hum; Tractor engine | Engine `/m/02mk9` | 2 |

기존 예제의 2/62에서 이 후보 설정은 **23/62**를 연결한다. 두 설정은 별도 파일이다.

| Mapping 상태 | 합계 | Planned | Ambiguous | Onset | Span | Ambience |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Supported | 23 | 9 | 14 | 8 | 12 | 3 |
| Unsupported | 39 | 10 | 29 | 10 | 25 | 4 |

Unsupported 39개는 자동으로 광범위한 class에 넣지 않았다. 주요 보류 사유:

- 복합 소스: shoe squeaks와 footfalls, racket hits와 grunts, engine과 splash 등은 여러 class의
  max 확률만으로 모든 구성 요소가 충족되었다고 할 수 없다.
- 동작 차이: 카메라 취급은 shutter, 팬 내려놓기는 팬 작동, 총 취급은 gunfire와 다르다.
- 음원 정체 차이: rabbit laughter와 dinosaur screams를 human voice에 바로 연결하지 않는다.
- 재료 차이: dirt/gravel spray는 liquid droplets를 뜻하는 Spray에 넣지 않는다.
- 불특정 ambience와 마법 효과: 조용한 야외나 generic game sound만으로 의도된 이벤트를 확인하기 어렵다.
- 계획 충돌: `Repeated hand-tool chopping impacts on wood`의 retained 계획은 타격 사이 scraping/rustling이다.
  Relabel·role 변경·merge를 먼저 검토한다.

Mapping 보강 전후 **mapping 필드를 제외한 62개 normalized stem 전체가 동일**한지 확인하였다.
Planned 19 / ambiguous 43, parent-only support, 선택 attempt와 원본 provenance는 유지된다.

## Metadata 연결 및 누락 처리 검증

아래 명령은 실제 WAV 없이도 metadata와 누락 처리를 재현한다. 두 번째 명령은
`missing_audio` 62개 때문에 **의도대로 종료 코드 1**을 반환하며 이후 evaluate는 별도로 실행한다.

```powershell
cd C:\multitrack-audio-filtering
.venv/Scripts/python.exe -m waves_sed adapt-waves --frozen-finals C:/WAVES/data/frozen_pass2/frozen_finals.json --frozen-reports C:/WAVES/data/frozen_pass2/frozen_reports.json --mappings configs/source_mappings.frozen-review.json --output outputs/real-data-readiness/frozen-review/stems.json
.venv/Scripts/python.exe -m waves_sed batch-infer --stems outputs/real-data-readiness/frozen-review/stems.json --output-dir outputs/real-data-readiness/frozen-review/batch
.venv/Scripts/python.exe -m waves_sed evaluate --stems outputs/real-data-readiness/frozen-review/stems.json --predictions outputs/real-data-readiness/frozen-review/batch/prediction-index.json --mappings configs/source_mappings.frozen-review.json --config configs/evaluation.example.json --filter-config configs/filter.example.json --output-dir outputs/real-data-readiness/frozen-review/reports
.venv/Scripts/python.exe -m pytest -q tests/test_mapping.py tests/test_ontology.py tests/test_phase3_cli.py
```

- Adapter: supported 23 / unsupported 39; 62개 모두 audio path 없음.
- Batch: `missing_audio` 62, model 초기화 0, 추론 0, 성공 prediction index 비어 있음.
- Evaluate: evaluation/detection unavailable 62, metric 기여 0, 평균 null.
- 기본 all-null 정책: unassigned 62, PASS/REVIEW/FAIL/UNSUPPORTED 판정 모두 0.
  Mapping의 unsupported 상태와 disabled 정책의 판정은 별개다.
- 관련 기존 테스트 **97 passed**. Config/문서만 변경하였으므로 모델 추론이나 전체 suite를 반복하지 않았다.

Git 제외 `outputs/real-data-readiness/frozen-review/`에 normalized `stems.json`, `batch/`, `reports/`가 있다.
추가 조사 산출물 `required-audio.csv`는 62개 selected audio ID, mapping/reference, 기록된 경로와
system role을 나란히 제공한다. `inventory.json`은 입력 SHA-256과 집계/불변성 검증 결과다.
이 두 조사 파일은 2026-09-29 대조 과정에서 별도로 내보냈으며 위 세 CLI 명령이 생성하는 파일은 아니다.
CSV의 경로는 파일을 찾기 위한 기록이며 `--audio-paths`용 JSON이나 확인된 음원 연결이 아니다.

## 실제 음원 평가와 threshold 보정 절차

1. 실제 실행 폴더를 확인하고 selected WAV를 metadata의 선택 attempt와 file/hash에 연결한다.
   Frozen 입력은 [Phase 3 형식](phase3-validation.md)의 명시적 `audio_id -> 실제 path` JSON을 만든다.
   Metadata·선택 attempt·경로를 확인한 항목부터 시작하고, 미확인 항목은 누락 상태로 유지한다.
2. 위 후보 mapping의 실제 음원을 청취/검토한다. Final role과 reference 상태를 검토하되,
   계획이 바뀐 43개를 편의상 planned로 바꾸지 않는다. 독립 시간 annotation이 있으면 출처와
   음원 식별 정보를 보존해 `ExternalVideoReference` API에 연결한다. 현재 CLI에는 외부 annotation
   파일 loader가 없으므로 자료의 실제 형식을 확인한 후 연결 범위를 정한다.
3. 새 출력 폴더에서 batch-infer → evaluate → visualize를 실행한다. Raw cache와 원본 SHA를
   보존하고 source/role별 검출·metric·기여 수를 확인한다. 운영 판정은 아직 all-null로 둔다.
4. 실제 이벤트 구간을 확인한 stem에서 controlled corruption을 수행한다. 삭제/복제가 실제
   missing/extra event를 만들었는지와 SED 반응을 구분한다. Synthetic demo 결과를 이 자료의 정답으로 쓰지 않는다.
5. Calibration 자료와 최종 확인 자료를 clip 단위로 나누는 계획을 먼저 고정한다. 같은 원본의
   stem·attempt·corruption을 양쪽에 섞지 않는다. 시간 검수와 품질 검수의 대상을 명시하고,
   그 판단에 맞는 source/role별 metric 분포를 조사한다. 기존 누음 평균만으로 시간 기준을 정하지 않는다.
6. Event extraction threshold/smoothing과 quality filter의 PASS/FAIL 경계를 별도 설정으로 기록한다.
   준비된 검수 자료에서 경계를 선정한 뒤 보류 자료에서 오수락·오거절·REVIEW 비율과 기여 수를 확인한다.
   표본이나 근거가 부족한 role은 null로 유지한다. 설정·mapping·모델·입력·분할의 hash와 검증 결과를
   함께 남겨야 운영 보정 결과로 부를 수 있다.

실제 평가의 선행 조건은 **생성된 stem WAV가 있는 실행 폴더와 선택 metadata의 연결**이다.
운영 보정에는 해당 음원에 대응하는 검수 결과/시간 주석과 판정 기준을 추가로 요구한다.
