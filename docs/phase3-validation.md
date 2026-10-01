# Phase 3: WAVES metadata와 명시적 AudioSet mapping

검증 기준일: **2026-09-27**.

Phase 3은 WAVES의 최종 stem과 생성 계획을 공통 metadata로 정규화하고, 최종 소리 설명을
ATST-F의 실제 출력 클래스에 연결하는 단계이다. 공식 AudioSet ontology를 읽어 클래스
관계를 확인하고, 명시적 수동 mapping으로 기존 raw 확률에서 목표 소리의 확률을 추출한다.

모델 추론은 다시 실행하지 않는다. 이 단계의 출력은 [Phase 4](phase4-validation.md)의
이벤트 추출·시간 지표 계산에 입력되며, 품질 판정은 [Phase 7](phase7-validation.md)의
별도 정책이 담당한다. 문서의 테스트 개수와 frozen 자료 집계는 위 기준일의 기록이다.

## 입출력과 코드 구성

입력은 materialized WAVES metadata 또는 frozen finals/reports이며, 필요하면 음원 경로
mapping과 source mapping을 함께 전달한다. 출력은 정규화된 stem JSON이다.
`map-source` 명령은 설명과 클래스의 연결 결과를 반환하며, 기존 NPZ를 전달하면 시간별
target 확률도 계산한다.

| 모듈 | 책임 |
| --- | --- |
| `metadata.py` | `StemMetadata`와 정규화 JSON 계약 |
| `adapters/waves.py` | final·candidate·선택 attempt 결합, 경로와 reference 상태 기록 |
| `ontology.py` | 공식 ontology 검증과 계층 탐색 |
| `mapping.py` | 명시적 description alias와 클래스 연결, target 확률 집계 |
| `phase3.py` | `adapt-waves`, `map-source` CLI |

NumPy와 설치된 패키지만으로 실행할 수 있다. Phase 2의 선택적 추론 의존성은 요구하지 않는다.

## 실행 방법

검증에 사용한 WAVES frozen 자료 전체를 정규화하는 명령은 다음과 같다.

```powershell
cd C:\multitrack-audio-filtering
.venv/Scripts/python.exe -m waves_sed adapt-waves --frozen-finals C:/WAVES/data/frozen_pass2/frozen_finals.json --frozen-reports C:/WAVES/data/frozen_pass2/frozen_reports.json --mappings configs/source_mappings.example.json --output outputs/phase3/frozen-stems.json
```

실제 WAVES 실행 결과가 준비된 경우에는 다음 형식으로 사용한다.
아래 경로는 실행 형식 예시이며 현재 checkout에 있는 파일을 뜻하지 않는다.

```powershell
.venv/Scripts/python.exe -m waves_sed adapt-waves --metadata C:/runs/demo/final/clip_01/metadata.json --sam-manifest C:/runs/demo/sam_manifest.json --dsp-report C:/runs/demo/dsp_reports/clip_01.json --mappings configs/source_mappings.example.json --output outputs/phase3/clip_01.json
```

SAM manifest나 DSP report 중 하나만 전달해도 된다. 둘 다 전달하면 공유하는 계획 필드의
일치 여부를 검사하며, 충돌 시 오류를 반환한다. 둘 다 생략하면 최종 metadata에 실제로 없는
description/support 등의 정보는 누락 상태로 남긴다. 이웃 파일을 자동 탐색하지 않는다.
`--mappings`를 생략하면 정규화만 한다. `--output`을 생략하면 JSON을 stdout에 출력한다.

Frozen 자료의 실제 음원 파일을 연결하려면 `--audio-paths paths.json`을 추가한다.
이 파일은 다음과 같은 **명시적 audio_id → path 객체**다. 예제 경로는 실제 파일 위치로 대체한다.

```json
{
  "fb5k_1229/01/a1": "audio/selected-weapon-stem.wav"
}
```

상대 경로는 `paths.json`이 있는 디렉터리를 기준으로 해석한다.
등록되지 않은 ID는 `missing_audio_path`, 명시한 파일이 없으면 `audio_file_not_found`로 남긴다.
`audio_id`에서 파일명을 만들거나 과거 Linux 평가 경로를 현재 로컬 경로로 추정하지 않는다.

## 정규화 계약과 시간 기준

정규화 결과의 주요 필드는 다음과 같다.

| 필드 | 의미 |
| --- | --- |
| `stem_id` | `<clip_key>::<candidate_id>`의 안정적인 식별자. Windows 파일명으로 직접 사용하지 않음 |
| `audio_path` | materialized `stems[].file`을 metadata 위치에 대해 해석하거나 명시적 frozen path mapping 사용 |
| `source_description` | 최종 `final_label`. `description_origin=final_label`을 명시 |
| `role` | 최종 `final_role` |
| `expected_intervals` | 선택된 parent candidate의 `activity_intervals`. 누락은 null, 명시적인 빈 목록은 [] |
| `planned_description`, `planned_label`, `planned_role` | Pass 1 원래 계획을 별도로 보존 |
| `optional_video_id` | 명시된 video_id만 사용. clip key나 영상 파일명으로 추정하지 않음 |
| `merged_candidate_ids`, `issues` | merge 이력, 누락·relabel·role 변경·기록 불일치 |
| `provenance` | 입력 경로/hash, 원본 final/parent/merged-child 자료, 선택 attempt, reference 상태 |

최종 WAVES JSON에는 description과 support가 없으므로 candidate ID로 중간 metadata를 조인한다.
Source mapping에는 **최종 label**을 사용한다. relabel 이전 description을 대신 매핑하지 않는다.
최종 label 자체가 없으면 source_description도 null이며 이전 계획을 대입하지 않는다.

시간 기준은 `reference_origin=waves_pass1_planned`, `support_policy=selected_parent_only`다.
Semantic merge는 대표 WAV 하나를 선택하므로 child들의 구간을 합집합으로 만들지 않는다.
Parent/child의 원래 구간은 provenance에 남아 있다.

`provenance.reference_status`:

- `planned`: 계획 구간이 있고 현재 기록에서 역할/의미 변경 등의 모호함이 발견되지 않음.
- `ambiguous`: relabel, role 변경, merge, 필요한 식별 정보 누락 또는 decision 불일치 등.
- `missing`: 사용할 계획 구간이 없거나 명시적인 목록이 비어 있음.

이 상태는 품질 PASS/REVIEW/FAIL이 아니다. `planned`도 실제 영상의 정답 시간이 아니다.
Phase 4는 기본적으로 `ambiguous`인 기준을 시간 지표 계산에 사용하지 않는다. 명시적인
설정으로 사용을 허용할 수 있으나, Phase 7에서 해당 역할의 판정 기준을 활성화하면
reference의 모호함을 근거로 `REVIEW`를 반환한다. 누락된 구간을 전체 clip으로 채우지 않는다.
실제 영상과의 동기화 검증에는
독립적인 외부 시간 기준이 필요하다.

입력의 duplicate ID/key, cross-clip 조인, 충돌하는 계획, 비유한/음수/역전 구간과 잘못된
선택 attempt는 오류로 처리한다. 정렬되지 않은 구간을 자동 정렬하거나 병합하지 않는다.
명시적 JSON 입력, 최종 음원, 실제 pipeline이 기록한 절대 원본 음원/영상 경로를 출력으로
덮어쓰지 않도록 검사한다. 상대 final WAV와 frozen mapping 경로는 위의 문서화된 기준으로 해석한다.

## ontology와 모델 vocabulary의 실제 차이

`AudioSetOntology`는 [공식 AudioSet ontology](https://github.com/audioset/ontology)의
revision `d417d32bf59c711abb5910fd2f76a0eb44697991` 원본 JSON을 읽는다.
632개 노드의 DAG를 검증하고 multiple parents, ancestor/descendant 탐색을 지원한다.
ID/name 중복, 없는 child, cycle을 거부한다. 기본 resource는 SHA-256도 확인한다.
라이선스는 데이터에 적용되는 CC BY-SA 4.0이며 attribution/provenance를 함께 배포한다.

공식 archive와 PretrainedSED의 실제 447-class 출력 목록을 비교한 결과는 다음과 같다.

| 항목 | 개수 |
| --- | ---: |
| 공식 ontology 노드 | 632 |
| ATST-F classifier output ID | 447 |
| 양쪽에 같은 ID가 있음 | 416 |
| 모델에는 있으나 ontology에는 없음 | 31 |
| 같은 ID의 표시명이 다름 | 11 |

예를 들어 모델의 `Knife` (`/m/04ctx`)와 `Pant (dog)` (`/t/dd00141`)는 해당 공식 archive에 없다.
차이를 숨기기 위해 데이터나 hierarchy를 수정하지 않는다.
`vocabulary_coverage`에는 누락 ID 전체와 이름 차이 목록이 들어 있다.
Mapping 결과의 `ontology_name`과 `model_name`도 구분한다.

API 예:

```python
from waves_sed.labels import load_labels
from waves_sed.ontology import AudioSetOntology

ontology = AudioSetOntology.load()  # 명시적 custom JSON path도 가능
coverage = ontology.vocabulary_coverage(*load_labels())
dog = ontology.resolve_name("Dog")  # 정확한 표시명만 허용
family = ontology.descendants(dog)  # 탐색일 뿐, source를 자동 매핑하지 않음
```

Custom ontology는 자체 경로/hash로 기록하며 공식 데이터의 revision/license로 표시하지 않는다.

## 수동 mapping 계약

기본 예제는 `configs/source_mappings.example.json`이다. 이 설정은 완전한 WAVES caption
사전이나 검증된 품질 규칙이 아니다. 후속 검토에서 확장한 후보 설정은
`configs/source_mappings.frozen-review.json`이며, 적용 범위와 검토 근거는
[실데이터 준비 현황](real-data-readiness.md)에 있다. 두 설정의 coverage를 구분한다.
Mapping에는 다음과 같이 정확한 description alias와 실제 class ID를 등록한다.

```json
{
  "schema_version": 1,
  "mappings": {
    "dog_barking": {
      "descriptions": ["dog barking", "barking dog"],
      "allowed_classes": ["/m/05tny_"],
      "foreign_classes": ["/m/09x0r"],
      "include_descendants": false
    }
  }
}
```

위 ID는 실제 metadata의 Bark와 Speech다. 대소문자와 연속 공백만 정규화한다.
Substring/fuzzy/embedding/LLM mapping은 하지 않는다. 겹치는 alias, 알 수 없는 ID/설정 필드,
중복 JSON key, allowed와 명시적 foreign의 충돌을 거부한다.

- 기본적으로 `allowed_classes`에 명시한 ID만 사용한다.
- `include_descendants=true`일 때만 공식 ontology의 자식들을 확장한다.
- 모델 metadata에만 존재하는 ID도 명시적 직접 mapping은 가능하다.
  단, 그 ID의 descendant 확장은 오류다. 없는 hierarchy를 추정하지 않는다.
- 모든 allowed ID 중 현재 prediction에 실제 존재하는 ID가 `target_class_ids`다.
  없는 ID는 `unavailable_class_ids`로 보고한다. 일부만 존재하면 존재하는 ID로 집계하고 누락 목록을 유지한다.
- `ontology_missing_class_ids`는 rule의 allowed/foreign ID 중 공식 hierarchy에 없는 ID다.
- 명시적 rule이 없거나 모델에서 사용 가능한 target ID가 하나도 없으면 `unsupported_mapping`이다.
- `foreign_classes`는 명시적 목록을 보존하는 용도이며 이 단계에서 판정하지 않는다.

`supported`는 source 설명과 모델 column을 연결할 수 있다는 뜻이다.
그 소리가 실제로 검출되었다거나 stem 품질이 좋다는 뜻이 아니다.

```powershell
.venv/Scripts/python.exe -m waves_sed map-source --description "dog barking" --mappings configs/source_mappings.example.json

# 기존 raw cache를 사용한 두 class의 max 계산 예시. 이 음원에 chopping이 있다는 의미는 아니다.
.venv/Scripts/python.exe -m waves_sed map-source --description "repeated chopping impacts" --mappings configs/source_mappings.example.json --prediction outputs/predictions/metro.npz --output outputs/phase3/metro-chop-mapping.json
```

Class ID로 column을 찾아 `max(P(c,t))`를 계산하므로 cache의 column 순서가 달라도 맞게 연결한다.
Threshold나 smoothing을 적용하지 않고 원래 시간 bin을 유지한다.
Mapping을 지원하지 않거나 raw cache가 없으면 `target_probability=null`이다.
지원되지 않는 source를 zero 확률이나 missing event로 바꾸지 않는다.

## 2026-09-27 검증 결과

`C:\WAVES` HEAD `07af161`의 frozen finals/reports 전체를 기본 예제 mapping으로
정규화하여 확인하였다.

| 결과 | 개수 |
| --- | ---: |
| Clip / final stem | 29 / 62 |
| `missing_audio_path` / `missing_video_id` | 62 / 62 |
| `relabelled` | 41 |
| `role_changed` | 2 |
| `merged_support_parent_only` | 10 |
| reference `ambiguous` / `planned` | 43 / 19 |
| 예제 mapping `supported` / `unsupported_mapping` | 2 / 60 |

검증 시점의 WAVES checkout에는 WAV가 없으므로 음원 경로는 모두 누락 상태로 나타났다.
실제 materialized schema와 상대 경로 조인은 test fixture로 검증하였다.
위의 `2 / 60`은 기본 예제 설정의 mapping coverage이며, 후속 후보 설정의 coverage와
구분한다.

Phase 2의 실제 metro NPZ 938개 frame에서 두 chopping-class column을 선택하여 집계한 결과는
NumPy의 직접 max 계산과 정확히 일치하였다. 이 검증에서 SED 추론은 실행하지 않았다.

테스트는 ontology graph/coverage, 명시적 mapping/unsupported, ID 순서별 max,
materialized/frozen 조인, merge/role 변경, 누락/충돌, malformed JSON, 출력 보호와
추론 라이브러리 import 없는 CLI를 포함한다. 검증 명령:

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/ruff.exe check src tests scripts
.venv/Scripts/ruff.exe format --check src tests scripts
uv build --wheel
```

당시 전체 테스트 결과는 **226 passed, 1 skipped**, 실행 시간은 9.83초로 나타났다.
실제 checkpoint 테스트 1개는 환경변수 미설정으로 skip되었으며, 기존 audioread
deprecation 경고 3개가 발생하였다. Phase 3은 inference 경로를 변경하지 않았으므로
Phase 2의 실제 모델 검증을 반복하지 않았다.

Ruff lint/format과 wheel build를 통과하였다. 별도의 깨끗한 환경에 wheel과 NumPy만
설치하여 ontology 632개·model vocabulary 447개를 읽고 명시적 mapping이 동작함을
확인하였다. 해당 환경에는 torch가 없었으며, 추론 의존성 없이 Phase 3 기능을 사용할 수
있음을 배포 패키지에서도 검증하였다.

## 검증 범위와 한계

검증한 항목은 metadata 조인의 일관성, 공식 ontology와 모델 vocabulary의 관계,
mapping 규칙 및 확률 column 집계이다. 실제 62개 stem의 음향 품질이나 영상 동기화를
평가한 결과는 아니다.

`supported` mapping은 평가할 클래스 column을 지정할 수 있음을 의미한다. 음원에 목표
소리가 있는지, 해당 클래스가 생성 음원에서 충분한 인식 정확도를 보이는지는 실제 음원과
독립적인 검수 자료로 확인해야 한다. 실제 파일 연결과 다음 평가 절차는
[실데이터 준비 현황](real-data-readiness.md)에 정리되어 있다.
