# Phase 7: 명시적인 설정에 따른 판정

Phase 7은 기존 temporal metric 위에 선택적인 PASS / REVIEW / FAIL / UNSUPPORTED 정책을
추가한다. 추론·확률·이벤트 추출·metric은 변경하지 않는다. `evaluate`와 `visualize`에
`--filter-config`를 전달할 때만 적용한다. 설정을 생략하면 이전 보고서 구조와 `decision=null`을
유지한다. 기본 예제의 모든 threshold도 null이다.

PASS는 명시한 temporal 조건을 충족한다는 뜻이다. WAVES planned support를 기준으로 한
판정은 내부 시간 일관성에 관한 것이며 영상 동기화나 전체 청각적 품질을 입증하지 않는다.
Outside-family 확률로 자동 FAIL을 만들지 않는다. WAV 이동·삭제 같은 후속 동작도 하지 않는다.

## 실행

새 dependency는 없다. 평가와 정책 적용은 기본 NumPy 환경에서 실행하며, 그림에는 기존
`[visualization]` 선택 의존성이 필요하다.

```powershell
# filter.example.json의 값은 모두 null이다. 필요한 기준을 별도 파일에 명시한다.
.venv/Scripts/python.exe -m waves_sed evaluate --stems outputs/stems.json --predictions outputs/batch/prediction-index.json --mappings configs/source_mappings.example.json --config configs/evaluation.example.json --filter-config configs/filter.example.json --output-dir outputs/filtered-reports
.venv/Scripts/python.exe -m waves_sed visualize --stems outputs/stems.json --predictions outputs/batch/prediction-index.json --mappings configs/source_mappings.example.json --config configs/evaluation.example.json --filter-config configs/filter.example.json --output-dir outputs/filtered-visuals
```

두 명령에 같은 평가/판정 설정을 전달한다. `--config`는 이벤트 추출과 metric 설정이고,
`--filter-config`는 계산된 metric의 판정 설정이다. 정책만 바꾸면 새 추론이 필요 없다.
`evaluate`는 FAIL/REVIEW/UNSUPPORTED도 정상 보고하며 성공 시 종료 코드 0이다.
잘못된 설정이나 파일 충돌은 1이다. 그림 생성 실패 정책은 Phase 5와 같다.

## 설정 schema와 경계값

```json
{
  "schema_version": 1,
  "filter": {
    "onset": {
      "min_event_recall": null,
      "max_mean_onset_error_ms": null
    },
    "span": {"min_tiou": null},
    "ambience": {"min_occupancy": null}
  }
}
```

역할이나 규칙을 생략해도 null이다. Boolean, 문자열 threshold, NaN/Infinity, 중복 JSON key,
알 수 없는 역할/규칙, 잘못된 band 순서와 범위를 거부한다. 정확히 읽은 설정 bytes의 SHA-256과
절대 경로를 보존하고, 모든 활성 숫자는 float의 pass/fail band로 정규화한다.

| 규칙 | 사용 metric | 단위와 범위 | 좋은 방향 |
| --- | --- | --- | --- |
| `onset.min_event_recall` | `event_recall` | 0~1 | 클수록 좋음 |
| `onset.max_mean_onset_error_ms` | `mean_onset_error_ms` | 0 이상, ms | 작을수록 좋음 |
| `span.min_tiou` | `temporal_iou` | 0~1 | 클수록 좋음 |
| `ambience.min_occupancy` | `occupancy_in_expected_span` | 0~1 | 클수록 좋음 |

값은 null, 숫자, 또는 `{"pass": 숫자, "fail": 숫자}`이다. 숫자는 pass와 fail에 같은 값을 둔
단일 경계다. 예를 들어 `min_event_recall: 0.9`는 0.9 이상이면 PASS, 미만이면 FAIL이다.
이 값은 사용자가 명시한 조건이며 자동으로 보정된 기준이 아니다.

| 종류 | PASS | REVIEW | FAIL | 유효 순서 |
| --- | --- | --- | --- | --- |
| min | 값 ≥ pass | fail ≤ 값 < pass | 값 < fail | fail ≤ pass |
| max | 값 ≤ pass | pass < 값 ≤ fail | 값 > fail | pass ≤ fail |

예를 들어 `max_mean_onset_error_ms: {"pass":100,"fail":300}`이면 100ms는 PASS,
300ms는 REVIEW, 300ms 초과는 FAIL이다. 숫자는 동작 설명용이며 권장 운영 threshold가 아니다.
별도 epsilon을 넣지 않고 계산된 metric과 명시한 경계를 그대로 비교한다.

## 판정 순서

1. 모든 규칙이 null이면 `decision=null`, audit 상태는 `disabled`다.
2. 활성 정책이 있지만 현재 역할의 규칙이 없으면 null/`not_configured`다.
   역할 자체를 지원하지 않으면 활성 정책에서 UNSUPPORTED다.
3. 현재 역할에 적용할 규칙이 있고 mapping이 미지원이면 UNSUPPORTED다.
4. Reference가 누락·사용 불가·모호하거나, 평가/검출 불가 또는 cache identity 미검증이면
   REVIEW다. 숫자 기준을 적용하지 않고 각 조건을 `blocked`로 남긴다.
5. 신뢰할 수 있는 평가 결과에는 활성 규칙만 적용한다. 필요한 metric이 null/누락/잘못된
   숫자이면 그 조건은 `unavailable`/REVIEW다. 나머지 조건의 유효한 FAIL이 있으면 최종 FAIL,
   그렇지 않고 REVIEW가 있으면 REVIEW, 모든 활성 조건이 PASS이면 PASS다.

Ambiguous WAVES reference를 `allow_ambiguous_reference=true`로 metric 계산에 사용했더라도
판정은 REVIEW다. Missing reference를 무음이나 정답 구간으로 바꾸지 않는다. External
reference의 명시적인 origin/status vocabulary는 유지한다.

검출이 없어 recall=0, 평균 onset 오차=null인 경우 오차 조건만 활성화하면 REVIEW다.
Recall 조건도 활성화하여 0이 기준에 미달하면 그 유효한 조건 때문에 FAIL이다. Null 오차를
0ms로 바꾸어 PASS시키지 않는다. 정책이 비활성이면 미지원 source도 판정은 null이며
기존 `mapping_status`와 `reason_codes`에서 미지원 상태를 확인한다.

## 보고서와 시각화

Stem의 `decision`은 문자열 또는 null이다. 설정을 전달한 경우에만 `decision_details`가 생긴다.
여기에는 status/reason_codes, 각 활성 규칙의 metric·관측 값·min/max 방향·pass/fail 경계·상태·결과,
정규화된 config, 원본 설정의 config_provenance(path/hash), implementation_version(현재 1)이 있다.
`provenance.filter_config`와 `filter_config_provenance`에도 같은 정책을 연결한다.

Metric provenance의 기존 `decision=none` convention은 **metric 계층 자체**가 판정하지
않는다는 의미로 유지했다. 최종 optional 판정은 stem의 decision/audit에 있다.

CSV에는 `decision_status`, `decision_reason_codes`, `decision_checks`, `decision_config`,
`decision_config_provenance`, `decision_implementation_version`을 추가한다. 구조화된 값은
JSON 셀이며 null 판정은 빈 셀이다. Clip/dataset/역할별 `decision_summary`는 네 label과
`unassigned` 개수 및 사유별 기여 stem 수를 제공한다. 집계 자체에는 PASS/FAIL을 부여하지 않는다.

집계와 그림은 정책으로 audit을 다시 계산하여 임의 label, 변경된 조건/metric/audit을 거부한다.
Label이 있는 결과를 서로 다른 정책 또는 기존 무설정 결과와 섞어 집계하지 않는다. 같은
정규화 정책의 다른 파일 경로는 허용한다. 모두 미판정인 혼합 결과는 policy context가 null이다.
기록된 audit 검증은 과거 정책 파일을 다시 열지 않으며, 현재 WAV 검증에는 `evaluate`를 다시 실행한다.

PNG/SVG와 HTML에는 판정, 사유, 모든 적용 조건을 표시한다. 그림 API에 정책 적용 보고서를
넘기면 동일한 `filter_config`도 제공해야 한다. 설정 input과 hardlink를 출력에서 보호하며,
CLI 처리 중 설정 파일이 바뀌면 보고서 또는 gallery/index 게시를 거부한다. 그림 일부가 먼저
생성될 수 있으므로 전체 출력을 단일 transaction으로 가정하지 않는다.

```python
from waves_sed.decision import FilterConfig, decide
from waves_sed.evaluation import evaluate_stem

policy = FilterConfig.load("my-filter.json")
report = evaluate_stem(stem, prediction, mapping, evaluation_config, filter_config=policy)
print(report["decision"], report["decision_details"]["checks"])
audit = decide(existing_report, policy)  # 보관한 report에 정책 자체만 계산
```

`decide`는 보고서를 변경하거나 파일/model을 읽지 않고 보고서의 evidence status와 metric을
사용한다. Waveform/cache를 현재 상태로 재검증하는 API는 아니다. `FilterConfig()`는 모두
비활성, `.from_dict()`는 파일 경로 없는 programmatic 정책이다.

## 검증 (2026-09-28)

- 신규 core 103개, CLI 21개, reporting 26개, visualization 27개: **177개 통과**.
- 최종 전체 **1019 passed, 1 skipped**, 47.32초. Skip은 환경변수가 없는 선택적 실제 모델 test다.
  기존 audioread deprecation warning 3개 외 오류 없음. 첫 전체 실행의 기존 시각화 subprocess
  test가 timeout했으나 같은 test 단독 실행은 1.44초에 통과했고, 이후 전체 재실행도 통과했다.
- 정확한 min/max 경계, 복수 조건, missing/invalid metric, ambiguous opt-in, 미지원, strict JSON,
  provenance, 변경된 audit 거부, 혼합 정책, CSV null/기여 수, 원본·hardlink·설정 변경 보호,
  cache 불변과 optional dependency 격리를 검증했다.

WAVES frozen 29 clips / 62 stems를 빈 prediction index로 평가했다. 검증용으로 모든 역할의
규칙을 활성화했을 때 REVIEW 2, UNSUPPORTED 60, PASS/FAIL 0이다. 기본 all-null 정책에서는
unassigned 62다. 각각 `outputs/phase7/frozen-reports`와 `frozen-disabled-reports`에 있다.
이는 결측 자료 처리 검증이며 실제 WAVES 음질 평가가 아니다.

실제 공개 음원 cache의 정책 적용 검증은 `scripts/verify_filter_policy.py`로 재현한다.
Phase 6에서 만든 명시적인 synthetic full-track demo만 허용하며, metadata/reference/cache
연결을 검사하고 metric·검출 구간 불변과 입력 hash 불변을 확인한다. 새 추론은 하지 않는다.

```powershell
.venv/Scripts/python.exe scripts/verify_filter_policy.py --demo-dir outputs/phase6/metro-demo --filter-config outputs/phase7/demonstration-filter.json --output-dir outputs/phase7/metro-policy-demo --plots
```

다시 실행할 때는 새 output-dir을 사용한다. 검증용 `demonstration-filter.json`은 onset recall0.9,
평균 오차50ms, span `{"pass":0.79,"fail":0.78}`, ambience occupancy0.8을 명시했다.
이 값은 여러 판정 경로를 보여주기 위한 예시이며 calibration 결과나 운영 권장값이 아니다.

실행 결과는 **PASS 5 / REVIEW 2 / FAIL 2**였다. 500ms shift와 shorten은 REVIEW,
1000ms shift와 remove는 FAIL이며 원본을 포함한 나머지 5개는 PASS다. 기존 9개 cache와
metric·검출 구간은 모두 그대로였고, 9개 PNG와 HTML을 생성했다. REVIEW 그림을 직접 열어
조건·관측값·경계와 파형이 읽기 좋게 배치되었음을 확인했다.

`outputs/phase7/metro-policy-demo/reports/dataset.json`과 `visuals/index.html`에서 확인한다.
별도 `.cache/phase7-package-smoke` 환경에 wheel+NumPy만 설치하고 같은 스크립트를 `--plots`
없이 실행했다. torch/torchaudio/librosa/soundfile/Matplotlib이 없으며 9개 stem 보고서와 summary가
완전히 일치했다. 결과는 `outputs/phase7/wheel-policy-demo/reports/`에 있다.
Ruff lint/format, `git diff --check`, wheel build도 통과했다.

## 남은 범위

Phase 1~7의 구현 순서는 완료했다. 남은 실험 과제는 실제 WAVES WAV 확보·경로 연결,
source별 명시적 mapping 확장, 신뢰 가능한 reference와 validation 자료에 기반한 threshold
calibration이다. Human evaluation platform, 학습, 영상 event detector, 파일 자동 삭제/이동은
이번 구현에 포함하지 않는다.
