# Phase 6: controlled corruption 평가

Phase 6는 원본 stem에서 독립적인 WAV 변형을 만들고, 원본 대조군과 변형의 평가 결과를
비교한다. `corrupt`가 음원을 만들며 기존 `batch-infer` → `evaluate` → `visualize`를
재사용한다. `compare-corruptions`는 저장된 보고서만 읽어 변화량을 JSON/CSV로 내보낸다.
원본 WAV, WAVES 코드, raw cache를 수정하지 않는다. 품질 판정은 계속 `decision=null`이다.

## 실행

WAV 생성에만 선택 의존성 soundfile 0.13.1이 필요하다. 비교는 기본 NumPy 환경에서
실행할 수 있다. 새 추론은 기존 `[atst]`, 그림은 `[visualization]` 설치가 필요하다.

```powershell
uv pip install --python .venv/Scripts/python.exe -e ".[corruption]"

# 실제 WAV 경로가 있는 정규화 manifest와 exact stem ID를 지정한다.
.venv/Scripts/python.exe -m waves_sed corrupt --stems outputs/stems.json --stem-id "실제 stem ID" --config configs/corruption.example.json --output-dir outputs/corruption/experiment
.venv/Scripts/python.exe -m waves_sed batch-infer --stems outputs/corruption/experiment/stems.json --output-dir outputs/corruption/batch
.venv/Scripts/python.exe -m waves_sed evaluate --stems outputs/corruption/experiment/stems.json --predictions outputs/corruption/batch/prediction-index.json --mappings configs/source_mappings.example.json --config configs/evaluation.example.json --output-dir outputs/corruption/reports
.venv/Scripts/python.exe -m waves_sed compare-corruptions --experiment outputs/corruption/experiment/experiment.json --report outputs/corruption/reports/dataset.json --output-dir outputs/corruption/comparison
.venv/Scripts/python.exe -m waves_sed visualize --stems outputs/corruption/experiment/stems.json --predictions outputs/corruption/batch/prediction-index.json --mappings configs/source_mappings.example.json --config configs/evaluation.example.json --output-dir outputs/corruption/visuals
```

Mapping은 해당 stem 설명에 맞는 명시적인 설정을 사용한다. 예제 mapping이 모든 source를
지원하는 것은 아니다. WAVES frozen 자료에 WAV가 없으면 먼저 실제 경로를 연결해야 한다.
새 변형은 실제 WAV bytes가 달라지므로 새 추론이 필요하다. 기존 확률의 시간축만 이동시켜
실제 모델의 반응으로 보고하지 않는다.

`corrupt`는 기존 실험 출력 파일을 덮어쓰지 않는다. 설정을 바꾸거나 중단 후 다시 생성할
때는 새 출력 디렉터리를 선택한다. 개별 구간이 clip 밖인 경우 그 변형을 `failed`로 기록하고
나머지는 계속 만든다. 하나라도 실패하면 종료 코드 1이며, 대조군과 성공한 변형은 남는다.
잘못된 설정, 없는 원본, source hash 불일치, 출력 충돌은 쓰기 전에 거부한다.
`compare-corruptions`는 비교 불가 항목도 정상 보고하므로 출력에 성공하면 종료 코드 0이다.
잘못된 schema/중복 JSON key/입력 충돌은 종료 코드 1이다.

## 설정과 파형 처리

설정은 `{"schema_version":1,"variants":[...]}`이다. Variant에는 고유한 `id`와
`operation`이 필요하다. `control` ID는 예약되어 있다. 사용하지 않는 필드, 중복 ID/key,
boolean 시간, NaN/Infinity와 알 수 없는 operation은 거부한다. 모든 시간은 초 단위다.

| Operation | 필수 추가 필드 | 처리 |
| --- | --- | --- |
| `shift` | `shift_seconds` | 선택 구간을 지우고 지정 시간만큼 이동하여 더함. `interval` 생략 시 전체 clip |
| `remove` | `interval: [start,end]` | 해당 구간의 모든 채널을 0으로 만듦 |
| `duplicate` | `interval`, `destination_seconds` | 원본 구간을 유지하고 목적지에 복제하여 더함 |
| `shorten` | `interval`, `duration_seconds` | 선택 구간의 앞부분을 지정 길이만 남기고 꼬리를 0으로 만듦 |
| `extend` | `interval`, `duration_seconds` | 선택 구간을 반복하여 지정 길이로 만들고 기존 구간을 대체·연장함 |

`interval`은 원본 clip 안의 반열린 구간이어야 한다. `shorten` 길이는 기존보다 짧고,
`extend`는 길어야 한다. 음수 shift도 지원한다. 음수 duplicate 목적지는 허용하지 않는다.
제공된 예제는 +100/200/500/1000ms 전체 이동과 0.5~1.0초 구간 제거/복제/길이 변경이다.
예제 구간은 실제 event annotation이 아니므로 대상 음원에 맞게 선택해야 한다.

정책은 다음과 같다.

- 원본 sample rate, 전체 frame 수, channel 수를 유지한다. 채널 평균이나 리샘플링을 하지 않는다.
- 각 변형은 원본에서 독립적으로 생성한다. 이전 변형 위에 다음 변형을 누적하지 않는다.
- `Decimal(str(seconds)) * sample_rate`를 가장 가까운 sample로 반올림한다. 정확히 절반이면
  0에서 멀어지는 방향이다. 실제 sample 경계와 구현된 shift/duration을 기록한다.
- Clip 밖으로 나간 삽입 부분은 버리고 빈 부분은 0이다. 잘린 frame 수는
  `cropped_source_sample_frames`이며, 무음 및 extend의 반복 부분도 포함한다.
- 삽입은 덧셈이다. 자동 normalization/clipping을 하지 않고 겹친 frame 수와 전후 peak를 남긴다.
  `extend`는 반복이며 time stretching이 아니다. `shorten`도 뒤쪽 음원을 당기지 않는다.
- 파형을 float32로 읽고 WAV `FLOAT`로 저장한다. `audio_changed`는 디코딩된 float32 sample의
  동일 여부다. 원본 대조군은 재인코딩하지 않고 원래 파일 경로와 bytes를 그대로 사용한다.
- Hard cut을 사용하며 fade를 넣지 않는다. 경계 artifact, 겹침에 따른 음량 변화, clip 끝 손실은
  모델 반응에 영향을 줄 수 있다. 중첩 복제가 분리된 extra event를 보장하지 않는다.

soundfile은 배열 dtype과 출력 파일 subtype을 별도로 다루므로 `format="WAV"`,
`subtype="FLOAT"`를 명시한다. API 계약은
[soundfile read/write 문서](https://python-soundfile.readthedocs.io/en/latest/#soundfile.write)를 따른다.
메모리 사용량은 원본 clip과 현재 변형의 길이에 비례한다. 매우 큰 연장 요청도 실제 clip 안의
부분만 할당하지만, 긴 원본 자체를 스트리밍 편집하는 기능은 없다.

## 산출물과 provenance

- `stems.json`: 원본 대조군과 성공한 변형만 포함하는 기존 schema 1 정규화 manifest.
- `experiment.json`: 원본 metadata/hash, 설정 경로/hash, 대조군, 변형별 spec/application,
  실패 사유, 생성 수, normalized manifest의 경로/hash.
- `audio/<SHA256(stem_id)>.wav`: 성공한 변형. 사용자 ID를 직접 파일명으로 쓰지 않는다.
- 대조군 ID는 `<원본>::control`, 변형 ID는 `<원본>::corruption::<variant id>`이다.

Description, role, expected intervals, reference origin/status와 ambiguity는 원본 그대로
보존한다. Expected 구간을 변형된 오디오에 맞춰 이동시키지 않는다. 각 derived stem의
`provenance.corruption.source_stem`에는 원래 metadata를 보존한다. 변형의 현재
`raw_final_stem.file/sha256`은 새 WAV를 가리키며 원래 raw provenance도 nested source에 남는다.

선택하지 않은 입력 stem, 입력 manifest/config, 원본 media, 보고서에 기록된 cache·source·report
경로와 hardlink alias까지 출력 preflight 대상이다. 파일은 각각 임시 파일에 쓴 뒤 atomic replace한다.
`experiment.json`을 마지막에 쓴다. 전체 실험이 단일 transaction인 것은 아니므로 중단 시
부분 WAV가 남을 수 있으며 새 디렉터리에서 재생성해야 한다. 생성 중 원본/설정이 바뀌면 manifest를
게시하지 않는다. 비교도 읽는 도중 입력 JSON이 바뀌면 쓰기 전에 거부한다.

## 비교와 해석

`comparison.json`과 `comparison.csv`는 각 변형을 원본 대조군에 연결한다. Report의 stem
metadata와 audio hash를 실험 manifest에 연결하고, 대조군이 원본 audio identity를 유지하는지
확인한다. Mapping, class vocabulary, 모델 provenance, metric/config, reference가 다른
두 보고서는 비교하지 않는다. 이 명령은 보고서에 기록된 증거를 비교하며 현재 WAV/NPZ 파일을
다시 열지 않는다. 현재 파일 검증이 필요하면 `evaluate`를 다시 실행한다.

| 결과 | 의미 |
| --- | --- |
| `metric_deltas` | 역할별 각 metric의 변형 값 − 대조군 값. 어느 한쪽이 null이면 null |
| `detection_count_delta` | 검출 구간 개수 차이 |
| `detected_support_iou` | 대조군·변형 검출 구간 합집합끼리의 IoU. 둘 다 비면 null |
| `single_event_onset_displacement_ms` | 양쪽 검출이 각각 하나일 때 시작점 차이, signed ms |
| `single_event_shift_error_ms` | 위 관측 변화량 − 실제 구현된 waveform shift, shift 조작에서만 계산 |

검출이 하나씩이라는 이유만으로 두 event의 identity를 보장하지 않는다. Missing/extra event
민감도는 onset 역할의 `missing_event_count`, `extra_event_count`, recall/precision 변화로
확인한다. Span/ambience는 기존 역할별 metric을 그대로 사용한다.

누락/평가 불가/다른 hash·설정·reference/변형 생성 실패/실제 파형 변화 없음은
`comparison_status=unavailable`과 reason code로 남는다. 0 변화나 통과로 대체하지 않는다.
요약은 비교 가능한 항목만 사용하고 metric별 유효 기여 수를 제공한다. `by_operation`은
조작 종류별 요약이며, 정해진 검출 정확도나 민감도 성공률을 주장하지 않는다.

## 검증 결과 (2026-09-28)

- Phase 6 테스트 **195 passed**: 파형 core 108, paired comparison 38, CLI 49.
- 전체 **842 passed, 1 skipped**, 69.67초. Skip은 환경변수를 지정하지 않은 선택적 실제
  checkpoint 테스트이고, 실제 모델은 아래 별도 실행으로 검증했다. 기존 audioread deprecation
  warning 3개 외 실패 없음. Ruff lint/format, `git diff --check`, wheel build 통과.
- Synthetic waveform에서 샘플 정확도·음수 이동·crop·overlap·비정상 설정·no-effect·원본 보존 검증.
  Synthetic score oracle에서 100/200/500/1000ms 이동, missing/extra, shorten/extend와
  metric/분모/null 동작을 검증했다. 이 테스트는 ATST 인식 성능 테스트가 아니다.
- 독립 wheel+NumPy 환경에서 torch/torchaudio/librosa/soundfile/Matplotlib 없이 실제 보고서
  8쌍을 재비교하여 결과가 일치했다. 이후 `[corruption]`만 설치하여 변형 WAV 8개 생성 성공.
  재생성한 8개 WAV의 디코딩된 sample/application도 모두 일치했다. WAV의 PEAK chunk 일부 bytes는
  달랐으므로 같은 semantic experiment ID가 같은 파일 SHA를 보장하지는 않는다.
  기존 cache 정책대로 실제 WAV bytes의 hash를 사용한다.
  `.cache/phase6-package-smoke`, `outputs/phase6/wheel-comparison`, `wheel-experiment`에 남았다.

공개 metro WAV 37.498594초에서 원본 대조군과 변형 8개를 실제 CPU 모델로 추론했다.
`backend_initializations=1`, `inferred_count=9`, `failed_count=0`, batch 시간 45.356초였다.
병행 테스트와 환경의 영향을 받으므로 처리 성능 benchmark로 사용하지 않는다.
모든 원본 hash와 expected 구간은 유지되었고, 8쌍 비교 및 9개 PNG/HTML이 생성되었다.

재현 명령은 다음과 같다. 다시 실행할 때 `--output-dir`은 새 경로로 바꾼다.

```powershell
.venv/Scripts/python.exe scripts/verify_corruption_pipeline.py --audio .cache/PretrainedSED/test_files/752547__iscence__milan_metro_coming_in_station.wav --class-id /m/0195fx --corruptions configs/corruption.example.json --output-dir outputs/phase6/metro-demo
```

이 스크립트는 **전체 clip의 synthetic support**를 외부 reference API로 명시한다.
WAVES 계획/영상 annotation을 만들거나 추정하지 않는다. 일반 `evaluate` CLI의 WAVES provider는
이 origin을 승인하지 않는다. 이 demo를 다시 평가하려면 스크립트의 명시적 reference API를 사용한다.
Mapping은 명시적인 Subway ID `/m/0195fx`, threshold 0.2, median 1, 최소 길이 0이다.
원본 검출은 `[0.08,30.0]`, synthetic 전체 길이 대비 IoU는 약 0.797897이었다.

| 실제 파형 이동 | 관측 시작점 변화 | 관측 변화 − 조작량 |
| --- | --- | --- |
| +100ms | +80ms | −20ms |
| +200ms | +160ms | −40ms |
| +500ms | +480ms | −20ms |
| +1000ms | +1000ms | 0ms |

0.5~1.0초 구간 제거는 검출을 쪼개 검출 개수 +2, shorten은 +1이었다. 복제는 덧셈 중첩
22,011 frames였지만 검출 구간은 변하지 않았다. Extend의 관측 시작점은 −40ms였다.
따라서 의도한 waveform 오류와 검출기의 반응은 별개이며, 임의 구간 삭제/복제를 곧바로
missing/extra semantic event의 정답으로 해석하지 않는다. 40ms 출력 bin과 10초 chunk/global
context도 이 결과에 영향을 줄 수 있다. 하나의 예제로 일반 민감도나 threshold를 보정하지 않는다.

실제 결과: `outputs/phase6/metro-demo/comparison/comparison.json`, `comparison.csv`,
`reports/dataset.json`, `visuals/index.html`. 원본/변형 WAV와 모델·cache는 Git에 넣지 않았다.
예제 음원의 출처와 라이선스는 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)에 있다.

## 다음 단계

실제 WAVES materialized WAV가 추가되면 source별 구간을 선택해 동일한 실험을 적용한다.
현재 checkout에 WAV가 없으므로 WAVES 생성 품질/영상 동기화/실제 perceptual quality는 미검증이다.
Phase 7은 별도 설정이 주어졌을 때만 사용하는 PASS/REVIEW/FAIL/UNSUPPORTED 정책이다.
Null threshold는 판정에서 제외하고 기본값으로 품질 판정을 만들지 않는다. Human evaluation
platform이나 training은 이번 범위에 포함하지 않는다.
