# 🧑‍🏫 MediaPipe Face Landmarker로 웹캠 얼굴 랜드마크 검출하기

> **강의노트** · 2026-10-07
> 웹캠 영상에서 얼굴의 **3D 랜드마크 478개**와 **표정 점수(블렌드쉐이프) 52개**를 실시간으로 뽑아 화면에 그리는 파이썬 프로그램을 만들어 봅니다.

---

## 📌 목차

1. [학습 목표](#1-학습-목표)
2. [한눈에 보는 전체 흐름](#2-한눈에-보는-전체-흐름)
3. [준비물 & 설치](#3-준비물--설치)
4. [Face Landmarker 핵심 개념](#4-face-landmarker-핵심-개념)
5. [코드 해설](#5-코드-해설)
6. [실행 방법](#6-실행-방법)
7. [자주 겪는 문제](#7-자주-겪는-문제)
8. [📖 용어 정리](#8--용어-정리)
9. [더 해볼 것](#9-더-해볼-것)

---

## 1. 학습 목표

이 강의를 마치면 다음을 할 수 있습니다.

- [x] MediaPipe **Tasks API**로 Face Landmarker 모델을 불러올 수 있다.
- [x] OpenCV로 웹캠 프레임을 읽어 MediaPipe에 넘길 수 있다.
- [x] 랜드마크 좌표(0~1로 정규화된 값)를 픽셀 좌표로 바꿔 화면에 그릴 수 있다.
- [x] 블렌드쉐이프 점수로 표정을 숫자로 읽을 수 있다.
- [x] `IMAGE` / `VIDEO` / `LIVE_STREAM` 실행 모드의 차이를 설명할 수 있다.

---

## 2. 한눈에 보는 전체 흐름

```
 ┌──────────┐   BGR 프레임   ┌───────────┐  RGB mp.Image  ┌──────────────────┐
 │  웹캠     │ ────────────▶ │  OpenCV    │ ─────────────▶ │  FaceLandmarker   │
 │ (cv2)    │               │ 좌우반전·변환 │               │ detect_for_video │
 └──────────┘               └───────────┘                └────────┬─────────┘
                                                                  │ 결과
                     ┌────────────────────────────────────────────┘
                     ▼
     face_landmarks (얼굴별 478개 점)   face_blendshapes (52개 표정 점수)
                     │                               │
                     └──────────────┬────────────────┘
                                    ▼
                         cv2로 선·막대 그리기 → cv2.imshow
```

**핵심 한 줄 요약:** `프레임 읽기 → RGB 변환 → mp.Image 생성 → detect_for_video(이미지, 타임스탬프) → 결과 그리기` 를 매 프레임 반복합니다.

---

## 3. 준비물 & 설치

### 3-1. 이번 실습 환경

| 항목 | 버전 |
|---|---|
| OS | Windows 11 |
| Python | 3.14 |
| mediapipe | 1.1.0 |
| opencv-python | 5.0.0 |

### 3-2. 라이브러리 설치

```bash
pip install -r requirements.txt
# 또는
pip install mediapipe opencv-python
```

### 3-3. 모델 파일 받기

Face Landmarker는 **모델 번들 파일(`.task`)** 이 있어야 동작합니다.
[공식 가이드](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)의 **Models → Versions → Latest** 링크에서 받습니다.

```
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

> 💡 이 저장소에는 `face_landmarker.task`(약 3.7MB)가 이미 들어 있습니다. 스크립트와 **같은 폴더**에 두면 됩니다.

### 3-4. 폴더 구조

```
face landmark/
├── face_landmark_webcam.py   # 실행 코드
├── face_landmarker.task      # 모델 번들 (float16)
├── requirements.txt
└── README.md                 # 이 강의노트
```

---

## 4. Face Landmarker 핵심 개념

### 4-1. 모델 번들 안에는 모델 3개가 들어 있다

| 순서 | 모델 | 입력 크기 | 하는 일 |
|---|---|---|---|
| ① | **Face Detector** (BlazeFace short-range) | 192×192 | 얼굴이 *어디에* 있는지 찾기 |
| ② | **Face Mesh V2** | 256×256 | 얼굴 위에 **3D 점 478개** 찍기 |
| ③ | **Blendshape** | 1×146×2 | 점들로 **표정 점수 52개** 계산 |

> ①에서 찾은 얼굴 영역을 ②가 정밀하게 보고, ②의 결과를 ③이 해석하는 **파이프라인** 구조입니다.

### 4-2. 실행 모드 (running_mode) 비교 ⭐

| 모드 | 용도 | 호출 함수 | 결과 받는 방식 |
|---|---|---|---|
| `IMAGE` | 사진 한 장 | `detect(image)` | 바로 반환 (동기) |
| `VIDEO` | 동영상·웹캠 프레임 | `detect_for_video(image, ts_ms)` | 바로 반환 (동기) |
| `LIVE_STREAM` | 실시간 카메라 | `detect_async(image, ts_ms)` | **콜백 함수**로 받음 (비동기) |

👉 이번 실습은 **`VIDEO` 모드**를 사용했습니다. 결과를 바로 받아서 그 프레임에 그리면 되므로 코드가 가장 단순합니다.

> ⚠️ `VIDEO`/`LIVE_STREAM` 모드는 **타임스탬프(ms)가 계속 증가**해야 합니다. 같은 값이나 더 작은 값을 넣으면 에러가 납니다.

### 4-3. 주요 설정값 (Options)

| 옵션 | 의미 | 기본값 | 이번 코드 |
|---|---|---|---|
| `num_faces` | 최대 몇 명까지 검출할지 | 1 | `--faces` 인자 |
| `min_face_detection_confidence` | 얼굴 *검출* 최소 신뢰도 | 0.5 | 0.5 |
| `min_face_presence_confidence` | 얼굴 *존재* 최소 신뢰도 | 0.5 | 0.5 |
| `min_tracking_confidence` | *추적* 최소 신뢰도 | 0.5 | 0.5 |
| `output_face_blendshapes` | 표정 점수 출력 여부 | False | **True** |
| `output_facial_transformation_matrixes` | 변환 행렬 출력 여부 | False | False |

> 💡 `num_faces=1`일 때만 랜드마크 **스무딩(떨림 보정)** 이 적용됩니다.

### 4-4. 결과(Result) 구조

```python
result.face_landmarks      # [얼굴][478개 점] → 각 점은 .x .y .z
result.face_blendshapes    # [얼굴][52개]     → 각 항목은 .category_name .score
result.facial_transformation_matrixes  # (옵션 켰을 때) 4×4 행렬
```

- `x`, `y`는 **0~1로 정규화된 값**입니다. 픽셀 좌표로 바꾸려면 `x * 너비`, `y * 높이`를 하면 됩니다.
- `z`는 깊이 값입니다. 작을수록 카메라에 가깝고, 머리 중심 기준의 상대값입니다.

---

## 5. 코드 해설

전체 코드: [`face_landmark_webcam.py`](face_landmark_webcam.py)

### STEP 1. 모델 옵션 만들기

```python
options = vision.FaceLandmarkerOptions(
    base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=vision.RunningMode.VIDEO,   # 웹캠 프레임을 순서대로 처리
    num_faces=args.faces,
    output_face_blendshapes=True,            # 표정 점수도 받기
)
```

### STEP 2. 웹캠 열기

```python
cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)  # Windows에서 빠르게 열림
```

> `CAP_DSHOW`(DirectShow)로 열리지 않으면 기본 백엔드로 다시 시도하도록 짰습니다.

### STEP 3. 프레임 → MediaPipe 이미지

```python
frame = cv2.flip(frame, 1)                         # 거울처럼 좌우 반전
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)       # OpenCV는 BGR, MediaPipe는 RGB!
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
```

> ⚠️ **자주 하는 실수:** BGR을 그대로 넣으면 색이 뒤바뀌어 검출 성능이 떨어집니다.

### STEP 4. 추론 (타임스탬프 필수)

```python
ts = int((time.monotonic() - start) * 1000)   # 경과 시간(ms)
if ts <= last_ts:                              # 반드시 증가하도록 보정
    ts = last_ts + 1
last_ts = ts
result = landmarker.detect_for_video(mp_image, ts)
```

### STEP 5. 랜드마크 그리기

```python
pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]   # 정규화 → 픽셀
for c in C.FACE_LANDMARKS_LIPS:                                # 연결선 목록
    cv2.line(img, pts[c.start], pts[c.end], color, 2)
```

`vision.FaceLandmarksConnections`에는 미리 정의된 **연결선 묶음**이 들어 있습니다.

| 상수 | 그리는 부위 |
|---|---|
| `FACE_LANDMARKS_TESSELATION` | 얼굴 전체 삼각형 메쉬 |
| `FACE_LANDMARKS_FACE_OVAL` | 얼굴 외곽선 |
| `FACE_LANDMARKS_LIPS` | 입술 |
| `FACE_LANDMARKS_LEFT_EYE` / `RIGHT_EYE` | 눈 |
| `FACE_LANDMARKS_LEFT_EYEBROW` / `RIGHT_EYEBROW` | 눈썹 |
| `FACE_LANDMARKS_LEFT_IRIS` / `RIGHT_IRIS` | 홍채 |
| `FACE_LANDMARKS_CONTOURS` | 주요 윤곽 전체 |

> 📝 예전 버전의 `mp.solutions.drawing_utils`는 mediapipe 1.x에서 **사라졌습니다.** 그래서 이번 코드는 `cv2.line`으로 직접 그렸습니다.

### STEP 6. 블렌드쉐이프(표정) 표시

```python
top = sorted(blendshapes, key=lambda b: b.score, reverse=True)[:8]   # 상위 8개
```

점수가 높은 표정 8개를 **막대 그래프**로 보여줍니다. 예를 들어 `jawOpen`(입 벌림), `eyeBlinkLeft`(왼눈 깜빡임), `mouthSmileRight`(오른쪽 미소) 같은 항목이 나옵니다.

---

## 6. 실행 방법

```bash
python face_landmark_webcam.py                       # 기본 카메라, 얼굴 1개
python face_landmark_webcam.py --camera 1 --faces 2  # 2번째 카메라, 얼굴 2개까지
```

| 키 | 동작 |
|---|---|
| `q` / `ESC` | 종료 |
| `m` | 메쉬 켜기/끄기 |
| `b` | 블렌드쉐이프 표시 켜기/끄기 |

화면 왼쪽 위에는 **FPS**와 **검출된 얼굴 수**가 표시됩니다.

---

## 7. 자주 겪는 문제

| 증상 | 원인 / 해결 |
|---|---|
| `모델 파일이 없습니다` | `face_landmarker.task`를 스크립트와 같은 폴더에 두세요. |
| `웹캠(0)을 열 수 없습니다` | 다른 프로그램(Zoom 등)이 카메라를 쓰는 중이거나 번호가 다릅니다. `--camera 1`을 시도해 보세요. |
| 타임스탬프 관련 에러 | `VIDEO` 모드는 ms 값이 계속 증가해야 합니다. (코드에서 보정함) |
| `Feedback manager ... Disabling support for feedback tensors.` 경고 | 내부 경고라서 **무시해도 됩니다.** |
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | 구버전 예제 코드입니다. mediapipe 1.x에서는 **Tasks API**(`mediapipe.tasks`)를 쓰세요. |

---

## 8. 📖 용어 정리

| 용어 | 한 줄 설명 |
|---|---|
| **MediaPipe** | Google에서 만든 온디바이스 ML 프레임워크입니다. 얼굴·손·포즈 인식 등을 바로 쓸 수 있는 형태로 제공합니다. |
| **Google AI Edge** | MediaPipe·LiteRT 등 *기기에서 직접 도는* AI 도구들을 묶은 Google 브랜드입니다. |
| **Tasks API** | MediaPipe의 최신 사용 방식입니다. `Options` 만들기 → `create_from_options` → `detect`의 3단계로 씁니다. |
| **Face Landmarker** | 얼굴 랜드마크, 블렌드쉐이프, 변환 행렬을 출력하는 MediaPipe Task입니다. |
| **랜드마크 (Landmark)** | 얼굴의 특정 위치(코끝, 입꼬리 등)를 나타내는 점입니다. 이 모델은 **478개**를 찍습니다. |
| **페이스 메쉬 (Face Mesh)** | 랜드마크들을 삼각형으로 이어 얼굴 표면을 표현한 그물망입니다. |
| **테셀레이션 (Tessellation)** | 면을 작은 다각형(삼각형)으로 빈틈없이 나누는 것입니다. 메쉬를 그릴 때 쓰는 연결선 목록이기도 합니다. |
| **홍채 (Iris)** | 눈동자 둘레입니다. 478개 중 마지막 10개 점(468~477)이 홍채에 해당합니다. |
| **블렌드쉐이프 (Blendshape)** | 표정을 **0~1 점수 52개**로 나타낸 것입니다 (예: `jawOpen` 0.8 = 입을 크게 벌림). 3D 아바타 애니메이션에 씁니다. |
| **변환 행렬 (Transformation Matrix)** | 표준 얼굴 모델을 실제 얼굴의 위치·회전에 맞추는 4×4 행렬입니다. AR 필터 효과에 씁니다. |
| **정규화 좌표 (Normalized Coordinate)** | 이미지 크기와 상관없이 0~1로 표현한 좌표입니다. 픽셀 좌표 = 정규화 값 × 이미지 크기입니다. |
| **모델 번들 (`.task`)** | 여러 모델(검출 + 메쉬 + 블렌드쉐이프)을 하나로 묶은 MediaPipe 모델 파일입니다. |
| **float16** | 16비트 실수 정밀도입니다. 32비트보다 모델이 작고 빠르지만 정확도는 거의 같습니다. |
| **BlazeFace** | Google의 초경량 얼굴 검출 모델입니다. 모바일 GPU에 최적화되어 있습니다. |
| **추론 (Inference)** | 학습된 모델에 새 입력을 넣어 결과를 얻는 과정입니다. |
| **실행 모드 (Running Mode)** | `IMAGE` / `VIDEO` / `LIVE_STREAM` 중에서 입력 종류에 맞게 고르는 처리 방식입니다. |
| **타임스탬프 (Timestamp)** | 프레임의 시간값(ms)입니다. 영상 모드에서 프레임 순서를 알려 주고 추적에 쓰입니다. |
| **동기 / 비동기** | 동기는 결과가 나올 때까지 기다리는 방식이고, 비동기는 결과가 나오면 콜백으로 받는 방식입니다. |
| **콜백 (Callback)** | 어떤 일이 끝나면 자동으로 호출되도록 등록해 두는 함수입니다 (`LIVE_STREAM`의 `result_callback`). |
| **신뢰도 (Confidence)** | 모델이 결과를 얼마나 확신하는지 나타내는 0~1 값입니다. 임계값보다 낮으면 결과를 버립니다. |
| **트래킹 (Tracking)** | 이전 프레임의 결과를 이용해 다음 프레임에서 빠르게 다시 찾는 것입니다. 매번 처음부터 검출하는 것보다 빠릅니다. |
| **스무딩 (Smoothing)** | 프레임마다 생기는 점의 떨림을 줄여 주는 보정입니다 (`num_faces=1`일 때만 적용). |
| **BGR / RGB** | 색상 채널 순서입니다. OpenCV는 BGR, MediaPipe는 RGB를 씁니다. |
| **OpenCV (`cv2`)** | 영상 처리 라이브러리입니다. 이번 실습에서는 웹캠 읽기, 그리기, 창 띄우기에 사용했습니다. |
| **DirectShow (`CAP_DSHOW`)** | Windows의 영상 장치 백엔드입니다. 카메라를 여는 속도가 빠른 편입니다. |
| **FPS** | Frames Per Second, 즉 초당 처리한 프레임 수입니다. |

---

## 9. 더 해볼 것

- [ ] `LIVE_STREAM` 모드 + `result_callback`으로 비동기 버전 만들기
- [ ] `eyeBlinkLeft/Right` 점수로 **졸음 감지기** 만들기
- [ ] `jawOpen`, `mouthSmile*` 점수로 **표정 인식 게임** 만들기
- [ ] `output_facial_transformation_matrixes=True`로 머리 회전 각도(yaw/pitch/roll) 구하기
- [ ] 특정 랜드마크 번호(예: 코끝 1번)에 스티커 이미지 붙이기 (AR 필터)

---

### 참고 자료

- [MediaPipe Face Landmarker 가이드](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)
- [Face Landmarker Python 가이드](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python)
- 모델 라이선스: Apache 2.0 (Google)
