"""MediaPipe Face Landmarker - 웹캠 실시간 얼굴 랜드마크 검출.

실행: python face_landmark_webcam.py [--camera 0] [--faces 1]
키:   q / ESC = 종료,  m = 메쉬 표시 토글,  b = 블렌드쉐이프 표시 토글
"""
import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).with_name("face_landmarker.task")
C = vision.FaceLandmarksConnections


def draw_connections(img, pts, connections, color, thickness=1):
    for c in connections:
        cv2.line(img, pts[c.start], pts[c.end], color, thickness, cv2.LINE_AA)


def draw_face(img, landmarks, show_mesh):
    h, w = img.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    if show_mesh:
        draw_connections(img, pts, C.FACE_LANDMARKS_TESSELATION, (192, 192, 192))
    draw_connections(img, pts, C.FACE_LANDMARKS_FACE_OVAL, (224, 224, 224), 2)
    draw_connections(img, pts, C.FACE_LANDMARKS_LIPS, (80, 80, 255), 2)
    draw_connections(img, pts, C.FACE_LANDMARKS_LEFT_EYE, (80, 255, 80), 2)
    draw_connections(img, pts, C.FACE_LANDMARKS_RIGHT_EYE, (255, 160, 80), 2)
    draw_connections(img, pts, C.FACE_LANDMARKS_LEFT_EYEBROW, (80, 255, 80), 2)
    draw_connections(img, pts, C.FACE_LANDMARKS_RIGHT_EYEBROW, (255, 160, 80), 2)
    draw_connections(img, pts, C.FACE_LANDMARKS_LEFT_IRIS, (0, 255, 255), 2)
    draw_connections(img, pts, C.FACE_LANDMARKS_RIGHT_IRIS, (0, 255, 255), 2)


def draw_blendshapes(img, blendshapes, top_k=8):
    top = sorted(blendshapes, key=lambda b: b.score, reverse=True)[:top_k]
    for i, b in enumerate(top):
        y = 60 + i * 22
        cv2.rectangle(img, (10, y - 12), (10 + int(b.score * 150), y + 4), (0, 200, 255), -1)
        cv2.putText(img, f"{b.category_name} {b.score:.2f}", (170, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--faces", type=int, default=1)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")

    options = vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=args.faces,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"웹캠({args.camera})을 열 수 없습니다.")

    show_mesh, show_blend = True, True
    start = time.monotonic()
    last_ts = -1
    prev = time.monotonic()

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("프레임을 읽지 못했습니다.")
                break
            frame = cv2.flip(frame, 1)  # 거울 모드

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int((time.monotonic() - start) * 1000)
            if ts <= last_ts:  # VIDEO 모드는 타임스탬프가 단조 증가해야 함
                ts = last_ts + 1
            last_ts = ts
            result = landmarker.detect_for_video(mp_image, ts)

            for face in result.face_landmarks:
                draw_face(frame, face, show_mesh)
            if show_blend and result.face_blendshapes:
                draw_blendshapes(frame, result.face_blendshapes[0])

            now = time.monotonic()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS {fps:.1f}  faces {len(result.face_landmarks)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Face Landmarker (q: quit, m: mesh, b: blendshapes)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                show_mesh = not show_mesh
            if key == ord("b"):
                show_blend = not show_blend

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
