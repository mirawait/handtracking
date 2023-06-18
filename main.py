import os.path
import time

import mediapipe as mp
from mediapipe.tasks.python import vision
import cv2 as cv
from visualization_utils import draw_landmarks_on_image

model_path = os.path.dirname(os.path.realpath(__file__)) + "/models/hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

detection_result = None


def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global detection_result
    detection_result = result
    #print('hand landmarker result: {}'.format(result))


options = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path),
                                running_mode=VisionRunningMode.LIVE_STREAM,
                                result_callback=print_result,
                                num_hands=2,
                                min_hand_detection_confidence=0.4,
                                min_tracking_confidence=0.5)

detector = vision.HandLandmarker.create_from_options(options)

if __name__ == "__main__":
    cam = cv.VideoCapture(0)
    cam.set(cv.CAP_PROP_FPS, 60)
    start_time = time.time()
    frames = 0
    if not cam:
        raise ValueError("Cam not found")
    while True:
        if time.time() - start_time >= 1.0:
            frames = 0
            start_time = time.time()
        _, img = cam.read()
        frames += 1
        img = cv.flip(img, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        with HandLandmarker.create_from_options(options) as detector:
            detector.detect_async(mp_image, 1)
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
        cv.putText(annotated_image, f"fps: {round(frames / (time.time() - start_time),2)}", (100, 100), cv.FONT_HERSHEY_SIMPLEX,
                   1,
                   255)
        cv.imshow("WEB CUM", cv.cvtColor(annotated_image, cv.COLOR_RGBA2RGB))
        if cv.waitKey(1) == 27:
            break
    cam.release()
