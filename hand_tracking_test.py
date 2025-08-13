#FILE NAME: hand_tracking_test.py
#CREATOR: Zachary Petrogeorge
#LAST UPDATED; 8/12/2025
#SOURCES: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python

# Impot mediapipe as follows to access Hand Landmarker functionality
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Creating live hand tracking task
import mediapipe as mp
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

mp_hands = mp.solutions.hands
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS
latest_results = None

# Creating a instance for hand landmarker with live stream
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestampe_ms: int):
    global latest_results
    latest_results = result


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='C:/Users/zptaz/OneDrive/Desktop/hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

# Starting live video feed
cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        
        if not ret: 
            print("Failed to cature frame")
            break

        # Convert frame from BGR to RGB, MediaPipe excpects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert from received from cv2 to mediapipe object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Send frame to hand landmarker asynchronously for live stream processing
        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        # Drawing hand landmarks on live frame
        if latest_results and latest_results.hand_landmarks:
            for hand_landmarks in latest_results.hand_landmarks:
                for connection in HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    start = hand_landmarks[start_idx]
                    end = hand_landmarks[end_idx]
                    start_point = (int(start.x * frame.shape[1]), int(start.y * frame.shape[0]))
                    end_point = (int(end.x * frame.shape[1]), int(end.y * frame.shape[0]))
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
                
                for landmark in hand_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x,y), 5, (0, 0, 255), -1)

        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()