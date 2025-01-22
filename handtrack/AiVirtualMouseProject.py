import cv2
import mediapipe as mp
import pyautogui
import time

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

hand_detector = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
drawing_utils = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

click_distance_threshold = 15

pinch_detected = False

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image from webcam.")
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    index_x, index_y, thumb_x, thumb_y = None, None, None, None

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark

            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 8:
                    index_x = screen_width / frame_width * x
                    index_y = screen_height / frame_height * y

                if id == 4:
                    thumb_x = screen_width / frame_width * x
                    thumb_y = screen_height / frame_height * y

            if index_x is not None and index_y is not None and thumb_x is not None and thumb_y is not None:
                if abs(index_y - thumb_y) < click_distance_threshold:
                    if not pinch_detected:
                        pyautogui.click()
                        pyautogui.click()
                        print("Pinch detected, double-click performed")

                        pinch_detected = True

                elif abs(index_y - thumb_y) > 40:
                    pinch_detected = False

                if abs(index_y - thumb_y) > 40:  
                    pyautogui.moveTo(index_x, index_y)

    cv2.imshow('Virtual Mouse', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
