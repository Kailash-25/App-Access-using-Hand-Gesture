from flask import Flask, render_template, jsonify
import subprocess
import threading
import time
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import signal
import os

app = Flask(__name__)

gesture_process = None

# Gesture Control Function
def run_gesture_control():
    global gesture_process
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

    screen_width, screen_height = pyautogui.size()
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    alpha = 0.3
    prev_cursor_x, prev_cursor_y = 0, 0
    swipe_threshold = 100
    prev_swipe_x, prev_swipe_y = None, None
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                def is_finger_raised(tip_idx, base_idx):
                    tip = hand_landmarks.landmark[tip_idx]
                    base = hand_landmarks.landmark[base_idx]
                    return tip.y < base.y

                index_tip = hand_landmarks.landmark[8]
                thumb_tip = hand_landmarks.landmark[4]
                middle_tip = hand_landmarks.landmark[12]

                cursor_x = np.interp(index_tip.x, [0, 1], [0, screen_width])
                cursor_y = np.interp(index_tip.y, [0, 1], [0, screen_height])

                smooth_x = alpha * cursor_x + (1 - alpha) * prev_cursor_x
                smooth_y = alpha * cursor_y + (1 - alpha) * prev_cursor_y
                prev_cursor_x, prev_cursor_y = smooth_x, smooth_y

                index_thumb_distance = np.linalg.norm(
                    np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y])
                )
                middle_thumb_distance = np.linalg.norm(
                    np.array([thumb_tip.x, thumb_tip.y]) - np.array([middle_tip.x, middle_tip.y])
                )

                if index_thumb_distance < 0.05:
                    pyautogui.click()
                    time.sleep(0.2)

                if middle_thumb_distance < 0.05:
                    pyautogui.rightClick()
                    time.sleep(0.2)

        cv2.imshow("Hand Gesture Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start', methods=['GET'])
def start_gesture():
    global gesture_process
    if gesture_process is None:
        gesture_process = threading.Thread(target=run_gesture_control)
        gesture_process.daemon = True
        gesture_process.start()
        return jsonify({'status': 'Gesture control started'})
    return jsonify({'status': 'Already running'})


@app.route('/stop', methods=['GET'])
def stop_gesture():
    global gesture_process
    if gesture_process is not None:
        os.kill(os.getpid(), signal.SIGKILL)
        gesture_process = None
        return jsonify({'status': 'Gesture control stopped'})
    return jsonify({'status': 'Not running'})


if __name__ == '__main__':
    app.run(debug=True)
