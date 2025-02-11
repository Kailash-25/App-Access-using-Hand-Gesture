from flask import Flask, render_template, jsonify
import threading
import time
import cv2
import mediapipe as mp
import pyautogui
import numpy as np

app = Flask(__name__)

gesture_thread = None
running = False
lock = threading.Lock()

# Gesture Control Function
def run_gesture_control():
    global running

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

    screen_width, screen_height = pyautogui.size()
    cap = cv2.VideoCapture(0)

    prev_cursor_x, prev_cursor_y = 0, 0
    alpha = 0.3
    swipe_threshold = 100
    prev_swipe_x, prev_swipe_y = None, None
    prev_time = time.time()

    with lock:
        running = True

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:

                def is_finger_raised(tip_idx, base_idx):
                    tip = hand_landmarks.landmark[tip_idx]
                    base = hand_landmarks.landmark[base_idx]
                    return tip.y < base.y  

                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                middle_tip = hand_landmarks.landmark[12]
                ring_tip = hand_landmarks.landmark[16]
                pinky_tip = hand_landmarks.landmark[20]

                thumb_raised = is_finger_raised(4, 2)
                index_raised = is_finger_raised(8, 6)
                middle_raised = is_finger_raised(12, 10)
                ring_raised = is_finger_raised(16, 14)
                pinky_raised = is_finger_raised(20, 18)

                # Cursor Movement (Index Finger Raised)
                cursor_x = np.interp(index_tip.x, [0, 1], [0, screen_width])
                cursor_y = np.interp(index_tip.y, [0, 1], [0, screen_height])

                smooth_x = alpha * cursor_x + (1 - alpha) * prev_cursor_x
                smooth_y = alpha * cursor_y + (1 - alpha) * prev_cursor_y
                prev_cursor_x, prev_cursor_y = smooth_x, smooth_y

                if thumb_raised and index_raised and middle_raised and not (ring_raised or pinky_raised):
                    pyautogui.moveTo(smooth_x, smooth_y, duration=0.05)

                # Click Gestures
                index_thumb_distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))
                middle_thumb_distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([middle_tip.x, middle_tip.y]))

                if index_thumb_distance < 0.05:  
                    pyautogui.click()
                    time.sleep(0.2)

                if middle_thumb_distance < 0.05:  
                    pyautogui.rightClick()
                    time.sleep(0.2)

                current_time = time.time()

                # App Switching (Swipe Left/Right)
                if thumb_raised and index_raised and middle_raised and ring_raised and pinky_raised:
                    if prev_swipe_x is not None and current_time - prev_time > 0.5:
                        delta_x = cursor_x - prev_swipe_x

                        if delta_x > swipe_threshold:
                            pyautogui.hotkey('alt', 'tab')  # Switch forward
                            prev_time = current_time

                        elif delta_x < -swipe_threshold:
                            pyautogui.hotkey('alt', 'shift', 'tab')  # Switch backward
                            prev_time = current_time

                    prev_swipe_x = cursor_x  

                # Close App (Swipe Down)
                if thumb_raised and index_raised and middle_raised and ring_raised and pinky_raised:
                    if prev_swipe_y is not None and current_time - prev_time > 0.5:
                        delta_y = cursor_y - prev_swipe_y

                        if delta_y > swipe_threshold:  
                            pyautogui.hotkey('alt', 'f4')
                            prev_time = current_time

                    prev_swipe_y = cursor_y  

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    with lock:
        running = False


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start', methods=['GET'])
def start_gesture():
    global gesture_thread
    with lock:
        if gesture_thread is None or not gesture_thread.is_alive():
            gesture_thread = threading.Thread(target=run_gesture_control)
            gesture_thread.daemon = True
            gesture_thread.start()
            return jsonify({'status': 'Gesture control started'})
    return jsonify({'status': 'Already running'})


@app.route('/stop', methods=['GET'])
def stop_gesture():
    global running
    with lock:
        if running:
            running = False
            return jsonify({'status': 'Stopping gesture control...'})
    return jsonify({'status': 'Not running'})


if __name__ == '__main__':
    app.run(debug=True)
