import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time


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

            
            cursor_x = np.interp(index_tip.x, [0, 1], [0, screen_width])
            cursor_y = np.interp(index_tip.y, [0, 1], [0, screen_height])

            
            smooth_x = alpha * cursor_x + (1 - alpha) * prev_cursor_x
            smooth_y = alpha * cursor_y + (1 - alpha) * prev_cursor_y
            prev_cursor_x, prev_cursor_y = smooth_x, smooth_y

            
            if thumb_raised and index_raised and middle_raised and not (ring_raised or pinky_raised):
                pyautogui.moveTo(smooth_x, smooth_y, duration=0.05)

            
            index_thumb_distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))
            middle_thumb_distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([middle_tip.x, middle_tip.y]))

            if index_thumb_distance < 0.05:  
                pyautogui.click()
                cv2.putText(frame, "Left Click", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                time.sleep(0.2)

            if middle_thumb_distance < 0.05:  
                pyautogui.rightClick()
                cv2.putText(frame, "Right Click", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                time.sleep(0.2)

            if thumb_raised and index_raised and middle_raised and ring_raised and pinky_raised:
                current_time = time.time()
                if prev_swipe_x is not None and current_time - prev_time > 0.5:
                    delta_x = cursor_x - prev_swipe_x

                    if delta_x > swipe_threshold:
                        pyautogui.hotkey('alt', 'tab')  
                        cv2.putText(frame, "Switching App ->", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                        prev_time = current_time

                    elif delta_x < -swipe_threshold:
                        pyautogui.hotkey('alt', 'shift', 'tab')  
                        cv2.putText(frame, "<- Switching App", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                        prev_time = current_time

                prev_swipe_x = cursor_x  

            if thumb_raised and index_raised and middle_raised and ring_raised and pinky_raised:
                if prev_swipe_y is not None and current_time - prev_time > 0.5:
                    delta_y = cursor_y - prev_swipe_y

                    if delta_y > swipe_threshold:  
                        pyautogui.hotkey('alt', 'f4')
                        cv2.putText(frame, "Closing App!", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        prev_time = current_time

                prev_swipe_y = cursor_y  

    cv2.imshow("Hand Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()







