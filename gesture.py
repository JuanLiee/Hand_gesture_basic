import cv2
import mediapipe as mp
from gtts import gTTS
import pygame
import threading
import time
import os
import warnings

warnings.filterwarnings("ignore")

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def play_audio(text):
    filename = f"voice_{text.replace(' ', '_')}.mp3"
    tts = gTTS(text=text, lang='en')
    tts.save(filename)

    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    pygame.mixer.quit()
    os.remove(filename)


def detect_gesture(landmarks):
   
    thumb_tip = landmarks.landmark[4]
    index_tip = landmarks.landmark[8]
    middle_tip = landmarks.landmark[12]
    ring_tip = landmarks.landmark[16]
    pinky_tip = landmarks.landmark[20]

    
    thumb_base = landmarks.landmark[2]
    index_base = landmarks.landmark[5]
    middle_base = landmarks.landmark[9]
    ring_base = landmarks.landmark[13]
    pinky_base = landmarks.landmark[17]


    # Hi! ✌️ Peace
    if (index_tip.y < index_base.y and middle_tip.y < middle_base.y and
        ring_tip.y > ring_base.y and pinky_tip.y > pinky_base.y):
        return "Hi!"

    # OK 👍 Thumbs up
    if (thumb_tip.y < thumb_base.y and index_tip.y > index_base.y and
        middle_tip.y > middle_base.y and ring_tip.y > ring_base.y and pinky_tip.y > pinky_base.y):
        return "OK"

    # Stop ✋ 
    if (index_tip.y < index_base.y and middle_tip.y < middle_base.y and
        ring_tip.y < ring_base.y and pinky_tip.y < pinky_base.y and thumb_tip.y < thumb_base.y):
        return "stop"

    # Thanks 🤟 Love sign
    if (index_tip.y < index_base.y and pinky_tip.y < pinky_base.y and
        middle_tip.y > middle_base.y and ring_tip.y > ring_base.y):
        return "Thanks"

    # My Name ☝️ Telunjuk tegak
    if (index_tip.y < index_base.y and middle_tip.y > middle_base.y and
        ring_tip.y > ring_base.y and pinky_tip.y > pinky_base.y):
        return "My Name"

   # Is Juan 
    if (pinky_tip.y < pinky_base.y and
    thumb_tip.y > thumb_base.y and index_tip.y > index_base.y and
    middle_tip.y > middle_base.y and ring_tip.y > ring_base.y):
        return "Is Juan"

    # Sorry ✌️ 
    if (index_tip.y < index_base.y and middle_tip.y < middle_base.y and
    ring_tip.y > ring_base.y and pinky_tip.y > pinky_base.y and thumb_tip.y > thumb_base.y):
        return "Sorry"

    return None

# main program
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera tidak bisa dibuka")
    exit()

last_gesture = None
last_time = 0

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        gesture = None
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = detect_gesture(hand_landmarks)

        if gesture:
            font = cv2.FONT_HERSHEY_SIMPLEX
            x, y = 50, 80

            cv2.putText(frame, gesture, (x + 2, y + 2), font, 2, (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(frame, gesture, (x, y), font, 2, (0, 255, 0), 2, cv2.LINE_AA)

            if gesture != last_gesture and time.time() - last_time > 1.5:
                threading.Thread(target=play_audio, args=(gesture,)).start()
                last_gesture = gesture
                last_time = time.time()

        cv2.imshow("Gesture Shortcut", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
