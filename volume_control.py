# volume_control.py

import cv2
import mediapipe as mp
import numpy as np
import math
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# --------------- Setup ----------------
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Audio control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_min, vol_max = volume.GetVolumeRange()[:2]

# Time tracking for FPS
p_time = 0

def fingers_up(lm_list):
    fingers = []

    # Thumb
    if lm_list[4][1] > lm_list[3][1]:  # Right hand
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for tip_id in [8, 12, 16, 20]:
        if lm_list[tip_id][2] < lm_list[tip_id - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers  # List of 5 values: 1 = up, 0 = down

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    lm_list = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    if lm_list:
        fingers = fingers_up(lm_list)

        x1, y1 = lm_list[4][1], lm_list[4][2]   # Thumb tip
        x2, y2 = lm_list[8][1], lm_list[8][2]   # Index tip

        length = math.hypot(x2 - x1, y2 - y1)

        # MUTE GESTURE: Pinky up & others down
        if fingers == [0, 0, 0, 0, 1]:
            volume.SetMasterVolumeLevel(vol_min, None)
            cv2.putText(img, "MUTED 🔇", (450, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        else:
            # Volume Control
            vol = np.interp(length, [30, 200], [vol_min, vol_max])
            volume.SetMasterVolumeLevel(vol, None)

            vol_bar = np.interp(length, [30, 200], [400, 150])
            vol_percent = np.interp(length, [30, 200], [0, 100])

            # Draw hand gesture line and points
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 3)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

            # Draw volume bar
            cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 3)
            cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, f'{int(vol_percent)} %', (40, 430),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # FPS Counter
    c_time = time.time()
    fps = 1 / (c_time - p_time) if p_time else 0
    p_time = c_time
    cv2.putText(img, f'FPS: {int(fps)}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)

    # Show final image
    cv2.imshow("Volume Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
