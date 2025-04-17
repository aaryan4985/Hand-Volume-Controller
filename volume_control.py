# volume_control.py

import cv2
import mediapipe as mp
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ---------------------- Setup MediaPipe ----------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# ---------------------- Setup Webcam ----------------------
cap = cv2.VideoCapture(0)

# ---------------------- Setup Pycaw (Volume Control) ----------------------
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_min, vol_max = volume.GetVolumeRange()[:2]  # Typically (-65.25, 0.0)

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

            if lm_list:
                x1, y1 = lm_list[4][1], lm_list[4][2]   # Thumb tip
                x2, y2 = lm_list[8][1], lm_list[8][2]   # Index tip

                # Draw fingers
                cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 3)

                # Midpoint
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(img, (cx, cy), 10, (0, 255, 255), cv2.FILLED)

                # Distance
                length = math.hypot(x2 - x1, y2 - y1)

                # Map distance to volume
                vol = np.interp(length, [30, 200], [vol_min, vol_max])
                volume.SetMasterVolumeLevel(vol, None)

                # Volume bar & percentage
                vol_bar = np.interp(length, [30, 200], [400, 150])
                vol_percent = np.interp(length, [30, 200], [0, 100])

                # Draw volume bar
                cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 3)
                cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, f'{int(vol_percent)} %', (40, 430),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    # Show webcam feed
    cv2.imshow("Hand Volume Controller", img)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------------- Cleanup ----------------------
cap.release()
cv2.destroyAllWindows()
