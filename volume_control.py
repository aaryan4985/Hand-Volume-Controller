# volume_control.py

# 📌 Step 1: Import required libraries
import cv2
import mediapipe as mp

# 📌 Step 2: Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)  # Detect 1 hand
mp_draw = mp.solutions.drawing_utils     # For drawing

# 📌 Step 3: Access the webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    success, img = cap.read()
    if not success:
        break

    # Convert to RGB for MediaPipe processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Check if a hand is detected
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Draw the hand landmarks
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    # Show the image
    cv2.imshow("Hand Volume Controller", img)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close window
cap.release()
cv2.destroyAllWindows()
