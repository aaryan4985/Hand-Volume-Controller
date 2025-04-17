# volume_control.py

import cv2
import mediapipe as mp
import math

# Step 1: Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)  # Detect only one hand
mp_draw = mp.solutions.drawing_utils     # For drawing landmarks

# Step 2: Open the webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    # Step 3: Convert to RGB (MediaPipe needs RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Step 4: If hand is detected
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm_list = []

            # Get landmark positions
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

            # Check for Thumb (id 4) and Index Tip (id 8)
            if lm_list:
                x1, y1 = lm_list[4][1], lm_list[4][2]  # Thumb tip
                x2, y2 = lm_list[8][1], lm_list[8][2]  # Index tip

                # Draw circles on thumb and index tips
                cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)

                # Draw line between the two points
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 3)

                # Calculate distance between fingers
                length = math.hypot(x2 - x1, y2 - y1)

                # Display distance on screen
                cv2.putText(img, f'Distance: {int(length)}', (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw hand landmarks
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    # Show the image
    cv2.imshow("Hand Volume Controller", img)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
