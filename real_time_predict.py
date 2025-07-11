import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Finger tip indices as per MediaPipe (THUMB is handled differently)
finger_tips = [8, 12, 16, 20]
finger_pip = [6, 10, 14, 18]  # PIP joints for comparison

# Start camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror the image
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    finger_count = 0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark list
            landmarks = hand_landmarks.landmark

            # Thumb (special case)
            if landmarks[4].x > landmarks[3].x:  # For right hand (flip for left)
                finger_count += 1

            # Other fingers
            for tip, pip in zip(finger_tips, finger_pip):
                if landmarks[tip].y < landmarks[pip].y:
                    finger_count += 1

    # Display count
    cv2.putText(frame, f"Fingers: {finger_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow("Finger Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
