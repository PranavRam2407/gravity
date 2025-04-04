import os
import cv2
import time
import mediapipe as mp

# Setup dataset directory
DATA_DIR = './data'
os.makedirs(DATA_DIR, exist_ok=True)

number_of_classes = 10
dataset_size = 100

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    os.makedirs(class_dir, exist_ok=True)

    print(f'\nCollecting data for class {j}')
    print('Press "Q" to start.')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from camera.")
            break

        cv2.putText(frame, 'Ready? Press "Q"!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            time.sleep(1)
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # If at least one hand is detected, save the image
        if results.multi_hand_landmarks:
            frame = cv2.resize(frame, (224, 224))
            cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
            counter += 1

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"Completed collecting data for class {j}")

cap.release()
cv2.destroyAllWindows()
