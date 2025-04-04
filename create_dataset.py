import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
    print("Error: No data found in './data' folder. Ensure you have collected images.")
    exit()

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, dir_)

    if not os.path.isdir(class_dir):
        continue  # Skip files, only process directories

    for img_path in os.listdir(class_dir):
        data_aux = []
        x_, y_ = [], []

        img = cv2.imread(os.path.join(class_dir, img_path))
        if img is None:
            print(f"Warning: Could not read {img_path} in {dir_}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

            data.append(data_aux)
            labels.append(int(dir_))  # Convert labels to integers

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data saved successfully to 'data.pickle'!")
