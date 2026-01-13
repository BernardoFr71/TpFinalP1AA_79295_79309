import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import time

# --- CONFIGURAÇÃO ---
OUTPUT_FILE = 'my_extra_data.csv'
TARGET_LABEL = 'C'
SAMPLES_TO_COLLECT = 200

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

data = []
cap = cv2.VideoCapture(0)

print(f"Pressiona 's' para começar a gravar {SAMPLES_TO_COLLECT} frames da letra '{TARGET_LABEL}'...")

collecting = False
counter = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if collecting and counter < SAMPLES_TO_COLLECT:
                # Extrair landmarks
                landmarks = hand_landmarks.landmark
                row = []
                for lm in landmarks:
                    row.extend([lm.x, lm.y, lm.z])

                full_row = [TARGET_LABEL, 'Right'] + row
                data.append(full_row)

                counter += 1
                cv2.putText(frame, f"Gravando: {counter}/{SAMPLES_TO_COLLECT}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if counter >= SAMPLES_TO_COLLECT:
        collecting = False
        cv2.putText(frame, "Concluido! Pressiona Q para sair.", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Coletor de Dados', frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('s'):
        collecting = True
        counter = 0
        data = []
        print("A gravar...")
    elif key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Guardar em CSV
if data:
    df = pd.DataFrame(data)

    if os.path.exists(OUTPUT_FILE):
        df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
    else:
        columns = ['label', 'hand'] + [f'lm_{i}_{c}' for i in range(21) for c in ['x', 'y', 'z']]
        df.columns = columns
        df.to_csv(OUTPUT_FILE, index=False)

    print(f"Dados guardados em {OUTPUT_FILE}")