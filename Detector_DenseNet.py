import cv2
import numpy as np
from tensorflow import keras

IMG_SIZE = 128
MODEL_NAME = 'DenseNet_Model'
emotion_labels = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']

model = keras.models.load_model(r'C:\Users\Lenovo\Desktop\Projet_faciale emotion recognatiion\DenseNet\best_DenseNetModel.keras')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    
    cv2.putText(frame, f'Number of Faces : {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 255), 2)

    for idx, (x, y, w, h) in enumerate(faces):
        roi_color = frame[y:y+h, x:x+w]  # Prend le ROI couleur
        roi = cv2.resize(roi_color, (IMG_SIZE, IMG_SIZE))
        roi = roi.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=0)  # (1, 100, 100, 3)
        prediction = model.predict(roi)[0]
        label = emotion_labels[np.argmax(prediction)]

        # Rectangle et label principal
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f'{label}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # Affichage des scores pour chaque émotion
        for i, (emo, score) in enumerate(zip(emotion_labels, prediction)):
            txt = f"{emo}: {score * 100:.1f}%"
            cv2.putText(frame, txt, (x + w + 10, y + 25 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 128), 2)

    # Afficher le nom du modèle en bas
    cv2.putText(frame, f"Model: {MODEL_NAME}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow('Emotion Detection - DenseNet', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
