import cv2
import joblib
import numpy as np

# Charger le modèle SVM et le scaler de normalisation
model = joblib.load("svm_model_rbf_emotion.pkl")
scaler = joblib.load("scaler_svm_emotion.pkl")

# Liste des émotions (labels de 1 à 7)
classes = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']

# Définir les couleurs BGR associées à chaque émotion
emotion_colors = {
    'surprise': (0, 255, 255),    # Jaune
    'fear': (255, 0, 0),          # Bleu
    'disgust': (0, 0, 128),       # Marron
    'happy': (0, 255, 0),         # Vert
    'sad': (255, 0, 255),         # Violet
    'angry': (0, 0, 255),         # Rouge
    'neutral': (128, 128, 128)    # Gris
}

# Initialiser le détecteur de visage Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Démarrer la capture vidéo
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Nom du modèle à afficher
MODEL_NAME = 'SVM_model'

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Conversion en niveaux de gris pour la détection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Conversion en RGB pour le traitement
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Détection des visages
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Afficher le nombre de visages détectés
    cv2.putText(frame, f'Number of Faces : {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 255), 2)

    for idx, (x, y, w, h) in enumerate(faces):
        # Extraire la région du visage et la redimensionner
        face_rgb = rgb_frame[y:y+h, x:x+w]
        resized_face = cv2.resize(face_rgb, (48, 48), interpolation=cv2.INTER_AREA)

        # Mise en forme pour SVM (1D, normalisé)
        face_array = np.array(resized_face).reshape(1, -1)
        normalized_face = scaler.transform(face_array)

        # Prédiction avec le modèle SVM
        prediction = model.predict(normalized_face)
        emotion_idx = prediction[0] - 1  # Si labels de 1 à 7
        emotion = classes[emotion_idx]

        # Couleur et affichage
        color = emotion_colors.get(emotion, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f'{emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Affichage du nom du modèle
    cv2.putText(frame, f"Model: {MODEL_NAME}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Afficher la fenêtre
    cv2.imshow('Emotion Detection - SVM', frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
