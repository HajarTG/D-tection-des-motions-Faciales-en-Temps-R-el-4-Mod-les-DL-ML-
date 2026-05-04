import customtkinter as ctk
import cv2
import numpy as np
from tensorflow import keras
from PIL import Image, ImageTk
import threading
from style_config import PALETTE, STYLES,EMOTION_COLORS

ctk.set_appearance_mode("light")  # "dark" ou "light"
ctk.set_default_color_theme("green")  # "blue", "green", "dark-blue"

class EmotionDetectionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Détection d'Émotions - Pro")
        self.geometry("900x600")
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Chargement du modèle
        self.model = keras.models.load_model(r'C:\Users\Lenovo\Desktop\Projet_faciale emotion recognatiion\CNN V2\best_CNNModel.keras')
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_labels = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']
        self.IMG_SIZE = 128

        # Variables
        self.is_camera_on = False
        self.cap = None
        self.thread = None

        # UI
        self.create_widgets()

    def create_widgets(self):
        # Titre
        self.title_label = ctk.CTkLabel(self, text="🧠 Système de Détection d'Émotions", font=ctk.CTkFont(size=22, weight="bold"))
        self.title_label.grid(row=0, column=0, columnspan=2, pady=(10, 5), sticky="ew")

        # Zone vidéo
        self.video_label = ctk.CTkLabel(self, text="", width=520, height=400, corner_radius=12, fg_color="#e8f5e9")
        self.video_label.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Panneau de contrôle
        self.control_frame = ctk.CTkFrame(self, corner_radius=12, width=220, fg_color="#f1f8e9")
        self.control_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        self.control_frame.grid_rowconfigure(7, weight=1)

        # Bouton caméra
        self.camera_button = ctk.CTkButton(self.control_frame, text="Démarrer la Caméra", command=self.toggle_camera, corner_radius=8, font=ctk.CTkFont(size=14, weight="bold"), height=36)
        self.camera_button.grid(row=0, column=0, pady=10, sticky="ew")

        # Spinner (animation de chargement)
        self.spinner = ctk.CTkProgressBar(self.control_frame, mode="indeterminate")
        self.spinner.grid(row=1, column=0, pady=10, sticky="ew")
        self.spinner.grid_remove()

        # Infos
        self.face_count_label = ctk.CTkLabel(self.control_frame, text="Nombre de visages: 0", font=ctk.CTkFont(size=16, weight="bold"))
        self.face_count_label.grid(row=2, column=0, pady=10, sticky="ew")
        self.model_label = ctk.CTkLabel(self.control_frame, text="Modèle: CNN V2", font=ctk.CTkFont(size=16, weight="bold"))
        self.model_label.grid(row=3, column=0, pady=10, sticky="ew")

        # Barres d'émotions
        self.bars_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        self.bars_frame.grid(row=4, column=0, pady=10, sticky="nsew")
        self.emotion_bars = []
        for i, emo in enumerate(self.emotion_labels):
            color = EMOTION_COLORS.get(emo, "#90A4AE")
            bar = ctk.CTkProgressBar(self.bars_frame, width=180, height=16, corner_radius=8, progress_color=color)
            bar.grid(row=i, column=1, padx=5, pady=3)
            label = ctk.CTkLabel(self.bars_frame, text=emo.capitalize(), width=80, font=ctk.CTkFont(size=14, weight="bold"))
            label.grid(row=i, column=0, sticky="w")
            value = ctk.CTkLabel(self.bars_frame, text="0.00", font=ctk.CTkFont(size=14, weight="bold"))
            value.grid(row=i, column=2, sticky="e", padx=(10, 0))
            self.emotion_bars.append((bar, value))

        # Instructions
        self.instructions = ctk.CTkLabel(
            self.control_frame,
            text="1. Cliquez sur 'Démarrer la Caméra'\n"
                 "2. Regardez la caméra\n"
                 "3. Les émotions seront détectées\n"
                 "4. Cliquez à nouveau pour arrêter",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#1B2631",
            justify="left"
        )
        self.instructions.grid(row=6, column=0, pady=(20, 10), sticky="ew")

    def toggle_camera(self):
        if not self.is_camera_on:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        self.is_camera_on = True
        self.camera_button.configure(text="Arrêter la Caméra")
        self.spinner.grid()
        self.spinner.start()
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.thread = threading.Thread(target=self.update_frame)
        self.thread.daemon = True
        self.thread.start()

    def stop_camera(self):
        self.is_camera_on = False
        self.camera_button.configure(text="Démarrer la Caméra")
        self.spinner.stop()
        self.spinner.grid_remove()
        if self.cap is not None:
            self.cap.release()
        self.video_label.configure(image=None)
        for bar, value in self.emotion_bars:
            bar.set(0)
            value.configure(text="0.00")

    def update_frame(self):
        while self.is_camera_on:
            ret, frame = self.cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            self.face_count_label.configure(text=f"Nombre de visages: {len(faces)}")
            predictions = None
            for (x, y, w, h) in faces:
                roi = frame[y:y+h, x:x+w]
                roi = cv2.resize(roi, (100, 100))  # Redimensionne en 100x100
                # S'assurer que c’est en couleur (3 canaux)
                if len(roi.shape) == 2 or roi.shape[2] != 3:
                  roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                roi = roi.astype('float32') / 255.0  # Normalisation si nécessaire
                roi = np.expand_dims(roi, axis=0)   # Ajoute la dimension batch

                prediction = self.model.predict(roi)[0]
                predictions = prediction
                label = self.emotion_labels[np.argmax(prediction)]
                # Utilise la même couleur pour la barre, le rectangle et le texte
                hex_color = EMOTION_COLORS.get(label, "#90A4AE")
                # Convertit le hex en RGB pour OpenCV (attention à l'ordre BGR !)
                bgr_color = tuple(int(hex_color[i:i+2], 16) for i in (5, 3, 1))
                cv2.rectangle(frame, (x, y), (x+w, y+h), bgr_color, 6)
                cv2.putText(frame, f'{label}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, bgr_color, 3)
            # Animation barres d'émotions
            if predictions is not None:
                for i, (bar, value) in enumerate(self.emotion_bars):
                    bar.set(float(predictions[i]))
                    value.configure(text=f"{predictions[i]:.2f}")
            else:
                for bar, value in self.emotion_bars:
                    bar.set(0)
                    value.configure(text="0.00")
            # Affichage vidéo
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image = image.resize((800, 600))
            photo = ImageTk.PhotoImage(image=image)
            self.video_label.configure(image=photo)
            self.video_label.image = photo
            self.after(10)


if __name__ == "__main__":
    app = EmotionDetectionApp()
    app.mainloop() 