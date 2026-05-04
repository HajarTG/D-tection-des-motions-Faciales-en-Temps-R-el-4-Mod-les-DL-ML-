import customtkinter as ctk
import cv2
import numpy as np
import joblib
from PIL import Image
import threading

# Use CTkImage instead of ImageTk to fix the HighDPI warning
from customtkinter import CTkImage


ctk.set_appearance_mode("light")
ctk.set_default_color_theme("green")

# Define emotion colors
EMOTION_COLORS = {
    "surprise": "#FFA726",  # Orange
    "fear": "#9C27B0",      # Purple
    "disgust": "#66BB6A",   # Green
    "happy": "#FDD835",     # Yellow
    "sad": "#42A5F5",       # Blue
    "angry": "#EF5350",     # Red
    "neutral": "#90A4AE"    # Gray
}

class EmotionDetectionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Détection d'Émotions - SVM")
        self.geometry("900x600")
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Load the SVM model and scaler
        try:
            self.model = joblib.load(r"C:\Users\Lenovo\Desktop\Projet_faciale emotion recognatiion\SVM\svm_model_rbf_emotion.pkl")
            self.scaler = joblib.load(r"C:\Users\Lenovo\Desktop\Projet_faciale emotion recognatiion\SVM\scaler_svm_emotion.pkl")
            
            # Print information about the model
            if hasattr(self.model, "coef_"):
                print("Model is expecting features of size:", self.model.coef_.shape[1])
            else:
                print("Model loaded successfully, but couldn't determine feature size")
                
        except Exception as e:
            print(f"Error loading model or scaler: {e}")
            # This would be better handled with a proper error dialog

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Define emotion labels - ensure these match your model's expected outputs
        self.emotion_labels =  ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']
        self.IMG_SIZE = 48  # Make sure this matches your model's expected input size

        self.is_camera_on = False
        self.cap = None
        self.thread = None

        self.create_widgets()

    def create_widgets(self):
        self.title_label = ctk.CTkLabel(self, text="🧠 Système de Détection d'Émotions", font=ctk.CTkFont(size=22, weight="bold"))
        self.title_label.grid(row=0, column=0, columnspan=2, pady=(10, 5), sticky="ew")

        self.video_label = ctk.CTkLabel(self, text="", width=520, height=400, corner_radius=12, fg_color="#e8f5e9")
        self.video_label.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.control_frame = ctk.CTkFrame(self, corner_radius=12, width=220, fg_color="#f1f8e9")
        self.control_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        self.control_frame.grid_rowconfigure(7, weight=1)

        self.camera_button = ctk.CTkButton(self.control_frame, text="Démarrer la Caméra", command=self.toggle_camera,
                                           corner_radius=8, font=ctk.CTkFont(size=14, weight="bold"), height=36)
        self.camera_button.grid(row=0, column=0, pady=10, padx=10, sticky="ew")

        self.spinner = ctk.CTkProgressBar(self.control_frame, mode="indeterminate")
        self.spinner.grid(row=1, column=0, pady=10, padx=10, sticky="ew")
        self.spinner.grid_remove()

        self.face_count_label = ctk.CTkLabel(self.control_frame, text="Nombre de visages: 0", font=ctk.CTkFont(size=16, weight="bold"))
        self.face_count_label.grid(row=2, column=0, pady=10, padx=10, sticky="ew")

        self.model_label = ctk.CTkLabel(self.control_frame, text="Modèle: SVM", font=ctk.CTkFont(size=16, weight="bold"))
        self.model_label.grid(row=3, column=0, pady=10, padx=10, sticky="ew")

        self.bars_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        self.bars_frame.grid(row=4, column=0, pady=10, padx=10, sticky="nsew")

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
        self.instructions.grid(row=6, column=0, pady=(20, 10), padx=10, sticky="ew")

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
            try:
                ret, frame = self.cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                
                # Update face count in the UI thread
                self.after(0, lambda: self.face_count_label.configure(text=f"Nombre de visages: {len(faces)}"))

                # Default predictions to None
                predictions = None

                for (x, y, w, h) in faces:
                    try:
                        # Extract face ROI - use color (RGB) image as the model expects
                        face_img = frame[y:y+h, x:x+w]
                        if face_img.size == 0:  # Skip empty ROIs
                            continue
                            
                        # Resize to expected input size
                        resized_face = cv2.resize(face_img, (self.IMG_SIZE, self.IMG_SIZE))
                        
                        # Flatten the RGB image (48x48x3 = 6912 features)
                        # OpenCV uses BGR, but we'll leave it as is unless model explicitly needs RGB
                        flattened_face = resized_face.flatten()
                        
                        # Check dimensions before prediction
                        print(f"Feature vector size: {flattened_face.shape[0]}")
                        
                        # Apply the scaler transformation
                        scaled_face = self.scaler.transform([flattened_face])
                        
                        # Make prediction
                        if hasattr(self.model, "predict_proba"):
                            prediction = self.model.predict_proba(scaled_face)[0]
                            predicted_class = np.argmax(prediction)
                        else:
                            predicted_class = self.model.predict(scaled_face)[0]
                            # Ensure predicted_class is valid
                            if predicted_class >= len(self.emotion_labels) or predicted_class < 0:
                                print(f"Warning: Invalid predicted class {predicted_class}, using default class 0")
                                predicted_class = 0
                                
                            # Create one-hot encoding for the prediction
                            prediction = np.zeros(len(self.emotion_labels))
                            prediction[predicted_class] = 1.0

                        emotion_label = self.emotion_labels[predicted_class]
                        predictions = prediction  # Update bars with this prediction

                        # Get color for detected emotion
                        hex_color = EMOTION_COLORS.get(emotion_label, "#90A4AE")

                        # Convert hex -> BGR (OpenCV uses BGR format)
                        b = int(hex_color[5:7], 16)
                        g = int(hex_color[3:5], 16)
                        r = int(hex_color[1:3], 16)
                        bgr_color = (b, g, r)

                        # Draw rectangle and label
                        cv2.rectangle(frame, (x, y), (x+w, y+h), bgr_color, 6)
                        cv2.putText(frame, f'{emotion_label}', (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, bgr_color, 3)
                                  
                    except Exception as e:
                        print(f"Error processing face: {e}")
                        continue

                # Update emotion bars in UI thread
                if predictions is not None:
                    def update_bars(pred):
                        for i, (bar, value) in enumerate(self.emotion_bars):
                            if i < len(pred):
                                bar.set(float(pred[i]))
                                value.configure(text=f"{pred[i]:.2f}")
                    self.after(0, lambda: update_bars(predictions))
                
                # Convert frame to RGB (PIL uses RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                
                # Resize the image to fit the label
                image_resized = image.resize((500, 375))
                
                # Convert to CTkImage instead of PhotoImage to avoid warnings
                photo = CTkImage(light_image=image_resized, size=(500, 375))
                
                # Update the video label in the UI thread
                self.after(0, lambda p=photo: self.video_label.configure(image=p))
                
                # Store reference to prevent garbage collection
                self.video_label.image = photo
                
            except Exception as e:
                print(f"Error in update_frame: {e}")
                import traceback
                traceback.print_exc()
                # Continue processing despite errors
                
            # Small delay to prevent hogging CPU
            self.after(10)

if __name__ == "__main__":
    app = EmotionDetectionApp()
    app.mainloop()