import json
import os
from typing import Optional
import numpy as np
from joblib import dump, load
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from app.services.db import get_all_users_with_embeddings
from app.services.storage import download_file_from_r2

def get_model_path():
    module_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(module_dir, "knn_model.jobl")

class FaceKNNModel:
    def __init__(self, model_path: str = None):
        if model_path is None:
            self.model_path = get_model_path()
        else:
            self.model_path = model_path
        self.knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self._load_model()

    def _load_model(self):
        model_path_abs = os.path.abspath(self.model_path)

        if os.path.exists(model_path_abs):
            try:
                if not os.access(model_path_abs, os.R_OK):
                    self.is_trained = False
                    return

                data = load(model_path_abs)
                self.knn = data['model']
                self.label_encoder = data['label_encoder']
                self.is_trained = data.get('is_trained', False)
            except Exception as e:
                print(f"DEBUG: Error al cargar el modelo: {str(e)}")
                self.is_trained = False
        else:
            print(f"DEBUG: El archivo del modelo no existe en {model_path_abs}")
            self.is_trained = False

    def save_model(self):
        # Asegurarse de que el directorio existe (aunque ya debería existir)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        dump({
            'model': self.knn,
            'label_encoder': self.label_encoder,
            'is_trained': self.is_trained
        }, self.model_path)
        print(f"DEBUG: Modelo guardado en {os.path.abspath(self.model_path)}")

    async def train_model(self):
        try:
            users = await get_all_users_with_embeddings()
            if not users:
                self.is_trained = False
                return

            X_train = []
            y_train = []

            for user in users:
                user_id = user['id']
                kp_url = user['KP']
                try:
                    temp_file_path = await download_file_from_r2(kp_url, os.getenv('BUCKET_KPS'))
                    with open(temp_file_path, 'r') as f:
                        stored_embedding = json.load(f)
                    # Usar dtype=np.float64 para mantener la máxima precisión
                    stored_embedding_array = np.array(stored_embedding['embedding'], dtype=np.float64)
                    X_train.append(stored_embedding_array)
                    y_train.append(user_id)
                    os.unlink(temp_file_path)
                except Exception as e:
                    print(f"Error al procesar usuario {user_id}: {str(e)}")
                    continue

            if not X_train:
                self.is_trained = False
                return

            # TERCERO: Debug y entrenamiento (FUERA del ciclo)
            print(f"DEBUG: y_train antes de encoding: {y_train}")
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            print(f"DEBUG: y_train después de encoding: {y_train_encoded}")
            for original, encoded in zip(y_train, y_train_encoded):
                print(f"DEBUG: User ID {original} -> Label {encoded}")

            X_train = np.array(X_train, dtype=np.float64)
            self.knn.fit(X_train, y_train_encoded)
            self.is_trained = True
            self.save_model()

        except Exception as e:
            print(f"Error al entrenar el modelo: {str(e)}")
            self.is_trained = False

    def predict(self, embedding: np.ndarray) -> Optional[int]:
        print(f"DEBUG: Inicio de predicción. Modelo entrenado: {self.is_trained}")
        if not self.is_trained:
            print("DEBUG: Modelo no entrenado")
            return None

        if len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1)
            print(f"DEBUG: Embedding reshaped a: {embedding.shape}")

        try:
            # Obtener distancias y índices de los vecinos más cercanos
            distances, indices = self.knn.kneighbors(embedding, n_neighbors=1)
            print(f"DEBUG: Distancias a vecinos más cercanos: {distances[0]}")
            print(f"DEBUG: Índices de vecinos más cercanos: {indices[0]}")

            # Acceder a las etiquetas de forma correcta
            training_labels = self.knn._y if hasattr(self.knn, '_y') else []
            if training_labels is not None and len(training_labels) > 0:
                neighbor_labels = [training_labels[idx] for idx in indices[0] if idx < len(training_labels)]
                print(f"DEBUG: Etiquetas de vecinos más cercanos: {neighbor_labels}")

            predicted_label = self.knn.predict(embedding)[0]
            print(f"DEBUG: Etiqueta predicha: {predicted_label}")
            predicted_user_id = self.label_encoder.inverse_transform([predicted_label])[0]
            print(f"DEBUG: ID de usuario predicho: {predicted_user_id}")
            return predicted_user_id
        except Exception as e:
            print(f"DEBUG: Error en predicción: {str(e)}")
            return None

# Instanciar el modelo con la ruta correcta
knn_model = FaceKNNModel()