import json
import os
from typing import Optional

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from app.services.db import get_all_users_with_embeddings
from app.services.storage import download_file_from_r2


class FaceKNNModel:
    def __init__(self, model_path: str = "knn_model.npz"):
        self.model_path = model_path
        self.knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            data = np.load(self.model_path, allow_pickle=True)
            self.knn = data['model']
            self.label_encoder = data['label_encoder']
            self.is_trained = True

    def save_model(self):
        np.savez(self.model_path, model=self.knn, label_encoder=self.label_encoder)

    async def rebuild_training_data(self):
        try:
            users = await get_all_users_with_embeddings()
            X_train = []
            y_train = []

            for user in users:
                user_id = user['id']
                kp_url = user['KP']

                # Descargar el embedding almacenado
                temp_file_path = await download_file_from_r2(kp_url, os.getenv('BUCKET_KPS'))
                with open(temp_file_path, 'r') as f:
                    stored_embedding = json.load(f)

                # Extraer el embedding del diccionario
                stored_embedding_array = np.array(stored_embedding['embedding'])
                X_train.append(stored_embedding_array)
                y_train.append(user_id)

                # Eliminar el archivo temporal
                os.unlink(temp_file_path)

            if not X_train:
                print("No hay datos de entrenamiento")
                return

            X_train = np.array(X_train)
            y_train = np.array(y_train)

            # Ajustar el label encoder
            self.label_encoder.fit(y_train)
            self.is_trained = False

        except Exception as e:
            print(f"Error al reconstruir los datos de entrenamiento: {e}")

    async def train_model(self):
        try:
            users = await get_all_users_with_embeddings()
            X_train = []
            y_train = []

            for user in users:
                user_id = user['id']
                kp_url = user['KP']

                # Descargar el embedding almacenado
                temp_file_path = await download_file_from_r2(kp_url, os.getenv('BUCKET_KPS'))
                with open(temp_file_path, 'r') as f:
                    stored_embedding = json.load(f)

                # Extraer el embedding del diccionario
                stored_embedding_array = np.array(stored_embedding['embedding'])
                X_train.append(stored_embedding_array)
                y_train.append(user_id)

                # Eliminar el archivo temporal
                os.unlink(temp_file_path)

            if not X_train:
                print("No hay datos de entrenamiento")
                return

            X_train = np.array(X_train)
            y_train = self.label_encoder.transform(np.array(y_train))

            # Entrenar el modelo KNN
            self.knn.fit(X_train, y_train)

            # Guardar el modelo entrenado
            self.save_model()
            self.is_trained = True

        except Exception as e:
            print(f"Error al entrenar el modelo: {e}")

    def predict(self, embedding: np.ndarray) -> Optional[int]:
        if not self.is_trained:
            return None

        # Predecir la clase del embedding
        predicted_label = self.knn.predict([embedding])[0]
        return self.label_encoder.inverse_transform([predicted_label])[0]

# Instanciar el modelo
knn_model = FaceKNNModel()