import os
import sys
import asyncio

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from app.models.knn_model import FaceKNNModel

async def train_model():
    knn_model = FaceKNNModel()
    print("DEBUG: Comenzando entrenamiento del modelo...")
    await knn_model.train_model()
    print("El modelo KNN ha sido entrenado y guardado correctamente.")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    asyncio.run(train_model())