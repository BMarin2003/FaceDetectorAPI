import cv2
import numpy as np
from typing import Optional
from fastapi import UploadFile, HTTPException
from skimage.feature import local_binary_pattern
from skimage import io
import tempfile
import os

async def extract_embedding_from_image(file: UploadFile) -> Optional[dict]:
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="El contenido del archivo está vacío")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        img = cv2.imread(temp_file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise HTTPException(status_code=400, detail="No se pudo leer la imagen")

        img = cv2.equalizeHist(img)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            raise HTTPException(status_code=400, detail="No se detectaron rostros en la imagen")

        (x, y, w, h) = faces[0]
        face_img = img[y:y+h, x:x+w]
        resized = cv2.resize(face_img, (128, 128))
        cropped_temp_file_path = temp_file_path + '_cropped.jpg'
        cv2.imwrite(cropped_temp_file_path, resized)

        radius = 3
        n_points = 8 * radius
        lbp_features = extract_lbp_features(cropped_temp_file_path, radius, n_points)

        os.unlink(temp_file_path)
        os.unlink(cropped_temp_file_path)

        if lbp_features is None:
            return None

        # Convertir a lista con máxima precisión
        lbp_features_list = [float(x) for x in lbp_features]

        # Imprimir detalles del embedding extraído
        print(f"DEBUG: Longitud del embedding extraído: {len(lbp_features_list)}")
        print(f"DEBUG: Primeros 5 valores del embedding extraído:")
        for i, val in enumerate(lbp_features_list[:5]):
            print(f"  Valor {i}: {val}")
        print(f"DEBUG: Últimos 5 valores del embedding extraído:")
        for i, val in enumerate(lbp_features_list[-5:], start=len(lbp_features_list)-5):
            print(f"  Valor {i}: {val}")

        return {"embedding": lbp_features_list}
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error al extraer el embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Error al extraer el embedding: {e}")

def extract_lbp_features(image_path: str, radius: int, n_points: int) -> np.ndarray:
    img = io.imread(image_path, as_gray=True)
    lbp = local_binary_pattern(img, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype(np.float64)
    hist /= (hist.sum() + 1e-7)
    return hist

