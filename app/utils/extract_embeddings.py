import cv2
import numpy as np
from typing import Optional
from fastapi import UploadFile

async def extract_embedding_from_image(file: UploadFile) -> Optional[dict]:
    try:
        # Leer el contenido del archivo
        content = await file.read()
        # Verificar que el contenido no esté vacío
        if not content:
            print("El contenido del archivo está vacío")
            return None
        # Convertir el contenido a un array de numpy
        nparr = np.frombuffer(content, np.uint8)
        # Verificar que el array no esté vacío
        if nparr.size == 0:
            print("El array de la imagen está vacío")
            return None
        # Decodificar la imagen
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Verificar que la imagen se haya decodificado correctamente
        if img is None:
            print("No se pudo decodificar la imagen")
            return None
        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Ecualización de histograma para mejorar el contraste
        gray = cv2.equalizeHist(gray)
        # Detectar rostros en la imagen
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            print("No se detectaron rostros en la imagen")
            return None
        # Recortar la región del rostro
        (x, y, w, h) = faces[0]
        face_img = gray[y:y+h, x:x+w]
        # Redimensionar la imagen del rostro
        resized = cv2.resize(face_img, (128, 128))
        # Extraer características utilizando LBP
        radius = 3
        n_points = 8 * radius
        lbp = cv2.LBPHFaceRecognizer_create(radius, n_points)
        lbp_features = lbp.compute(resized)
        # Normalizar las características
        normalized_features = lbp_features / 255.0
        # Añadir una dimensión para el canal (grises)
        embedding = {"embedding": normalized_features.tolist()}
        return embedding
    except Exception as e:
        print(f"Error al extraer el embedding: {e}")
        return None
