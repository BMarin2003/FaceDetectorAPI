import json
import os
import uuid
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, FastAPI
from sklearn.metrics.pairwise import cosine_similarity

from app.models.knn_model import FaceKNNModel
from app.schemas.user import UserInDB
from app.services import db, storage
from app.utils.extract_embeddings import extract_embedding_from_image

router = APIRouter()



# Instanciar el modelo KNN
knn_model = FaceKNNModel()
print(f"Modelo KNN inicializado correctamente")

@router.get("/")
async def read_root():
    return {"message": "Bienvenido a la API de FaceDetector"}

async def initialize_model():
    if not knn_model.is_trained:
        print("Entrenando modelo KNN...")
        await knn_model.train_model()
        print("Modelo KNN entrenado exitosamente")
    else:
        print("Modelo KNN ya estaba entrenado")

@router.on_event("startup")
async def startup_event():
    await initialize_model()

@router.post("/users/", response_model=dict)
async def create_user(
        upaoID: int = Form(...),
        nombres: str = Form(...),
        apellidos: str = Form(...),
        correo: str = Form(...),
        requisitoriado: bool = Form(False),
        foto: UploadFile = File(...)
):
    try:
        if not foto.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo subido no es una imagen")

        content = await foto.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="El archivo subido está vacío")

        # Procesar nombres para archivos
        nombres_sin_espacios = nombres.replace(" ", "_")
        apellidos_sin_espacios = apellidos.replace(" ", "_")
        filename_photo = f"{nombres_sin_espacios}_{apellidos_sin_espacios}.jpg"

        # Subir foto
        foto_url = await storage.upload_file_to_r2(foto, bucket=os.getenv('BUCKET_PHOTOS'), filename=filename_photo)

        # Crear usuario
        user_data = {
            "upaoID": upaoID,
            "nombres": nombres,
            "apellidos": apellidos,
            "correo": correo,
            "requisitoriado": requisitoriado,
        }
        result = await db.create_user(user_data)
        user_id = result['result'][0]['meta']['last_row_id']

        # Extraer embedding
        await foto.seek(0)
        embedding = await extract_embedding_from_image(foto)
        if not embedding:
            raise HTTPException(status_code=400, detail="No se detectaron rostros en la imagen")

        # Subir embedding
        embedding_filename = f"{upaoID}_{uuid.uuid4().hex[:8]}.json"
        kp_url = await storage.upload_json_to_r2(
            embedding,
            bucket=os.getenv('BUCKET_KPS'),
            filename=embedding_filename
        )

        # Actualizar usuario con URLs
        await db.add_face_photo(user_id, foto_url, kp_url)
        await db.update_user(user_id, {"foto": foto_url, "KP": kp_url})

        # Reentrenar modelo
        await knn_model.train_model()
        print(f"Usuario {nombres} {apellidos} creado exitosamente")

        return {"message": "Usuario creado exitosamente", "user_id": user_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al crear el usuario: {e}")


@router.get("/users/{user_id}", response_model=UserInDB)
async def get_user(user_id: int):
    user = await db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return user


@router.get("/users/", response_model=List[UserInDB])
async def list_users():
    users = await db.list_users()
    return users


@router.get("/users/{user_id}/photos", response_model=List[dict])
async def list_photos(user_id: int):
    user = await db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    faces = await db.get_faces_by_user_id(user_id)
    photos = []
    for face in faces:
        photos.append({
            "id": face['id'],
            "foto_url": face['foto'],
            "kp_url": face['KP']
        })
    return photos


@router.put("/users/{user_id}", response_model=dict)
async def update_user(
        user_id: int,
        nombres: Optional[str] = None,
        apellidos: Optional[str] = None,
        correo: Optional[str] = None,
        requisitoriado: Optional[bool] = None,
        conservar: Optional[bool] = False,
        foto: UploadFile = File(None)
):
    user = await db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    update_data = {}
    if nombres is not None:
        update_data['nombres'] = nombres
    if apellidos is not None:
        update_data['apellidos'] = apellidos
    if correo is not None:
        update_data['correo'] = correo
    if requisitoriado is not None:
        update_data['requisitoriado'] = requisitoriado

    if foto:
        if not foto.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo subido no es una imagen")

        content = await foto.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="El archivo subido está vacío")

        # Procesar nueva foto
        nombres_sin_espacios = user['nombres'].replace(" ", "_")
        apellidos_sin_espacios = user['apellidos'].replace(" ", "_")
        filename_photo = f"{nombres_sin_espacios}_{apellidos_sin_espacios}_{uuid.uuid4().hex[:8]}.jpg"
        foto_url = await storage.upload_file_to_r2(foto, bucket=os.getenv('BUCKET_PHOTOS'), filename=filename_photo)

        # Extraer embedding
        await foto.seek(0)
        embedding = await extract_embedding_from_image(foto)
        if not embedding:
            raise HTTPException(status_code=400, detail="No se detectaron rostros en la imagen")

        embedding_filename = f"{user_id}_{uuid.uuid4().hex[:8]}.json"
        kp_url = await storage.upload_json_to_r2(
            embedding,
            bucket=os.getenv('BUCKET_KPS'),
            filename=embedding_filename
        )

        # Eliminar foto anterior si no se conserva
        if not conservar and user.get('foto') and user.get('KP'):
            await delete_photo_files(user_id, user['foto'], user['KP'])

        update_data['foto'] = foto_url
        update_data['KP'] = kp_url
        await db.add_face_photo(user_id, foto_url, kp_url)
        await knn_model.train_model()

    await db.update_user(user_id, update_data)
    return {"message": "Usuario actualizado exitosamente"}


@router.put("/users/{user_id}/change_profile_photo/{photo_id}", response_model=dict)
async def change_profile_photo(user_id: int, photo_id: int):
    user = await db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    faces = await db.get_faces_by_user_id(user_id)
    photo_to_set = None
    for face in faces:
        if face['id'] == photo_id:
            photo_to_set = face
            break

    if not photo_to_set:
        raise HTTPException(status_code=404, detail="Foto no encontrada")

    update_data = {
        "foto": photo_to_set['foto'],
        "KP": photo_to_set['KP']
    }
    await db.update_user(user_id, update_data)
    await knn_model.train_model()

    return {"message": "Foto de perfil actualizada exitosamente"}


async def delete_photo_files(user_id: int, foto_url: str, kp_url: str):
    await storage.delete_file_from_r2(foto_url, os.getenv('BUCKET_PHOTOS'))
    await storage.delete_file_from_r2(kp_url, os.getenv('BUCKET_KPS'))

    faces = await db.get_faces_by_user_id(user_id)
    for face in faces:
        if face['foto'] == foto_url and face['KP'] == kp_url:
            await db.delete_face(face['id'])
            break


@router.delete("/users/{user_id}", response_model=dict)
async def delete_user(user_id: int):
    user = await db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    faces = await db.get_faces_by_user_id(user_id)
    for face in faces:
        await storage.delete_file_from_r2(face['foto'], os.getenv('BUCKET_PHOTOS'))
        await storage.delete_file_from_r2(face['KP'], os.getenv('BUCKET_KPS'))
        await db.delete_face(face['id'])

    await db.delete_user(user_id)
    await knn_model.train_model()

    return {"message": "Usuario eliminado exitosamente"}


@router.delete("/users/photo/{user_id}/{photo_id}", response_model=dict)
async def delete_photo(user_id: int, photo_id: int):
    faces = await db.get_faces_by_user_id(user_id)
    photo_to_delete = None
    for face in faces:
        if face['id'] == photo_id:
            photo_to_delete = face
            break

    if not photo_to_delete:
        raise HTTPException(status_code=404, detail="Foto no encontrada")

    await storage.delete_file_from_r2(photo_to_delete['foto'], os.getenv('BUCKET_PHOTOS'))
    await storage.delete_file_from_r2(photo_to_delete['KP'], os.getenv('BUCKET_KPS'))
    await db.delete_face(photo_id)
    await knn_model.train_model()

    return {"message": "Foto eliminada exitosamente"}


@router.post("/users/photo/")
async def add_photo_to_user(usuario_id: int = Form(...), file: UploadFile = File(...)):
    user = await db.get_user_by_id(usuario_id)
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="El archivo subido no es una imagen")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="El archivo subido está vacío")

    nombres_sin_espacios = user['nombres'].replace(" ", "_")
    apellidos_sin_espacios = user['apellidos'].replace(" ", "_")
    filename_photo = f"{nombres_sin_espacios}_{apellidos_sin_espacios}_{uuid.uuid4().hex[:8]}.jpg"
    foto_url = await storage.upload_file_to_r2(file, bucket=os.getenv('BUCKET_PHOTOS'), filename=filename_photo)

    await file.seek(0)
    embedding = await extract_embedding_from_image(file)
    if not embedding:
        raise HTTPException(status_code=400, detail="No se detectaron rostros en la imagen")

    embedding_filename = f"{usuario_id}_{uuid.uuid4().hex[:8]}.json"
    kp_url = await storage.upload_json_to_r2(
        embedding,
        bucket=os.getenv('BUCKET_KPS'),
        filename=embedding_filename
    )

    await db.add_face_photo(usuario_id, foto_url, kp_url)
    await knn_model.train_model()

    return {"message": "Foto agregada exitosamente", "foto_url": foto_url}

async def simulate_authority_notification(user_data: dict) -> dict:
    notification_data = {
        "alert_type": "SECURITY_ALERT",
        "timestamp": "2025-06-21T10:30:00Z",
        "detected_user": {
            "id": user_data['id'],
            "upaoID": user_data['upaoID'],
            "nombres": user_data['nombres'],
            "apellidos": user_data['apellidos'],
            "correo": user_data['correo']
        },
        "authority_notification": {
            "status": "SIMULATED",
            "message": "Notificación enviada a la Policía Nacional del Perú (SIMULADO)",
            "reference_code": f"REQ-{user_data['id']}-{uuid.uuid4().hex[:8].upper()}",
            "priority": "URGENT"
        },
        "recommended_actions": [
            "Verificar identidad del individuo",
            "Contactar a seguridad del campus",
            "Mantener distancia de seguridad",
            "Documentar el incidente"
        ]
    }

    print(
        f"🚨 ALERTA DE SEGURIDAD: Usuario requisitoriado detectado - {user_data['nombres']} {user_data['apellidos']} (ID: {user_data['id']})")
    print(
        f"📞 Notificación simulada enviada con código: {notification_data['authority_notification']['reference_code']}")

    return notification_data


@router.post("/compare/")
async def compare_external_image(file: UploadFile = File(...)):
    try:
        # Extraer embedding de la imagen externa
        embedding_result = await extract_embedding_from_image(file)
        if embedding_result is None:
            raise HTTPException(status_code=400, detail="No se detectaron rostros en la imagen")

        # Convertir a array numpy
        embedding_array = np.array(embedding_result['embedding'], dtype=np.float64)
        if len(embedding_array.shape) == 1:
            embedding_array = embedding_array.reshape(1, -1)

        # Predecir usando el modelo KNN
        predicted_user_id = knn_model.predict(embedding_array)
        if predicted_user_id is None:
            raise HTTPException(status_code=404, detail="No se encontró ninguna coincidencia en la base de datos")

        predicted_user_id = int(predicted_user_id)
        matched_user = await db.get_user_by_id(predicted_user_id)

        if not matched_user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado en la base de datos")

        # Calcular similitud con el usuario encontrado
        kp_url = matched_user['KP']
        temp_file_path = await storage.download_file_from_r2(kp_url, os.getenv('BUCKET_KPS'))
        with open(temp_file_path, 'r') as f:
            stored_embedding = json.load(f)

        stored_embedding_array = np.array(stored_embedding['embedding'], dtype=np.float64)
        if len(stored_embedding_array.shape) == 1:
            stored_embedding_array = stored_embedding_array.reshape(1, -1)

        similarity = cosine_similarity(embedding_array, stored_embedding_array)[0][0]
        os.unlink(temp_file_path)

        # Preparar respuesta base
        response_data = {
            "message": "Coincidencia encontrada",
            "user": matched_user,
            "similarity": round(similarity, 4),
            "confidence": "high" if similarity > 0.8 else "medium" if similarity > 0.6 else "low"
        }

        if matched_user.get('requisitoriado') == 'true' or matched_user.get('requisitoriado') is True:
            security_alert = await simulate_authority_notification(matched_user)

            response_data.update({
                "SECURITY_ALERT": True,
                "alert_level": "CRITICAL",
                "alert_message": "USUARIO REQUISITORIADO DETECTADO",
                "security_notification": security_alert,
                "immediate_actions_required": True
            })

            print(
                f"ALERTA CRÍTICA: Usuario requisitoriado detectado - {matched_user['nombres']} {matched_user['apellidos']}")

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error en comparación de imagen: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al comparar la imagen: {str(e)}")
