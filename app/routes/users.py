import json
import os
import uuid
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from keras.src.losses import cosine_similarity

from app.utils import extract_embeddings
from app.utils.extract_embeddings import extract_embedding_from_image
from app.models.knn_model import knn_model
from app.schemas.user import UserInDB
from app.services import db, storage
from app.services.db import get_all_users_with_embeddings

router = APIRouter()

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

        nombres_sin_espacios = nombres.replace(" ", "_")
        apellidos_sin_espacios = apellidos.replace(" ", "_")

        filename_photo = f"{nombres_sin_espacios}_{apellidos_sin_espacios}.jpg"
        foto_url = await storage.upload_file_to_r2(foto, bucket=os.getenv('BUCKET_PHOTOS'), filename=filename_photo)

        user_data = {
            "upaoID": upaoID,
            "nombres": nombres,
            "apellidos": apellidos,
            "correo": correo,
            "requisitoriado": requisitoriado,
        }
        result = await db.create_user(user_data)
        user_id = result['result'][0]['meta']['last_row_id']

        await foto.seek(0)

        embedding = await extract_embeddings.extract_embedding_from_image(foto)
        if embedding:
            embedding_filename = f"{upaoID}_{uuid.uuid4().hex[:8]}.json"
            kp_url = await storage.upload_json_to_r2(
                embedding,
                bucket=os.getenv('BUCKET_KPS'),
                filename=embedding_filename
            )

            await db.add_face_photo(user_id, foto_url, kp_url)

            await db.update_user(user_id, {"foto": foto_url, "KP": kp_url})

        return {"message": "Usuario creado", "result": result}
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

    # Obtener todas las fotos del usuario de la tabla "caras"
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
        # Verificar que el archivo es una imagen
        if not foto.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo subido no es una imagen")

        # Leer el contenido del archivo subido
        content = await foto.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="El archivo subido está vacío")

        # Reemplazar espacios con guiones bajos en el nombre del archivo
        nombres_sin_espacios = user['nombres'].replace(" ", "_")
        apellidos_sin_espacios = user['apellidos'].replace(" ", "_")

        # Subir la nueva foto a R2
        filename_photo = f"{nombres_sin_espacios}_{apellidos_sin_espacios}_{uuid.uuid4().hex[:8]}.jpg"
        foto_url = await storage.upload_file_to_r2(foto, bucket=os.getenv('BUCKET_PHOTOS'), filename=filename_photo)

        # Reiniciar el puntero del archivo antes de extraer el embedding
        await foto.seek(0)

        # Extraer embeddings de la nueva foto
        embedding = await extract_embeddings.extract_embedding_from_image(foto)
        if embedding:
            # Subir el embedding como JSON a R2
            embedding_filename = f"{user_id}_{uuid.uuid4().hex[:8]}.json"
            kp_url = await storage.upload_json_to_r2(
                embedding,
                bucket=os.getenv('BUCKET_KPS'),
                filename=embedding_filename
            )

            # Si no se conserva la foto anterior, eliminar la foto y el embedding antiguos
            if not conservar and user.get('foto') and user.get('KP'):
                await delete_photo_files(user_id, user['foto'], user['KP'])

            # Actualizar la foto y el KP del usuario en la tabla "estudiantes"
            update_data['foto'] = foto_url
            update_data['KP'] = kp_url

            # Añadir la nueva foto a la tabla "caras"
            await db.add_face_photo(user_id, foto_url, kp_url)

    # Actualizar los datos del usuario
    result = await db.update_user(user_id, update_data)

    return {"message": "Usuario actualizado", "result": result}

@router.put("/users/{user_id}/change_profile_photo/{photo_id}", response_model=dict)
async def change_profile_photo(user_id: int, photo_id: int):
    user = await db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    # Obtener la foto de la tabla "caras"
    faces = await db.get_faces_by_user_id(user_id)
    photo_to_set = None
    for face in faces:
        if face['id'] == photo_id:
            photo_to_set = face
            break

    if not photo_to_set:
        raise HTTPException(status_code=404, detail="Foto no encontrada")

    # Actualizar la foto y el KP del usuario en la tabla "estudiantes"
    update_data = {
        "foto": photo_to_set['foto'],
        "KP": photo_to_set['KP']
    }
    result = await db.update_user(user_id, update_data)

    return {"message": "Foto de perfil actualizada", "result": result}

async def delete_photo_files(user_id: int, foto_url: str, kp_url: str):
    # Eliminar la foto y el embedding de R2
    await storage.delete_file_from_r2(foto_url, os.getenv('BUCKET_PHOTOS'))
    await storage.delete_file_from_r2(kp_url, os.getenv('BUCKET_KPS'))

    # Obtener y eliminar la fila correspondiente en la tabla "caras"
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

    # Obtener todas las fotos del usuario de la tabla "caras"
    faces = await db.get_faces_by_user_id(user_id)
    for face in faces:
        # Eliminar la foto y el embedding de R2
        await storage.delete_file_from_r2(face['foto'], os.getenv('BUCKET_PHOTOS'))
        await storage.delete_file_from_r2(face['KP'], os.getenv('BUCKET_KPS'))

    # Borrar todas las entradas en la tabla "caras" para este usuario
    for face in faces:
        await db.delete_face(face['id'])

    # Borrar el usuario
    await db.delete_user(user_id)

    return {"message": "Usuario eliminado"}

@router.delete("/users/photo/{user_id}/{photo_id}", response_model=dict)
async def delete_photo(user_id: int, photo_id: int):
    # Obtener la foto de la tabla "caras"
    faces = await db.get_faces_by_user_id(user_id)
    photo_to_delete = None
    for face in faces:
        if face['id'] == photo_id:
            photo_to_delete = face
            break

    if not photo_to_delete:
        raise HTTPException(status_code=404, detail="Foto no encontrada")

    # Eliminar la foto y el embedding de R2
    await storage.delete_file_from_r2(photo_to_delete['foto'], os.getenv('BUCKET_PHOTOS'))
    await storage.delete_file_from_r2(photo_to_delete['KP'], os.getenv('BUCKET_KPS'))

    # Eliminar la fila correspondiente en la tabla "caras"
    await db.delete_face(photo_id)

    return {"message": "Foto eliminada"}

@router.post("/users/photo/")
async def add_photo_to_user(usuario_id: int = Form(...), file: UploadFile = File(...)):
    user = await db.get_user_by_id(usuario_id)
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    # Verificar que el archivo es una imagen
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="El archivo subido no es una imagen")

    # Leer el contenido del archivo subido
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="El archivo subido está vacío")

    # Reemplazar espacios con guiones bajos en el nombre del archivo
    nombres_sin_espacios = user['nombres'].replace(" ", "_")
    apellidos_sin_espacios = user['apellidos'].replace(" ", "_")

    # Subir la foto a R2
    filename_photo = f"{nombres_sin_espacios}_{apellidos_sin_espacios}_{uuid.uuid4().hex[:8]}.jpg"
    foto_url = await storage.upload_file_to_r2(file, bucket=os.getenv('BUCKET_PHOTOS'), filename=filename_photo)

    # Reiniciar el puntero del archivo antes de extraer el embedding
    await file.seek(0)

    # Extraer embeddings de la foto
    embedding = await extract_embeddings.extract_embedding_from_image(file)
    if embedding:
        # Subir el embedding como JSON a R2
        embedding_filename = f"{usuario_id}_{uuid.uuid4().hex[:8]}.json"
        kp_url = await storage.upload_json_to_r2(
            embedding,
            bucket=os.getenv('BUCKET_KPS'),
            filename=embedding_filename
        )

        # Añadir la foto y el embedding a la tabla "caras"
        await db.add_face_photo(usuario_id, foto_url, kp_url)

    return {"message": "Foto agregada", "foto_url": foto_url}

@router.post("/compare/")
async def compare_external_image(file: UploadFile = File(...)):
    try:
        # Extraer embedding de la imagen externa
        embedding = await extract_embedding_from_image(file)
        if embedding is None:
            raise HTTPException(status_code=400, detail="No se pudo extraer el embedding de la imagen")

        # Convertir el embedding a un array de numpy
        embedding_array = np.array(embedding['embedding'])

        # Predecir la clase utilizando el modelo KNN
        predicted_user_id = knn_model.predict(embedding_array)

        if predicted_user_id is None:
            raise HTTPException(status_code=404, detail="No se encontró ninguna coincidencia")

        # Obtener los datos del usuario predicho
        matched_user = await db.get_user_by_id(predicted_user_id)

        return {"message": "Coincidencia encontrada", "user": matched_user}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al comparar la imagen: {e}")

async def compare_embeddings(embedding: np.ndarray):
    try:
        # Obtener todos los usuarios con sus embeddings
        users = await get_all_users_with_embeddings()

        # Comparar el embedding con los embeddings almacenados
        for user in users:
            # Descargar el embedding almacenado
            kp_url = user['KP']
            embedding_data = await storage.download_file_from_r2(kp_url, os.getenv('BUCKET_KPS'))
            with open(embedding_data, 'r') as f:
                stored_embedding = json.load(f)

            # Extraer el embedding del diccionario
            stored_embedding_array = np.array(stored_embedding['embedding'])

            # Calcular la similitud coseno
            similarity = cosine_similarity([embedding], [stored_embedding_array])[0][0]
            if similarity > 0.7:  # Umbral de similitud
                return user

        return None
    except Exception as e:
        print(f"Error al comparar embeddings: {e}")
        return None