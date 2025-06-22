import os
import asyncio
import sys
import uuid
from app.utils.extract_embeddings import extract_embedding_from_image
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.db import create_user, update_user, add_face_photo
from app.services.storage import upload_file_to_r2, upload_json_to_r2
from fastapi import UploadFile
from io import BytesIO

async def load_initial_data(photos_dir: str):
    for filename in os.listdir(photos_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Parsear el nombre del archivo para obtener la información del usuario
            parts = filename.split('_')
            upaoID = parts[0]
            nombres = parts[1]
            apellidos = parts[2]

            # Corregir el parsing del correo
            correo_parts = parts[3].split('.')
            correo = f"{correo_parts[0]}.edu.pe"

            # Crear el usuario en la base de datos
            user_data = {
                "upaoID": upaoID,
                "nombres": nombres,
                "apellidos": apellidos,
                "correo": correo,
                "requisitoriado": False
            }
            print(f"Creando usuario con datos: {user_data}")
            result = await create_user(user_data)
            print(f"Resultado de la creación del usuario: {result}")
            user_id = result['result'][0]['meta']['last_row_id']

            # Leer el archivo una sola vez
            photo_path = os.path.join(photos_dir, filename)
            with open(photo_path, 'rb') as photo_file:
                content = photo_file.read()
            if not content:
                print(f"El archivo {filename} está vacío")
                continue

            # Crear dos BytesIO separados
            foto_buffer_r2 = BytesIO(content)
            foto_buffer_embedding = BytesIO(content)

            # Para R2
            foto_buffer_r2.seek(0)
            temp_foto_r2 = UploadFile(
                filename=filename,
                file=foto_buffer_r2
            )

            # Para embedding
            foto_buffer_embedding.seek(0)
            temp_foto_embedding = UploadFile(
                filename=filename,
                file=foto_buffer_embedding
            )

            # Subir la foto a R2
            filename_photo = filename
            foto_url = await upload_file_to_r2(temp_foto_r2, bucket=os.getenv('BUCKET_PHOTOS'), filename=filename_photo)
            print(f"URL de la foto subida: {foto_url}")

            # Extraer embeddings de la foto
            embedding = await extract_embedding_from_image(temp_foto_embedding)
            print(f"Embedding extraído: {embedding}")

            if embedding:
                # Subir el embedding como JSON a R2
                embedding_filename = f"{upaoID}_{uuid.uuid4().hex[:8]}.json"
                kp_url = await upload_json_to_r2(
                    embedding,
                    bucket=os.getenv('BUCKET_KPS'),
                    filename=embedding_filename
                )
                print(f"URL del embedding (KP) subida: {kp_url}")

                # Subir la foto y el embedding a la tabla "caras"
                await add_face_photo(user_id, foto_url, kp_url)

                # Actualizar el usuario con la URL de la foto y el KP
                update_data = {"foto": foto_url, "KP": kp_url}
                print(f"Actualizando usuario con ID {user_id} con datos: {update_data}")
                await update_user(user_id, update_data)
            else:
                print(f"No se pudo extraer el embedding para la foto {filename}")

# Ejecutar el script para cargar los datos iniciales
if __name__ == "__main__":
    asyncio.run(load_initial_data('C:/Users/bryan/OneDrive/Desktop/fotos'))