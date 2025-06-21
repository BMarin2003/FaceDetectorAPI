import os
import boto3
import json
from fastapi import UploadFile, HTTPException
from dotenv import load_dotenv
import tempfile
from io import BytesIO

load_dotenv()
ENDPOINT_URL = os.getenv('ENDPOINT_URL')
ACCESS_KEY = os.getenv('ACCESS_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')
BUCKET_PHOTOS = os.getenv('BUCKET_PHOTOS')
BUCKET_KPS = os.getenv('BUCKET_KPS')
PUBLIC_R2_URL_IMAGES = os.getenv('PUBLIC_R2_URL_IMAGES')
PUBLIC_R2_URL_JSON = os.getenv('PUBLIC_R2_URL_JSON')

session = boto3.session.Session()
s3_client = session.client('s3',
                           endpoint_url=ENDPOINT_URL,
                           aws_access_key_id=ACCESS_KEY,
                           aws_secret_access_key=SECRET_KEY)


async def upload_file_to_r2(file: UploadFile, bucket: str, filename: str) -> str:
    # Reiniciar el puntero del archivo antes de leer su contenido
    await file.seek(0)

    # Leer el contenido del archivo subido
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="El archivo subido está vacío")

    # Subir el contenido del archivo a R2
    s3_client.put_object(Bucket=bucket, Key=filename, Body=content, ACL='public-read')

    # Devolver la URL pública del archivo
    if bucket == BUCKET_PHOTOS:
        return f"{PUBLIC_R2_URL_IMAGES}/{filename}"
    elif bucket == BUCKET_KPS:
        return f"{PUBLIC_R2_URL_JSON}/{filename}"
    else:
        raise ValueError("Bucket no reconocido")


async def upload_json_to_r2(data: dict, bucket: str, filename: str) -> str:
    """
    Sube un diccionario como archivo JSON a R2
    """
    try:
        # Serializar el diccionario a JSON
        json_content = json.dumps(data, indent=2)

        # Convertir a bytes
        content_bytes = json_content.encode('utf-8')

        # Crear un BytesIO object
        json_buffer = BytesIO(content_bytes)
        json_buffer.seek(0)

        # Crear un UploadFile temporal
        temp_json_file = UploadFile(
            filename=filename,
            file=json_buffer
        )

        # Usar la función existente para subir
        return await upload_file_to_r2(temp_json_file, bucket, filename)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al subir JSON a R2: {e}")


def extract_key_from_url(url: str) -> str:
    return url.split('/')[-1]


async def delete_file_from_r2(url: str, bucket: str):
    key = extract_key_from_url(url)
    s3_client.delete_object(Bucket=bucket, Key=key)

async def download_file_from_r2(url: str, bucket: str) -> str:
    key = extract_key_from_url(url)
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        s3_client.download_fileobj(bucket, key, temp_file)
        return temp_file.name