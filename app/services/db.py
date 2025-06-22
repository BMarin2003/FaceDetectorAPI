import os
import httpx
from dotenv import load_dotenv

load_dotenv()
API_TOKEN = os.getenv('API_TOKEN')
DATABASE_ID = os.getenv('DATABASE_ID')
ACCOUNT_ID = os.getenv('ACCOUNT_ID')

BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/d1/database/{DATABASE_ID}"

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

# --- Estudiantes ---
async def get_user_by_id(user_id: int):
    query = "SELECT * FROM estudiantes WHERE id = ?"
    params = [user_id]
    print("Query:", query)
    print("Parameters:", params)
    async with httpx.AsyncClient() as client:
        payload = {"sql": query, "params": params}
        print("Payload:", payload)
        response = await client.post(f"{BASE_URL}/query", json=payload, headers=headers)
        print("Response status code:", response.status_code)
        print("Response content:", response.content)
        response.raise_for_status()
        data = response.json()
        # Ajusta la extracción de resultados según la estructura de la respuesta
        results = data['result'][0]['results'] if data['result'] else []
        return results[0] if results else None

async def get_user_by_email(correo: str):
    query = "SELECT * FROM estudiantes WHERE correo = ?"
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/query", json={"sql": query, "parameters": [correo]}, headers=headers)
        response.raise_for_status()
        data = response.json()
        results = data.get('result', {}).get('results', [])
        return results[0] if results else None

async def list_users():
    query = "SELECT * FROM estudiantes"
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/query", json={"sql": query}, headers=headers)
        response.raise_for_status()
        data = response.json()
        try:
            users = data['result'][0]['results']
            return users
        except (KeyError, IndexError, TypeError) as e:
            raise Exception(f"Error extrayendo usuarios: {e} - data: {data}")

async def create_user(user_data: dict):
    query = """
    INSERT INTO estudiantes (upaoID, nombres, apellidos, correo, requisitoriado)
    VALUES (?, ?, ?, ?, ?)
    """
    params = [user_data['upaoID'], user_data['nombres'], user_data['apellidos'], user_data['correo'], user_data['requisitoriado']]
    print("Query:", query)
    print("Parameters:", params)
    async with httpx.AsyncClient() as client:
        payload = {"sql": query, "params": params}
        print("Payload:", payload)
        response = await client.post(f"{BASE_URL}/query", json=payload, headers=headers)
        print("Response status code:", response.status_code)
        print("Response content:", response.content)
        response.raise_for_status()
        return response.json()

async def update_user(user_id: int, update_data: dict):
    set_clause = ", ".join([f"{k} = ?" for k in update_data.keys()])
    params = list(update_data.values()) + [user_id]
    query = f"UPDATE estudiantes SET {set_clause} WHERE id = ?"
    print(f"Query: {query}")
    print(f"Parameters: {params}")
    async with httpx.AsyncClient() as client:
        payload = {"sql": query, "params": params}
        print(f"Payload: {payload}")
        response = await client.post(f"{BASE_URL}/query", json=payload, headers=headers)
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.content}")
        response.raise_for_status()
        return response.json()

async def delete_user(user_id: int):
    query = "DELETE FROM estudiantes WHERE id = ?"
    params = [user_id]
    async with httpx.AsyncClient() as client:
        payload = {"sql": query, "params": params}
        response = await client.post(f"{BASE_URL}/query", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

# --- Fotos adicionales ---
async def add_face_photo(usuario_id: int, foto_url: str, kp_url: str):
    query = """
    INSERT INTO caras (foto, KP, usuario_id)
    VALUES (?, ?, ?)
    """
    params = [foto_url, kp_url, usuario_id]
    print("Query:", query)
    print("Parameters:", params)
    async with httpx.AsyncClient() as client:
        payload = {"sql": query, "params": params}
        print("Payload:", payload)
        response = await client.post(f"{BASE_URL}/query", json=payload, headers=headers)
        print("Response status code:", response.status_code)
        print("Response content:", response.content)
        response.raise_for_status()
        return response.json()

async def get_faces_by_user_id(usuario_id: int):
    query = "SELECT * FROM caras WHERE usuario_id = ?"
    params = [usuario_id]
    print("Query:", query)
    print("Parameters:", params)
    async with httpx.AsyncClient() as client:
        payload = {"sql": query, "params": params}
        response = await client.post(f"{BASE_URL}/query", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        results = data['result'][0]['results'] if data['result'] else []
        return results

async def delete_face(face_id: int):
    query = "DELETE FROM caras WHERE id = ?"
    params = [face_id]
    print("Query:", query)
    print("Parameters:", params)
    async with httpx.AsyncClient() as client:
        payload = {"sql": query, "params": params}
        response = await client.post(f"{BASE_URL}/query", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

async def get_all_users_with_photos():
    query = "SELECT id, foto FROM estudiantes WHERE foto IS NOT NULL"
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/query", json={"sql": query}, headers=headers)
        response.raise_for_status()
        data = response.json()
        results = data['result'][0]['results'] if data['result'] else []
        return results

async def get_all_users_with_embeddings():
    query = """
    SELECT e.id, e.nombres, e.apellidos, e.correo, c.KP
    FROM estudiantes e
    JOIN caras c ON e.id = c.usuario_id
    WHERE c.KP IS NOT NULL
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/query", json={"sql": query}, headers=headers)
        response.raise_for_status()
        data = response.json()
        results = data['result'][0]['results'] if data['result'] else []
        return results

