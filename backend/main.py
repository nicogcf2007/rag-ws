# --- START OF FILE api.py ---

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uvicorn
from pydantic import BaseModel
from typing import List, Optional, Set
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
# <--- MODIFICADO: Asegurarse de importar Langchain correctamente ---
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # <--- MODIFICADO: Importar Document
import asyncio
import json
import time
import re
from pathlib import Path
import traceback
import platform
import sys
import uuid # <--- MODIFICADO: Para IDs únicos

# Cargar variables de entorno
load_dotenv()

# Configuración de rutas para Tesseract y Poppler (Mantener como estaba)
TESSERACT_PATH = os.environ.get('TESSERACT_CMD', 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe')
POPPLER_PATH = os.environ.get('POPPLER_PATH', 'C:\\Poppler\\bin')

# Verificar si estamos en Windows y ajustar las rutas según sea necesario (Mantener como estaba)
if platform.system() == "Windows":
    if not os.path.exists(TESSERACT_PATH):
        print(f"Advertencia: No se encontró Tesseract en {TESSERACT_PATH}")
    if not os.path.exists(POPPLER_PATH):
        print(f"Advertencia: No se encontró Poppler en {POPPLER_PATH}")

# Configuración
app = FastAPI(title="RAG API")
DATA_PATH = "data"
EMBEDDINGS_PATH = "embeddings_db"

# Configurar CORS (Mantener como estaba)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Asegurar que existan las carpetas necesarias (Mantener como estaba)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings_db")
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(EMBEDDINGS_PATH, exist_ok=True)

# Initialize OpenAI client (Mantener como estaba)
client = OpenAI()

# Initialize ChromaDB client (Mantener como estaba)
chroma_client = chromadb.PersistentClient(path=EMBEDDINGS_PATH)

# Modelos de datos (Mantener como estaba)
class QueryRequest(BaseModel):
    query: str
    num_results: int = 3

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

# <--- MODIFICADO: Prompt por defecto del sistema ---
DEFAULT_SYSTEM_PROMPT = "Eres un asistente útil que responde preguntas basadas únicamente en el contexto proporcionado. Si la respuesta no se encuentra en el contexto, indica que no tienes suficiente información."

# Cliente OpenAI (Ya inicializado arriba)

# WebSocketManager para manejar conexiones (Mantener como estaba)
class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"WebSocket connected: {websocket.client}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"WebSocket disconnected: {websocket.client}")

    async def broadcast(self, message: str):
        # <--- MODIFICADO: Manejar desconexiones durante el broadcast ---
        disconnected_websockets = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                disconnected_websockets.append(connection)
                print(f"WebSocket found disconnected during broadcast: {connection.client}")
            except Exception as e:
                print(f"Error broadcasting to {connection.client}: {e}")
                disconnected_websockets.append(connection) # Considerar desconectar si hay error

        # Limpiar conexiones desconectadas después del broadcast
        for ws in disconnected_websockets:
            self.disconnect(ws)


websocket_manager = WebSocketManager()

# Endpoint WebSocket genérico para progreso (Mantener como estaba)
@app.websocket("/ws/progress")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Mantener la conexión abierta, se puede recibir un 'ping' o similar
            data = await websocket.receive_text()
            # print(f"Received from {websocket.client}: {data}") # Para depuración
    except WebSocketDisconnect:
        print(f"WebSocket /ws/progress explicit disconnect: {websocket.client}")
        websocket_manager.disconnect(websocket)
    except Exception as e:
        print(f"Error in /ws/progress websocket loop for {websocket.client}: {e}")
        websocket_manager.disconnect(websocket)


# <--- MODIFICADO: Eliminar el endpoint /ws/upload-progress separado ---
# Ya no es necesario, usaremos /ws/progress para todo

# Función para extraer texto de documentos (MODIFICADO para mejor metadata en Word/Excel)
async def extract_text_from_document(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    raw_documents: List[Document] = [] # Especificar tipo

    try:
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path, extract_images=False) # extract_images=False puede mejorar rendimiento si no se usa OCR siempre
            raw_documents = loader.load()
            # Asegurarse que la metadata de página existe y es numérica (base 0)
            for doc in raw_documents:
                doc.metadata['source'] = os.path.basename(file_path)
                if 'page' not in doc.metadata or not isinstance(doc.metadata['page'], int):
                     # Intentar extraer de page_number si existe, sino asignar -1 o 0
                     page_num_str = str(doc.metadata.get('page_number', '0'))
                     doc.metadata['page'] = int(page_num_str) -1 if page_num_str.isdigit() and int(page_num_str)>0 else 0

        elif file_extension == '.docx':
            loader = UnstructuredWordDocumentLoader(file_path, mode="elements")
            raw_documents = loader.load()
            # <--- MODIFICADO: Añadir metadata útil si es posible ---
            # Unstructured puede dar 'page_number' a veces, o 'category' (Title, ListItem, etc.)
            for i, doc in enumerate(raw_documents):
                 doc.metadata['source'] = os.path.basename(file_path)
                 doc.metadata['page'] = doc.metadata.get('page_number', 0) -1 # Base 0
                 doc.metadata['element_type'] = doc.metadata.get('category', 'Unknown')
                 doc.metadata['chunk_index'] = i # Guardar índice como fallback

        elif file_extension in ['.xlsx', '.xlsm']:
             # <--- MODIFICADO: Modo 'elements' es mejor para tablas ---
            loader = UnstructuredExcelLoader(file_path, mode="elements")
            raw_documents = loader.load()
             # <--- MODIFICADO: Añadir metadata útil si es posible ---
             # Puede incluir 'page_number' (nombre de hoja) y 'category' (Table)
            for i, doc in enumerate(raw_documents):
                 doc.metadata['source'] = os.path.basename(file_path)
                 # 'page_number' en Excel a menudo es el nombre de la hoja, no un número
                 doc.metadata['sheet_name'] = doc.metadata.get('page_number', 'UnknownSheet')
                 doc.metadata['page'] = 0 # Página no aplica directamente a Excel
                 doc.metadata['element_type'] = doc.metadata.get('category', 'Unknown')
                 doc.metadata['chunk_index'] = i # Guardar índice como fallback
        else:
            print(f"Formato de archivo no soportado: {file_extension}")
            return []

    except Exception as e:
        print(f"Error cargando documento {file_path}: {e}")
        traceback.print_exc()
        # Intentar OCR si falla la carga directa de PDF (lógica de OCR mantenida pero simplificada)
        if file_extension == '.pdf':
            print("Intentando extracción con OCR como fallback...")
            # (Lógica de OCR aquí - similar a la versión anterior, pero asegurándose de añadir metadata correcta)
            # ... (mantener lógica OCR con Tesseract/Poppler si es necesaria) ...
            # Ejemplo simplificado si la lógica OCR se mantiene:
            # ocr_texts = await run_ocr_extraction(file_path) # Función hipotética
            # return ocr_texts
            pass # Por ahora, si falla la carga normal, devolvemos lista vacía
        return []

    # Filtrar documentos vacíos que algunos loaders pueden generar
    raw_documents = [doc for doc in raw_documents if doc.page_content.strip()]
    print(f"Documento {os.path.basename(file_path)} cargado. {len(raw_documents)} elementos extraídos.")
    return raw_documents


# Función para procesar documentos y generar embeddings (MODIFICADO para IDs únicos)
async def process_document_chunks(
    chunks: List[Document],
    context_name: str,
    filename: str,
    websocket_manager: WebSocketManager, # <-- Añadido
    file_index: int,                    # <-- Añadido (0-based index of current file)
    total_files: int                    # <-- Añadido
):
    """Procesa una lista de chunks de documento, enviando progreso detallado."""
    if not chunks:
        print(f"No chunks to process for {filename} in context {context_name}")
        return 0

    total_chunks_in_file = len(chunks)
    print(f"Procesando {total_chunks_in_file} chunks para {filename} (Archivo {file_index + 1}/{total_files})")
    embeddings = []
    documents_content = []
    metadatas = []
    ids = []
    processed_chunk_count = 0
    last_reported_percentage = -1 # Para throttling

    # Constante para throttling (envía actualización cada N chunks)
    UPDATE_EVERY_N_CHUNKS = 5

    for i, chunk in enumerate(chunks):
        page_content = chunk.page_content.strip()
        if not page_content:
            print(f"Skipping empty chunk {i} for file {filename}")
            continue

        try:
            # Generar embedding
            start_embed_time = time.time()
            response = client.embeddings.create(
                input=page_content,
                model="text-embedding-3-small"
            )
            # print(f"Embedding chunk {i+1}/{total_chunks_in_file} for {filename} took {time.time() - start_embed_time:.2f}s")

            if response and response.data:
                embeddings.append(response.data[0].embedding)
            else:
                print(f"Warning: No embedding generated for chunk {i} of {filename}. Using fallback.")
                embeddings.append([0.0] * 1536)

            # Preparar datos para ChromaDB
            documents_content.append(page_content)
            chunk_metadata = {
                "source": filename,
                "page": chunk.metadata.get("page", 0),
                "element_type": chunk.metadata.get("element_type", "N/A"),
                "sheet_name": chunk.metadata.get("sheet_name", "N/A"),
                "chunk_index": i
            }
            metadatas.append(chunk_metadata)
            ids.append(f"{filename}_chunk_{i}_{uuid.uuid4()}")
            processed_chunk_count += 1

            # --- MODIFICADO: Calcular y enviar progreso detallado ---
            progress_within_file = (i + 1) / total_chunks_in_file
            # Calcula el porcentaje global basado en archivos completados + progreso en el archivo actual
            overall_percentage = int(
                ((file_index / total_files) + (progress_within_file / total_files)) * 100
            )
            overall_percentage = min(overall_percentage, 100) # Asegurar que no pase de 100

            # Throttling: Enviar solo si el porcentaje cambió o cada N chunks, o en el último chunk
            # if overall_percentage > last_reported_percentage or (i + 1) % UPDATE_EVERY_N_CHUNKS == 0 or (i + 1) == total_chunks_in_file:
            # Simplificado: Enviar cada N chunks o al final
            if (i + 1) % UPDATE_EVERY_N_CHUNKS == 0 or (i + 1) == total_chunks_in_file:
                print(f"WS Update: File {file_index+1}/{total_files}, Chunk {i+1}/{total_chunks_in_file}, Overall: {overall_percentage}%")
                await websocket_manager.broadcast(json.dumps({
                    "type": "progress",
                    "current_file": filename,
                    "files_processed": file_index, # Archivos completamente terminados antes de este
                    "total_files": total_files,
                    "percentage": overall_percentage,
                    "status_message": f"Procesando {filename}: Chunk {i + 1}/{total_chunks_in_file}"
                }))
                last_reported_percentage = overall_percentage

        except Exception as e:
            print(f"Error procesando chunk {i} de {filename}: {str(e)}")
            # Considerar si continuar o fallar el archivo entero
            continue # Saltar este chunk si falla

    if not documents_content:
        print(f"No valid content to add for {filename} after processing chunks.")
        return 0

    # Añadir a ChromaDB (esto también puede tardar, pero menos que los embeddings individuales)
    try:
        print(f"Almacenando {len(documents_content)} embeddings en ChromaDB para {filename}")
        start_add_time = time.time()
        collection = chroma_client.get_or_create_collection(name=context_name)
        collection.add(
            documents=documents_content,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Embeddings para {filename} almacenados en {time.time() - start_add_time:.2f}s.")
        return len(documents_content)
    except Exception as e:
        print(f"Error añadiendo chunks a ChromaDB para {filename}: {e}")
        traceback.print_exc()
        return 0
# Rutas para gestión de contextos (sin cambios mayores)
@app.post("/create-context")
async def create_context(request: Request):
    try:
        data = await request.json()
        context_name = data.get('context_name')

        if not context_name or not validate_context_name(context_name):
            raise HTTPException(status_code=400, detail="Nombre de contexto inválido o vacío")

        # Verificar si el contexto ya existe (opcional, depende de la lógica deseada)
        context_data_path = os.path.join(DATA_PATH, context_name)
        if os.path.exists(context_data_path):
             print(f"Contexto '{context_name}' ya existe.")
             # Podrías retornar un mensaje diferente o un código 409 Conflict
             # return JSONResponse(status_code=409, content={"message": f"Contexto '{context_name}' ya existe"})

        # Crear directorios
        os.makedirs(context_data_path, exist_ok=True)
        # No es necesario crear carpeta de embeddings aquí, Chroma se encarga

        # Crear la colección en ChromaDB explícitamente si se desea
        try:
            chroma_client.create_collection(name=context_name)
            print(f"Colección ChromaDB '{context_name}' creada.")
        except chromadb.errors.DuplicateCollectionError:
             print(f"Colección ChromaDB '{context_name}' ya existía.")
        except Exception as e:
            print(f"Error creando colección ChromaDB '{context_name}': {e}")
            # Decide si esto debe impedir la creación del contexto
            # raise HTTPException(status_code=500, detail=f"Error al inicializar la base de datos del contexto: {e}")


        return {"message": f"Contexto '{context_name}' creado/verificado correctamente"}
    except Exception as e:
        print(f"Error en create_context: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")


@app.get("/get-contexts")
async def get_contexts():
    try:
        # Listar directorios en DATA_PATH como contextos
        # <--- MODIFICADO: Asegurarse que sólo sean directorios ---
        contexts = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
        # Opcionalmente, verificar si existe colección correspondiente en ChromaDB
        valid_contexts = []
        for context_name in contexts:
             try:
                 # Intenta obtener la colección para confirmar que existe en la DB
                 chroma_client.get_collection(name=context_name)
                 valid_contexts.append(context_name)
             except Exception:
                 # Si no existe la colección, podrías decidir no listarlo o marcarlo
                 print(f"Advertencia: Directorio de datos '{context_name}' existe pero no su colección en ChromaDB.")
                 # Descomentar la siguiente línea para sólo listar contextos con colección
                 # continue
                 valid_contexts.append(context_name) # Por ahora, lo listamos si la carpeta existe

        return {"contexts": sorted(valid_contexts)} # Ordenar alfabéticamente
    except Exception as e:
        print(f"Error en get_contexts: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

# Endpoint para subir documentos (MODIFICADO para progreso y mejor procesamiento)
@app.post("/upload-documents") # <-- Renombrado para más precisión
async def upload_documents(context_name: str = Form(...), files: List[UploadFile] = File(...)):
    context_path = os.path.join(DATA_PATH, context_name)
    if not os.path.exists(context_path) or not os.path.isdir(context_path):
         print(f"Contexto '{context_name}' no encontrado, intentando crear...")
         try:
             os.makedirs(context_path, exist_ok=True)
             chroma_client.get_or_create_collection(name=context_name)
             print(f"Contexto '{context_name}' creado automáticamente.")
         except Exception as e:
             print(f"Error creando contexto automáticamente: {e}")
             raise HTTPException(status_code=404, detail=f"Contexto '{context_name}' no encontrado y no se pudo crear.")

    results = []
    total_files = len(files)
    processed_files_count = 0
    total_chunks_processed = 0

    await websocket_manager.broadcast(json.dumps({
        "type": "start_upload",
        "total_files": total_files,
        "context": context_name
    }))

    for index, file in enumerate(files): # index es 0-based
        filename = file.filename
        file_path = os.path.join(context_path, filename)
        status = "Error"
        chunks_count = 0
        file_details = ""

        # Enviar estado inicial para este archivo
        await websocket_manager.broadcast(json.dumps({
            "type": "progress",
            "current_file": filename,
            "files_processed": index, # Archivos completados *antes* de este
            "total_files": total_files,
            "percentage": int((index / total_files) * 100) if total_files > 0 else 0, # Porcentaje basado en archivos iniciados
            "status_message": f"Iniciando archivo {index + 1}/{total_files}: {filename}"
        }))

        try:
            start_time_file = time.time()
            # 1. Guardar archivo
            print(f"Guardando archivo: {filename}")
            with open(file_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)

            # 2. Extraer texto
            print(f"Extrayendo texto de: {filename}")
            text_documents = await extract_text_from_document(file_path)

            if not text_documents:
                status = "Error: No se pudo extraer texto"
                file_details = "Archivo vacío, corrupto o formato no soportado."
                print(f"Warning: {status} para {filename}")
            else:
                # 3. Dividir en chunks
                print(f"Dividiendo texto de {filename}...")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(text_documents)
                print(f"{len(chunks)} chunks creados para {filename}")

                if not chunks:
                    status = "Error: No se generaron chunks"
                    file_details = "No se pudo dividir el texto extraído."
                    print(f"Warning: {status} para {filename}")
                else:
                    # --- MODIFICADO: Llamar a process_document_chunks con WS manager ---
                    num_added = await process_document_chunks(
                        chunks, context_name, filename,
                        websocket_manager, index, total_files # Pasar info necesaria
                    )

                    if num_added > 0:
                        status = "Procesado correctamente"
                        chunks_count = num_added
                        total_chunks_processed += num_added
                    elif status != "Error: No se pudo extraer texto": # Evitar sobrescribir error anterior
                        status = "Error: No se guardaron chunks"
                        file_details = "Error al generar embeddings o guardar en DB."
                        print(f"Warning: {status} para {filename}")

            processed_files_count += 1 # Incrementar contador de archivos que *intentaron* procesarse

            # Mensaje final para *este* archivo (opcional, ya que el progreso dentro de chunks es más útil)
            # await websocket_manager.broadcast(json.dumps({ ... }))
            print(f"Archivo {filename} ({(time.time() - start_time_file):.2f}s) finalizado con estado: {status}")


        except Exception as e:
            processed_files_count += 1 # Contar como procesado aunque falle
            status = "Error General"
            file_details = str(e)
            print(f"Error fatal procesando archivo {filename}: {e}")
            traceback.print_exc()
            await websocket_manager.broadcast(json.dumps({ # Enviar error específico del archivo
                "type": "error_file", "filename": filename, "message": file_details,
                "files_processed": processed_files_count, "total_files": total_files,
                "percentage": int((processed_files_count / total_files) * 100) if total_files > 0 else 0
            }))

        finally:
            # --- INICIO: LÓGICA DE BORRADO DEL ARCHIVO ---
            # Intentar eliminar el archivo temporal del disco después de procesarlo (o fallar)
            if os.path.exists(file_path):
                try:
                    print(f"Intentando eliminar archivo temporal: {file_path}")
                    os.remove(file_path) # <--- ¡AQUÍ SE BORRA EL ARCHIVO!
                    print(f"Archivo temporal {file_path} eliminado exitosamente.")
                except OSError as remove_error:
                    # Es importante capturar errores aquí para que un fallo al borrar
                    # no interrumpa el proceso general ni la respuesta final.
                    print(f"ALERTA: No se pudo eliminar el archivo temporal {file_path}. Error: {remove_error}")
                    # Podrías añadir un log más formal aquí si lo necesitas.
                    # No relanzamos la excepción, solo informamos.
            else:
                # Esto podría pasar si el guardado inicial falló.
                print(f"Archivo temporal {file_path} no encontrado para eliminar (posiblemente no se guardó).")
            # --- FIN: LÓGICA DE BORRADO DEL ARCHIVO ---

            # Guardar resultado para este archivo (esta línea ya debería existir)
            results.append({"filename": filename, "status": status, "chunks": chunks_count, "details": file_details})

    # Mensaje final global
    final_status_message = f"Proceso completado para '{context_name}'. {processed_files_count}/{total_files} archivos procesados. Total chunks: {total_chunks_processed}."
    await websocket_manager.broadcast(json.dumps({
        "type": "complete_upload", "message": final_status_message,
        "results": results, "percentage": 100
    }))
    print(final_status_message)
    return JSONResponse(content={"message": "Proceso completado", "results": results})


@app.get("/get-pdfs") # Debería llamarse /get-documents
async def get_documents(context_name: str):
    context_data_path = os.path.join(DATA_PATH, context_name)
    if not os.path.exists(context_data_path) or not os.path.isdir(context_data_path):
        # Si el contexto no existe en data, retornar vacío (o error 404)
        # raise HTTPException(status_code=404, detail=f"Contexto '{context_name}' no encontrado.")
        return {"pdfs": []} # Mantener comportamiento actual

    try:
        # Listar archivos soportados
        supported_extensions = ('.pdf', '.docx', '.xlsx', '.xlsm')
        doc_files = [f for f in os.listdir(context_data_path) if f.lower().endswith(supported_extensions) and os.path.isfile(os.path.join(context_data_path, f))]
        return {"pdfs": sorted(doc_files)} # Devolver ordenados
    except Exception as e:
        print(f"Error en get_documents para {context_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno al listar documentos: {str(e)}")


# Endpoint /upload obsoleto? Parece que /upload-pdf hace lo mismo. Comentar o eliminar si no se usa.
# @app.post("/upload")
# async def upload_files(...): ...


# Endpoint /query obsoleto? Parece que /ask hace lo mismo con más detalle. Comentar o eliminar si no se usa.
# @app.post("/query")
# async def query_documents(...): ...


# Endpoint para hacer preguntas (MODIFICADO)
@app.post("/ask")
async def ask_question(request: Request):
    try:
        data = await request.json()
        question = data.get("question")
        context_name = data.get("context_name")
        # <--- MODIFICADO: Aceptar prompt del sistema opcional ---
        system_prompt = data.get("system_prompt", DEFAULT_SYSTEM_PROMPT) # Usar default si no viene

        if not question or not context_name:
            raise HTTPException(status_code=400, detail="Parámetros 'question' y 'context_name' son requeridos.")

        # Validar nombre de contexto
        if not validate_context_name(context_name):
             raise HTTPException(status_code=400, detail=f"Nombre de contexto inválido: '{context_name}'")

        print(f"Recibida pregunta para contexto '{context_name}': {question}")
        # print(f"Usando system prompt: {system_prompt}") # Para depuración

        # Verificar que la colección del contexto existe
        try:
            collection = chroma_client.get_collection(name=context_name)
            print(f"Colección '{context_name}' encontrada.")
        except Exception as e:
            print(f"Error obteniendo colección '{context_name}': {e}")
            # Podrías intentar get_or_create_collection si quieres crearlo si no existe
            # collection = chroma_client.get_or_create_collection(name=context_name)
            raise HTTPException(status_code=404, detail=f"Contexto '{context_name}' no encontrado en la base de datos.")

        # Generar embedding para la consulta
        print("Generando embedding para la pregunta...")
        start_time = time.time()
        try:
            query_response = client.embeddings.create(
                input=question,
                model="text-embedding-3-small" # Asegurarse que el modelo es correcto
            )
            query_embedding = query_response.data[0].embedding
            print(f"Embedding generado en {time.time() - start_time:.2f}s")
        except Exception as e:
             print(f"Error generando embedding para la pregunta: {e}")
             raise HTTPException(status_code=500, detail=f"Error al procesar la pregunta (embedding): {str(e)}")


        # Buscar documentos relevantes en ChromaDB
        print("Buscando documentos relevantes...")
        start_time = time.time()
        try:
            # Ajustar n_results según sea necesario
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=20, # Número de chunks a recuperar
                include=['documents', 'metadatas', 'distances'] # Incluir distancias puede ser útil
            )
            print(f"Búsqueda completada en {time.time() - start_time:.2f}s. Encontrados {len(results.get('ids', [[]])[0])} resultados.")
        except Exception as e:
            print(f"Error buscando en ChromaDB: {e}")
            raise HTTPException(status_code=500, detail=f"Error al buscar información relevante: {str(e)}")

        # Procesar resultados de ChromaDB
        if not results or not results.get('ids') or not results['ids'][0]:
            print("No se encontraron documentos relevantes.")
            # Devolver respuesta indicando que no se encontró info
            return JSONResponse(content={
                "answer": "No pude encontrar información relevante en los documentos cargados para responder a tu pregunta.",
                "metadata": {
                    "document_name": None,
                    "page_number": None,
                    "all_source_documents": [] # <--- MODIFICADO: Campo para múltiples fuentes
                }
            })

        # Extraer textos, metadatos y calcular fuentes únicas
        relevant_docs = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0] if results.get('distances') else [0.0] * len(relevant_docs)

        # <--- MODIFICADO: Detectar múltiples fuentes ---
        source_files: Set[str] = set()
        detailed_sources = []
        primary_doc_name = "Desconocido"
        primary_page_number = None

        for i, meta in enumerate(metadatas):
             doc_name = meta.get('source', f'Desconocido_{i}')
             source_files.add(doc_name)
             # Para Word/Excel, 'page' puede ser 0 o sheet_name. Mostrar lo más útil.
             page_info = meta.get('page', 0)
             if isinstance(page_info, int): # Es número de página (PDF o fallback)
                  page_str = f"Página {page_info + 1}" if page_info >= 0 else "N/A" # Ajustar base 0
             else: # Probablemente nombre de hoja de Excel
                  page_str = f"Hoja '{page_info}'"

             chunk_idx = meta.get('chunk_index', i)
             distance = distances[i]
             # Guardar info detallada de cada fuente recuperada
             detailed_sources.append(f"{doc_name} ({page_str}, Chunk {chunk_idx}, Distancia: {distance:.4f})")

             # Guardar el nombre y página del primer documento como primario
             if i == 0:
                 primary_doc_name = doc_name
                 primary_page_number = page_info + 1 if isinstance(page_info, int) and page_info >= 0 else page_info # Mantener string si es nombre de hoja


        print(f"Documentos fuente encontrados: {source_files}")
        print(f"Fuentes detalladas: {detailed_sources}")

        # Construir el contexto para el LLM
        context_text = "\n\n---\n\n".join(relevant_docs) # Separador claro entre chunks

        # Construir prompt final
        # <--- MODIFICADO: Usar el system_prompt recibido o el default ---
        final_prompt = f"""Contexto de documentos:
        {context_text}

        Pregunta del usuario:
        {question}

        Instrucción: Responde la pregunta del usuario basándote *única y exclusivamente* en el "Contexto de documentos" proporcionado arriba. No inventes información. Si la respuesta no está en el contexto, dilo explícitamente."""


        # Generar respuesta usando LLM
        print("Generando respuesta con LLM...")
        start_time = time.time()
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini", # Asegurarse que es el modelo deseado
                messages=[
                    {"role": "system", "content": system_prompt}, # <--- MODIFICADO: Usar prompt del sistema
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.2, # Ajustar temperatura para respuestas más/menos creativas
            )
            answer = response.choices[0].message.content
            print(f"Respuesta generada en {time.time() - start_time:.2f}s")

        except Exception as e:
             print(f"Error llamando a OpenAI API: {e}")
             raise HTTPException(status_code=500, detail=f"Error al generar la respuesta (LLM): {str(e)}")


        # <--- MODIFICADO: Formatear respuesta final ---
        metadata = {
            "document_name": primary_doc_name,
            "page_number": primary_page_number, # Puede ser número o string (nombre de hoja)
            "all_source_documents": sorted(list(source_files)) # Lista ordenada de todas las fuentes
        }

        # Añadir nota si viene de múltiples documentos
        if len(source_files) > 1:
            answer += f"\n\n*Nota: La información proviene de múltiples documentos: {', '.join(metadata['all_source_documents'])}.*"


        result = {
            "answer": answer,
            "metadata": metadata
        }

        return JSONResponse(content=result)

    except HTTPException as he:
        # Re-lanzar excepciones HTTP para que FastAPI las maneje
        raise he
    except Exception as e:
        # Capturar cualquier otro error inesperado
        print(f"Error inesperado en /ask: {e}")
        traceback.print_exc()
        # Devolver respuesta de error genérica
        return JSONResponse(
             status_code=500,
             content={
                "answer": f"Lo siento, ocurrió un error interno al procesar tu pregunta. Por favor, inténtalo de nuevo más tarde.",
                "metadata": {
                    "document_name": None,
                    "page_number": None,
                    "all_source_documents": [],
                    "error": str(e) # Incluir error solo si es seguro/para depuración
                }
            }
        )


# Endpoint para eliminar contexto (MODIFICADO para usar ChromaDB correctamente)
@app.delete("/delete-context/{context_name}")
async def delete_specific_context(context_name: str):
    print(f"Solicitud para eliminar contexto: {context_name}")
    if not validate_context_name(context_name):
        raise HTTPException(status_code=400, detail="Nombre de contexto inválido")

    context_data_path = os.path.join(DATA_PATH, context_name)
    deleted_data = False
    deleted_collection = False

    # 1. Eliminar colección de ChromaDB
    try:
        print(f"Intentando eliminar colección ChromaDB: {context_name}")
        chroma_client.delete_collection(name=context_name)
        print(f"Colección ChromaDB '{context_name}' eliminada.")
        deleted_collection = True
    except Exception as e:
        # Podría ser que la colección no exista, lo cual está bien en un delete
        print(f"No se pudo eliminar la colección '{context_name}' (puede que no existiera): {e}")
        # No lanzar error aquí, continuar para borrar carpeta de datos

    # 2. Eliminar carpeta de datos
    if os.path.exists(context_data_path) and os.path.isdir(context_data_path):
        try:
            print(f"Intentando eliminar directorio de datos: {context_data_path}")
            shutil.rmtree(context_data_path)
            print(f"Directorio de datos '{context_data_path}' eliminado.")
            deleted_data = True
        except Exception as e:
            print(f"Error eliminando directorio de datos {context_data_path}: {e}")
            # Lanzar error si la colección se borró pero los datos no (o viceversa)
            raise HTTPException(status_code=500, detail=f"Error al eliminar los archivos del contexto: {str(e)}")
    else:
        print(f"Directorio de datos '{context_data_path}' no encontrado.")
        # Considerar esto como éxito si la colección tampoco existía o se borró

    if deleted_data or deleted_collection:
         return {"message": f"Contexto '{context_name}' eliminado correctamente (Datos: {'Sí' if deleted_data else 'No'}, Colección DB: {'Sí' if deleted_collection else 'No'})"}
    else:
         # Si ni la carpeta ni la colección existían, retornar 404
         raise HTTPException(status_code=404, detail=f"Contexto '{context_name}' no encontrado.")


# Endpoint /delete-context obsoleto? El anterior con {context_name} es más RESTful.
# @app.delete("/delete-context")
# async def delete_context(...): ...


# Funciones y Endpoints para guardar/cargar historial (Sin cambios mayores)
def save_chat_history(context_name: str, history: list):
    context_path = Path(DATA_PATH) / context_name
    history_file = context_path / "chat_history.json"
    try:
        # Asegurarse que el directorio existe
        context_path.mkdir(parents=True, exist_ok=True)
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2) # Guardar con formato e indentación
        print(f"Historial guardado para {context_name} en {history_file}")
    except Exception as e:
        print(f"Error guardando historial para {context_name}: {e}")
        # Podrías querer lanzar una excepción aquí

def load_chat_history(context_name: str) -> list:
    history_file = Path(DATA_PATH) / context_name / "chat_history.json"
    if not history_file.exists():
        print(f"No se encontró archivo de historial para {context_name}")
        return []
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
            print(f"Historial cargado para {context_name} desde {history_file}")
            return history
    except json.JSONDecodeError:
        print(f"Error: Archivo de historial para {context_name} está corrupto o mal formateado.")
        return [] # Devolver vacío si está corrupto
    except Exception as e:
        print(f"Error cargando historial para {context_name}: {e}")
        return []

@app.post("/save-chat-history")
async def save_chat_history_endpoint(request: Request):
    try:
        data = await request.json()
        context_name = data.get('context_name')
        history = data.get('history') # Asume que es una lista de objetos

        if not context_name or history is None: # Verificar que history no sea None
            raise HTTPException(status_code=400, detail="Parámetros 'context_name' e 'history' (lista) son requeridos.")
        if not validate_context_name(context_name):
             raise HTTPException(status_code=400, detail="Nombre de contexto inválido")

        save_chat_history(context_name, history)
        return {"message": "Historial guardado correctamente"}
    except Exception as e:
        print(f"Error en /save-chat-history: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno al guardar el historial: {str(e)}")

@app.post("/load-chat-history")
async def load_chat_history_endpoint(request: Request):
    try:
        data = await request.json()
        context_name = data.get('context_name')

        if not context_name:
            raise HTTPException(status_code=400, detail="Falta el nombre del contexto ('context_name')")
        if not validate_context_name(context_name):
             raise HTTPException(status_code=400, detail="Nombre de contexto inválido")

        history = load_chat_history(context_name)
        return {"history": history}
    except Exception as e:
        print(f"Error en /load-chat-history: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno al cargar el historial: {str(e)}")


# Función de validación de nombres de contexto (Sin cambios)
def validate_context_name(context_name: str):
    if not context_name: return False
    # Permitir un poco más de longitud y quizás puntos o espacios (reemplazados internamente si es necesario)
    if len(context_name) > 63: return False # Chroma tiene límite de longitud
    # Permitir letras, números, guión bajo, guión medio. Evitar otros caracteres especiales.
    # El nombre debe empezar y terminar con letra/número.
    if not re.match(r"^[a-zA-Z0-9](?:[a-zA-Z0-9_-]*[a-zA-Z0-9])?$", context_name):
         # Explicar la regla de validación
         print(f"Validación fallida para '{context_name}'. Regla: ^[a-zA-Z0-9](?:[a-zA-Z0-9_-]*[a-zA-Z0-9])?$")
         return False
    # Reglas adicionales de ChromaDB (evitar ..)
    if ".." in context_name:
        return False
    return True

# Iniciar servidor (Sin cambios)
if __name__ == "__main__":
    # Considerar configurar logging
    # import logging
    # logging.basicConfig(level=logging.INFO)
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

# --- END OF FILE api.py ---