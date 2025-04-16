# --- START OF FILE api.py ---

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uvicorn
from pydantic import BaseModel
from typing import List, Optional, Set, Dict, Any, Union # <--- MODIFICADO: Añadir Dict, Any, Set
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
# import sys # No parece usarse
import uuid # <--- MODIFICADO: Para IDs únicos

# Cargar variables de entorno
load_dotenv()

# Configuración de rutas para Tesseract y Poppler (Mantener como estaba - AJUSTA ESTAS RUTAS SI ES NECESARIO)
TESSERACT_PATH = os.environ.get('TESSERACT_CMD', 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe' if platform.system() == "Windows" else '/usr/bin/tesseract')
POPPLER_PATH = os.environ.get('POPPLER_PATH', 'C:\\Poppler\\bin' if platform.system() == "Windows" else None) # Poppler a menudo está en el PATH en Linux/Mac

# Verificar si estamos en Windows y ajustar las rutas según sea necesario (Mantener como estaba)
# <--- MODIFICADO: Mejorar verificación y mensajes ---
if platform.system() == "Windows":
    if not Path(TESSERACT_PATH).is_file():
        print(f"ADVERTENCIA: Tesseract ejecutable no encontrado en {TESSERACT_PATH}. La extracción OCR fallará.")
    if POPPLER_PATH and not Path(POPPLER_PATH).is_dir():
        print(f"ADVERTENCIA: Directorio de Poppler no encontrado en {POPPLER_PATH}. La extracción de PDF (especialmente escaneados) puede fallar.")
    elif POPPLER_PATH:
         # Añadir Poppler al PATH si se especificó y existe, crucial para PyPDFLoader/Unstructured
         os.environ["PATH"] += os.pathsep + POPPLER_PATH
         print(f"INFO: Poppler añadido al PATH desde: {POPPLER_PATH}")
else:
     # En Linux/Mac, verificar si tesseract está en el PATH si no se especificó TESSERACT_CMD
     if 'TESSERACT_CMD' not in os.environ and shutil.which('tesseract') is None:
          print("ADVERTENCIA: El comando 'tesseract' no se encontró en el PATH. La extracción OCR fallará.")

# Configuración
app = FastAPI(title="RAG API")
BASE_DIR = Path(__file__).resolve().parent # Usar pathlib para mejor manejo de rutas
DATA_PATH = BASE_DIR / "data"
EMBEDDINGS_PATH = BASE_DIR / "embeddings_db"

# Configurar CORS (Lógica dinámica mantenida)
# --- Inicio Configuración CORS ---
local_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173", # Añadido puerto común de Vite
    "http://127.0.0.1:5173",
]
deployed_frontend_url = os.environ.get("FRONTEND_URL")
allowed_origins = local_origins
if deployed_frontend_url:
    print(f"INFO: Añadiendo URL del frontend desplegado a origenes CORS: {deployed_frontend_url}")
    allowed_origins.append(deployed_frontend_url)
else:
    print("INFO: Variable de entorno FRONTEND_URL no establecida. Solo se permitirán orígenes locales por defecto.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- Fin Configuración CORS ---

# Asegurar que existan las carpetas necesarias
DATA_PATH.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_PATH.mkdir(parents=True, exist_ok=True)
print(f"INFO: Usando directorio de datos: {DATA_PATH}")
print(f"INFO: Usando directorio de embeddings: {EMBEDDINGS_PATH}")

# Initialize OpenAI client
# <--- MODIFICADO: Añadir manejo de error si falta la API Key ---
try:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("ERROR CRÍTICO: La variable de entorno OPENAI_API_KEY no está configurada.")
        # Considerar salir si la clave es esencial para el funcionamiento
        # sys.exit(1)
        client = None # Marcar cliente como None si falta la clave
    else:
        client = OpenAI(api_key=openai_api_key)
        # Probar conexión básica (opcional)
        # try:
        #    client.models.list()
        #    print("INFO: Conexión con API de OpenAI exitosa.")
        # except Exception as api_err:
        #    print(f"ERROR: No se pudo conectar a la API de OpenAI: {api_err}")
        #    client = None
except Exception as e:
    print(f"ERROR CRÍTICO: Fallo al inicializar el cliente OpenAI: {e}")
    client = None


# Initialize ChromaDB client
try:
    chroma_client = chromadb.PersistentClient(path=str(EMBEDDINGS_PATH)) # path debe ser string
    print(f"INFO: Cliente ChromaDB inicializado. Colecciones existentes: {[c.name for c in chroma_client.list_collections()]}")
except Exception as e:
    print(f"ERROR CRÍTICO: Fallo al inicializar ChromaDB en {EMBEDDINGS_PATH}: {e}")
    # Considerar manejo de error más robusto si la DB es esencial
    chroma_client = None

# Modelos de datos (Pydantic)
class QueryRequest(BaseModel): # No usado actualmente por /ask
    query: str
    num_results: int = 3

class QueryResponse(BaseModel): # No usado actualmente por /ask
    answer: str
    sources: List[str]

class AskRequest(BaseModel): # <--- NUEVO: Modelo para /ask ---
    question: str
    context_name: str
    system_prompt: Optional[str] = None

class AskResponseMetadata(BaseModel): # <--- NUEVO: Modelo para metadata en /ask ---
    document_name: Optional[str] = None
    page_number: Optional[Union[int, str]] = None # Puede ser número o nombre de hoja
    all_source_documents: List[str] = []
    error: Optional[str] = None # Para devolver errores específicos

class AskResponse(BaseModel): # <--- NUEVO: Modelo para /ask ---
    answer: str
    metadata: AskResponseMetadata

class ContextRequest(BaseModel): # <--- NUEVO: Modelo para crear/cargar contexto ---
    context_name: str

class HistoryItem(BaseModel): # <--- NUEVO: Modelo para historial ---
     query: str
     answer: str

class HistoryRequest(BaseModel): # <--- NUEVO: Modelo para guardar historial ---
    context_name: str
    history: List[HistoryItem]


# <--- MODIFICADO: Prompt por defecto del sistema ---
DEFAULT_SYSTEM_PROMPT = "Eres un asistente útil que responde preguntas basadas únicamente en el contexto proporcionado. Si la respuesta no se encuentra en el contexto, indica que no tienes suficiente información."


# WebSocketManager para manejar conexiones
class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"WebSocket conectado: {websocket.client.host}:{websocket.client.port}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"WebSocket desconectado: {websocket.client.host}:{websocket.client.port}")

    async def broadcast(self, message: str):
        disconnected_websockets = []
        # Crear una copia para iterar en caso de que la lista cambie durante el broadcast
        connections_to_broadcast = self.active_connections[:]
        for connection in connections_to_broadcast:
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                disconnected_websockets.append(connection)
                print(f"WebSocket desconectado durante broadcast: {connection.client.host}:{connection.client.port}")
            except Exception as e:
                # Capturar otros errores como ConnectionClosedOK, etc.
                print(f"Error enviando a WebSocket {connection.client.host}:{connection.client.port}: {type(e).__name__} - {e}")
                disconnected_websockets.append(connection) # Eliminar si hay error

        # Limpiar conexiones desconectadas después del broadcast
        for ws in disconnected_websockets:
            # Asegurarse de no intentar desconectar dos veces si ya se eliminó
            if ws in self.active_connections:
                 self.disconnect(ws)

websocket_manager = WebSocketManager()

# --- Endpoints ---

# Endpoint de "Health Check" simple
@app.get("/health", status_code=200, tags=["Utilities"])
async def health_check():
    """ Endpoint simple para verificar que el servidor está corriendo. """
    # Podría expandirse para verificar conexión a DB, OpenAI, etc.
    db_ok = chroma_client is not None
    openai_ok = client is not None
    return {"status": "ok", "database_connected": db_ok, "openai_client_initialized": openai_ok}


# Endpoint WebSocket genérico para progreso
@app.websocket("/ws/progress")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    client_info = f"{websocket.client.host}:{websocket.client.port}" # Info para logs
    try:
        while True:
            # Mantener la conexión abierta. Podemos recibir 'ping' o mensajes de control.
            data = await websocket.receive_text()
            print(f"WS /ws/progress: Mensaje recibido de {client_info}: {data}")
            # Podrías implementar lógica aquí si el cliente envía comandos específicos
            # Ejemplo: if data == "ping": await websocket.send_text("pong")
    except WebSocketDisconnect:
        print(f"WS /ws/progress: Desconexión explícita de {client_info}")
    except Exception as e:
        print(f"WS /ws/progress: Error en el loop para {client_info}: {type(e).__name__} - {e}")
    finally:
        # Asegurar desconexión al salir del loop por cualquier razón
        websocket_manager.disconnect(websocket)


# Función para extraer texto de documentos (MODIFICADO para mejor metadata y manejo de errores)
async def extract_text_from_document(file_path: Path) -> List[Document]:
    """ Extrae texto y metadata de un archivo usando Langchain Loaders. """
    file_extension = file_path.suffix.lower()
    raw_documents: List[Document] = []
    print(f"Procesando archivo: {file_path.name} (Ext: {file_extension})")

    try:
        if file_extension == '.pdf':
            # Intentar con PyPDFLoader primero
            try:
                loader = PyPDFLoader(str(file_path), extract_images=False) # extract_images=False para rendimiento
                raw_documents = loader.load()
                # Normalizar metadata de página
                for i, doc in enumerate(raw_documents):
                    doc.metadata['source'] = file_path.name
                    page_num = doc.metadata.get('page', -1) # PyPDFLoader usa 'page' (0-based)
                    if not isinstance(page_num, int) or page_num < 0:
                         # Intentar extraer de otra metadata si 'page' no es útil
                         try:
                              page_num_str = str(doc.metadata.get('page_number', '0')) # fallback a 0
                              page_num = int(page_num_str) if page_num_str.isdigit() else 0
                         except: page_num = 0
                    doc.metadata['page'] = page_num # Asegurar que existe 'page' 0-based
                    doc.metadata['chunk_index_in_doc'] = i # Índice original dentro de las páginas/elementos del loader

            except Exception as pdf_err:
                print(f"Error con PyPDFLoader para {file_path.name}: {pdf_err}. Intentando fallback (si aplica)...")
                # Aquí podrías añadir lógica de fallback con OCR si es necesario
                # from langchain_community.document_loaders import PyPDFium2Loader, PyMuPDFLoader
                # loader = PyMuPDFLoader(str(file_path)) # Probar otro loader
                # raw_documents = loader.load()
                # O lógica OCR con Tesseract/Poppler
                raw_documents = [] # Dejar vacío si el primer intento falla por ahora


        elif file_extension == '.docx':
            # <--- MODIFICADO: Usar Unstructured con modo 'elements' ---
            loader = UnstructuredWordDocumentLoader(str(file_path), mode="elements")
            raw_documents = loader.load()
            for i, doc in enumerate(raw_documents):
                 doc.metadata['source'] = file_path.name
                 # Unstructured puede dar 'page_number' (1-based)
                 page_num = doc.metadata.get('page_number')
                 doc.metadata['page'] = int(page_num) - 1 if isinstance(page_num, int) and page_num > 0 else 0 # Convertir a 0-based
                 doc.metadata['element_type'] = doc.metadata.get('category', 'Unknown') # e.g., 'Title', 'ListItem'
                 doc.metadata['chunk_index_in_doc'] = i

        elif file_extension in ['.xlsx', '.xlsm']:
            # <--- MODIFICADO: Usar Unstructured con modo 'elements' ---
            loader = UnstructuredExcelLoader(str(file_path), mode="elements")
            raw_documents = loader.load()
            for i, doc in enumerate(raw_documents):
                 doc.metadata['source'] = file_path.name
                 # 'page_number' en Excel a menudo es el nombre de la hoja
                 sheet_name = doc.metadata.get('page_number', f'Sheet_{i+1}') # Fallback a Sheet_N
                 doc.metadata['sheet_name'] = str(sheet_name)
                 doc.metadata['page'] = 0 # Página numérica no aplica directamente
                 doc.metadata['element_type'] = doc.metadata.get('category', 'Unknown') # e.g., 'Table'
                 doc.metadata['chunk_index_in_doc'] = i
        # <--- NUEVO: Añadir soporte para TXT y CSV ---
        elif file_extension == '.txt':
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(str(file_path), encoding='utf-8') # Especificar encoding
            raw_documents = loader.load()
            # Añadir metadata básica para TXT
            for i, doc in enumerate(raw_documents):
                 doc.metadata['source'] = file_path.name
                 doc.metadata['page'] = 0
                 doc.metadata['element_type'] = 'Text'
                 doc.metadata['chunk_index_in_doc'] = i
        elif file_extension == '.csv':
             from langchain_community.document_loaders import CSVLoader
             # CSVLoader carga por fila, puede generar muchos documentos pequeños
             loader = CSVLoader(str(file_path), encoding='utf-8', source_column='source') # Asumir columna 'source' si existe
             raw_documents = loader.load()
             for i, doc in enumerate(raw_documents):
                  if 'source' not in doc.metadata: # Añadir filename si 'source_column' no funcionó
                      doc.metadata['source'] = file_path.name
                  doc.metadata['row'] = doc.metadata.get('row', i + 1) # CSVLoader añade 'row' (1-based)
                  doc.metadata['page'] = 0
                  doc.metadata['element_type'] = 'CSVRow'
                  doc.metadata['chunk_index_in_doc'] = i

        else:
            print(f"Formato de archivo no soportado: {file_extension} para {file_path.name}")
            return [] # Retornar lista vacía para formatos no soportados

    except Exception as e:
        print(f"ERROR FATAL: Error procesando documento {file_path.name}: {e}")
        traceback.print_exc()
        return [] # Retornar lista vacía en caso de error grave

    # Filtrar documentos vacíos que algunos loaders pueden generar
    # <--- MODIFICADO: Hacer el filtrado y log más informativo ---
    original_count = len(raw_documents)
    raw_documents = [doc for doc in raw_documents if doc.page_content and doc.page_content.strip()]
    filtered_count = len(raw_documents)
    if original_count != filtered_count:
         print(f"INFO: Filtrados {original_count - filtered_count} elementos vacíos de {file_path.name}.")

    print(f"Documento {file_path.name} cargado. {filtered_count} elementos extraídos.")
    return raw_documents


# Función para procesar documentos y generar embeddings (MODIFICADO para IDs únicos y progreso WS)
async def process_document_chunks(
    chunks: List[Document],
    context_name: str,
    filename: str,
    websocket_manager: WebSocketManager, # <-- Añadido
    file_index: int,                    # <-- Añadido (0-based index of current file)
    total_files: int                    # <-- Añadido
) -> int: # Retorna el número de chunks añadidos exitosamente
    """Procesa una lista de chunks de documento, genera embeddings y los añade a ChromaDB."""
    if not chunks:
        print(f"INFO: No hay chunks para procesar para {filename} en contexto {context_name}")
        return 0
    if not client:
        print(f"ERROR: Cliente OpenAI no inicializado, no se pueden generar embeddings para {filename}.")
        return 0
    if not chroma_client:
        print(f"ERROR: Cliente ChromaDB no inicializado, no se pueden guardar embeddings para {filename}.")
        return 0


    total_chunks_in_file = len(chunks)
    print(f"Procesando {total_chunks_in_file} chunks para {filename} (Archivo {file_index + 1}/{total_files})")
    embeddings = []
    documents_content = []
    metadatas = []
    ids = []
    processed_chunk_count = 0
    # last_reported_percentage = -1 # Para throttling basado en % global

    # Constante para throttling (envía actualización cada N chunks)
    UPDATE_EVERY_N_CHUNKS = 10 # Ajustar según necesidad (menos frecuente para archivos grandes)

    start_time_file_processing = time.time()

    for i, chunk in enumerate(chunks):
        page_content = chunk.page_content
        # Limpiar un poco el texto antes de embeber? (Opcional)
        page_content = page_content.strip()
        page_content = re.sub(r'\s+', ' ', page_content) # Reemplazar múltiples espacios/saltos

        if not page_content:
            # print(f"Skipping empty chunk {i} for file {filename}") # Log redundante si ya filtramos
            continue

        try:
            # --- Generar embedding ---
            start_embed_time = time.time()
            response = client.embeddings.create(
                input=[page_content], # Enviar como lista, aunque sea un solo elemento
                model="text-embedding-3-small" # Usar el modelo apropiado
            )
            # print(f"Embedding chunk {i+1}/{total_chunks_in_file} for {filename} took {time.time() - start_embed_time:.2f}s") # Debug

            if response and response.data:
                chunk_embedding = response.data[0].embedding
            else:
                print(f"ADVERTENCIA: No se generó embedding para chunk {i} de {filename}. Saltando chunk.")
                continue # Saltar este chunk si falla el embedding

            # --- Preparar datos para ChromaDB ---
            embeddings.append(chunk_embedding)
            documents_content.append(page_content) # Guardar contenido limpio
            # <--- MODIFICADO: Metadata más completa ---
            chunk_metadata = {
                "source": filename,
                "page": chunk.metadata.get("page", 0), # 0-based page or 0
                "sheet_name": chunk.metadata.get("sheet_name", "N/A"), # Excel sheet name
                "element_type": chunk.metadata.get("element_type", "N/A"), # Type of element (Title, Table, etc.)
                "chunk_index_in_doc": chunk.metadata.get("chunk_index_in_doc", i), # Original index from loader
                "overall_chunk_index": i # Index within the final list of chunks for this file
            }
            metadatas.append(chunk_metadata)
            # <--- MODIFICADO: Usar UUID para IDs únicos ---
            ids.append(f"{context_name}_{filename}_chunk_{i}_{uuid.uuid4()}") # ID más robusto
            processed_chunk_count += 1

            # --- Calcular y enviar progreso detallado vía WebSocket ---
            progress_within_file = (i + 1) / total_chunks_in_file
            # Calcula el porcentaje global basado en archivos completados + progreso en el archivo actual
            overall_percentage = int(
                ((file_index / total_files) + (progress_within_file / total_files)) * 100
            )
            overall_percentage = min(overall_percentage, 100) # Asegurar que no pase de 100

            # Throttling: Enviar cada N chunks o en el último chunk
            if (i + 1) % UPDATE_EVERY_N_CHUNKS == 0 or (i + 1) == total_chunks_in_file:
                # print(f"WS Update: File {file_index+1}/{total_files}, Chunk {i+1}/{total_chunks_in_file}, Overall: {overall_percentage}%") # Debug
                await websocket_manager.broadcast(json.dumps({
                    "type": "progress",
                    "current_file": filename,
                    "files_processed": file_index, # Archivos completamente terminados antes de este
                    "total_files": total_files,
                    "percentage": overall_percentage,
                    "status_message": f"Procesando {filename}: Chunk {i + 1}/{total_chunks_in_file}"
                }))
                # last_reported_percentage = overall_percentage # Para throttling por %

        except Exception as e:
            print(f"ERROR: Error procesando chunk {i} de {filename}: {str(e)}")
            # Considerar si continuar o fallar el archivo entero
            continue # Saltar este chunk si falla

    # --- Añadir a ChromaDB ---
    if not documents_content:
        print(f"INFO: No hay contenido válido para añadir para {filename} después de procesar chunks.")
        return 0

    try:
        print(f"Almacenando {len(documents_content)} embeddings en ChromaDB para {filename} en colección '{context_name}'...")
        start_add_time = time.time()
        # <--- MODIFICADO: Asegurarse que la colección existe antes de añadir ---
        collection = chroma_client.get_or_create_collection(name=context_name) # Get or create es seguro
        collection.add(
            embeddings=embeddings,
            documents=documents_content,
            metadatas=metadatas,
            ids=ids
        )
        add_duration = time.time() - start_add_time
        total_duration = time.time() - start_time_file_processing
        print(f"Embeddings para {filename} ({len(documents_content)} chunks) almacenados en {add_duration:.2f}s. Tiempo total procesamiento archivo: {total_duration:.2f}s.")
        return len(documents_content) # Retorna el número de chunks añadidos
    except Exception as e:
        print(f"ERROR FATAL: Error añadiendo chunks a ChromaDB para {filename}: {e}")
        traceback.print_exc()
        # Enviar error por WebSocket si falla la adición a la DB
        await websocket_manager.broadcast(json.dumps({
            "type": "error_file", "filename": filename,
            "message": f"Error al guardar embeddings en DB: {str(e)}",
            "files_processed": file_index, "total_files": total_files,
            "percentage": int(((file_index + 1) / total_files) * 100) if total_files > 0 else 0 # Marcar como completado (con error)
        }))
        return 0 # Indica que no se añadieron chunks

# --- Rutas API ---

@app.post("/create-context", status_code=201, tags=["Context Management"])
async def create_context(context_req: ContextRequest):
    """ Crea un nuevo contexto (directorio de datos y colección en ChromaDB). """
    context_name = context_req.context_name
    print(f"Solicitud para crear contexto: {context_name}")

    if not validate_context_name(context_name):
        raise HTTPException(status_code=400, detail=f"Nombre de contexto inválido: '{context_name}'. Debe ser alfanumérico, guiones bajos/medios permitidos, máx 63 chars, sin '..'.")
    if not chroma_client:
         raise HTTPException(status_code=503, detail="Servicio de base de datos (ChromaDB) no disponible.")

    context_data_path = DATA_PATH / context_name
    collection_exists = False
    try:
        chroma_client.get_collection(name=context_name)
        collection_exists = True
        print(f"INFO: Colección ChromaDB '{context_name}' ya existe.")
    except Exception:
        collection_exists = False
        print(f"INFO: Colección ChromaDB '{context_name}' no encontrada, se intentará crear.")

    try:
        # Crear directorio de datos si no existe
        context_data_path.mkdir(parents=True, exist_ok=True)
        print(f"INFO: Directorio de datos '{context_data_path}' asegurado.")

        # Crear colección en ChromaDB si no existía
        if not collection_exists:
            chroma_client.create_collection(name=context_name)
            print(f"INFO: Colección ChromaDB '{context_name}' creada exitosamente.")
        # Si ya existía, simplemente confirmamos
        return {"message": f"Contexto '{context_name}' asegurado (directorio y colección DB)."}

    except chromadb.errors.DuplicateCollectionError:
         # Esto no debería pasar si ya verificamos, pero por si acaso
         print(f"ADVERTENCIA: Intento de crear colección duplicada '{context_name}' (ya existía).")
         return {"message": f"Contexto '{context_name}' ya existía."}
    except Exception as e:
        print(f"ERROR: Error creando/verificando contexto '{context_name}': {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error interno del servidor al gestionar el contexto: {str(e)}")


@app.get("/get-contexts", tags=["Context Management"])
async def get_contexts():
    """ Obtiene la lista de contextos disponibles (basado en colecciones de ChromaDB). """
    if not chroma_client:
         print("ADVERTENCIA: get_contexts llamado pero ChromaDB no está disponible.")
         return {"contexts": []}
    try:
        collections = chroma_client.list_collections()
        context_names = sorted([col.name for col in collections])
        print(f"Contextos encontrados en ChromaDB: {context_names}")
        return {"contexts": context_names}
    except Exception as e:
        print(f"Error obteniendo lista de contextos de ChromaDB: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error interno del servidor al listar contextos: {str(e)}")


@app.post("/upload-documents", tags=["Document Management"])
async def upload_documents(context_name: str = Form(...), files: List[UploadFile] = File(...)):
    """ Sube uno o más archivos a un contexto existente, los procesa y genera embeddings. """
    print(f"Solicitud de subida para contexto '{context_name}' con {len(files)} archivo(s).")
    if not validate_context_name(context_name):
        raise HTTPException(status_code=400, detail=f"Nombre de contexto inválido: '{context_name}'")
    if not chroma_client:
         raise HTTPException(status_code=503, detail="Servicio de base de datos (ChromaDB) no disponible para guardar embeddings.")
    if not client:
         raise HTTPException(status_code=503, detail="Servicio de OpenAI no disponible para generar embeddings.")

    # Verificar que la colección del contexto existe en ChromaDB
    try:
        collection = chroma_client.get_or_create_collection(name=context_name) # Crea si no existe
        print(f"INFO: Usando colección ChromaDB '{collection.name}'")
    except Exception as e:
        print(f"ERROR: No se pudo obtener/crear la colección ChromaDB '{context_name}': {e}")
        raise HTTPException(status_code=404, detail=f"Contexto (colección DB) '{context_name}' no encontrado o no se pudo crear.")

    context_data_path = DATA_PATH / context_name
    context_data_path.mkdir(parents=True, exist_ok=True) # Asegurar que la carpeta existe

    results = []
    total_files = len(files)
    processed_files_count = 0
    total_chunks_processed_session = 0

    # Enviar mensaje inicial al WebSocket
    await websocket_manager.broadcast(json.dumps({
        "type": "start_upload",
        "total_files": total_files,
        "context": context_name
    }))

    start_session_time = time.time()

    for index, file in enumerate(files):
        # Sanitize filename (opcional pero recomendado)
        # filename = secure_filename(file.filename) # Necesitarías una función `secure_filename`
        filename = file.filename # Usar nombre original por ahora
        # Crear ruta completa usando pathlib
        file_path = context_data_path / filename
        status = "Error" # Estado por defecto
        chunks_count = 0
        file_details = ""
        file_start_time = time.time()

        # Enviar estado inicial para este archivo al WebSocket
        await websocket_manager.broadcast(json.dumps({
            "type": "progress",
            "current_file": filename,
            "files_processed": index, # Archivos completados *antes* de este
            "total_files": total_files,
            # Porcentaje inicial (basado en archivos ya iniciados)
            "percentage": int((index / total_files) * 100) if total_files > 0 else 0,
            "status_message": f"Iniciando archivo {index + 1}/{total_files}: {filename}"
        }))

        try:
            # 1. Guardar archivo temporalmente
            print(f"Guardando archivo: {file_path}")
            try:
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                print(f"Archivo guardado temporalmente en: {file_path}")
            except Exception as save_err:
                print(f"ERROR: No se pudo guardar el archivo {filename}: {save_err}")
                raise HTTPException(status_code=500, detail=f"Error al guardar archivo {filename}") # Detener si no se puede guardar

            # 2. Extraer texto
            print(f"Extrayendo texto de: {filename}")
            text_documents = await extract_text_from_document(file_path) # Usa pathlib Path

            if not text_documents:
                status = "Error"
                file_details = "No se pudo extraer texto (vacío, corrupto o formato no soportado)."
                print(f"ADVERTENCIA: {file_details} para {filename}")
                # Enviar error específico del archivo por WS
                await websocket_manager.broadcast(json.dumps({
                    "type": "error_file", "filename": filename, "message": file_details,
                    "files_processed": index, "total_files": total_files,
                    "percentage": int(((index + 1) / total_files) * 100) if total_files > 0 else 0
                }))

            else:
                # 3. Dividir en chunks
                print(f"Dividiendo texto de {filename}...")
                # <--- MODIFICADO: Ajustar chunk_size/overlap si es necesario ---
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150, # Reducir overlap ligeramente?
                    length_function=len,
                    is_separator_regex=False,
                )
                chunks = text_splitter.split_documents(text_documents)
                print(f"{len(chunks)} chunks creados para {filename}")

                if not chunks:
                    status = "Error"
                    file_details = "No se generaron chunks a partir del texto extraído."
                    print(f"ADVERTENCIA: {file_details} para {filename}")
                    await websocket_manager.broadcast(json.dumps({
                       "type": "error_file", "filename": filename, "message": file_details,
                       "files_processed": index, "total_files": total_files,
                       "percentage": int(((index + 1) / total_files) * 100) if total_files > 0 else 0
                   }))
                else:
                    # 4. Procesar chunks (generar embeddings y guardar en DB)
                    # <--- MODIFICADO: Llamar a process_document_chunks con WS manager ---
                    num_added = await process_document_chunks(
                        chunks, context_name, filename,
                        websocket_manager, index, total_files # Pasar info necesaria
                    )

                    if num_added > 0:
                        status = "Procesado" # Cambiado de "Procesado correctamente"
                        chunks_count = num_added
                        total_chunks_processed_session += num_added
                    # Si num_added es 0, process_document_chunks ya habrá logueado/enviado error WS
                    elif status != "Error": # Evitar sobrescribir error de extracción
                         status = "Error"
                         file_details = file_details or "No se guardaron chunks en DB (error de embedding o DB)."
                         print(f"ADVERTENCIA: {file_details} para {filename}")
                         # El error WS ya se envió desde process_document_chunks si falló allí

            # Incrementar contador de archivos que *intentaron* procesarse (incluso si fallaron)
            processed_files_count += 1

            file_duration = time.time() - file_start_time
            print(f"Archivo {filename} ({(file_duration):.2f}s) finalizado con estado: {status}. Chunks añadidos: {chunks_count}")

        except HTTPException as he:
             # Re-lanzar HTTPException para que FastAPI la maneje (p.ej., error al guardar archivo)
             status = "Error Crítico"
             file_details = str(he.detail)
             processed_files_count += 1 # Contar como procesado aunque falle aquí
             # Asegurarse de enviar error por WS
             await websocket_manager.broadcast(json.dumps({
                    "type": "error_file", "filename": filename, "message": file_details,
                    "files_processed": processed_files_count, # Usar contador actualizado
                    "total_files": total_files,
                    "percentage": int((processed_files_count / total_files) * 100) if total_files > 0 else 0
                }))
             # No relanzar, añadir al resultado y continuar con otros archivos si los hay

        except Exception as e:
            # Capturar otros errores inesperados durante el procesamiento del archivo
            processed_files_count += 1 # Contar como procesado aunque falle
            status = "Error Inesperado"
            file_details = str(e)
            print(f"ERROR INESPERADO procesando archivo {filename}: {e}")
            traceback.print_exc()
            # Enviar error por WS
            await websocket_manager.broadcast(json.dumps({
                "type": "error_file", "filename": filename, "message": file_details,
                "files_processed": processed_files_count, # Usar contador actualizado
                "total_files": total_files,
                "percentage": int((processed_files_count / total_files) * 100) if total_files > 0 else 0
            }))

        finally:
            # --- INICIO: LÓGICA DE BORRADO DEL ARCHIVO ---
            if file_path.exists(): # Verificar si el archivo existe antes de intentar borrar
                try:
                    print(f"Intentando eliminar archivo temporal: {file_path}")
                    os.remove(file_path) # Usar os.remove con Path convertido a str o directamente file_path.unlink()
                    # file_path.unlink() # Alternativa con pathlib
                    print(f"Archivo temporal {file_path} eliminado exitosamente.")
                except OSError as remove_error:
                    print(f"ALERTA: No se pudo eliminar el archivo temporal {file_path}. Error: {remove_error}")
                    # Guardar este error en los detalles si no hubo otro error peor
                    if status != "Error Crítico" and status != "Error Inesperado":
                         file_details = f"{file_details} (Advertencia: no se pudo eliminar el archivo temporal: {remove_error})"
            else:
                # Si el archivo no existe aquí, probablemente falló al guardarse.
                print(f"INFO: Archivo temporal {file_path} no encontrado para eliminar (posiblemente no se guardó).")
            # --- FIN: LÓGICA DE BORRADO DEL ARCHIVO ---

            # Guardar resultado para este archivo
            results.append({"filename": filename, "status": status, "chunks": chunks_count, "details": file_details})

    # Mensaje final global al WebSocket
    session_duration = time.time() - start_session_time
    final_status_message = f"Proceso completado para '{context_name}' en {session_duration:.2f}s. {processed_files_count}/{total_files} archivos procesados. Total chunks añadidos en sesión: {total_chunks_processed_session}."
    print(final_status_message)
    await websocket_manager.broadcast(json.dumps({
        "type": "complete_upload",
        "message": final_status_message,
        "results": results, # Enviar resultados detallados
        "percentage": 100
    }))

    # Devolver respuesta HTTP (podría ser 200 OK incluso si hubo errores parciales)
    # El frontend debe mirar los 'results' para ver el detalle
    # Opcional: Cambiar a código 207 Multi-Status si hubo errores parciales
    # status_code = 207 if any(r['status'] != 'Procesado' for r in results) else 200
    return JSONResponse(content={"message": "Proceso de subida finalizado.", "results": results})

# Endpoint /get-pdfs obsoleto? No se usa en el frontend y no sigue la lógica de ChromaDB.
# Se podría reimplementar como /get-documents-in-context/{context_name} si es necesario,
# consultando los metadatos en ChromaDB para obtener la lista de archivos ('source').
# Por ahora, lo comentamos.
# @app.get("/get-pdfs", tags=["Document Management"])
# async def get_pdfs(context_name: str): ...


# Endpoint /upload obsoleto? Parece que /upload-documents hace el trabajo.
# @app.post("/upload", tags=["Document Management"])
# async def upload_files(...): ...


# Endpoint /query obsoleto? Parece que /ask hace lo mismo con más detalle.
# @app.post("/query", tags=["Querying"])
# async def query_documents(...): ...


@app.post("/ask", response_model=AskResponse, tags=["Querying"])
async def ask_question(ask_req: AskRequest):
    """ Recibe una pregunta, busca en un contexto y genera una respuesta usando un LLM. """
    question = ask_req.question
    context_name = ask_req.context_name
    # <--- MODIFICADO: Usar prompt del sistema recibido o el default ---
    system_prompt = ask_req.system_prompt or DEFAULT_SYSTEM_PROMPT # Usa el default si es None o ""

    print(f"Recibida pregunta para contexto '{context_name}': \"{question[:100]}...\"")
    # print(f"Usando system prompt: \"{system_prompt[:100]}...\"") # Debug

    if not client or not chroma_client:
         print("ERROR: OpenAI o ChromaDB no están inicializados.")
         # Devolver error 503 Service Unavailable
         error_meta = AskResponseMetadata(error="Servicios backend no disponibles.")
         return JSONResponse(status_code=503, content=AskResponse(answer="Lo siento, uno de los servicios necesarios no está disponible.", metadata=error_meta).model_dump())

    if not validate_context_name(context_name):
         raise HTTPException(status_code=400, detail=f"Nombre de contexto inválido: '{context_name}'")

    # Verificar que la colección del contexto existe
    try:
        collection = chroma_client.get_collection(name=context_name)
        print(f"INFO: Colección '{context_name}' encontrada en ChromaDB.")
    except Exception as e:
        print(f"ERROR: Colección '{context_name}' no encontrada en ChromaDB: {e}")
        raise HTTPException(status_code=404, detail=f"Contexto (colección DB) '{context_name}' no encontrado.")

    # Generar embedding para la consulta
    print("Generando embedding para la pregunta...")
    start_time = time.time()
    try:
        query_response = client.embeddings.create(
            input=[question], # Enviar como lista
            model="text-embedding-3-small" # Asegurarse que coincide con el modelo de indexación
        )
        query_embedding = query_response.data[0].embedding
        print(f"Embedding generado en {time.time() - start_time:.2f}s")
    except Exception as e:
         print(f"ERROR: Fallo al generar embedding para la pregunta: {e}")
         raise HTTPException(status_code=500, detail=f"Error al procesar la pregunta (embedding): {str(e)}")

    # Buscar documentos relevantes en ChromaDB
    print("Buscando documentos relevantes en ChromaDB...")
    start_time = time.time()
    try:
        # <--- MODIFICADO: Ajustar n_results y considerar where clause si es necesario ---
        # n_results: Número de *chunks* a recuperar. Ajustar según la longitud esperada del contexto.
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=15, # Recuperar más chunks para contexto más amplio? Ajustar.
            include=['documents', 'metadatas', 'distances'] # Incluir metadatos y distancias
            # Ejemplo de where clause:
            # where={"source": "mi_documento_especifico.pdf"}
            # where_document={"$contains":"palabra_clave"}
        )
        num_results = len(results.get('ids', [[]])[0])
        print(f"Búsqueda completada en {time.time() - start_time:.2f}s. Encontrados {num_results} chunks relevantes.")
    except Exception as e:
        print(f"ERROR: Error buscando en ChromaDB: {e}")
        raise HTTPException(status_code=500, detail=f"Error al buscar información relevante: {str(e)}")

    # Procesar resultados de ChromaDB
    metadata_response = AskResponseMetadata() # Inicializar metadata
    if not results or not results.get('ids') or not results['ids'][0]:
        print("ADVERTENCIA: No se encontraron documentos relevantes en ChromaDB.")
        metadata_response.error = "No relevant documents found."
        return AskResponse(
            answer="No pude encontrar información relevante en los documentos cargados para responder a tu pregunta.",
            metadata=metadata_response
        )

    relevant_docs_content = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0] if results.get('distances') else [0.0] * len(relevant_docs_content)

    # <--- MODIFICADO: Extraer fuentes únicas y metadata detallada ---
    source_files: Set[str] = set()
    detailed_sources_info = [] # Para logs o debug
    primary_doc_name = None
    primary_page_info = None

    for i, meta in enumerate(metadatas):
         doc_name = meta.get('source', f'Desconocido_{i}')
         source_files.add(doc_name)

         page_num = meta.get('page', -1) # 0-based page number
         sheet_name = meta.get('sheet_name', 'N/A')
         element_type = meta.get('element_type', 'N/A')
         chunk_idx = meta.get('overall_chunk_index', i) # Usar índice del chunk procesado
         distance = distances[i]

         # Determinar la información de página/hoja a mostrar
         page_display = "N/A"
         if sheet_name != 'N/A':
              page_display = f"Hoja '{sheet_name}'"
         elif isinstance(page_num, int) and page_num >= 0:
              page_display = f"Página {page_num + 1}" # Mostrar 1-based

         source_info_str = f"{doc_name} ({page_display}, ChunkIdx: {chunk_idx}, Type: {element_type}, Dist: {distance:.4f})"
         detailed_sources_info.append(source_info_str)

         # Guardar el nombre y página/hoja del *primer* resultado como referencia principal
         if i == 0:
             primary_doc_name = doc_name
             if sheet_name != 'N/A':
                 primary_page_info = sheet_name # Guardar nombre de hoja
             elif isinstance(page_num, int) and page_num >= 0:
                 primary_page_info = page_num + 1 # Guardar número de página 1-based
             else:
                 primary_page_info = None # No hay info de página/hoja útil

    print(f"Documentos fuente únicos encontrados: {source_files}")
    # print(f"Fuentes detalladas recuperadas: {detailed_sources_info}") # Debug

    # Actualizar metadata de la respuesta
    metadata_response.document_name = primary_doc_name
    metadata_response.page_number = primary_page_info # Puede ser int (página) o str (hoja)
    metadata_response.all_source_documents = sorted(list(source_files))

    # Construir el contexto para el LLM
    context_text = "\n\n---\n\n".join(relevant_docs_content) # Separador claro entre chunks

    # Construir prompt final para el LLM
    final_user_prompt = f"""Basándote *única y exclusivamente* en el siguiente contexto extraído de documentos, responde a la pregunta del usuario. No utilices conocimiento externo. Si la respuesta no está en el contexto, indícalo claramente.

Contexto de documentos:
\"\"\"
{context_text}
\"\"\"

Pregunta del usuario:
{question}
"""

    # Generar respuesta usando LLM
    print("Generando respuesta con LLM...")
    start_time = time.time()
    try:
        # <--- MODIFICADO: Usar modelo y parámetros adecuados ---
        completion = client.chat.completions.create(
            model="gpt-4o-mini", # O el modelo que prefieras/tengas disponible
            messages=[
                {"role": "system", "content": system_prompt}, # Usar prompt del sistema (default o del request)
                {"role": "user", "content": final_user_prompt}
            ],
            temperature=0.1, # Temperatura baja para respuestas basadas en contexto
            max_tokens=1000, # Limitar longitud de respuesta si es necesario
        )
        answer = completion.choices[0].message.content
        # Opcional: Limpiar saltos de línea iniciales/finales
        answer = answer.strip() if answer else "No se pudo generar una respuesta."

        print(f"Respuesta LLM generada en {time.time() - start_time:.2f}s")

    except Exception as e:
         print(f"ERROR: Fallo en la llamada a la API de OpenAI: {e}")
         traceback.print_exc()
         # Devolver error específico en la respuesta
         metadata_response.error = f"Error al generar respuesta del LLM: {str(e)}"
         return JSONResponse(
              status_code=500,
              content=AskResponse(answer="Lo siento, ocurrió un error al intentar generar la respuesta final.", metadata=metadata_response).model_dump()
          )

    # <--- MODIFICADO: Añadir nota si viene de múltiples documentos ---
    if len(source_files) > 1:
        source_list_str = ", ".join(metadata_response.all_source_documents)
        answer += f"\n\n*Nota: La información para esta respuesta puede provenir de múltiples documentos ({source_list_str}). La referencia principal indicada es la del fragmento más relevante.*"

    # Devolver respuesta final
    return AskResponse(answer=answer, metadata=metadata_response)


@app.delete("/delete-context/{context_name}", status_code=200, tags=["Context Management"])
async def delete_specific_context(context_name: str):
    """ Elimina un contexto completo (colección de ChromaDB y directorio de datos). """
    print(f"Solicitud para eliminar contexto: {context_name}")
    if not validate_context_name(context_name):
        raise HTTPException(status_code=400, detail="Nombre de contexto inválido.")
    if not chroma_client:
        raise HTTPException(status_code=503, detail="Servicio de base de datos (ChromaDB) no disponible.")

    context_data_path = DATA_PATH / context_name
    deleted_data = False
    deleted_collection = False
    collection_existed = False
    data_existed = context_data_path.is_dir() # Verificar si el directorio existe

    # 1. Intentar eliminar colección de ChromaDB
    try:
        # Primero verificar si existe para saber si realmente se borró algo
        try:
             chroma_client.get_collection(name=context_name)
             collection_existed = True
             print(f"INFO: Colección ChromaDB '{context_name}' encontrada, procediendo a eliminar.")
        except Exception:
             collection_existed = False
             print(f"INFO: Colección ChromaDB '{context_name}' no encontrada (nada que eliminar en DB).")

        if collection_existed:
            chroma_client.delete_collection(name=context_name)
            print(f"INFO: Colección ChromaDB '{context_name}' eliminada exitosamente.")
            deleted_collection = True

    except Exception as e:
        # Si falla la eliminación después de confirmar que existía, es un problema
        print(f"ERROR: No se pudo eliminar la colección ChromaDB '{context_name}': {e}")
        traceback.print_exc()
        # No lanzar error aquí todavía, intentar borrar carpeta de datos de todas formas
        # Pero marcar que la eliminación de la colección falló si existía
        if collection_existed: deleted_collection = False

    # 2. Eliminar carpeta de datos si existe
    if data_existed:
        try:
            print(f"Intentando eliminar directorio de datos: {context_data_path}")
            shutil.rmtree(context_data_path)
            print(f"INFO: Directorio de datos '{context_data_path}' eliminado exitosamente.")
            deleted_data = True
        except Exception as e:
            print(f"ERROR: Error eliminando directorio de datos {context_data_path}: {e}")
            traceback.print_exc()
            # Si falló la eliminación de datos, es un error grave, informar
            raise HTTPException(status_code=500, detail=f"Error al eliminar los archivos del contexto '{context_name}': {str(e)}. La colección DB puede haber sido eliminada.")
    else:
        print(f"INFO: Directorio de datos '{context_data_path}' no encontrado (nada que eliminar en filesystem).")

    # Determinar mensaje final y código de estado
    if deleted_data or deleted_collection:
         return {"message": f"Contexto '{context_name}' eliminado. Datos: {'Eliminados' if deleted_data else ('No encontrados' if not data_existed else 'Error al eliminar')}, Colección DB: {'Eliminada' if deleted_collection else ('No encontrada' if not collection_existed else 'Error al eliminar')}"}
    elif not data_existed and not collection_existed:
         # Si ni la carpeta ni la colección existían, retornar 404 Not Found
         raise HTTPException(status_code=404, detail=f"Contexto '{context_name}' no encontrado para eliminar.")
    else:
         # Caso raro donde algo falló pero no se lanzó excepción antes
          return JSONResponse(status_code=500, content={"message": f"Contexto '{context_name}' no se pudo eliminar completamente. Revise los logs."})


# Endpoint /delete-context obsoleto? El anterior con {context_name} es más RESTful.
# @app.delete("/delete-context", tags=["Context Management"])
# async def delete_context(...): ...


# --- Funciones y Endpoints para guardar/cargar historial ---

def _get_history_file_path(context_name: str) -> Path:
    """ Helper para obtener la ruta del archivo de historial. """
    # <--- MODIFICADO: Asegurarse de que el directorio base existe ---
    context_dir = DATA_PATH / context_name
    context_dir.mkdir(parents=True, exist_ok=True) # Crea el directorio si no existe
    return context_dir / "chat_history.json"

def save_chat_history(context_name: str, history: List[Dict[str, str]]):
    """ Guarda el historial de chat para un contexto en un archivo JSON. """
    history_file = _get_history_file_path(context_name)
    try:
        # Guardar como JSON array de objetos {"query": ..., "answer": ...}
        history_to_save = [{"query": item.get("query", ""), "answer": item.get("answer", "")} for item in history]
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_to_save, f, ensure_ascii=False, indent=2)
        print(f"Historial ({len(history_to_save)} items) guardado para '{context_name}' en {history_file}")
    except Exception as e:
        print(f"ERROR: Error guardando historial para '{context_name}': {e}")
        # Considerar lanzar excepción o manejar de otra forma

def load_chat_history(context_name: str) -> List[Dict[str, str]]:
    """ Carga el historial de chat para un contexto desde un archivo JSON. """
    history_file = _get_history_file_path(context_name)
    if not history_file.is_file():
        print(f"INFO: No se encontró archivo de historial para '{context_name}' en {history_file}.")
        return []
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            history_data = json.load(f)
            # Validar que sea una lista (opcional pero bueno)
            if not isinstance(history_data, list):
                 print(f"ERROR: Archivo de historial para '{context_name}' no contiene una lista JSON.")
                 return []
            # Validar estructura interna (opcional)
            validated_history = []
            for item in history_data:
                if isinstance(item, dict) and "query" in item and "answer" in item:
                    validated_history.append({"query": str(item["query"]), "answer": str(item["answer"])})
                else:
                    print(f"ADVERTENCIA: Item inválido encontrado en historial de '{context_name}', saltando: {item}")

            print(f"Historial ({len(validated_history)} items) cargado para '{context_name}' desde {history_file}")
            return validated_history
    except json.JSONDecodeError:
        print(f"ERROR: Archivo de historial para '{context_name}' ({history_file}) está corrupto (JSON inválido).")
        # Opcional: Mover/renombrar archivo corrupto
        # corrupt_file_path = history_file.with_suffix(".corrupt.json")
        # history_file.rename(corrupt_file_path)
        return []
    except Exception as e:
        print(f"ERROR: Error cargando historial para '{context_name}': {e}")
        return []

@app.post("/save-chat-history", status_code=200, tags=["Chat History"])
async def save_chat_history_endpoint(history_req: HistoryRequest):
    """ Guarda el historial de chat proporcionado para un contexto. """
    context_name = history_req.context_name
    history = history_req.history # Es List[HistoryItem] gracias a Pydantic

    if not validate_context_name(context_name):
        raise HTTPException(status_code=400, detail="Nombre de contexto inválido.")

    # Convertir List[HistoryItem] a List[Dict[str, str]] para la función de guardado
    history_dicts = [item.model_dump() for item in history]

    try:
        save_chat_history(context_name, history_dicts)
        return {"message": "Historial guardado correctamente"}
    except Exception as e:
        # La función save_chat_history ya loguea el error
        raise HTTPException(status_code=500, detail=f"Error interno al guardar el historial: {str(e)}")

@app.post("/load-chat-history", response_model=Dict[str, List[HistoryItem]], tags=["Chat History"])
async def load_chat_history_endpoint(context_req: ContextRequest):
    """ Carga el historial de chat para un contexto. """
    context_name = context_req.context_name

    if not validate_context_name(context_name):
        raise HTTPException(status_code=400, detail="Nombre de contexto inválido.")

    try:
        history_dicts = load_chat_history(context_name)
        # Convertir List[Dict] de vuelta a List[HistoryItem] para cumplir el response_model
        history_items = [HistoryItem(**item) for item in history_dicts]
        return {"history": history_items}
    except Exception as e:
        # La función load_chat_history ya loguea el error
        raise HTTPException(status_code=500, detail=f"Error interno al cargar el historial: {str(e)}")


# Función de validación de nombres de contexto (Sin cambios, pero con comentario sobre reglas)
def validate_context_name(context_name: Optional[str]) -> bool:
    """ Valida si un nombre de contexto es válido para ChromaDB y filesystem. """
    if not context_name: return False

    # Reglas de ChromaDB (aproximadas, consultar su documentación para precisión):
    # 1. Longitud: 3 a 63 caracteres.
    # 2. Contenido: Letras minúsculas (a-z), números (0-9), guión bajo (_), guión medio (-), punto (.).
    # 3. No puede empezar ni terminar con punto (.).
    # 4. No puede tener dos puntos seguidos (..).
    # 5. No puede ser una dirección IP.

    # Simplificando un poco para filesystem y uso común:
    if not (3 <= len(context_name) <= 63):
        print(f"Validación fallida (longitud): '{context_name}' (len={len(context_name)})")
        return False
    # Permitir letras (mayúsculas y minúsculas), números, guión bajo, guión medio.
    # No permitir puntos para evitar problemas con filesystem/URLs.
    # Debe empezar y terminar con letra/número.
    if not re.match(r"^[a-zA-Z0-9](?:[a-zA-Z0-9_-]*[a-zA-Z0-9])?$", context_name):
         print(f"Validación fallida (caracteres): '{context_name}'. Regla: ^[a-zA-Z0-9](?:[a-zA-Z0-9_-]*[a-zA-Z0-9])?$")
         return False
    # Regla adicional de Chroma (evitar ..) - ya cubierta por el regex anterior si no permitimos puntos.
    # if ".." in context_name:
    #     return False
    return True

# --- Ejecución del Servidor ---
if __name__ == "__main__":
    # Configurar logging básico (opcional pero recomendado)
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING) # Reducir verbosidad de uvicorn

    # Verificar dependencias clave
    if not client:
         logging.warning("El cliente OpenAI no está inicializado. Las funciones de embedding y RAG no funcionarán.")
    if not chroma_client:
         logging.warning("El cliente ChromaDB no está inicializado. Las funciones de almacenamiento y búsqueda vectorial no funcionarán.")

    # Obtener puerto de variable de entorno (útil para Render) o usar default 8000
    port = int(os.environ.get("PORT", 8000))
    print(f"INFO: Iniciando servidor Uvicorn en host 0.0.0.0 puerto {port}")

    # Nota: reload=True es útil para desarrollo, pero debería ser False en producción.
    # Render gestiona el reinicio por sí mismo al hacer deploy.
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False) # CAMBIAR reload=False para producción

# --- END OF FILE api.py ---