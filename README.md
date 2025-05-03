# RAG System (Retrieval Augmented Generation)

Este proyecto implementa un sistema RAG (Generación Aumentada por Recuperación) que te permite cargar documentos, procesarlos y realizar consultas sobre ellos utilizando lenguaje natural a través de una interfaz web interactiva.

## Descripción General

El sistema utiliza técnicas de procesamiento de lenguaje natural (PLN) para extraer información relevante de los documentos cargados y un modelo de lenguaje grande (LLM) para generar respuestas coherentes y contextualizadas a las preguntas del usuario, basadas en la información recuperada de dichos documentos.

## Características Principales

-   Carga y procesamiento de múltiples tipos de documentos (ej. PDF, TXT, DOCX - especificar si aplica).
-   Extracción y vectorización de texto para búsqueda semántica eficiente.
-   Interfaz de chat para realizar preguntas en lenguaje natural sobre los documentos.
-   Generación de respuestas basadas en la información extraída y potenciadas por un LLM.
-   Interfaz web construida con React y FastAPI para el backend.

## Estructura del Proyecto

-   `backend/`: Contiene la API de Python (FastAPI), la lógica de procesamiento de documentos, la interacción con la base de datos vectorial y el modelo de lenguaje.
-   `frontend/`: Contiene la aplicación frontend construida con React y TypeScript.

## Tecnologías Utilizadas

### Backend
-   FastAPI (Framework web de Python)
-   [Librería de Embeddings/Vector DB - ej. Langchain, LlamaIndex, ChromaDB, FAISS]
-   [Librería de OCR/Extracción de Texto - ej. PyMuPDF, python-docx, Tesseract OCR]
-   [Modelo de Lenguaje - ej. OpenAI API, Hugging Face Transformers]
-   Python 3.8+

### Frontend
-   React 19
-   TypeScript
-   Tailwind CSS
-   API Fetch / Axios

## Primeros Pasos

### Requisitos Previos Indispensables

-   Python 3.8+
-   Node.js 18+ y npm (o yarn/pnpm)
-   **Dependencias Externas Cruciales:** Este proyecto requiere la instalación de dependencias adicionales específicas que **NO** se gestionan únicamente con `pip`. Por favor, consulta **OBLIGATORIAMENTE** el archivo `install-dependencies.md` para obtener las instrucciones detalladas de instalación de herramientas como Tesseract OCR y otras librerías necesarias para el correcto funcionamiento del procesamiento de documentos. **¡Omitir este paso impedirá que el sistema funcione!**
-   Claves API (si aplica): Necesitarás configurar las claves API para los servicios externos utilizados (ej. OpenAI, modelos de Hugging Face, etc.).

### Instalación y Configuración

1.  **Clona el repositorio:**
    ```bash
    git clone https://github.com/tuusuario/tu-repo-rag.git
    cd tu-repo-rag
    ```
    *(Reemplaza `tuusuario/tu-repo-rag` con la URL real)*

2.  **Instala Dependencias Externas:**
    *   **SIGUE LAS INSTRUCCIONES DETALLADAS EN `install-dependencies.md`.** Este paso es fundamental.

3.  **Configuración del Backend (FastAPI):**
    *   Navega al directorio `backend`:
        ```bash
        cd backend
        ```
    *   Crea y activa un entorno virtual:
        ```bash
        python -m venv venv
        # Windows: venv\Scripts\activate
        # macOS/Linux: source venv/bin/activate
        ```
    *   Instala las dependencias de Python:
        ```bash
        pip install -r requirements.txt
        ```
    *   **Configura las variables de entorno:** Renombra el archivo `.env.example` a `.env` y completa **TODAS** las variables requeridas con tus propias credenciales y configuraciones (claves API, rutas, configuraciones de base de datos vectorial, etc.). **Consulta los comentarios dentro de `.env.example` para obtener orientación.**

4.  **Configuración del Frontend React:**
    *   Navega al directorio `frontend`:
        ```bash
        # Desde backend/: cd ../frontend
        # Desde la raíz: cd frontend
        cd ../frontend
        ```
    *   Instala las dependencias de Node.js:
        ```bash
        npm install
        ```
    *   **Configura las variables de entorno:** Renombra el archivo `.env.example` a `.env.local` (o simplemente `.env`) y completa **TODAS** las variables requeridas, especialmente la URL para conectarse a tu API backend. **Consulta los comentarios dentro de `.env.example` para obtener orientación.**

### Ejecución

1.  **Iniciar el Backend:**
    *   Desde el directorio `backend` (con el entorno virtual activado):
        ```bash
        uvicorn main:app --reload --port 8000
        ```
        *(Ajusta `main:app` y el puerto si es necesario)*

2.  **Iniciar el Frontend:**
    *   Desde el directorio `frontend`:
        ```bash
        npm run dev
        ```

3.  **Acceso:**
    *   Abre tu navegador y visita la URL (usualmente `http://localhost:3000`).

## Notas Importantes

-   **Dependencias Externas:** La correcta instalación de las dependencias listadas en `install-dependencies.md` es crucial.
-   **Variables de Entorno:** Asegúrate de configurar **correctamente todos los archivos `.env`** tanto en el backend como en el frontend, siguiendo las indicaciones de los archivos `.env.example`. Un error aquí es una causa común de problemas.
-   **Procesamiento Inicial:** Dependiendo de la implementación, puede que necesites ejecutar algún script inicial para procesar documentos de ejemplo o configurar la base de datos vectorial la primera vez. Consulta la documentación adicional si existe.
