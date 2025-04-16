import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import './App.css'; // Asegúrate que este archivo existe y contiene las bases de Tailwind si no usas un CDN/setup

// --- Constantes (sin cambios) ---
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const WS_PROTOCOL = window.location.protocol === 'https:' ? 'wss' : 'ws'; // Usa wss si estás en https
const WS_URL = process.env.REACT_APP_WS_URL || `${WS_PROTOCOL}://localhost:8000`; // Usa wss:// para producción
const DEFAULT_SYSTEM_PROMPT_FRONTEND = "Eres un asistente útil que responde preguntas basadas únicamente en el contexto proporcionado. Si la respuesta no se encuentra en el contexto, indica que no tienes suficiente información.";

// --- Interfaces (sin cambios) ---
interface UploadResult { filename: string; status: string; chunks: number; details?: string; }
interface ChatMetadata { document_name: string | null; page_number: number | string | null; all_source_documents?: string[]; }
interface ChatResponse { answer: string; metadata: ChatMetadata; }
interface HistoryItem { query: string; answer: string; }

function App() {
  // --- Estados (con NUEVO estado backendAwake) ---
  const [selectedFiles, setSelectedFiles] = useState<File[] | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResults, setUploadResults] = useState<UploadResult[] | null>(null);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false); // Usado para cargas generales (contextos, historial)
  const [error, setError] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [currentFile, setCurrentFile] = useState<string>('');
  const [filesRemaining, setFilesRemaining] = useState<number>(0);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [contexts, setContexts] = useState<string[]>([]);
  const [selectedContext, setSelectedContext] = useState<string | null>(null);
  const [newContextName, setNewContextName] = useState('');
  const [isCreatingContext, setIsCreatingContext] = useState(false);
  const [chatHistory, setChatHistory] = useState<HistoryItem[]>([]);
  const [chatMessages, setChatMessages] = useState<{ role: string; content: string }[]>([]);
  const [systemPrompt, setSystemPrompt] = useState<string>(DEFAULT_SYSTEM_PROMPT_FRONTEND);
  const [uploadStatusMessage, setUploadStatusMessage] = useState<string>('');
  const [isContextPanelVisible, setIsContextPanelVisible] = useState(false);
  const [isBackendWakingUp, setIsBackendWakingUp] = useState(true); // Estado para indicar si se está intentando despertar
  const [backendAwake, setBackendAwake] = useState(false); // NUEVO: Indica si el intento de despertar finalizó

  // --- Refs (sin cambios) ---
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);


  // --- Funciones (NUEVA: wakeUpBackend) ---

  /**
   * Intenta hacer ping al backend para "despertarlo".
   * Idealmente, usa un endpoint ligero como '/health' o '/'.
   */
  const wakeUpBackend = async () => {
    console.log("INIT: Intentando despertar al backend...");
    setIsBackendWakingUp(true); // Inicia indicador visual si lo deseas
    try {
      // Puedes cambiar '/health' por '/' o cualquier otro endpoint GET simple que tengas
      // Usar /get-contexts también funciona, pero es menos ideal que un endpoint dedicado.
      await axios.get(`${API_URL}/health`, { timeout: 15000 }); // Timeout de 15s por si tarda en arrancar
      console.log("INIT: Ping al backend exitoso (o al menos respondió).");
      // No necesitas hacer nada con la respuesta, solo saber que respondió.
    } catch (err: any) {
      // Es NORMAL que esto falle si el backend estaba dormido y tarda en arrancar.
      // O si el endpoint /health no existe (prueba con '/' o '/get-contexts').
      console.warn(`INIT: Ping al backend falló o no respondió a tiempo (esto puede ser normal si estaba dormido): ${err.message}`);
      // Puedes mostrar un mensaje o simplemente continuar. El WS intentará conectar igualmente.
      // setError("El backend no respondió al ping inicial, puede tardar en conectar.");
    } finally {
      console.log("INIT: Intento de despertar finalizado.");
      setIsBackendWakingUp(false); // Termina indicador visual
      setBackendAwake(true); // Señala que el proceso terminó y el WS puede intentar conectar.
    }
  };

  // --- Efectos ---

  // EFECTO 1: Despertar al backend al montar el componente
  useEffect(() => {
    wakeUpBackend();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Se ejecuta solo una vez al montar

  // EFECTO 2: Conectar al WebSocket DESPUÉS de intentar despertar al backend
  useEffect(() => {
    // No intentar conectar hasta que el intento de 'wake up' haya terminado
    if (!backendAwake) {
        console.log("WS: Esperando a que termine el intento de despertar al backend...");
        return;
    }

    // Lógica WebSocket (movida aquí)
    const connectWebSocket = () => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) { console.log("WS: Ya conectado."); return; }
        if (wsRef.current) { console.log("WS: Limpiando conexión anterior."); wsRef.current.close(); wsRef.current = null; }

        console.log("WS: Intentando conectar a /ws/progress...");
        setUploadStatusMessage('Conectando al servidor...'); // Mensaje inicial
        const ws = new WebSocket(`${WS_URL}/ws/progress`);
        wsRef.current = ws;

        ws.onopen = () => {
            console.log('WS: Conectado');
            setUploadStatusMessage(''); // Limpiar mensaje de conexión
            setError(null); // Limpiar errores previos de conexión WS si los hubo
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data); console.log('WS: Mensaje Recibido:', data);
                switch (data.type) {
                    case 'start_upload':
                        console.log("WS: Manejando 'start_upload'");
                        setUploading(true); setUploadProgress(0); setUploadResults(null); setSuccessMessage(null); setError(null); setCurrentFile(''); setFilesRemaining(data.total_files || 0); setUploadStatusMessage(`Iniciando subida de ${data.total_files} archivo(s)...`);
                        break;
                    case 'progress':
                        const rawPercentage = data.percentage;
                        const newProgress = Math.min(Math.max(0, parseInt(String(rawPercentage ?? 0), 10)), 100);
                        console.log(`WS: Manejando 'progress'. Recibido: ${rawPercentage}%. Calculado: ${newProgress}%.`);
                        setUploadProgress(prevProgress => newProgress); // Usar valor calculado
                        setCurrentFile(data.current_file || '');
                        const remaining = (data.total_files || 0) - (data.files_processed || 0) - 1; // -1 porque procesados no incluye el actual
                        setFilesRemaining(remaining >= 0 ? remaining : 0);
                        setUploadStatusMessage(data.status_message || `Procesando ${data.current_file || ''}...`);
                        break;
                    case 'error_file':
                        console.log("WS: Manejando 'error_file'");
                        setError(prev => `${prev ? prev + '\n' : ''}Error subiendo ${data.filename}: ${data.message}`);
                        // Actualizar progreso si viene en el mensaje de error, si no, mantener el último conocido
                        const errorProgress = Math.min(Math.max(0, parseInt(String(data.percentage ?? uploadProgress), 10)), 100);
                        setUploadProgress(prev => errorProgress);
                        setUploadStatusMessage(`Error con ${data.filename}.`);
                        // No detener 'uploading' aquí, esperar a 'complete_upload' o cierre de WS
                        break;
                    case 'complete_upload':
                        console.log("WS: Manejando 'complete_upload'");
                        setUploading(false);
                        setUploadProgress(100);
                        setSuccessMessage(data.message || 'Subida completada con posibles errores (ver detalles).'); // Mensaje más genérico si hay errores
                        setUploadResults(data.results || []);
                        setCurrentFile('');
                        setFilesRemaining(0);
                        setUploadStatusMessage('');
                        setSelectedFiles(null); // Limpiar archivos seleccionados
                        // Recargar contextos podría ser útil aquí si la subida afecta la lista
                        // fetchContexts(); // Descomentar si es necesario
                        break;
                    default:
                        console.warn('WS: Tipo de mensaje no manejado:', data.type);
                }
            } catch (e) {
                console.error('WS: Error procesando mensaje:', e, event.data);
            }
        };

        ws.onerror = (error) => {
            console.error('WS: Error:', error);
            // Distinguir si es error inicial de conexión o durante operación
            if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
                 setError('Error de conexión WebSocket. El backend puede estar iniciándose o no disponible.');
                 setUploadStatusMessage('Error de conexión WS.');
            } else {
                 setError('Error en la conexión WebSocket existente.');
            }
            setUploading(false); // Detener subida si hay error de WS
            wsRef.current = null;
        };

        ws.onclose = (event) => {
            console.log(`WS: Desconectado: Código ${event.code}, Razón: ${event.reason}`);
            wsRef.current = null;
            // Si la desconexión no fue limpia (code 1000) y estábamos subiendo, mostrar error.
            if (uploading && event.code !== 1000) {
                setError("Se perdió la conexión durante la subida.");
                setUploading(false);
            }
             // Si no estábamos subiendo y la desconexión no fue iniciada por el cliente (código 1000), puede indicar un problema.
             // Podrías intentar reconectar aquí si es necesario, pero cuidado con bucles infinitos.
             // if (!uploading && event.code !== 1000) {
             //    setUploadStatusMessage('Desconectado.');
             //    // Opcional: Intentar reconectar después de un delay
             //    // setTimeout(connectWebSocket, 5000);
             // } else {
                 setUploadStatusMessage(''); // Limpiar mensaje si fue cierre normal o ya se manejó el error
             // }
        };
    };

    // Llama a la función para conectar
    connectWebSocket();

    // Función de limpieza para cuando el componente se desmonte
    return () => {
        if (wsRef.current) {
            console.log("WS: Cerrando conexión al desmontar.");
            wsRef.current.onclose = null; // Evitar triggers de onclose durante el desmontaje manual
            wsRef.current.onerror = null; // Evitar triggers de onerror
            wsRef.current.close(1000, "Componente desmontado"); // Código 1000 indica cierre normal
            wsRef.current = null;
        }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [backendAwake]); // <- Dependencia clave: Solo se ejecuta cuando backendAwake cambia (a true)

  // EFECTO 3: Cargar contextos iniciales (se ejecuta independientemente del WS)
  useEffect(() => {
    fetchContexts();
  }, []); // Se ejecuta solo una vez al montar

  // EFECTO 4: Scroll automático del chat (sin cambios)
  useEffect(() => {
    scrollToBottom();
  }, [chatMessages]);

  // --- Funciones de manejo (sin cambios en la lógica interna, solo se movieron comentarios/logs) ---
  const scrollToBottom = () => { messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); };
  const fetchContexts = async () => { setLoading(true); setError(null); try { const r = await axios.get(`${API_URL}/get-contexts`); setContexts(r.data.contexts || []); } catch (e) { console.error("Error fetching contexts:", e); setError('Error al obtener contextos.'); } finally { setLoading(false); } };
  const handleFileUpload = async (filesToUpload: File[] | null) => {
    if (!selectedContext || !filesToUpload || filesToUpload.length === 0) { setError('Selecciona un contexto y al menos un archivo.'); return; }
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) { setError('Conexión WebSocket no establecida. Refresca la página o espera un momento.'); return; }

    // Reiniciar estados antes de la petición HTTP
    setUploading(true); // Se pone true aquí, el WS lo confirmará con 'start_upload'
    setUploadProgress(0);
    setSuccessMessage(null);
    setError(null);
    setUploadResults(null);
    setFilesRemaining(filesToUpload.length);
    setCurrentFile(filesToUpload[0]?.name || '');
    setUploadStatusMessage('Enviando petición de subida...'); // Mensaje inicial antes del WS

    try {
        const formData = new FormData();
        formData.append('context_name', selectedContext);
        filesToUpload.forEach(file => formData.append('files', file, file.name));

        console.log(`Enviando ${filesToUpload.length} archivo(s) a /upload-documents para el contexto '${selectedContext}'...`);
        // La respuesta de esta petición POST ya no es tan relevante si el progreso va por WS
        await axios.post(`${API_URL}/upload-documents`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        });
        console.log("Petición HTTP /upload-documents enviada correctamente. Esperando mensajes del WebSocket...");
        // No ponemos uploading=false aquí, esperamos al mensaje 'complete_upload' del WS
        setUploadStatusMessage('Petición recibida, esperando inicio del procesamiento...');
    } catch (err: any) {
        console.error('Error en la petición HTTP /upload-documents:', err);
        const errorMsg = `Error al iniciar la subida: ${err.response?.data?.detail || err.message || 'Error desconocido'}`;
        setError(errorMsg);
        // Si falla la petición HTTP, revertimos el estado de subida
        setUploading(false);
        setUploadProgress(0);
        setUploadStatusMessage('Fallo al iniciar la subida.');
        setCurrentFile('');
        setFilesRemaining(0);
    }
};
  const handleAskQuestion = async (questionToSend: string) => {
    if (!selectedContext || !questionToSend.trim() || loading || uploading) return;
    const userMessage = { role: 'user' as const, content: questionToSend }; setChatMessages(prev => [...prev, userMessage]); const currentQuery = query; setQuery(''); setLoading(true); setError(null);
    try {
        const payload: { question: string; context_name: string; system_prompt?: string } = { question: questionToSend, context_name: selectedContext }; const finalSystemPrompt = systemPrompt.trim() || DEFAULT_SYSTEM_PROMPT_FRONTEND; payload.system_prompt = finalSystemPrompt;
        console.log("Enviando a /ask:", payload); const response = await axios.post<ChatResponse>(`${API_URL}/ask`, payload); const data = response.data; console.log("Respuesta de /ask:", data); let metadataString = "";
        if (data.metadata) { const docName = data.metadata.document_name || 'N/A'; const pageNum = data.metadata.page_number; metadataString = `Documento: ${docName}`; if (pageNum !== null && pageNum !== undefined) { metadataString += `\nReferencia: ${typeof pageNum === 'number' ? `Página ${pageNum}` : pageNum}`; }}
        const assistantContent = data.answer + (metadataString ? `\n\n---\n${metadataString}` : ""); const assistantMessage = { role: 'assistant' as const, content: assistantContent }; setChatMessages(prev => [...prev, assistantMessage]); setChatHistory(prevHistory => [...prevHistory, { query: currentQuery, answer: data.answer }]);
    } catch (err: any) { console.error('Error asking question:', err); const errorMessage = `Error: ${err.response?.data?.detail || err.message || 'Inténtalo de nuevo.'}`; setError(errorMessage); setChatMessages(prev => [...prev, { role: 'assistant', content: `Lo siento, ocurrió un error. ${errorMessage}` }]);
    } finally { setLoading(false); }
  };
  const handleDeleteContext = async (contextName: string) => {
    if (!window.confirm(`¿Estás seguro de que quieres eliminar el contexto "${contextName}" y todos sus documentos asociados? Esta acción no se puede deshacer.`)) return;
    setLoading(true); setError(null);
    try { await axios.delete(`${API_URL}/delete-context/${contextName}`); await fetchContexts(); if (selectedContext === contextName) { setSelectedContext(null); setChatMessages([]); setChatHistory([]); setSelectedFiles(null); setUploadResults(null); setSuccessMessage(null); setSystemPrompt(DEFAULT_SYSTEM_PROMPT_FRONTEND); }
    } catch (err: any) { console.error("Error deleting context:", err); setError(`Error al eliminar '${contextName}': ${err.response?.data?.detail || err.message}`); } finally { setLoading(false); }
  };
  const handleCreateContext = async () => {
    const nameToCreate = newContextName.trim(); if (!nameToCreate) return; setIsCreatingContext(true); setError(null); try { await axios.post(`${API_URL}/create-context`, { context_name: nameToCreate }); setNewContextName(''); await fetchContexts(); setSelectedContext(nameToCreate); // Opcional: seleccionar el nuevo contexto
         setChatMessages([{ role: 'assistant', content: `Contexto '${nameToCreate}' creado y seleccionado.` }]); // Mensaje inicial
        } catch (err: any) { console.error("Error creating context:", err); setError(`Error al crear contexto: ${err.response?.data?.detail || err.message}`); } finally { setIsCreatingContext(false); }
  };
  const handleContextSelect = async (context: string) => {
    if (context === selectedContext || loading || uploading || isCreatingContext) return;
    if (isContextPanelVisible && window.innerWidth < 1024) { setIsContextPanelVisible(false); }

    setLoading(true); setError(null); // Usar setLoading general para el cambio de contexto
    setChatMessages([]); setChatHistory([]); setSelectedFiles(null); setUploadResults(null); setSuccessMessage(null); setUploadProgress(0); setCurrentFile(''); setFilesRemaining(0); setUploadStatusMessage(''); setSystemPrompt(DEFAULT_SYSTEM_PROMPT_FRONTEND);
    setSelectedContext(context); // Selecciona el contexto visualmente

    try {
      // Cargar historial del contexto seleccionado
      console.log(`Cargando historial para el contexto: ${context}`);
      const response = await axios.post<{ history: HistoryItem[] }>(`${API_URL}/load-chat-history`, { context_name: context });
      const loadedHistory = response.data.history || [];
      console.log(`Historial cargado: ${loadedHistory.length} items.`);
      setChatHistory(loadedHistory);

      // Poblar mensajes de chat desde el historial
      if (loadedHistory.length > 0) {
        const messagesFromHistory = loadedHistory.flatMap(item => [
          { role: 'user' as const, content: item.query },
          { role: 'assistant' as const, content: item.answer }
        ]);
        setChatMessages(messagesFromHistory);
      } else {
        setChatMessages([{ role: 'assistant', content: `Contexto '${context}' cargado. No hay historial previo.` }]);
      }
    } catch (err: any) {
      console.error("Error loading history:", err);
      setError(`Error cargando historial para '${context}': ${err.response?.data?.detail || err.message}`);
      setChatMessages([{ role: 'assistant', content: `No se pudo cargar el historial para '${context}'.` }]);
    } finally {
      setLoading(false); // Termina la carga del contexto
    }
  };

  // --- Lógica de Dropzone (sin cambios) ---
  const onDrop = useCallback((acceptedFiles: File[]) => {
    setSuccessMessage(null); setError(null); setUploadResults(null); // Limpiar mensajes/resultados previos
    setSelectedFiles(acceptedFiles);
  }, []);
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel.sheet.macroEnabled.12': ['.xlsm'],
      // Añade aquí más tipos si los soportas en el backend
      'text/plain': ['.txt'],
      'text/csv': ['.csv'],
    },
    multiple: true,
    disabled: uploading || !selectedContext, // Deshabilitar si está subiendo O si no hay contexto seleccionado
  });
  const handleRemoveFile = (fileNameToRemove: string) => {
    setSelectedFiles(prevFiles => prevFiles ? prevFiles.filter(file => file.name !== fileNameToRemove) : null);
  };

  // --- JSX (sin cambios estructurales importantes, solo textos o indicadores) ---
  return (
    <div className="flex h-screen bg-base-100 text-base-content overflow-hidden" data-theme="light">

      {/* Overlay/Backdrop */}
      {isContextPanelVisible && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-30 lg:hidden" onClick={() => setIsContextPanelVisible(false)} aria-hidden="true"></div>
      )}

      {/* Panel izquierdo (Contextos) */}
      <div className={`fixed inset-y-0 left-0 z-40 w-64 bg-base-200 p-4 border-r border-base-300 flex flex-col overflow-y-auto transform transition-transform duration-300 ease-in-out ${isContextPanelVisible ? 'translate-x-0' : '-translate-x-full'} lg:relative lg:translate-x-0 lg:flex lg:z-auto`} aria-label="Panel de Contextos" id="context-panel">
        <button onClick={() => setIsContextPanelVisible(false)} className="absolute top-2 right-2 btn btn-ghost btn-sm lg:hidden" aria-label="Cerrar panel de contextos">
           <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" /></svg>
        </button>
        <h2 className="text-lg font-bold mb-4 flex-shrink-0 mt-6 lg:mt-0">Contextos</h2>
        <div className="mb-4 flex-shrink-0">
          <input type="text" value={newContextName} onChange={(e) => setNewContextName(e.target.value)} placeholder="Nombre del nuevo contexto" className="input input-bordered input-sm w-full mb-2 bg-base-100" disabled={isCreatingContext || loading || uploading}/>
          <button onClick={handleCreateContext} disabled={!newContextName.trim() || isCreatingContext || loading || uploading} className={`btn btn-sm btn-primary w-full ${isCreatingContext ? 'loading' : ''}`}> {isCreatingContext ? 'Creando...' : 'Crear Contexto'} </button>
        </div>
        <div className="flex-grow overflow-y-auto">
           {/* Indicador mientras se cargan los contextos iniciales */}
          {(loading && contexts.length === 0) && <div className="text-center py-4"><span className="loading loading-spinner text-primary"></span><p className='text-xs mt-2'>Cargando contextos...</p></div>}
          <ul className="menu menu-sm bg-base-100 rounded-box p-2 w-full">
            {contexts.map((context) => (
              <li key={context}>
                <a onClick={() => handleContextSelect(context)} className={`flex justify-between items-center rounded group ${selectedContext === context ? 'active bg-primary text-primary-content font-semibold' : 'hover:bg-base-300'}`}>
                  <span className="truncate flex-grow mr-2">{context}</span>
                  <button onClick={(e) => { e.stopPropagation(); handleDeleteContext(context); }} className={`btn btn-xs btn-ghost text-error/70 hover:bg-error hover:text-error-content ml-auto flex-shrink-0 p-1 opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity ${selectedContext === context ? 'opacity-100' : '' }`} aria-label={`Eliminar contexto ${context}`} title={`Eliminar ${context}`} disabled={loading || uploading || isCreatingContext}>
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd" /></svg>
                  </button>
                </a>
              </li>
            ))}
          </ul>
        </div>
          {/* Indicador de estado del backend/WS en la parte inferior del panel */}
          <div className="mt-4 pt-2 border-t border-base-300 text-xs text-base-content/60 flex-shrink-0">
              {isBackendWakingUp ? (
                  <div className='flex items-center justify-center gap-1'> <span className="loading loading-dots loading-xs"></span> Despertando backend...</div>
              ) : wsRef.current && wsRef.current.readyState === WebSocket.OPEN ? (
                  <div className='text-success/80 text-center'>Conectado</div>
              ) : error && error.includes('WebSocket') ? (
                   <div className='text-error/80 text-center' title={error}>Error de Conexión WS</div>
              ) : (
                  <div className='text-warning/80 text-center'>WS Desconectado</div>
              )}
          </div>
      </div>

      {/* Panel principal (Chat y Subida) */}
      <div className="flex-1 flex flex-col p-4 bg-base-100 overflow-hidden">
        <button onClick={() => setIsContextPanelVisible(true)} className="btn btn-ghost btn-sm lg:hidden mb-2 self-start" aria-label="Abrir panel de contextos" aria-controls="context-panel" aria-expanded={isContextPanelVisible}>
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16m-7 6h7" /></svg>
           {selectedContext && <span className="ml-2 text-xs font-normal truncate max-w-[150px]">{selectedContext}</span>}
        </button>

        {/* Contenido principal */}
        {selectedContext ? (
          <>
            {/* Encabezado y Subida */}
            <div className="pb-4 border-b border-base-300 mb-4 flex-shrink-0 space-y-3">
              <h1 className="text-xl font-semibold truncate hidden lg:block">
                Contexto: <span className='font-bold text-primary'>{selectedContext}</span>
              </h1>
              <div {...getRootProps()} className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors duration-200 ease-in-out ${isDragActive ? 'border-primary bg-primary/10' : 'border-base-300 hover:border-primary/50'} ${uploading || !selectedContext ? 'cursor-not-allowed opacity-60 bg-base-200' : 'bg-base-100 hover:bg-base-200'}`}>
                <input {...getInputProps()} />
                 { !selectedContext ? <p className="text-base-content/50 text-sm">Selecciona un contexto para poder subir archivos.</p> :
                   isDragActive ? <p className="text-primary font-semibold">Suelta los archivos aquí...</p> : <p className="text-base-content/70 text-sm"> Arrastra archivos aquí, o haz clic para seleccionar (.pdf, .docx, .xlsx, .xlsm, .txt, .csv) </p>
                 }
              </div>

              {selectedFiles && selectedFiles.length > 0 && !uploading && (
                <div className="space-y-1 text-sm bg-base-200 p-2 rounded-md max-h-32 overflow-y-auto">
                  <p className="font-medium text-xs mb-1 text-base-content/70">Archivos seleccionados:</p>
                  <ul className="list-none pl-0">
                    {selectedFiles.map((file) => (
                      <li key={file.name} className="flex justify-between items-center text-xs py-0.5 group">
                        <span className="truncate pr-2">{file.name} - <span className='text-base-content/60'>{Math.round(file.size / 1024)} KB</span></span>
                        <button onClick={() => handleRemoveFile(file.name)} className="btn btn-xs btn-ghost text-error opacity-50 group-hover:opacity-100 transition-opacity" aria-label={`Quitar ${file.name}`} title={`Quitar ${file.name}`}>✕</button>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {selectedFiles && selectedFiles.length > 0 && (
                 <div className="flex justify-start items-center gap-4">
                  <button className={`btn btn-sm btn-secondary ${uploading ? 'loading cursor-wait' : ''}`} onClick={() => handleFileUpload(selectedFiles)} disabled={uploading || loading || !selectedContext}> {uploading ? 'Subiendo...' : `Subir ${selectedFiles.length} archivo(s)`} </button>
                   {/* Botón para cancelar/limpiar selección */}
                   {!uploading && <button className='btn btn-xs btn-ghost text-base-content/50 hover:text-error' onClick={() => setSelectedFiles(null)}>Cancelar selección</button>}
                </div>
              )}

               {/* Progreso y Mensajes de Subida */}
               {(uploading || uploadStatusMessage) && (
                 <div className="mt-3 space-y-1">
                   <p className="text-sm text-base-content/80 text-center font-medium">{uploadStatusMessage || 'Iniciando...'}</p>
                   {uploading && <progress className="progress progress-primary w-full" value={uploadProgress} max="100"></progress>}
                   {uploading && <div className="text-xs text-center text-base-content/60 tabular-nums"> {currentFile && `Archivo: ${currentFile} (${filesRemaining} restantes) - `} {`${uploadProgress}%`} </div>}
                 </div>
               )}
               {/* Mostrar resultados de subida (éxitos y errores) */}
               {uploadResults && !uploading && (
                 <div className="mt-3 p-2 text-xs border border-base-300 rounded-md bg-base-200/50 max-h-40 overflow-y-auto">
                     <p className='font-medium mb-1 text-sm'>Resultado de la subida:</p>
                     <ul>
                         {uploadResults.map((res, idx) => (
                             <li key={idx} className={`py-0.5 ${res.status === 'error' ? 'text-error' : 'text-success'}`}>
                                 <span className='font-semibold'>{res.filename}:</span> {res.status === 'error' ? `Error - ${res.details}` : `Éxito (${res.chunks} chunks)`}
                             </li>
                         ))}
                     </ul>
                 </div>
               )}
               {/* Mensaje general de éxito (opcional si ya están los resultados) */}
               {/* {successMessage && !uploading && !uploadResults && <div className="mt-3 alert alert-success p-2 text-sm shadow-sm"><span>{successMessage}</span></div>} */}
               {/* Mensaje de error general (si no es un error de archivo específico mostrado en resultados) */}
               {error && (!uploading || (error.includes('WebSocket') && !(wsRef.current && wsRef.current.readyState === WebSocket.OPEN) ) ) && <div className="mt-3 alert alert-error p-2 text-sm shadow-sm"><span>{error}</span></div>}

            </div>

            {/* Área de Chat */}
            <div className="flex-grow overflow-y-auto mb-4 pr-2 min-h-0 chat-area">
               {chatMessages.length === 0 && !loading && ( <p className='text-center text-base-content/50 mt-10'>Haz una pregunta sobre el contexto o sube documentos.</p> )}
              {chatMessages.map((message, index) => (
                <div key={index} className={`chat ${message.role === 'user' ? 'chat-end' : 'chat-start'}`}>
                  <div className={`chat-bubble ${message.role === 'user' ? 'chat-bubble-primary' : 'chat-bubble-secondary'} max-w-xl`} style={{ whiteSpace: 'pre-wrap', overflowWrap: 'break-word' }}>
                    {message.content}
                  </div>
                </div>
              ))}
               <div ref={messagesEndRef} />
            </div>

            {/* Input de pregunta y Prompt */}
            <form onSubmit={(e) => { e.preventDefault(); handleAskQuestion(query); }} className="flex flex-col gap-2 pt-4 border-t border-base-300 flex-shrink-0">
                 <textarea rows={2} value={systemPrompt} onChange={(e) => setSystemPrompt(e.target.value)} placeholder={`Prompt del sistema (por defecto: ${DEFAULT_SYSTEM_PROMPT_FRONTEND})`} title="Define el comportamiento del asistente (opcional)." className="textarea textarea-bordered w-full text-sm bg-base-100 focus:outline-none focus:ring-1 focus:ring-primary resize-none" disabled={loading || uploading}/>
                 <div className="flex gap-2">
                    <input type="text" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Escribe tu pregunta aquí..." className="input input-bordered w-full bg-base-100 focus:outline-none focus:ring-1 focus:ring-primary" disabled={loading || uploading || !selectedContext}/>
                    <button type="submit" disabled={!query.trim() || loading || uploading || !selectedContext} className={`btn btn-primary ${loading ? 'loading' : ''}`}> {loading ? 'Pensando...' : 'Enviar'} </button>
                 </div>
             </form>
          </>
        ) : (
          // Vista cuando no hay contexto seleccionado
          <div className="flex flex-col items-center justify-center h-full text-center">
             {(loading || isBackendWakingUp) ? <div className="loading loading-lg text-primary mb-4"></div> : <p className="text-xl text-base-content/60">Selecciona o crea un contexto en el panel izquierdo para comenzar.</p>}
             {isBackendWakingUp && <p className="text-sm text-base-content/50 mt-2">Intentando conectar con el servidor...</p>}
             {error && <div className="mt-4 alert alert-error max-w-md mx-auto"><span>{error}</span></div>}
          </div>
        )}
      </div> {/* Fin del panel principal */}
    </div> // Fin del contenedor principal
  );
}

export default App;