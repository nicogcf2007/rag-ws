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
  // --- Estados (sin cambios, EXCEPTO el nuevo estado para el panel) ---
  const [selectedFiles, setSelectedFiles] = useState<File[] | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResults, setUploadResults] = useState<UploadResult[] | null>(null);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
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

  // --- Refs (sin cambios) ---
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // --- *** NUEVO ESTADO PARA PANEL RESPONSIVE *** ---
  const [isContextPanelVisible, setIsContextPanelVisible] = useState(false);
  // ---------------------------------------------------

  // --- Efectos (sin cambios) ---
  useEffect(() => { scrollToBottom(); }, [chatMessages]);
  useEffect(() => { fetchContexts(); }, []);
  useEffect(() => {
    // Lógica WebSocket persistente (igual que antes)
    const connectWebSocket = () => {
      // ... (código WebSocket idéntico al original) ...
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) { console.log("WS: Ya conectado."); return; }
        if (wsRef.current) { console.log("WS: Limpiando conexión anterior."); wsRef.current.close(); wsRef.current = null; }
        console.log("WS: Intentando conectar a /ws/progress...");
        const ws = new WebSocket(`${WS_URL}/ws/progress`);
        wsRef.current = ws;
        ws.onopen = () => { console.log('WS: Conectado'); setUploadStatusMessage(''); };
        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data); console.log('WS: Mensaje Recibido:', data);
                switch (data.type) {
                    case 'start_upload': console.log("WS: Manejando 'start_upload'"); setUploading(true); setUploadProgress(0); setUploadResults(null); setSuccessMessage(null); setError(null); setCurrentFile(''); setFilesRemaining(data.total_files || 0); setUploadStatusMessage(`Iniciando subida de ${data.total_files} archivo(s)...`); break;
                    case 'progress': const rawPercentage = data.percentage; const newProgress = Math.min(Math.max(0, parseInt(String(rawPercentage ?? 0), 10)), 100); console.log(`WS: Manejando 'progress'. Recibido: ${rawPercentage}%. Calculado: ${newProgress}%.`); setUploadProgress(prevProgress => newProgress); setCurrentFile(data.current_file || ''); const remaining = (data.total_files || 0) - (data.files_processed || 0) - 1; setFilesRemaining(remaining >= 0 ? remaining : 0); setUploadStatusMessage(data.status_message || `Procesando ${data.current_file || ''}...`); break;
                    case 'error_file': console.log("WS: Manejando 'error_file'"); setError(prev => `${prev ? prev + '\n' : ''}Error subiendo ${data.filename}: ${data.message}`); const errorProgress = Math.min(Math.max(0, parseInt(String(data.percentage ?? uploadProgress), 10)), 100); setUploadProgress(prev => errorProgress); setUploadStatusMessage(`Error con ${data.filename}.`); break;
                    case 'complete_upload': console.log("WS: Manejando 'complete_upload'"); setUploading(false); setUploadProgress(100); setSuccessMessage(data.message || 'Subida completada.'); setUploadResults(data.results || []); setCurrentFile(''); setFilesRemaining(0); setUploadStatusMessage(''); setSelectedFiles(null); break;
                    default: console.warn('WS: Tipo de mensaje no manejado:', data.type);
                }
            } catch (e) { console.error('WS: Error procesando mensaje:', e, event.data); }
        };
        ws.onerror = (error) => { console.error('WS: Error:', error); setError('Error de conexión WebSocket.'); setUploading(false); setUploadStatusMessage('Error de conexión.'); wsRef.current = null; };
        ws.onclose = (event) => { console.log(`WS: Desconectado: Código ${event.code}, Razón: ${event.reason}`); wsRef.current = null; setUploadStatusMessage(''); if (uploading && event.code !== 1000) { setError("Se perdió la conexión durante la subida."); setUploading(false); } };
    };
    connectWebSocket();
    return () => { if (wsRef.current) { console.log("WS: Cerrando conexión al desmontar."); wsRef.current.onclose = null; wsRef.current.onerror = null; wsRef.current.close(1000, "Componente desmontado"); wsRef.current = null; } };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // --- Funciones (MODIFICADA handleContextSelect) ---
  const scrollToBottom = () => { messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); };
  const fetchContexts = async () => { setLoading(true); setError(null); try { const r = await axios.get(`${API_URL}/get-contexts`); setContexts(r.data.contexts || []); } catch (e) { console.error("Error ctx:", e); setError('Error al obtener contextos.'); } finally { setLoading(false); } };
  const handleFileUpload = async (filesToUpload: File[] | null) => {
    // ... (código idéntico al original) ...
    if (!selectedContext || !filesToUpload || filesToUpload.length === 0) { setError('Selecciona contexto y archivos.'); return; }
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) { setError('Conexión no establecida. Refresca o intenta de nuevo.'); return; }
    setUploading(true); setUploadProgress(0); setSuccessMessage(null); setError(null); setUploadResults(null); setFilesRemaining(filesToUpload.length); setCurrentFile(filesToUpload[0]?.name || ''); setUploadStatusMessage('Preparando subida...');
    try {
        const formData = new FormData(); formData.append('context_name', selectedContext); filesToUpload.forEach(file => formData.append('files', file, file.name));
        console.log(`Enviando ${filesToUpload.length} archivo(s) a /upload-documents...`);
        await axios.post(`${API_URL}/upload-documents`, formData, { headers: { 'Content-Type': 'multipart/form-data' }, });
        console.log("Petición HTTP /upload-documents enviada.");
    } catch (err: any) { console.error('Error HTTP /upload-documents:', err); const errorMsg = `Error al subir: ${err.response?.data?.detail || err.message || 'Error desconocido'}`; setError(errorMsg); setUploading(false); setUploadProgress(0); setUploadStatusMessage('Fallo en la subida.'); }
  };
  const handleAskQuestion = async (questionToSend: string) => {
    // ... (código idéntico al original) ...
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
     // ... (código idéntico al original) ...
    if (!window.confirm(`Eliminar contexto "${contextName}"?`)) return; setLoading(true); setError(null);
    try { await axios.delete(`${API_URL}/delete-context/${contextName}`); await fetchContexts(); if (selectedContext === contextName) { setSelectedContext(null); setChatMessages([]); setChatHistory([]); setSelectedFiles(null); setUploadResults(null); setSuccessMessage(null); setSystemPrompt(DEFAULT_SYSTEM_PROMPT_FRONTEND); }
    } catch (err: any) { console.error("Error deleting context:", err); setError(`Error al eliminar '${contextName}': ${err.response?.data?.detail || err.message}`); } finally { setLoading(false); }
  };
  const handleCreateContext = async () => {
    // ... (código idéntico al original) ...
    const nameToCreate = newContextName.trim(); if (!nameToCreate) return; setIsCreatingContext(true); setError(null); try { await axios.post(`${API_URL}/create-context`, { context_name: nameToCreate }); setNewContextName(''); await fetchContexts(); } catch (err: any) { console.error("Error creating context:", err); setError(`Error al crear: ${err.response?.data?.detail || err.message}`); } finally { setIsCreatingContext(false); }
  };

  // *** MODIFICADO PARA CERRAR PANEL EN MÓVIL ***
  const handleContextSelect = async (context: string) => {
    if (context === selectedContext || loading || uploading) return;

    // --- Cierra el panel lateral si está abierto en pantallas pequeñas ---
    if (isContextPanelVisible && window.innerWidth < 1024) { // 1024px es el breakpoint 'lg' por defecto en Tailwind
      setIsContextPanelVisible(false);
    }
    // --------------------------------------------------------------------

    setLoading(true); setError(null);
    setChatMessages([]); setChatHistory([]); setSelectedFiles(null); setUploadResults(null); setSuccessMessage(null); setUploadProgress(0); setCurrentFile(''); setFilesRemaining(0); setUploadStatusMessage(''); setSystemPrompt(DEFAULT_SYSTEM_PROMPT_FRONTEND);
    setSelectedContext(context);
    try {
      const response = await axios.post<{ history: HistoryItem[] }>(`${API_URL}/load-chat-history`, { context_name: context }); const loadedHistory = response.data.history || []; setChatHistory(loadedHistory);
      if (loadedHistory.length > 0) { const messagesFromHistory = loadedHistory.flatMap(item => [{ role: 'user' as const, content: item.query }, { role: 'assistant' as const, content: item.answer }]); setChatMessages(messagesFromHistory); } else { setChatMessages([{ role: 'assistant', content: `Contexto '${context}' cargado.` }]); }
    } catch (err: any) { console.error("Error loading history:", err); setError(`Error cargando historial: ${err.response?.data?.detail || err.message}`); setChatMessages([{ role: 'assistant', content: `No se pudo cargar historial para '${context}'.` }]); } finally { setLoading(false); }
  };

  // --- Lógica de Dropzone (sin cambios) ---
  const onDrop = useCallback((acceptedFiles: File[]) => {
    setSuccessMessage(null); setError(null); setSelectedFiles(acceptedFiles);
  }, []);
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel.sheet.macroEnabled.12': ['.xlsm'],
    },
    multiple: true,
    disabled: uploading,
  });
  const handleRemoveFile = (fileNameToRemove: string) => {
    setSelectedFiles(prevFiles => prevFiles ? prevFiles.filter(file => file.name !== fileNameToRemove) : null);
  };

  // --- JSX ---
  return (
    // Contenedor principal: Mantiene la altura y el color base
    <div className="flex h-screen bg-base-100 text-base-content overflow-hidden" data-theme="light">

      {/* --- Overlay/Backdrop para cerrar el panel en móvil --- */}
      {isContextPanelVisible && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-30 lg:hidden"
          onClick={() => setIsContextPanelVisible(false)}
          aria-hidden="true"
        ></div>
      )}

      {/* --- Panel izquierdo (Contextos) - MODIFICADO PARA RESPONSIVE --- */}
      <div
        className={`
          fixed inset-y-0 left-0 z-40 w-64 bg-base-200 p-4 border-r border-base-300
          flex flex-col overflow-y-auto
          transform transition-transform duration-300 ease-in-out
          ${isContextPanelVisible ? 'translate-x-0' : '-translate-x-full'}
          lg:relative lg:translate-x-0 lg:flex lg:z-auto
        `}
        aria-label="Panel de Contextos"
        id="context-panel"
      >
        {/* --- Botón para cerrar el panel (visible solo en móvil cuando está abierto) --- */}
        <button
            onClick={() => setIsContextPanelVisible(false)}
            className="absolute top-2 right-2 btn btn-ghost btn-sm lg:hidden"
            aria-label="Cerrar panel de contextos"
        >
           <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" /></svg>
        </button>

        <h2 className="text-lg font-bold mb-4 flex-shrink-0 mt-6 lg:mt-0">Contextos</h2>
        {/* Crear contexto (sin cambios internos) */}
        <div className="mb-4 flex-shrink-0">
          <input type="text" value={newContextName} onChange={(e) => setNewContextName(e.target.value)} placeholder="Nuevo contexto" className="input input-bordered input-sm w-full mb-2 bg-base-100" disabled={isCreatingContext || loading}/>
          <button onClick={handleCreateContext} disabled={!newContextName.trim() || isCreatingContext || loading} className={`btn btn-sm btn-primary w-full ${isCreatingContext ? 'loading' : ''}`}> {isCreatingContext ? 'Creando...' : 'Crear'} </button>
        </div>
        {/* Lista de contextos (sin cambios internos) */}
        <div className="flex-grow overflow-y-auto">
          {loading && contexts.length === 0 && <div className="loading loading-spinner text-primary mx-auto my-4"></div>}
          <ul className="menu menu-sm bg-base-100 rounded-box p-2 w-full">
            {contexts.map((context) => (
              <li key={context}>
                <a onClick={() => handleContextSelect(context)} className={`flex justify-between items-center rounded ${selectedContext === context ? 'active bg-primary text-primary-content' : 'hover:bg-base-300'}`}>
                  <span className="truncate flex-grow mr-2">{context}</span>
                  <button onClick={(e) => { e.stopPropagation(); handleDeleteContext(context); }} className="btn btn-xs btn-ghost text-error hover:bg-error hover:text-error-content ml-auto flex-shrink-0 p-1" aria-label={`Eliminar ${context}`} disabled={loading || uploading}>
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" /></svg>
                  </button>
                </a>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* --- Panel principal (Chat y Subida) --- */}
      <div className="flex-1 flex flex-col p-4 bg-base-100 overflow-hidden"> {/* overflow-hidden previene scroll doble */}

        {/* --- Botón para ABRIR el panel de contextos (visible solo en móvil) --- */}
        <button
          onClick={() => setIsContextPanelVisible(true)}
          className="btn btn-ghost btn-sm lg:hidden mb-2 self-start" // Lo ponemos arriba a la izquierda
          aria-label="Abrir panel de contextos"
          aria-controls="context-panel"
          aria-expanded={isContextPanelVisible}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16m-7 6h7" /></svg>
        </button>

        {/* Contenido del panel principal */}
        {selectedContext ? (
          <>
            {/* Encabezado y Subida - CON DROPZONE (sin cambios internos) */}
            <div className="pb-4 border-b border-base-300 mb-4 flex-shrink-0 space-y-3">
              <h1 className="text-xl font-semibold truncate lg:hidden">
                Contexto: {selectedContext}
              </h1>
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors duration-200 ease-in-out ${isDragActive ? 'border-primary bg-primary/10' : 'border-base-300 hover:border-base-content/50'} ${uploading ? 'cursor-not-allowed opacity-50 bg-base-200' : 'bg-base-100 hover:bg-base-200'}`}
              >
                <input {...getInputProps()} />
                { isDragActive ? <p className="text-primary font-semibold">Suelta los archivos aquí...</p> : <p className="text-base-content/70 text-sm"> Arrastra archivos aquí, o haz clic para seleccionar (.pdf, .docx, .xlsx, .xlsm) </p> }
              </div>

              {selectedFiles && selectedFiles.length > 0 && !uploading && (
                <div className="space-y-1 text-sm bg-base-200 p-2 rounded-md max-h-32 overflow-y-auto">
                  <p className="font-medium text-xs mb-1 text-base-content/70">Archivos a subir:</p>
                  <ul className="list-none pl-0">
                    {selectedFiles.map((file) => (
                      <li key={file.name} className="flex justify-between items-center text-xs py-0.5 group">
                        <span className="truncate pr-2">{file.name}</span>
                        <button onClick={() => handleRemoveFile(file.name)} className="btn btn-xs btn-ghost text-error opacity-50 group-hover:opacity-100 transition-opacity" aria-label={`Quitar ${file.name}`} title={`Quitar ${file.name}`} disabled={uploading}>✕</button>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {selectedFiles && selectedFiles.length > 0 && (
                 <div className="flex justify-start">
                  <button className={`btn btn-sm btn-secondary ${uploading ? 'loading' : ''}`} onClick={() => handleFileUpload(selectedFiles)} disabled={uploading || loading}> {uploading ? 'Subiendo...' : `Subir ${selectedFiles.length} archivo(s)`} </button>
                </div>
              )}

              {uploading && (
                <div className="mt-3 space-y-1">
                  <p className="text-sm text-base-content/80 text-center font-medium">{uploadStatusMessage || 'Iniciando...'}</p>
                  <progress className="progress progress-primary w-full" value={uploadProgress} max="100"></progress>
                  <div className="text-xs text-center text-base-content/60 tabular-nums"> {currentFile && `Archivo: ${currentFile} - `} {`${uploadProgress}%`} </div>
                </div>
              )}
              {successMessage && !uploading && <div className="mt-3 alert alert-success p-2 text-sm shadow-sm"><span>{successMessage}</span></div>}
              {error && (!uploading || error.includes('WebSocket')) && <div className="mt-3 alert alert-error p-2 text-sm shadow-sm"><span>{error}</span></div>}
            </div>

            {/* Área de Chat (con overflow-y-auto ajustado) */}
            <div className="flex-grow overflow-y-auto mb-4 pr-2 min-h-0"> {/* Añadido min-h-0 para asegurar que flex-grow funcione bien */}
               {chatMessages.length === 0 && !loading && ( <p className='text-center text-base-content/50 mt-10'>Inicia la conversación o sube documentos.</p> )}
              {chatMessages.map((message, index) => (
                <div key={index} className={`chat ${message.role === 'user' ? 'chat-end' : 'chat-start'}`}>
                  <div className={`chat-bubble ${message.role === 'user' ? 'chat-bubble-primary' : 'chat-bubble-secondary'}`} style={{ whiteSpace: 'pre-wrap', overflowWrap: 'break-word' }}>
                    {message.content}
                  </div>
                </div>
              ))}
               <div ref={messagesEndRef} />
            </div>

            {/* Input de pregunta y Prompt (sin cambios internos) */}
            <form onSubmit={(e) => { e.preventDefault(); handleAskQuestion(query); }} className="flex flex-col gap-2 pt-4 border-t border-base-300 flex-shrink-0">
                 <textarea rows={3} value={systemPrompt} onChange={(e) => setSystemPrompt(e.target.value)} placeholder="Define el comportamiento del asistente (opcional)..." className="textarea textarea-bordered w-full text-sm bg-base-100 focus:outline-none focus:border-primary" disabled={loading || uploading}/>
                 <div className="flex gap-2">
                    <input type="text" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Escribe tu pregunta..." className="input input-bordered w-full bg-base-100" disabled={loading || uploading}/>
                    <button type="submit" disabled={!query.trim() || loading || uploading} className={`btn btn-primary ${loading ? 'loading' : ''}`}> {loading ? 'Pensando...' : 'Enviar'} </button>
                 </div>
             </form>
          </>
        ) : (
          // Vista cuando no hay contexto seleccionado
          <div className="flex flex-col items-center justify-center h-full text-center">
             {loading ? <div className="loading loading-lg text-primary"></div> : <p className="text-xl text-base-content/50">Selecciona o crea un contexto.</p>}
             {error && <div className="mt-4 alert alert-error"><span>{error}</span></div>}
          </div>
        )}
      </div> {/* Fin del panel principal */}
    </div> // Fin del contenedor principal
  );
}

export default App;