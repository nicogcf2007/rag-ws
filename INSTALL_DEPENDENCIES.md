# Instalación de Dependencias para OCR

Este proyecto utiliza OCR (Reconocimiento Óptico de Caracteres) para extraer texto de PDFs escaneados. Para que funcione correctamente, necesitas instalar dos dependencias principales:

## 1. Tesseract OCR

### Windows
1. Descarga el instalador de Tesseract desde [aquí](https://github.com/UB-Mannheim/tesseract/wiki)
2. Ejecuta el instalador y asegúrate de seleccionar el idioma español durante la instalación
3. Añade Tesseract a tu PATH de sistema (normalmente `C:\Program Files\Tesseract-OCR`)

### Linux
```
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-spa  # Para soporte en español
```

### macOS
```
brew install tesseract
brew install tesseract-lang  # Para soporte multilenguaje
```

## 2. Poppler (requerido por pdf2image)

### Windows
1. Descarga Poppler para Windows desde [aquí](https://github.com/oschwartz10612/poppler-windows/releases)
2. Extrae el archivo ZIP en una ubicación como `C:\Poppler`
3. Añade la carpeta `bin` a tu PATH: `C:\Poppler\bin`
4. Reinicia tu aplicación o servidor

### Linux
```
sudo apt-get install poppler-utils
```

### macOS
```
brew install poppler
```

## Verificación

Para verificar que las dependencias están correctamente instaladas:

1. Abre una terminal o línea de comandos
2. Ejecuta `tesseract --version` para verificar Tesseract
3. Ejecuta `pdfinfo -v` (Windows) o `pdftoppm -v` (Linux/macOS) para verificar Poppler

Si alguno de estos comandos no funciona, revisa que las rutas estén correctamente añadidas al PATH del sistema.

## Configuración Alternativa

Si no puedes añadir Poppler al PATH, puedes especificar la ruta directamente en el código:

```python
from pdf2image import convert_from_path

# Especificar la ruta a Poppler
images = convert_from_path(file_path, poppler_path='C:\Poppler\bin')
```

Esta configuración puede hacerse modificando el archivo `api.py` en la función `extract_text_from_pdf`.