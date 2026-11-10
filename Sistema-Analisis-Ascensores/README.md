# Sistema de Análisis y Detección de Sobrecarga en Ascensores

Aplicación para monitorear la ocupación de ascensores mediante una interfaz web,
un backend y un microservicio de visión por computadora.

## Componentes

- `frontend/`: páginas y estilos de la interfaz web.
- `backend/`: API principal, autenticación y acceso a la base SQLite.
- `microservicio-vision/`: API de detección y seguimiento con YOLO.
- `docs/`: documentación del proyecto.

## Requisitos

- Python instalado y disponible como `python` en PowerShell.
- Para visión por computadora, las dependencias indicadas en
  `microservicio-vision/requirements.txt`.

## Instalación y ejecución en Windows

Desde la carpeta raíz del proyecto, abre una terminal PowerShell e instala las
dependencias del backend:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload
```

En otra terminal, inicia el microservicio de visión:

```powershell
cd microservicio-vision
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8001
```

Cada entorno virtual se crea localmente y no se comparte en Git. La base de
datos SQLite se crea al iniciar el backend; los archivos locales de base de
datos tampoco se suben al repositorio.
