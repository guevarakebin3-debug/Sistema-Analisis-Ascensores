# Fuente opcional de cámara RTSP

`RTSPSource` permite abrir una cámara IP que transmita por RTSP y leer sus
fotogramas con OpenCV. Es independiente del flujo actual: el microservicio no
la abre ni intenta conectarse a una cámara automáticamente.

## Uso cuando tengas una cámara

Configura la URL RTSP en una variable de entorno, sin escribirla en el código:

```powershell
$env:AFOROVISION_RTSP_INTERIOR = "rtsp://USUARIO:CLAVE@IP:554/RUTA"
```

La URL y las credenciales dependen del fabricante y de la configuración de la
cámara. En la misma terminal, desde `microservicio-vision`, se puede probar la
conexión así:

```powershell
@'
from cameras import RTSPSource

with RTSPSource.from_environment("AFOROVISION_RTSP_INTERIOR") as source:
    print("Fotograma recibido:", source.read().shape)
'@ | python
```

Para dos cámaras se pueden usar variables distintas, por ejemplo
`AFOROVISION_RTSP_INTERIOR` y `AFOROVISION_RTSP_EXTERIOR`. La variable de
PowerShell configurada con `$env:` solo dura mientras siga abierta esa
terminal.

Este módulo solo entrega fotogramas; todavía no está conectado a la interfaz
Monitoreo ni sustituye la carga de videos. La integración con el análisis se
puede hacer después de probar la cámara y confirmar su URL RTSP.
