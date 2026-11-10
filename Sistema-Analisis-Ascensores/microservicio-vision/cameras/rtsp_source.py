import os
from urllib.parse import urlsplit

import cv2
import numpy as np


class RTSPSource:
    """Optional OpenCV reader for one RTSP camera stream."""

    def __init__(self, url: str):
        try:
            parsed_url = urlsplit(url)
            valid_url = (
                parsed_url.scheme.lower() in {"rtsp", "rtsps"}
                and parsed_url.hostname is not None
            )
        except ValueError:
            valid_url = False

        if not valid_url:
            raise ValueError("La dirección debe ser una URL RTSP válida.")

        self._url = url
        self._capture = None

    @classmethod
    def from_environment(cls, variable_name: str) -> "RTSPSource":
        url = os.getenv(variable_name)
        if not url:
            raise RuntimeError(
                f"No está configurada la variable de entorno {variable_name}."
            )
        return cls(url)

    def open(self) -> "RTSPSource":
        if self._capture is not None and self._capture.isOpened():
            return self

        self.close()
        capture = cv2.VideoCapture(self._url)
        if not capture.isOpened():
            capture.release()
            raise ConnectionError(
                "No se pudo abrir la cámara RTSP. Verifica la dirección, "
                "las credenciales y el acceso a la red."
            )

        self._capture = capture
        return self

    def read(self) -> np.ndarray:
        if self._capture is None or not self._capture.isOpened():
            raise RuntimeError("La cámara RTSP no está abierta.")

        received, frame = self._capture.read()
        if not received or frame is None:
            raise ConnectionError(
                "La cámara RTSP no entregó un fotograma."
            )

        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "RTSPSource":
        return self.open()

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.close()
