from ultralytics import YOLO


class DetectorYOLO:
    def __init__(self, modelo: str = "yolo11n.pt"):
        self.modelo = YOLO(modelo)

    def detectar(self, imagen):
        resultados = self.modelo(imagen, verbose=False)
        return resultados

    def seguir(self, imagen):
        resultados = self.modelo.track(
        imagen,
        persist=True,
        tracker="microservicio-vision/config/bytetrack.yaml",
        verbose=False,
        conf=0.3
    )
        return resultados