
import os
import smtplib
from email.message import EmailMessage


def enviar_correo(destinatario: str, asunto: str, contenido: str):
    correo_emisor = os.getenv("AFOROVISION_EMAIL")
    contrasena_app = os.getenv("AFOROVISION_EMAIL_APP_PASSWORD")

    if not correo_emisor or not contrasena_app:
        raise RuntimeError(
            "Faltan las credenciales de correo en las variables de entorno."
        )

    mensaje = EmailMessage()
    mensaje["Subject"] = asunto
    mensaje["From"] = correo_emisor
    mensaje["To"] = destinatario
    mensaje.set_content(contenido)

    with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as servidor:
        servidor.ehlo()
        servidor.starttls()
        servidor.ehlo()
        servidor.login(correo_emisor, contrasena_app)
        servidor.send_message(mensaje)
