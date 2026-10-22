"""
Lancement du serveur Label Studio, dans un processus separe de l'app.

Label Studio sert les images directement depuis le disque ("local files") : le dossier
racine autorise est le dossier de l'app, d'ou les chemins en ?d=projects/... dans les taches.
"""
import os
import socket
import subprocess
from pathlib import Path

from app.config import Config

LS_PORT = Config.LABEL_STUDIO_PORT


def _configure_local_files() -> None:
    """Autorise Label Studio a servir les fichiers situes sous le dossier de l'app."""
    os.environ['LOCAL_FILES_DOCUMENT_ROOT'] = Path.cwd().as_posix()
    os.environ['LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT'] = Path.cwd().as_posix()
    os.environ['LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED'] = 'true'


def is_label_studio_running(port: int = LS_PORT) -> bool:
    """Vrai si un serveur repond deja sur le port de Label Studio."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def launch_LS() -> bool:
    """
    Lance Label Studio dans un processus non bloquant, sur le port de la configuration.

    Le token d'API (LABEL_STUDIO_API_TOKEN) est transmis s'il existe. Rien n'est lance si
    Label Studio repond deja sur ce port : sinon chaque visite de la page de lancement
    demarrerait une nouvelle instance.

    :return: True si Label Studio a ete lance, False s'il tournait deja.
    """
    if is_label_studio_running():
        print(f"Label Studio tourne déjà sur le port {LS_PORT}.")
        return False

    _configure_local_files()

    cmd: list[str] = ['label-studio', 'start', '--port', str(LS_PORT)]
    token = os.environ.get("LABEL_STUDIO_API_TOKEN")
    if token:
        cmd += ['--user-token', token, '--enable-legacy-api-token']

    # La commande n'est pas affichee telle quelle : elle contient le token, qui est un secret
    print(f"Lancement de Label Studio sur le port {LS_PORT} ({'avec' if token else 'sans'} token d'API)")
    subprocess.Popen(cmd, env=os.environ.copy())
    return True


def launch_LS2() -> str:
    """
    Lance Label Studio sans token, pour que l'utilisateur puisse en creer un dans son interface.

    Utilise le meme port que launch_LS (et non le port par defaut de Label Studio) :
    sinon deux instances distinctes, avec deux bases de donnees, pourraient tourner en meme temps.
    """
    launch_LS()
    return "Label studio launched, send you API TOKEN to the form"