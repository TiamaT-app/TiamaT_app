"""
Service pour le lancement de Label Studio en arrière-plan.
"""

import os
from threading import Thread

from ..src.scripts.run_label_studio import launch_LS
from .project_service import projects_folder


def configure_label_studio_root() -> None:
    """
    Définit la racine que Label Studio est autorisé à scanner pour servir des
    fichiers locaux. Racine fixe = projects_folder (le parent de TOUS les
    projets), pas le dossier du projet courant -- c'est ce qui permet de
    piloter Label Studio via son API sans relancer le serveur a chaque
    changement de projet.
    """
    os.environ["LOCAL_FILES_DOCUMENT_ROOT"] = projects_folder.as_posix()
    os.environ["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"


def launch_label_studio_async(on_ready=None) -> Thread:
    """
    Lance Label Studio dans un thread daemon pour ne pas bloquer la requête
    Si on_ready est fourni, l'appelle une fois LS confirmé comme démarré.
    """
    def _run():
        launch_LS()
        if on_ready:
            on_ready()
    
    thread = Thread(target=_run)
    thread.daemon = True
    thread.start()
    
    return thread