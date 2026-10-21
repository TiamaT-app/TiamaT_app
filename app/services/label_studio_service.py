"""
Service pour le lancement de Label Studio en arrière-plan.
"""

import os
from threading import Thread

from ..src.scripts.run_label_studio import launch_LS
from .project_service import projects_folder


def configure_label_studio_root() -> None:
    """
    Définit la racine que Label Studio est autorisé à scanner pour servir des fichiers locaux.

    La racine est le dossier de l'app (le parent de projects/), et non projects/ lui-même :
    les chemins des tâches commencent par ?d=projects/..., ils sont relatifs à cette racine.
    C'est aussi la racine utilisée par launch_LS : les deux doivent rester identiques.
    Ces variables ne sont lues qu'au démarrage de Label Studio : elles n'ont pas d'effet
    sur une instance déjà lancée.
    """
    root = projects_folder.parent.as_posix()
    os.environ["LOCAL_FILES_DOCUMENT_ROOT"] = root
    os.environ["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = root
    os.environ["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"


def launch_label_studio_async(on_ready=None) -> Thread:
    """
    Lance Label Studio dans un thread daemon pour ne pas bloquer la requête.

    Si on_ready est fourni, il est appelé juste après le lancement, ou tout de suite si
    Label Studio tournait déjà. C'est on_ready (setup_label_studio_project...) qui attend
    ensuite que Label Studio réponde, avec wait_for_label_studio.
    """
    def _run():
        launch_LS()
        if on_ready:
            on_ready()

    thread = Thread(target=_run)
    thread.daemon = True
    thread.start()

    return thread