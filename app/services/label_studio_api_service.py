"""
Service d'automatisation Label Studio via son API REST :
- création d'un projet nommé d'apres le nom du projet déclaré dans TiamaT
- configuration du storage d'entrée (images) et de sortie (annotations)
- chaque fichier devient une tâche individuelle ("Files" / use_blob_urls=True)
- synchronisation initiale pour faire apparaître les tâches dans LS

Necessite dans l'environnement :
- LABEL_STUDIO_URL (ex: http://localhost:8080) -- a defaut, localhost:8080
- LABEL_STUDIO_API_TOKEN -- token utilisateur, visible sur la page
  "Account & Settings" de Label Studio une fois connectee a l'interface
"""

import os
import time
from pathlib import Path

import requests

from ..src.modules.manipulate_files import open_json_file

LS_URL = os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080")
LS_TOKEN = os.environ.get("LABEL_STUDIO_API_TOKEN")

IMAGE_REGEX_FILTER = r".*\.(jpe?g|png|tiff)$"


def _headers():
    return {"Authorization": f"Token {LS_TOKEN}"}


def wait_for_label_studio(timeout=30, interval=1) -> bool:
    """
    Attend que le serveur Label Studio reponde. Comme LS est lance dans un
    thread juste avant ces appels (cf. labeling_routes.py), sans cette
    attente les appels API qui suivent risqueraient d'arriver avant que le
    serveur ait fini de demarrer.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{LS_URL}/api/projects", headers=_headers(), timeout=2)
            if r.status_code in (200, 401, 403):
                # le serveur repond, meme une erreur d'auth confirme qu'il est up
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(interval)
    return False


def create_ls_project(project_name: str) -> int:
    """Cree un projet Label Studio et renvoie son ID."""
    response = requests.post(
        f"{LS_URL}/api/projects",
        headers=_headers(),
        json={"title": project_name},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()["id"]


def configure_ls_storage(project_id: int, chemin_images: Path, chemin_labels: Path) -> dict:
    """
    Configure le storage source (images, une tache par fichier) et le
    storage cible (annotations) pour le projet donne.
    Renvoie {"import_storage_id": ..., "export_storage_id": ...}.
    """
    import_response = requests.post(
        f"{LS_URL}/api/storages/localfiles",
        headers=_headers(),
        json={
            "project": project_id,
            "path": str(chemin_images),
            "regex_filter": IMAGE_REGEX_FILTER,
            "use_blob_urls": True,  # "Files" : une tache par fichier
            "title": "ground_truth_images",
        },
        timeout=10,
    )
    import_response.raise_for_status()
    import_storage_id = import_response.json()["id"]

    export_response = requests.post(
        f"{LS_URL}/api/storages/export/localfiles",
        headers=_headers(),
        json={
            "project": project_id,
            "path": str(chemin_labels),
            "title": "ground_truth_annotations",
        },
        timeout=10,
    )
    export_response.raise_for_status()
    export_storage_id = export_response.json()["id"]

    return {"import_storage_id": import_storage_id, "export_storage_id": export_storage_id}


def sync_ls_storage(import_storage_id: int) -> None:
    """Declenche la synchronisation initiale : cree une tache par image deja presente."""
    response = requests.post(
        f"{LS_URL}/api/storages/localfiles/{import_storage_id}/sync",
        headers=_headers(),
        timeout=30,
    )
    response.raise_for_status()


def setup_label_studio_project(project_name: str, chemin_images: Path, chemin_labels: Path) -> int:
    """
    Orchestre les trois etapes : creation du projet, configuration du
    storage, synchronisation initiale. Renvoie l'ID du projet LS cree.
    """
    if not wait_for_label_studio():
        raise RuntimeError("Label Studio n'a pas repondu a temps -- le serveur a-t-il bien demarre ?")

    project_id = create_ls_project(project_name)
    storage_ids = configure_ls_storage(project_id, chemin_images, chemin_labels)
    sync_ls_storage(storage_ids["import_storage_id"])
    return project_id


def update_ls_annotation(annotation_id: int, result: list) -> dict:
    """
    Met a jour le champ "result" d'une annotation existante via PATCH.
    Necessaire car le storage d'export est a sens unique (DB -> disque) :
    modifier le fichier JSON directement et resynchroniser ne remonte jamais
    la correction dans Label Studio.
    """
    response = requests.patch(
        f"{LS_URL}/api/annotations/{annotation_id}/",
        headers=_headers(),
        json={"result": result},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def push_corrected_annotation(json_file: str) -> dict:
    """
    Lit un fichier JSON d'annotation corrige (deja passe par
    change_id_and_path) et pousse sa correction vers Label Studio via l'API.
    """
    data = open_json_file(json_file)
    return update_ls_annotation(annotation_id=data["id"], result=data["result"])