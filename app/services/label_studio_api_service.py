"""
Service d'automatisation Label Studio via son API REST :
- création d'un projet nommé d'aprés le nom du projet déclaré dans TiamaT
- configuration du storage d'entrée (images) et de sortie (annotations)
- chaque fichier devient une tâche individuelle ("Files" / use_blob_urls=True)
- synchronisation initiale pour faire apparaître les tâches dans LS

Nécessite dans l'environnement :
- LABEL_STUDIO_URL (ex: http://localhost:8080) -- à defaut, localhost:8080
- LABEL_STUDIO_API_TOKEN -- token utilisateur, visible sur la page
  "Account & Settings" de Label Studio une fois connectée à l'interface
"""

import os
import time
from pathlib import Path

from label_studio_sdk import LabelStudio

from ..src.modules.generate_labels import get_labels, build_ls_label_config

from app.config import Config

LS_PORT = Config.LABEL_STUDIO_PORT
LS_URL = os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080")
LS_TOKEN = os.environ.get("LABEL_STUDIO_API_TOKEN")
IMAGE_REGEX_FILTER = r".*\.(jpe?g|png|tiff)$"


def _get_client() -> LabelStudio:
    """Renvoie un client LS authentifié."""
    return LabelStudio(base_url=LS_URL, api_key=LS_TOKEN)

def get_ls_url() -> str:
    """Renvoie l'URL courante de Label Studio (potentiellement mise a jour par wait_for_label_studio)."""
    return LS_URL

def get_existing_project_id(project_name: str) -> int | None:
    """Cherche un projet par son titre exact. Renvoie son ID ou None."""
    client = _get_client()
    for project in client.projects.list():
        if project.title == project_name:
            return project.id
    return None


def create_ls_project(project_name: str) -> int:
    """Cree un projet Label Studio et renvoie son ID."""
    client = _get_client()
    project = client.projects.create(title=project_name)
    return project.id # type: ignore[assignment]


def configure_ls_storage(project_id: int, chemin_images: Path, chemin_labels: Path) -> dict:
    """
    Configure le storage source (images) et cible (annotations).
    Renvoie {"import_storage_id": ..., "export_storage_id": ...}.
    """
    client = _get_client()

    import_storage = client.import_storage.local.create(
        project=project_id,
        path=str(chemin_images),
        regex_filter=IMAGE_REGEX_FILTER,
        use_blob_urls=True,
        title="ground_truth_images",
    )

    export_storage = client.export_storage.local.create(
        project=project_id,
        path=str(chemin_labels),
        title="ground_truth_annotations",
    )

    return {
        "import_storage_id": import_storage.id,
        "export_storage_id": export_storage.id,
    }


def sync_ls_storage(import_storage_id: int) -> None:
    """Declenche la synchronisation : cree une tache par image presente."""
    client = _get_client()
    client.import_storage.local.sync(id=import_storage_id)


def get_import_storage_id(project_id: int) -> int | None:
    """Recupere l'ID du storage d'import d'un projet existant."""
    client = _get_client()
    storages = list(client.import_storage.local.list(project=project_id))
    if storages:
        return storages[0].id
    return None


def update_ls_label_config(project_id: int, labels_file: Path) -> None:
    """Met a jour le label config avec les classes du labels.txt."""
    client = _get_client()
    labels = get_labels(labels_file)
    label_config = build_ls_label_config(labels)
    client.projects.update(id=project_id, label_config=label_config)


def update_ls_annotation(annotation_id: int, result: list) -> dict:
    """Met a jour le champ result d'une annotation existante."""
    client = _get_client()
    annotation = client.annotations.update(id=annotation_id, result=result)
    return annotation.dict()


def push_corrected_annotation(json_file: str) -> dict:
    """Pousse une annotation corrigee vers Label Studio."""
    from ..src.modules.manipulate_files import open_json_file
    data = open_json_file(json_file)
    return update_ls_annotation(annotation_id=data["id"], result=data["result"])


def wait_for_label_studio(
    ports: list[int] | None = None,
    timeout: int = 180,
    interval: int = 1,
    initial_delay: int = 0,
) -> str | None:
    """
    Attend que LS reponde en essayant plusieurs ports.
    Met a jour LS_URL avec le port trouve.
    """
    global LS_URL
    if ports is None:
        ports = [LS_PORT, 8081, 8082]
    base_url = LS_URL.rsplit(':', 1)[0]
    time.sleep(initial_delay)
    deadline = time.time() + timeout
    while time.time() < deadline:
        for port in ports:
            url = f"{base_url}:{port}"
            try:
                client = LabelStudio(base_url=url, api_key=LS_TOKEN)
                client.projects.list()
                LS_URL = url
                return url
            except Exception:
                pass
        time.sleep(interval)
    return None


def setup_label_studio_project(
    project_name: str,
    chemin_images: Path,
    chemin_labels: Path,
    labels_file: Path | None = None,
) -> tuple[int, str]:
    """
    Cree ou met a jour un projet LS.
    Renvoie (project_id, ls_url).
    """
    ls_url = wait_for_label_studio()
    if ls_url is None:
        raise RuntimeError("Label Studio n'a pas repondu a temps.")

    project_id = get_existing_project_id(project_name)

    if project_id is None:
        project_id = create_ls_project(project_name)
        storage_ids = configure_ls_storage(project_id, chemin_images, chemin_labels)
        sync_ls_storage(storage_ids["import_storage_id"])
    else:
        if labels_file and labels_file.exists():
            update_ls_label_config(project_id, labels_file)
        import_storage_id = get_import_storage_id(project_id)
        if import_storage_id:
            sync_ls_storage(import_storage_id)

    return project_id, ls_url