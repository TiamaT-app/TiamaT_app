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
from..src.modules.manipulate_files import open_json_file, convert_gt_annotation_to_ls_task

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
    Créé ou met a jour un projet LS.
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
        client = _get_client()
        if labels_file and labels_file.exists():
            update_ls_label_config(project_id, labels_file)
        
        # Récupérer les taches existantes pour ne pas écraser les annotations en cours
        existing_tasks = {
            task.data["image"]: task.id 
            for task in client.tasks.list(project=project_id)
        }
        
        # Images présentes sur le disque
        img_folder = chemin_images  # projects/{nom}/image_inputs/ground_truth_images
        img_exts = {'.jpg', '.jpeg', '.png', '.tiff'}
        images_on_disk = {
            f"/data/local-files/?d=projects/{project_name}/image_inputs/ground_truth_images/{f.name}"
            for f in img_folder.iterdir()
            if f.is_file() and not f.name.startswith('.') and f.suffix.lower() in img_exts
        }
        
        # Annotations GT existantes
        gt_annotations = {
            data["task"]["data"]["image"]: data
            for f in chemin_labels.iterdir()
            if f.is_file() and not f.name.startswith('.')
            for data in [open_json_file(f)]
        }
        
        # Importer uniquement les nouvelles images (pas déjà dans LS)
        new_tasks = []
        for img_url in images_on_disk:
            if img_url not in existing_tasks:
                if img_url in gt_annotations:
                    # Nouvelle image avec annotation existante
                    new_tasks.append({
                        "data": {"image": img_url},
                        "annotations": [{"result": gt_annotations[img_url]["result"]}]
                    })
                else:
                    # Nouvelle image sans annotation
                    new_tasks.append({"data": {"image": img_url}})
        
        if new_tasks:
            client.projects.import_tasks(id=project_id, request=new_tasks)

    return project_id, ls_url


def setup_ls_correction_project(
    project_name: str,
    labels_folder_path: Path,
    img_folder_path: Path,
    ls_result_file: Path,
    label_config_file: Path,
) -> tuple[int, str]:
    """
    Créé ou met à jour le projet de correction des predictions du modèle.
    Le nom du projet LS sera {project_name}_corrections.
    - Premier lancement : création + storage export + import des tâches avec predictions
    - Projet existant : mise à jour du label config uniquement, pas de nouvel import
    Renvoie (project_id, ls_url).
    """
    ls_url = wait_for_label_studio()
    if ls_url is None:
        raise RuntimeError("Label Studio n'a pas repondu a temps.")

    correction_project_name = f"{project_name}_corrections"
    client = _get_client()
    project_id = get_existing_project_id(correction_project_name)

    label_config = label_config_file.read_text(encoding="utf-8")

    if project_id is None:
        ls_result = open_json_file(ls_result_file)
        project = client.projects.create(
            title=correction_project_name,
            label_config=label_config,
        )
        project_id = project.id
        # storage d'import pour que LS sache où trouver les images
        # pas de sync -- les taches sont importées via import_tasks()
        client.import_storage.local.create(
            project=project_id,
            path=str(img_folder_path),
            regex_filter=IMAGE_REGEX_FILTER,
            use_blob_urls=True,
            title="eval_images",
        )
        client.export_storage.local.create(
            project=project_id,
            path=str(labels_folder_path),
            title="prediction_corrections",
        )
        client.projects.import_tasks(id=project_id, request=ls_result)  # type: ignore
    else:
        # Projet existant : vider les taches et reimporter les nouvelles
        client.projects.update(id=project_id, label_config=label_config)
        # Récupérer tous les IDs des tâches et les supprimer une par une 
        # A MODIFIER ? Peut être long pour les dossiers avec beaucoup de tâches
            # -> A voir à l'usage
        tasks = list(client.tasks.list(project=project_id))
        for task in tasks:
            client.tasks.delete(id=task.id)
        ls_result = open_json_file(ls_result_file)
        client.projects.import_tasks(id=project_id, request=ls_result) # type:ignore

    return project_id, ls_url # type:ignore

