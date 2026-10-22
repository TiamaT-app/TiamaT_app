"""
Service d'automatisation Label Studio via son API REST :
- création d'un projet nommé d'aprés le nom du projet déclaré dans TiamaT
- configuration du storage d'entrée (images) et de sortie (annotations)
- chaque fichier devient une tâche individuelle ("Files" / use_blob_urls=True)
- synchronisation initiale pour faire apparaître les tâches dans LS

Nécessite dans l'environnement :
- LABEL_STUDIO_URL (ex: http://localhost:8080) -- à défaut, localhost et le port de la configuration
- LABEL_STUDIO_API_TOKEN -- token utilisateur, visible sur la page
  "Account & Settings" de Label Studio une fois connectée à l'interface

Les chemins d'images sont toujours comparés après décodage (normalize_image_url) :
Label Studio enregistre la forme encodée pour les URL (noam%C3%A9, f%201.jpg), alors que
TiamaT construit la forme brute (noamé, f 1.jpg). Comparés tels quels, les deux chemins
d'une même image semblent différents.
"""

import hashlib
import json
import os
import time
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote

from label_studio_sdk import LabelStudio

from ..src.modules.generate_labels import get_labels, build_ls_label_config
from ..src.modules.manipulate_files import open_json_file
from ..src.scripts.run_label_studio import is_label_studio_running

from app.config import Config

LS_PORT = Config.LABEL_STUDIO_PORT
LS_URL = os.environ.get("LABEL_STUDIO_URL", f"http://localhost:{LS_PORT}")
# (?i) : extensions en majuscules comprises (.JPG des appareils photo)
IMAGE_REGEX_FILTER = r"(?i).*\.(jpe?g|png|tiff?)$"
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}


def _token() -> str | None:
    """
    Token d'API, lu à chaque appel : il peut avoir été ajouté par le formulaire de l'app
    après son démarrage. Lu une seule fois à l'import, il resterait vide jusqu'au redémarrage.
    """
    return os.environ.get("LABEL_STUDIO_API_TOKEN")


def _get_client() -> LabelStudio:
    """Renvoie un client LS authentifié."""
    return LabelStudio(base_url=LS_URL, api_key=_token())


def normalize_image_url(url: str) -> str:
    """
    Forme unique d'un chemin d'image, pour comparer les chemins entre eux :
    décodé (%C3%A9 -> é, %20 -> espace), séparateurs uniformisés, et Unicode normalisé
    (macOS peut écrire un même « é » de deux façons, une lettre ou un e suivi d'un accent).
    """
    return unicodedata.normalize("NFC", unquote(url).replace("\\", "/"))


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


def delete_ls_projects(project_name: str, corrections_only: bool = False) -> list[str]:
    """
    Supprime dans Label Studio les projets d'un projet TiamaT : le projet d'annotation
    ({project_name}) et le projet de correction ({project_name}_corrections).

    - À la suppression d'un projet TiamaT : les deux. Sans cela, un nouveau projet TiamaT du même
      nom retrouverait dans Label Studio (où les projets sont recherchés par titre) les tâches et
      annotations de l'ancien.
    - En fin de cycle (corrections_only=True) : seulement le projet de correction. Ses images et
      ses corrections ont rejoint la vérité terrain ; il sera recréé, vide, au prochain test.

    Les fichiers d'annotation sur le disque ne sont pas concernés.

    :return: les titres des projets supprimés.
    :raises RuntimeError: si Label Studio n'est pas lancé ou si le token d'API manque.
    """
    if not _token():
        raise RuntimeError("le token d'API de Label Studio n'est pas encore enregistré")
    port = int(LS_URL.rsplit(':', 1)[1])
    if not is_label_studio_running(port):
        raise RuntimeError("Label Studio n'est pas lancé")

    client = _get_client()
    titles = {f"{project_name}_corrections"}
    if not corrections_only:
        titles.add(project_name)
    deleted = []
    for project in client.projects.list():
        if project.title in titles:
            client.projects.delete(id=project.id)
            deleted.append(project.title)
    return sorted(deleted)


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


def _labels_in_config(label_config: str | None) -> list[str]:
    """Classes déclarées dans une configuration Label Studio (balises <Label value="...">)."""
    if not label_config:
        return []
    try:
        return [label.get("value") for label in ET.fromstring(label_config).iter("Label") if label.get("value")]
    except ET.ParseError:
        return []


def update_ls_label_config(project_id: int, labels_file: Path) -> None:
    """
    Met a jour le label config avec les classes du labels.txt.

    Les classes déjà déclarées dans Label Studio sont conservées, même si elles ne sont pas
    (encore) dans labels.txt : une classe créée dans l'interface de Label Studio n'apparaît dans
    labels.txt qu'au calcul des statistiques. La supprimer de la configuration la ferait
    disparaître de l'interface, et Label Studio refuse une configuration qui ne contient plus
    des classes utilisées par des annotations existantes.
    """
    client = _get_client()
    labels = list(get_labels(labels_file).values())
    current_config = getattr(client.projects.get(id=project_id), "label_config", None)
    labels += [label for label in _labels_in_config(current_config) if label not in labels]
    client.projects.update(id=project_id, label_config=build_ls_label_config(labels))


def update_ls_annotation(annotation_id: int, result: list) -> dict:
    """Met a jour le champ result d'une annotation existante."""
    client = _get_client()
    annotation = client.annotations.update(id=annotation_id, result=result)
    return annotation.dict()


def push_corrected_annotation(json_file: str) -> dict:
    """Pousse une annotation corrigee vers Label Studio."""
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

    :raises PermissionError: si Label Studio répond mais refuse le token d'API. Sans cette
        distinction, un token invalide faisait attendre la fin du délai (3 minutes), puis
        afficher « Label Studio n'a pas répondu », un message trompeur.
    """
    global LS_URL
    if ports is None:
        ports = [LS_PORT, 8081, 8082]
    base_url = LS_URL.rsplit(':', 1)[0]
    time.sleep(initial_delay)
    deadline = time.time() + timeout
    while time.time() < deadline:
        for port in ports:
            if not is_label_studio_running(port):
                continue
            url = f"{base_url}:{port}"
            try:
                client = LabelStudio(base_url=url, api_key=_token())
                client.projects.list()
                LS_URL = url
                return url
            except Exception as e:
                if getattr(e, "status_code", None) in (401, 403):
                    raise PermissionError(
                        "Label Studio refuse le token d'API : vérifiez celui de la page "
                        "« Account & Settings » de Label Studio.") from e
                # Serveur encore en cours de démarrage : on réessaie
        time.sleep(interval)
    return None


def _images_on_disk(project_name: str, chemin_images: Path) -> list[str]:
    """Chemins Label Studio (?d=projects/...) des images de vérité terrain du projet."""
    return [
        f"/data/local-files/?d=projects/{project_name}/image_inputs/ground_truth_images/{f.name}"
        for f in sorted(Path(chemin_images).iterdir())
        if f.is_file() and not f.name.startswith('.') and f.suffix.lower() in IMG_EXTENSIONS
    ]


def setup_label_studio_project(
    project_name: str,
    chemin_images: Path,
    chemin_labels: Path,
    labels_file: Path | None = None,
) -> tuple[int, str]:
    """
    Créé ou met a jour le projet LS d'annotation de la vérité terrain.
    Renvoie (project_id, ls_url).

    - Projet inexistant : création, storages, configuration des classes (si labels.txt existe),
      puis synchronisation, qui crée une tâche par image.
    - Projet existant : mise à jour des classes, puis import des seules images absentes de
      Label Studio, avec leur annotation si la vérité terrain en contient une (c'est le cas
      des images corrigées en fin de cycle).
    """
    ls_url = wait_for_label_studio()
    if ls_url is None:
        raise RuntimeError("Label Studio n'a pas repondu a temps.")

    project_id = get_existing_project_id(project_name)

    if project_id is None:
        project_id = create_ls_project(project_name)
        if labels_file and Path(labels_file).exists():
            update_ls_label_config(project_id, labels_file)
        storage_ids = configure_ls_storage(project_id, chemin_images, chemin_labels)
        sync_ls_storage(storage_ids["import_storage_id"])
        return project_id, ls_url

    client = _get_client()
    if labels_file and Path(labels_file).exists():
        update_ls_label_config(project_id, labels_file)

    # Tâches déjà dans Label Studio, pour ne pas les importer une 2e fois
    # (ni écraser les annotations en cours)
    existing_images = {normalize_image_url(task.data["image"]) for task in client.tasks.list(project=project_id)}

    # Annotations de vérité terrain sur le disque, indexées par chemin d'image normalisé
    gt_annotations = {}
    for f in Path(chemin_labels).iterdir():
        if f.is_file() and not f.name.startswith('.'):
            data = open_json_file(f)
            gt_annotations[normalize_image_url(data["task"]["data"]["image"])] = data

    # Importer uniquement les nouvelles images (pas déjà dans LS)
    new_tasks = []
    for img_url in _images_on_disk(project_name, chemin_images):
        key = normalize_image_url(img_url)
        if key in existing_images:
            continue
        if key in gt_annotations:
            # Nouvelle image avec annotation existante (par exemple une image corrigée en fin de cycle)
            new_tasks.append({"data": {"image": img_url},
                              "annotations": [{"result": gt_annotations[key]["result"]}]})
        else:
            # Nouvelle image sans annotation
            new_tasks.append({"data": {"image": img_url}})

    if new_tasks:
        client.projects.import_tasks(id=project_id, request=new_tasks)

    return project_id, ls_url


def _tasks_fingerprint(tasks: list) -> str:
    """
    Empreinte des tâches à corriger : images ET prédictions. Deux lots de tâches ont la même
    empreinte seulement s'ils proposent les mêmes prédictions sur les mêmes images.
    """
    normalized = sorted(
        (normalize_image_url(task["data"]["image"]), json.dumps(task.get("predictions", []), sort_keys=True))
        for task in tasks)
    return "TiamaT:" + hashlib.sha256(json.dumps(normalized).encode("utf-8")).hexdigest()


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
    - Projet existant, mêmes tâches (mêmes images et mêmes prédictions) : seule la configuration
      des classes est mise à jour. Recharger la page de correction ne supprime donc pas les
      corrections en cours (sinon, une image corrigée à nouveau aurait deux fichiers de correction).
    - Projet existant, nouvelles tâches (nouvelles images, ou autre modèle testé sur les mêmes
      images) : les anciennes tâches sont remplacées.
    L'empreinte des tâches importées est enregistrée dans la description du projet Label Studio.
    Renvoie (project_id, ls_url).
    """
    ls_url = wait_for_label_studio()
    if ls_url is None:
        raise RuntimeError("Label Studio n'a pas repondu a temps.")

    correction_project_name = f"{project_name}_corrections"
    client = _get_client()
    project_id = get_existing_project_id(correction_project_name)

    label_config = Path(label_config_file).read_text(encoding="utf-8")
    ls_result = open_json_file(ls_result_file)
    fingerprint = _tasks_fingerprint(ls_result)

    if project_id is None:
        project = client.projects.create(
            title=correction_project_name,
            label_config=label_config,
            description=fingerprint,
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
        return project_id, ls_url # type:ignore

    client.projects.update(id=project_id, label_config=label_config)

    if getattr(client.projects.get(id=project_id), "description", None) == fingerprint:
        # Mêmes tâches : on les garde, avec les corrections déjà faites
        return project_id, ls_url # type:ignore

    # Nouvelles tâches : remplacer les anciennes, en une seule requête plutôt qu'une par tâche
    client.tasks.delete_all_tasks(id=project_id)
    client.projects.import_tasks(id=project_id, request=ls_result) # type:ignore
    client.projects.update(id=project_id, description=fingerprint)

    return project_id, ls_url # type:ignore