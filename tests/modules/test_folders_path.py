"""Tests de la reconstruction des chemins de dossiers d'un projet TiamaT.

Les résultats sont comparés sous forme de Path (et non de chaînes), pour que
les tests passent aussi bien sous macOS/Linux (/) que sous Windows (\\).
"""

from pathlib import Path

import pytest

from app.src.modules.folders_path import (
    get_corrections_folder_inference,
    get_correctedLabels_folder,
    get_data_folder,
    get_ground_truth_folder_training,
    get_img_folder_inference,
    get_img_folder_training,
    get_results_folder,
)


@pytest.fixture
def project_folder(tmp_path):
    # Simule <racine>/projects/manuscrits dans un dossier temporaire.
    return tmp_path / "projects" / "manuscrits"


@pytest.fixture
def yolo_model_folder(tmp_path):
    # Simule <racine>/runs/train/modele_v1 (structure produite par YOLO).
    return tmp_path / "runs" / "train" / "modele_v1"


# --- Dossiers internes au projet ---------------------------------------------

@pytest.mark.parametrize(
    "fonction, sous_chemin",
    [
        (get_img_folder_training, ("image_inputs", "ground_truth_images")),
        (get_img_folder_inference, ("image_inputs", "eval_images")),
        (get_ground_truth_folder_training, ("annotations", "ground_truth")),
        (get_corrections_folder_inference, ("annotations", "prediction_corrections")),
    ],
    ids=["images_entrainement", "images_inference", "ground_truth", "corrections"],
)
def test_dossiers_du_projet(project_folder, fonction, sous_chemin):
    assert Path(fonction(project_folder)) == project_folder.joinpath(*sous_chemin)


@pytest.mark.parametrize(
    "fonction",
    [get_img_folder_training, get_img_folder_inference,
     get_ground_truth_folder_training, get_corrections_folder_inference],
)
def test_accepte_str_et_path(project_folder, fonction):
    # Les fonctions sont appelées tantôt avec une chaîne, tantôt avec un Path.
    assert fonction(str(project_folder)) == fonction(project_folder)


def test_renvoie_une_chaine(project_folder):
    assert isinstance(get_img_folder_training(project_folder), str)


# --- Dossier data -------------------------------------------------------------

def test_data_folder_chemin_absolu(tmp_path, project_folder):
    # <racine>/projects/manuscrits -> <racine>/data/manuscrits
    assert Path(get_data_folder(project_folder)) == tmp_path / "data" / "manuscrits"


def test_data_folder_chemin_relatif():
    # Même logique avec un chemin relatif, par exemple lancé depuis un sous-dossier.
    assert Path(get_data_folder("tiamat/projects/manuscrits")) == Path("tiamat/data/manuscrits")


def test_data_folder_hors_du_projet(project_folder):
    # Garde-fou contre le bug déjà rencontré : data/ ne doit jamais se
    # retrouver à l'intérieur du dossier du projet.
    assert project_folder not in Path(get_data_folder(project_folder)).parents


# --- Dossiers de résultats YOLO -----------------------------------------------

def test_results_folder(tmp_path, project_folder, yolo_model_folder):
    # <racine>/runs/train/modele_v1 -> <racine>/runs/predict/manuscrits_modele_v1
    attendu = tmp_path / "runs" / "predict" / "manuscrits_modele_v1"
    assert Path(get_results_folder(project_folder, yolo_model_folder)) == attendu


def test_corrected_labels_dans_results_folder(project_folder, yolo_model_folder):
    # Les deux fonctions dupliquent le même calcul : on vérifie qu'elles restent cohérentes.
    results = Path(get_results_folder(project_folder, yolo_model_folder))
    corrected = Path(get_correctedLabels_folder(project_folder, yolo_model_folder))
    assert corrected == results / "correctedLabels"