"""Tests des fonctions de manipulation de fichiers (JSON, images, listes de fichiers)."""

import json
from pathlib import Path

import pytest

from app.src.modules.manipulate_files import (
    change_ids_and_path,
    convert_gt_annotation_to_ls_task,
    exclude_training_images,
    find_image_path,
    get_files,
    get_gt_project_id,
    load_data_from_files,
    open_json_file,
    save_json_file,
)


@pytest.fixture
def annotation_ls():
    """Une annotation au format d'export Label Studio, réduite à l'essentiel."""
    return {
        "id": 999,
        "project": 3,
        "result": [{"value": {"x": 10, "y": 20, "width": 30, "height": 40,
                              "rectanglelabels": ["miniature"]}}],
        "task": {"project": 3,
                 "data": {"image": "/data/local-files/?d=projects/manuscrits/image_inputs/eval_images/f12.jpg"}},
    }


# --- open_json_file / save_json_file ---------------------------------------------

def test_save_puis_open_json_aller_retour(tmp_path, annotation_ls):
    fichier = tmp_path / "annotation.json"
    save_json_file(fichier, annotation_ls)
    assert open_json_file(fichier) == annotation_ls


def test_json_accents(tmp_path):
    fichier = tmp_path / "annotation.json"
    save_json_file(fichier, {"classe": "lettrine ornée"})
    assert open_json_file(fichier) == {"classe": "lettrine ornée"}


# --- change_ids_and_path ------------------------------------------------------------

def test_change_ids_and_path(tmp_path, annotation_ls):
    # Le fichier s'appelle 42.json : son id doit devenir 42.
    fichier = tmp_path / "42.json"
    save_json_file(fichier, annotation_ls)

    change_ids_and_path(fichier, project_id=7)

    data = open_json_file(fichier)
    assert data["id"] == 42
    assert data["project"] == 7
    assert data["task"]["project"] == 7
    assert data["task"]["data"]["image"].endswith("image_inputs/ground_truth_images/f12.jpg")
    assert data["result"] == annotation_ls["result"]  # les annotations ne sont pas touchées


# --- get_files ---------------------------------------------------------------------

def test_get_files_filtre_par_extension_et_cherche_dans_les_sous_dossiers(tmp_path):
    (tmp_path / "sous_dossier").mkdir()
    for nom in ["a.txt", "b.jpg", "sous_dossier/c.txt"]:
        (tmp_path / nom).write_text("")

    noms = sorted(f.name for f in get_files(tmp_path, "txt"))
    assert noms == ["a.txt", "c.txt"]


# --- exclude_training_images ---------------------------------------------------------

def test_exclude_training_images(tmp_path):
    fichiers = [tmp_path / "f1.txt", tmp_path / "f2.txt", tmp_path / "f3.txt"]
    # Les images d'entraînement sont données avec leur extension, les fichiers comparés sans.
    restants = exclude_training_images(fichiers, ["f1.jpg", "f3.png"])
    assert restants == [tmp_path / "f2.txt"]


# --- load_data_from_files ------------------------------------------------------------

def test_load_data_from_files_concatene_les_lignes(tmp_path):
    (tmp_path / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.05 0.05\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("  2 0.9 0.9 0.1 0.1  \n", encoding="utf-8")

    lignes = load_data_from_files([tmp_path / "a.txt", tmp_path / "b.txt"])
    assert lignes == ["0 0.5 0.5 0.2 0.2", "1 0.1 0.1 0.05 0.05", "2 0.9 0.9 0.1 0.1"]


# --- find_image_path -------------------------------------------------------------------

@pytest.mark.parametrize("extension", [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".JPG", ".TIF"])
def test_find_image_path_trouve_chaque_extension(tmp_path, extension):
    image = tmp_path / f"f12{extension}"
    image.write_bytes(b"")
    assert find_image_path(tmp_path, "f12").samefile(image)


def test_find_image_path_image_absente(tmp_path):
    with pytest.raises(FileNotFoundError, match="f12"):
        find_image_path(tmp_path, "f12")


# --- get_gt_project_id ------------------------------------------------------------------

def test_get_gt_project_id_lit_le_projet(tmp_path, annotation_ls):
    save_json_file(tmp_path / "1.json", annotation_ls)
    assert get_gt_project_id(tmp_path) == 3


def test_get_gt_project_id_dossier_vide(tmp_path):
    assert get_gt_project_id(tmp_path) is None


def test_get_gt_project_id_ignore_les_fichiers_caches(tmp_path):
    # Sur Mac, un .DS_Store apparaît dans presque tous les dossiers ouverts dans le Finder.
    (tmp_path / ".DS_Store").write_text("")
    assert get_gt_project_id(tmp_path) is None


# --- convert_gt_annotation_to_ls_task ----------------------------------------------------

def test_convert_gt_annotation_to_ls_task(tmp_path, annotation_ls):
    fichier = tmp_path / "1.json"
    save_json_file(fichier, annotation_ls)

    tache = convert_gt_annotation_to_ls_task(fichier)

    assert tache == {
        "data": {"image": annotation_ls["task"]["data"]["image"]},
        "annotations": [{"result": annotation_ls["result"]}],
    }


def test_get_files_ignore_les_fichiers_et_dossiers_caches(tmp_path):
    # ._f12.jpg : fichier créé par macOS sur les disques externes.
    # .ipynb_checkpoints : dossier caché créé par Jupyter.
    (tmp_path / ".ipynb_checkpoints").mkdir()
    for nom in ["f12.jpg", "._f12.jpg", ".ipynb_checkpoints/f13.jpg"]:
        (tmp_path / nom).write_text("")

    assert [f.name for f in get_files(tmp_path, "jpg")] == ["f12.jpg"]