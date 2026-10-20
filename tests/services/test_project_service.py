"""Tests du service de gestion des projets (liste, création, suppression)."""

import pandas as pd
import pytest

from app.services.project_service import (
    create_project_structure,
    delete_project,
    list_existing_projects,
)


def creer_projet_complet(racine, nom, modeles=()):
    """Fabrique un projet avec ses dossiers data/, output/train/ et output/predict/."""
    create_project_structure(nom)
    (racine / "data" / nom).mkdir(parents=True)
    if modeles:
        pd.DataFrame({"model_name": list(modeles)}).to_csv(
            racine / "projects" / nom / f"{nom}.csv", index=False)
    for modele in modeles:
        (racine / "output" / "train" / modele / "weights").mkdir(parents=True)
        (racine / "output" / "predict" / f"{nom}_{modele}").mkdir(parents=True)


# --- list_existing_projects / create_project_structure ------------------------

def test_create_project_structure_cree_les_quatre_dossiers(espace_projets):
    chemin = create_project_structure("manuscrits")
    assert chemin == espace_projets / "projects" / "manuscrits"
    for sous_dossier in ["image_inputs/ground_truth_images", "image_inputs/eval_images",
                         "annotations/ground_truth", "annotations/prediction_corrections"]:
        assert (chemin / sous_dossier).is_dir()


def test_create_project_structure_ne_touche_pas_un_projet_existant(espace_projets):
    chemin = create_project_structure("manuscrits")
    (chemin / "image_inputs" / "ground_truth_images" / "f1.jpg").write_bytes(b"")
    create_project_structure("manuscrits")
    assert (chemin / "image_inputs" / "ground_truth_images" / "f1.jpg").exists()


def test_list_existing_projects_ignore_fichiers_et_dossiers_caches(espace_projets):
    create_project_structure("manuscrits")
    (espace_projets / "projects" / ".DS_Store").write_text("")
    (espace_projets / "projects" / ".cache").mkdir()
    assert list_existing_projects() == ["manuscrits"]


# --- delete_project --------------------------------------------------------------

def test_delete_project_supprime_tout_le_projet(espace_projets):
    creer_projet_complet(espace_projets, "manuscrits", ["manuscrits_20260301_101500"])

    delete_project("manuscrits")

    assert not (espace_projets / "projects" / "manuscrits").exists()
    assert not (espace_projets / "data" / "manuscrits").exists()
    assert not (espace_projets / "output" / "train" / "manuscrits_20260301_101500").exists()
    assert not (espace_projets / "output" / "predict"
                / "manuscrits_manuscrits_20260301_101500").exists()


def test_delete_project_ne_touche_pas_aux_autres_projets(espace_projets):
    creer_projet_complet(espace_projets, "manuscrits", ["manuscrits_20260301_101500"])
    creer_projet_complet(espace_projets, "incunables", ["incunables_20260502_090000"])

    delete_project("manuscrits")

    assert (espace_projets / "projects" / "incunables").is_dir()
    assert (espace_projets / "data" / "incunables").is_dir()
    assert (espace_projets / "output" / "train" / "incunables_20260502_090000").is_dir()
    assert (espace_projets / "output" / "predict"
            / "incunables_incunables_20260502_090000").is_dir()


def test_delete_project_sans_modele_ni_data(espace_projets):
    # Un projet tout juste créé : ni CSV, ni data/, ni output/.
    create_project_structure("manuscrits")
    delete_project("manuscrits")
    assert not (espace_projets / "projects" / "manuscrits").exists()


def test_delete_project_dossier_de_modele_deja_supprime(espace_projets):
    creer_projet_complet(espace_projets, "manuscrits", ["manuscrits_20260301_101500"])
    (espace_projets / "output" / "train" / "manuscrits_20260301_101500" / "weights").rmdir()
    (espace_projets / "output" / "train" / "manuscrits_20260301_101500").rmdir()

    delete_project("manuscrits")  # ne doit pas s'arrêter à mi-chemin

    assert not (espace_projets / "projects" / "manuscrits").exists()


@pytest.mark.parametrize("nom", ["", ".", "..", "inexistant", "manuscrits/../incunables"],
                         ids=["vide", "point", "point_point", "inexistant", "chemin_detourne"])
def test_delete_project_refuse_un_nom_invalide(espace_projets, nom):
    # Garde-fou essentiel : un nom vide désignerait le dossier projects/ lui-même.
    creer_projet_complet(espace_projets, "manuscrits")
    creer_projet_complet(espace_projets, "incunables")

    with pytest.raises(ValueError):
        delete_project(nom)

    assert (espace_projets / "projects" / "manuscrits").is_dir()
    assert (espace_projets / "projects" / "incunables").is_dir()
    assert (espace_projets / "data" / "manuscrits").is_dir()


def test_delete_project_nom_de_modele_detourne_dans_le_csv(espace_projets):
    # Un nom de modèle contenant ../ ne doit pas faire supprimer un dossier
    # en dehors de output/train/.
    victime = espace_projets / "output" / "victime"
    victime.mkdir(parents=True)
    create_project_structure("manuscrits")
    pd.DataFrame({"model_name": ["../victime"]}).to_csv(
        espace_projets / "projects" / "manuscrits" / "manuscrits.csv", index=False)

    delete_project("manuscrits")

    assert victime.is_dir()