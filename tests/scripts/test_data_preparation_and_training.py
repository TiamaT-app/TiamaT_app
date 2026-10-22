"""Tests de la préparation des données et de l'enchaînement d'un entraînement.

L'entraînement YOLO lui-même (yolo_training) est remplacé par un mock :
on teste tout ce qui se passe autour, sans entraîner de vrai modèle.
"""

import os

import cv2
import numpy as np
import pytest
import yaml

import app.src.scripts.data_preparation_and_training as dpt
from app.services.project_service import create_project_structure


@pytest.fixture
def projet(espace_projets):
    """Un projet avec son dossier data/ : labels.txt, images et annotations YOLO."""
    projet = create_project_structure("manuscrits")
    data = espace_projets / "data" / "manuscrits"
    (data / "images").mkdir(parents=True)
    (data / "labels").mkdir()
    (data / "labels.txt").write_text("'0': 'miniature'\n'1': 'lettrine d'or'\n", encoding="utf-8")
    return projet


def data(espace_projets):
    return espace_projets / "data" / "manuscrits"


def ajouter_image(espace_projets, nom, boites=("0 0.5 0.5 0.2 0.4",), taille=(200, 100)):
    """Ajoute une image (largeur x hauteur) et son annotation YOLO dans data/."""
    largeur, hauteur = taille
    cv2.imwrite(str(data(espace_projets) / "images" / f"{nom}.jpg"), np.full((hauteur, largeur, 3), 255, np.uint8))
    if boites is not None:
        (data(espace_projets) / "labels" / f"{nom}.txt").write_text("\n".join(boites) + "\n", encoding="utf-8")


def noms(dossier):
    return sorted(p.name for p in dossier.iterdir()) if dossier.exists() else []


# --- clean_comma -------------------------------------------------------------------------

def test_clean_comma(espace_projets, projet):
    (data(espace_projets) / "labels" / "f1.txt").write_text("0, 0.5, 0.5, 0.2, 0.4\n", encoding="utf-8")
    dpt.clean_comma(projet)
    assert (data(espace_projets) / "labels" / "f1.txt").read_text() == "0 0.5 0.5 0.2 0.4\n"


# --- Augmentation de données ---------------------------------------------------------------

def test_augmentation_conserve_la_position_relative_des_boites(espace_projets, projet):
    # La transformation redimensionne l'image : la boîte doit rester au même endroit
    # en coordonnées relatives (à un pixel près).
    ajouter_image(espace_projets, "f1", ["1 0.5 0.5 0.2 0.4"])

    dpt.generate_transformed_data(projet)

    image_pt = cv2.imread(str(data(espace_projets) / "images" / "f1_PT.jpg"))
    hauteur, largeur = image_pt.shape[:2]
    assert 60 <= largeur <= 160 and 30 <= hauteur <= 80  # entre 30 % et 80 % de 200 x 100
    classe, *coordonnees = (data(espace_projets) / "labels" / "f1_PT.txt").read_text().split()
    assert classe == "1"
    assert [float(v) for v in coordonnees] == pytest.approx([0.5, 0.5, 0.2, 0.4], abs=2 / min(largeur, hauteur))


def test_augmentation_ignore_les_images_sans_annotation(espace_projets, projet):
    ajouter_image(espace_projets, "f1", boites=None)
    dpt.generate_transformed_data(projet)
    assert noms(data(espace_projets) / "images") == ["f1.jpg"]


def test_augmentation_ne_transforme_pas_les_copies_d_un_cycle_precedent(espace_projets, projet):
    # Sans cette précaution, chaque cycle créerait f1_PT_PT, puis f1_PT_PT_PT…
    # et le dataset doublerait à chaque entraînement.
    ajouter_image(espace_projets, "f1")
    dpt.generate_transformed_data(projet)
    dpt.generate_transformed_data(projet)
    assert noms(data(espace_projets) / "images") == ["f1.jpg", "f1_PT.jpg"]


# --- Répartition entraînement / validation -------------------------------------------------------

def test_repartition_80_20(espace_projets, projet):
    for i in range(10):
        ajouter_image(espace_projets, f"f{i}")

    dpt.create_training_dataset(projet, None, preexisting_distribution=False)

    datasets = espace_projets / "data" / "datasets" / "manuscrits"
    assert len(noms(datasets / "images" / "train")) == 8
    assert len(noms(datasets / "images" / "val")) == 2
    # Chaque image est accompagnée de son annotation
    assert len(noms(datasets / "labels" / "train")) == 8
    assert noms(data(espace_projets) / "images") == []  # les images ont été déplacées


def test_la_validation_ne_contient_que_des_images_d_origine(espace_projets, projet):
    # Sinon le modèle serait validé sur des quasi-copies de ses images d'entraînement,
    # et les métriques de validation seraient surestimées.
    for i in range(10):
        ajouter_image(espace_projets, f"f{i}")
    dpt.generate_transformed_data(projet)

    dpt.create_training_dataset(projet, None, preexisting_distribution=False)

    datasets = espace_projets / "data" / "datasets" / "manuscrits"
    val = noms(datasets / "images" / "val")
    train = noms(datasets / "images" / "train")
    assert len(val) == 2 and not any("_PT" in nom for nom in val)
    assert len(train) == 16  # 8 originaux + leurs 8 copies
    # Une image de validation n'a pas de copie en entraînement
    assert not any(nom.replace(".jpg", "_PT.jpg") in train for nom in val)
    # Les copies des images de validation sont supprimées, avec leur annotation
    assert noms(data(espace_projets) / "images") == []
    assert noms(data(espace_projets) / "labels") == []


def test_deux_images_une_dans_chaque_ensemble(espace_projets, projet):
    ajouter_image(espace_projets, "f1")
    ajouter_image(espace_projets, "f2")

    dpt.create_training_dataset(projet, None, preexisting_distribution=False)

    datasets = espace_projets / "data" / "datasets" / "manuscrits" / "images"
    assert len(noms(datasets / "train")) == 1
    assert len(noms(datasets / "val")) == 1


def test_une_seule_image_erreur_claire(espace_projets, projet):
    ajouter_image(espace_projets, "f1")
    with pytest.raises(ValueError, match="Au moins 2 images"):
        dpt.create_training_dataset(projet, None, preexisting_distribution=False)


def test_fichier_yaml_valide_avec_apostrophe(espace_projets, projet):
    dpt.write_yaml_file(projet)

    contenu = yaml.safe_load((espace_projets / "data" / "datasets" / "manuscrits" / "manuscrits.yaml").read_text(encoding="utf-8"))
    assert contenu["names"] == {0: "miniature", 1: "lettrine d'or"}
    assert contenu["train"] == "images/train"
    assert contenu["path"] == str(espace_projets / "data" / "datasets" / "manuscrits")


# --- Enchaînement complet ----------------------------------------------------------------------

@pytest.fixture
def faux_yolo_training(mocker, espace_projets):
    """Remplace l'entraînement YOLO : crée seulement le dossier du modèle, comme le ferait YOLO."""
    def entrainer(project_folder, use_model, img_size, epochs, batch, workers, dropout, model_name, pretrained_model):
        (espace_projets / "output" / "train" / model_name / "weights").mkdir(parents=True)
        return model_name
    return mocker.patch.object(dpt, "yolo_training", side_effect=entrainer)


@pytest.fixture
def etat_entrainement(monkeypatch):
    monkeypatch.setenv("complete", "False")


def test_entrainement_complet(espace_projets, projet, faux_yolo_training, etat_entrainement):
    for i in range(5):
        ajouter_image(espace_projets, f"f{i}")

    dpt.training(projet, 10, 0.1, "yolo26s.pt", "manuscrits_20260301_101500")

    assert os.environ["complete"] == "True"
    modele = espace_projets / "output" / "train" / "manuscrits_20260301_101500"
    assert noms(modele) == ["dataset_statistics", "labels.txt", "manuscrits.yaml", "weights"]
    # Les images sont de retour dans data/, prêtes pour le cycle suivant :
    # les 5 originaux, et les copies augmentées des 4 images d'entraînement
    assert len(noms(data(espace_projets) / "images")) == 9
    assert not (espace_projets / "data" / "datasets" / "manuscrits").exists()
    assert faux_yolo_training.call_args.kwargs["epochs"] == 10


def test_entrainement_en_echec_remet_les_donnees_en_place(espace_projets, projet, faux_yolo_training, etat_entrainement):
    # Si YOLO plante, les images doivent revenir dans data/ : sinon le prochain
    # essai échouerait avec « No images found », et la page d'attente tournerait sans fin.
    for i in range(5):
        ajouter_image(espace_projets, f"f{i}")
    faux_yolo_training.side_effect = RuntimeError("CUDA out of memory")

    dpt.training(projet, 10, 0.1, "yolo26s.pt", "manuscrits_20260301_101500")

    assert os.environ["complete"] == "Error"
    # Les 5 originaux et les copies des 4 images d'entraînement sont revenus dans data/
    assert len(noms(data(espace_projets) / "images")) == 9
    assert not (espace_projets / "data" / "datasets" / "manuscrits").exists()


def test_deux_cycles_d_entrainement(espace_projets, projet, faux_yolo_training, etat_entrainement):
    for i in range(5):
        ajouter_image(espace_projets, f"f{i}")

    dpt.training(projet, 10, 0.1, "yolo26s.pt", "manuscrits_20260301_101500")
    dpt.training(projet, 10, 0.1, "yolo26s.pt", "manuscrits_20260415_143000")

    assert os.environ["complete"] == "True"
    images = noms(data(espace_projets) / "images")
    assert not any("_PT_PT" in nom for nom in images)
    assert len([nom for nom in images if "_PT" not in nom]) == 5