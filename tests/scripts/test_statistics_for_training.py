"""Tests des statistiques calculées sur le dataset d'un projet (data/images et data/labels)."""

import pandas as pd
import pytest

import app.src.scripts.statistics_for_training as stats
from app.services.project_service import create_project_structure


@pytest.fixture
def projet(espace_projets):
    """Projet avec un dossier data/ contenant labels.txt, images/ et labels/."""
    projet = create_project_structure("manuscrits")
    data = espace_projets / "data" / "manuscrits"
    (data / "images").mkdir(parents=True)
    (data / "labels").mkdir()
    (data / "labels.txt").write_text("'0': 'miniature'\n'1': 'initiale'\n'2': 'bordure'\n", encoding="utf-8")
    return projet


def data(espace_projets):
    return espace_projets / "data" / "manuscrits"


def ajouter(espace_projets, nom, lignes=None, extension=".jpg"):
    """Ajoute une image et, si lignes n'est pas None, son fichier d'annotation YOLO."""
    (data(espace_projets) / "images" / f"{nom}{extension}").write_bytes(b"")
    if lignes is not None:
        (data(espace_projets) / "labels" / f"{nom}.txt").write_text(
            "".join(f"{ligne}\n" for ligne in lignes), encoding="utf-8")


def lire_csv(espace_projets, nom):
    return pd.read_csv(data(espace_projets) / "dataset_statistics" / nom, sep=";")


# --- Fichiers pris en compte ------------------------------------------------------------------

def test_get_annotation_files(espace_projets, projet):
    ajouter(espace_projets, "f1", ["0 0.5 0.5 0.1 0.1"])
    ajouter(espace_projets, "f2", ["1 0.5 0.5 0.1 0.1"], extension=".tif")
    ajouter(espace_projets, "f3")                                   # image sans annotation
    ajouter(espace_projets, "f1_PT", ["0 0.5 0.5 0.1 0.1"])        # copie de l'augmentation
    (data(espace_projets) / "labels" / "orpheline.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    fichiers = stats.get_annotation_files(data(espace_projets) / "images", data(espace_projets) / "labels")

    assert sorted(f.name for f in fichiers) == ["f1.txt", "f2.txt"]


def test_encoding_signale_un_fichier_latin1(espace_projets, projet, capsys):
    ajouter(espace_projets, "f1")
    (data(espace_projets) / "labels" / "f1.txt").write_bytes("0 é\n".encode("latin-1"))

    stats.encoding(projet)  # plantait avant : get_annotation_files renvoyait du texte, sans .name

    assert "f1.txt is encoded in ISO-8859-1" in capsys.readouterr().out


# --- Comptages -------------------------------------------------------------------------------------

def test_annotations_par_image(espace_projets, projet):
    ajouter(espace_projets, "f1", ["0 0.5 0.5 0.1 0.1", "", "1 0.2 0.2 0.1 0.1"])  # ligne vide ignorée
    ajouter(espace_projets, "f2", ["0 0.5 0.5 0.1 0.1", "0 0.1 0.1 0.1 0.1", "2 0.3 0.3 0.1 0.1"])

    stats.annotations_per_img(projet)

    df = lire_csv(espace_projets, "annotations_per_img.csv")
    assert df["image_name"].tolist() == ["f2.jpg", "f1.jpg"]   # de la plus annotée à la moins annotée
    assert df["annotations_nb"].tolist() == [3, 2]


def test_resultats_globaux(espace_projets, projet):
    ajouter(espace_projets, "f1", ["0 0.5 0.5 0.1 0.1", "1 0.2 0.2 0.1 0.1"])
    ajouter(espace_projets, "f2", [""])     # annotation vide
    ajouter(espace_projets, "f3")           # pas d'annotation

    stats.get_global_results(projet)

    df = lire_csv(espace_projets, "global_data.csv")
    assert dict(zip(df["metric"], df["value"])) == {
        "Number of files without annotations": 2,
        "Total number of annotations": 2,
    }


# --- Distribution des classes ------------------------------------------------------------------

def test_distribution_des_classes(espace_projets, projet):
    ajouter(espace_projets, "f1", ["0 0.5 0.5 0.1 0.1", "1 0.2 0.2 0.1 0.1", ""])
    ajouter(espace_projets, "f2", ["0 0.5 0.5 0.1 0.1"])

    stats.classes_distribution(projet)

    df = lire_csv(espace_projets, "class_distribution.csv")
    # Toutes les classes, dans l'ordre de leur numéro, y compris celles sans annotation
    assert df["class_name"].tolist() == ["miniature", "initiale", "bordure"]
    assert df["nb_occurrences"].tolist() == [2, 1, 0]
    assert (data(espace_projets) / "dataset_statistics" / "class_distribution.png").stat().st_size > 0


def test_distribution_ignore_les_copies_augmentees(espace_projets, projet):
    # Après un entraînement dans le même cycle, les copies _PT sont encore dans data/images.
    ajouter(espace_projets, "f1", ["0 0.5 0.5 0.1 0.1"])
    ajouter(espace_projets, "f1_PT", ["0 0.5 0.5 0.1 0.1"])

    stats.classes_distribution(projet)

    assert lire_csv(espace_projets, "class_distribution.csv")["nb_occurrences"].tolist() == [1, 0, 0]


def test_distribution_numero_absent_de_labels(espace_projets, projet):
    ajouter(espace_projets, "f1", ["7 0.5 0.5 0.1 0.1"])

    stats.classes_distribution(projet)  # plantait avant avec une KeyError

    assert "classe inconnue (7)" in lire_csv(espace_projets, "class_distribution.csv")["class_name"].tolist()


def test_distribution_sans_aucune_annotation(espace_projets, projet):
    (data(espace_projets) / "labels.txt").write_text("", encoding="utf-8")

    stats.classes_distribution(projet)  # plantait avant : max() d'une liste vide

    assert (data(espace_projets) / "dataset_statistics" / "class_distribution.png").exists()


def test_deux_graphiques_successifs_ne_se_superposent_pas(espace_projets, projet):
    # Avec pyplot, un 2e graphique pouvait se dessiner sur les restes du 1er.
    ajouter(espace_projets, "f1", ["0 0.5 0.5 0.1 0.1"])
    stats.classes_distribution(projet)
    premier = (data(espace_projets) / "dataset_statistics" / "class_distribution.png").read_bytes()

    stats.classes_distribution(projet)

    assert (data(espace_projets) / "dataset_statistics" / "class_distribution.png").read_bytes() == premier


# --- clean_LS ----------------------------------------------------------------------------------------

def test_clean_ls_retire_le_prefixe_de_label_studio(espace_projets, projet):
    ajouter(espace_projets, "a1b2c3d4-f1", ["0 0.5 0.5 0.1 0.1"])
    ajouter(espace_projets, "folio_long_sans_prefixe", ["0 0.5 0.5 0.1 0.1"])

    stats.clean_LS(projet, annotated_with_LS=True)

    assert sorted(p.name for p in (data(espace_projets) / "images").iterdir()) == \
        ["f1.jpg", "folio_long_sans_prefixe.jpg"]
    # Chaque annotation garde le nom de sa propre image
    assert sorted(p.name for p in (data(espace_projets) / "labels").iterdir()) == \
        ["f1.txt", "folio_long_sans_prefixe.txt"]


def test_clean_ls_desactive(espace_projets, projet):
    ajouter(espace_projets, "a1b2c3d4-f1")
    stats.clean_LS(projet, annotated_with_LS=False)
    assert [p.name for p in (data(espace_projets) / "images").iterdir()] == ["a1b2c3d4-f1.jpg"]