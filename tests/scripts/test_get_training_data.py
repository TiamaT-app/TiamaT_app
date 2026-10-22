"""Tests de la préparation du dataset d'entraînement à partir des annotations Label Studio."""

import json

import pandas as pd
import pytest
from PIL import Image

from app.services.project_service import create_project_structure
from app.src.modules.generate_labels import get_labels
from app.src.scripts.get_training_data import (
    clean_image_name,
    create_annotations_file,
    create_csv_file,
    create_dataset,
    get_img_training_data,
    normalized_image_name,
)


@pytest.fixture
def projet(espace_projets):
    """Chemin absolu d'un projet vide, comme le reçoivent les scripts."""
    return create_project_structure("manuscrits")


def images(projet):
    return projet / "image_inputs" / "ground_truth_images"


def annotations(projet):
    return projet / "annotations" / "ground_truth"


def creer_image(chemin, taille=(40, 20)):
    Image.new("RGB", taille, "white").save(chemin)


def creer_annotation(projet, numero, chemin_image, boites):
    """Écrit une annotation au format d'export Label Studio : boites = [(classe, x, y, w, h), ...]."""
    resultat = [{"value": {"x": x, "y": y, "width": w, "height": h, "rectanglelabels": [classe]}}
                for classe, x, y, w, h in boites]
    contenu = {"id": numero, "result": resultat, "task": {"data": {"image": chemin_image}}}
    (annotations(projet) / f"{numero}.json").write_text(json.dumps(contenu), encoding="utf-8")


def chemin_ls(nom_image):
    """Chemin d'image tel que Label Studio l'enregistre pour les fichiers locaux."""
    return f"/data/local-files/?d=manuscrits/image_inputs/ground_truth_images/{nom_image}"


# --- Noms des images ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "nom, attendu",
    [
        ("épître_01.jpg", "epitre_01.jpg"),
        ("Œuvre complète.PNG", "oeuvre_complete.png"),
        ("Folio 12-recto.jpeg", "folio_12_recto.jpeg"),
        ("Ελληνικά.jpg", "ellenika.jpg"),
        ("../../evasion.jpg", "evasion.jpg"),
        ("★★.jpg", "image.jpg"),
    ],
    ids=["accents", "ligature_majuscules", "espaces_tiret", "grec", "remontee", "symboles_seuls"],
)
def test_normalized_image_name(nom, attendu):
    assert normalized_image_name(nom) == attendu


def test_clean_image_name_renomme_les_images(projet):
    (images(projet) / "Épître 01.JPG").write_bytes(b"")
    clean_image_name(projet)
    assert [p.name for p in images(projet).iterdir()] == ["epitre_01.jpg"]


def test_clean_image_name_changement_de_casse_seulement(projet):
    # Sur Mac, F1.jpg et f1.jpg désignent le même fichier : le renommage
    # ne doit pas être pris pour un conflit avec un autre fichier.
    (images(projet) / "F1.jpg").write_bytes(b"")
    clean_image_name(projet)
    assert [p.name for p in images(projet).iterdir()] == ["f1.jpg"]


def test_clean_image_name_n_ecrase_pas_une_autre_image(projet):
    (images(projet) / "folio_e.jpg").write_bytes(b"premiere")
    (images(projet) / "folio_é.jpg").write_bytes(b"seconde")

    clean_image_name(projet)

    assert (images(projet) / "folio_e.jpg").read_bytes() == b"premiere"
    assert (images(projet) / "folio_é.jpg").read_bytes() == b"seconde"


def test_clean_image_name_ignore_les_fichiers_caches(projet):
    (images(projet) / ".DS_Store").write_bytes(b"")
    clean_image_name(projet)
    assert (images(projet) / ".DS_Store").exists()


def test_clean_image_name_sans_effet_sur_les_noms_deja_normalises(projet):
    # Les images importées par l'app ont déjà un nom normalisé : rien à renommer.
    nom = normalized_image_name("Épître 01.JPG")
    (images(projet) / nom).write_bytes(b"")
    clean_image_name(projet)
    assert [p.name for p in images(projet).iterdir()] == [nom]


# --- create_csv_file -------------------------------------------------------------------------

def test_create_csv_file(projet):
    creer_image(images(projet) / "f1.jpg", (40, 20))
    creer_image(images(projet) / "f2.tif", (10, 10))
    (images(projet) / "._f1.jpg").write_bytes(b"pas une image")  # créé par macOS sur les disques externes

    create_csv_file(projet)

    df = pd.read_csv(images(projet) / "manuscrits_data.csv", sep=";").sort_values("Image_name")
    assert df["Image_name"].tolist() == ["f1", "f2"]
    assert df["Width"].tolist() == [40, 10]
    assert df["Image_size"].tolist() == [800, 100]


# --- create_annotations_file --------------------------------------------------------------------

def ecrire_labels(espace_projets, classes):
    data = espace_projets / "data" / "manuscrits"
    data.mkdir(parents=True, exist_ok=True)
    (data / "labels.txt").write_text("".join(f"'{i}': '{c}'\n" for i, c in enumerate(classes)),
                                     encoding="utf-8")
    return data


def test_create_annotations_file_convertit_au_format_yolo(espace_projets, projet):
    data = ecrire_labels(espace_projets, ["initiale", "miniature"])
    creer_annotation(projet, 1, chemin_ls("f1.jpg"), [("miniature", 10, 20, 30, 40)])

    create_annotations_file(projet)

    ligne = (data / "labels" / "f1.txt").read_text().split()
    assert ligne[0] == "1"
    assert [float(v) for v in ligne[1:]] == pytest.approx([0.25, 0.4, 0.3, 0.4])


@pytest.mark.parametrize(
    "chemin_image",
    [
        "/data/local-files/?d=manuscrits%5Cimage_inputs%5Cground_truth_images%5Cf1.jpg",
        "/data/local-files/?d=manuscrits/image_inputs/ground_truth_images/f%201.jpg",
    ],
    ids=["chemin_windows", "espace_encode"],
)
def test_create_annotations_file_chemins_encodes(espace_projets, projet, chemin_image):
    # Label Studio encode les caractères spéciaux du chemin (%5C pour \\ sous Windows, %20 pour un espace).
    data = ecrire_labels(espace_projets, ["miniature"])
    creer_annotation(projet, 1, chemin_image, [("miniature", 10, 20, 30, 40)])

    create_annotations_file(projet)

    assert [p.name for p in (data / "labels").iterdir()] in (["f1.txt"], ["f 1.txt"])


def test_create_annotations_file_classe_absente_de_labels(espace_projets, projet):
    # Une classe absente de labels.txt ne doit jamais finir en « unknown-class »
    # dans un fichier YOLO : l'erreur doit être claire et nommer la classe.
    ecrire_labels(espace_projets, ["miniature"])
    creer_annotation(projet, 1, chemin_ls("f1.jpg"), [("enluminure", 10, 20, 30, 40)])

    with pytest.raises(ValueError, match="enluminure"):
        create_annotations_file(projet)


# --- get_img_training_data ---------------------------------------------------------------------

def test_get_img_training_data_copie_les_images(espace_projets, projet):
    for nom in ["f1.jpg", "f2.tif", "notes.txt", ".DS_Store"]:
        (images(projet) / nom).write_bytes(b"")

    get_img_training_data(projet)

    copiees = sorted(p.name for p in (espace_projets / "data" / "manuscrits" / "images").iterdir())
    assert copiees == ["f1.jpg", "f2.tif"]


# --- create_dataset : la chaîne complète ----------------------------------------------------------

def test_create_dataset_chaine_complete(espace_projets, projet):
    creer_image(images(projet) / "f1.jpg")
    creer_image(images(projet) / "f2.jpg")
    creer_annotation(projet, 1, chemin_ls("f1.jpg"), [("miniature", 10, 20, 30, 40)])
    creer_annotation(projet, 2, chemin_ls("f2.jpg"), [("initiale", 0, 0, 50, 50)])

    create_dataset(projet, manually_downloaded=False)

    data = espace_projets / "data" / "manuscrits"
    assert get_labels(data / "labels.txt") == {"0": "initiale", "1": "miniature"}
    assert sorted(p.name for p in (data / "labels").iterdir()) == ["f1.txt", "f2.txt"]
    assert sorted(p.name for p in (data / "images").iterdir()) == ["f1.jpg", "f2.jpg"]


def test_create_dataset_conserve_les_identifiants_existants(espace_projets, projet):
    # Au cycle suivant, labels.txt existe déjà : les classes gardent leur numéro.
    ecrire_labels(espace_projets, ["miniature", "initiale"])
    creer_image(images(projet) / "f1.jpg")
    creer_annotation(projet, 1, chemin_ls("f1.jpg"), [("initiale", 10, 20, 30, 40)])

    create_dataset(projet, manually_downloaded=False)

    data = espace_projets / "data" / "manuscrits"
    assert get_labels(data / "labels.txt") == {"0": "miniature", "1": "initiale"}
    assert (data / "labels" / "f1.txt").read_text().startswith("1 ")


def test_create_dataset_nouvelle_classe_au_cycle_suivant(espace_projets, projet):
    # Une classe ajoutée dans Label Studio après le premier cycle est ajoutée
    # à la suite dans labels.txt, sans changer le numéro des classes existantes.
    ecrire_labels(espace_projets, ["miniature", "initiale"])
    creer_image(images(projet) / "f1.jpg")
    creer_annotation(projet, 1, chemin_ls("f1.jpg"), [("enluminure", 10, 20, 30, 40)])

    create_dataset(projet, manually_downloaded=False)

    data = espace_projets / "data" / "manuscrits"
    assert get_labels(data / "labels.txt") == {"0": "miniature", "1": "initiale", "2": "enluminure"}
    assert (data / "labels" / "f1.txt").read_text().startswith("2 ")