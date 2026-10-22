"""Tests de la génération et de la mise à jour des classes (labels.txt, config Label Studio)."""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from app.src.modules.generate_labels import (
    add_new_labels,
    build_ls_label_config,
    create_labels_file,
    generate_random_colours,
    get_labeling_code,
    get_labels,
)


# --- Outils pour fabriquer un faux projet dans un dossier temporaire ----------

def ecrire_annotation(dossier: Path, nom: str, classes: list[str]) -> None:
    """Écrit un fichier JSON au format d'export Label Studio, une boîte par classe."""
    dossier.mkdir(parents=True, exist_ok=True)
    result = [{"value": {"x": 10, "y": 10, "width": 5, "height": 5,
                         "rectanglelabels": [c]}} for c in classes]
    (dossier / nom).write_text(json.dumps({"result": result}), encoding="utf-8")


def ecrire_labels(fichier: Path, classes: list[str]) -> None:
    """Écrit un labels.txt au format TiamaT : 'id': 'nom' sur chaque ligne."""
    fichier.parent.mkdir(parents=True, exist_ok=True)
    lignes = [f"'{i}': '{c}'\n" for i, c in enumerate(classes)]
    fichier.write_text("".join(lignes), encoding="utf-8")


@pytest.fixture
def project_folder(tmp_path):
    return tmp_path / "projects" / "manuscrits"


@pytest.fixture
def yolo_model_folder(tmp_path):
    dossier = tmp_path / "runs" / "train" / "modele_v1"
    ecrire_labels(dossier / "labels.txt", ["miniature", "initiale"])
    return dossier


# --- get_labels ----------------------------------------------------------------

def test_get_labels_lit_le_fichier(tmp_path):
    fichier = tmp_path / "labels.txt"
    ecrire_labels(fichier, ["miniature", "initiale", "bordure"])
    assert get_labels(fichier) == {"0": "miniature", "1": "initiale", "2": "bordure"}


def test_get_labels_accents(tmp_path):
    fichier = tmp_path / "labels.txt"
    ecrire_labels(fichier, ["lettrine ornée", "décor"])
    assert get_labels(fichier) == {"0": "lettrine ornée", "1": "décor"}


# --- create_labels_file -------------------------------------------------------

def test_create_labels_file_classes_uniques(tmp_path, project_folder):
    gt = project_folder / "annotations" / "ground_truth"
    ecrire_annotation(gt, "img1.json", ["miniature", "initiale"])
    ecrire_annotation(gt, "img2.json", ["initiale", "bordure"])

    create_labels_file(project_folder)

    labels = get_labels(tmp_path / "data" / "manuscrits" / "labels.txt")
    assert sorted(labels.values()) == ["bordure", "initiale", "miniature"]
    assert sorted(labels.keys()) == ["0", "1", "2"]


def test_create_labels_file_ignore_fichiers_caches(tmp_path, project_folder):
    # macOS crée des fichiers .DS_Store : ils ne doivent pas être lus comme des annotations.
    gt = project_folder / "annotations" / "ground_truth"
    ecrire_annotation(gt, "img1.json", ["miniature"])
    (gt / ".DS_Store").write_text("pas du json")

    create_labels_file(project_folder)  # ne doit pas planter


def test_create_labels_file_ordre_stable(tmp_path, project_folder):
    # Les identifiants doivent être reproductibles : relancer la fonction sur les
    # mêmes annotations doit donner exactement le même labels.txt. On impose
    # l'ordre alphabétique, le plus simple à garantir.
    gt = project_folder / "annotations" / "ground_truth"
    ecrire_annotation(gt, "img1.json", ["miniature", "initiale", "bordure", "texte", "rubrique", "decor"])

    create_labels_file(project_folder)

    labels = get_labels(tmp_path / "data" / "manuscrits" / "labels.txt")
    assert list(labels.values()) == sorted(labels.values())


# --- add_new_labels -------------------------------------------------------------

def test_add_new_labels_sans_nouvelle_classe(tmp_path, project_folder, yolo_model_folder):
    corrections = project_folder / "annotations" / "prediction_corrections"
    ecrire_annotation(corrections, "img1.json", ["initiale"])

    add_new_labels(project_folder, yolo_model_folder)

    resultat = tmp_path / "runs" / "predict" / "manuscrits_modele_v1" / "labels.txt"
    assert get_labels(resultat) == {"0": "miniature", "1": "initiale"}


def test_add_new_labels_ajoute_les_nouvelles_classes_a_la_suite(tmp_path, project_folder, yolo_model_folder):
    corrections = project_folder / "annotations" / "prediction_corrections"
    ecrire_annotation(corrections, "img1.json", ["miniature", "bordure"])
    ecrire_annotation(corrections, "img2.json", ["texte"])

    add_new_labels(project_folder, yolo_model_folder)

    labels = get_labels(tmp_path / "runs" / "predict" / "manuscrits_modele_v1" / "labels.txt")
    # Les classes existantes gardent leur identifiant : le modèle les connaît sous ces numéros.
    assert labels["0"] == "miniature"
    assert labels["1"] == "initiale"
    # Les nouvelles prennent les identifiants suivants.
    assert {labels["2"], labels["3"]} == {"bordure", "texte"}


def test_add_new_labels_ne_modifie_pas_le_fichier_du_modele(project_folder, yolo_model_folder):
    avant = (yolo_model_folder / "labels.txt").read_text(encoding="utf-8")
    corrections = project_folder / "annotations" / "prediction_corrections"
    ecrire_annotation(corrections, "img1.json", ["bordure"])

    add_new_labels(project_folder, yolo_model_folder)

    assert (yolo_model_folder / "labels.txt").read_text(encoding="utf-8") == avant


# --- generate_random_colours ---------------------------------------------------

def test_generate_random_colours_format_hexadecimal():
    for _ in range(100):
        assert re.fullmatch(r"#[0-9a-f]{6}", generate_random_colours())


# --- build_ls_label_config ----------------------------------------------------

def test_build_ls_label_config_xml_valide():
    xml = build_ls_label_config(["miniature", "initiale"])
    racine = ET.fromstring(xml)  # échoue si le XML est mal formé
    valeurs = [label.get("value") for label in racine.iter("Label")]
    assert valeurs == ["miniature", "initiale"]


def test_build_ls_label_config_accepte_un_dictionnaire():
    xml = build_ls_label_config({"0": "miniature", "1": "initiale"})
    valeurs = [label.get("value") for label in ET.fromstring(xml).iter("Label")]
    assert valeurs == ["miniature", "initiale"]


def test_build_ls_label_config_caracteres_speciaux():
    # Un nom de classe avec &, < ou " ne doit pas casser le XML envoyé à Label Studio.
    xml = build_ls_label_config(["texte & image", 'titre "rubriqué"'])
    valeurs = [label.get("value") for label in ET.fromstring(xml).iter("Label")]
    assert valeurs == ["texte & image", 'titre "rubriqué"']


# --- get_labeling_code -------------------------------------------------------------

def test_get_labeling_code_ecrit_le_fichier(project_folder, yolo_model_folder):
    template, fichier = get_labeling_code(project_folder, yolo_model_folder)
    assert fichier.name == "manuscrits_labeling_code.txt"
    assert fichier.read_text(encoding="utf-8") == template