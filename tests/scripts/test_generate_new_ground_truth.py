"""Tests de l'intégration des images corrigées dans la vérité terrain du projet (fin de cycle)."""

import json

import pandas as pd
from PIL import Image

import app.src.scripts.generate_new_ground_truth as gng


def correction(cycle, nom_fichier, image, projet_ls=5):
    (cycle["corrections"] / nom_fichier).write_text(json.dumps({
        "id": 999, "project": projet_ls, "result": [],
        "task": {"project": projet_ls,
                 "data": {"image": f"/data/local-files/?d=projects/manuscrits/image_inputs/eval_images/{image}"}}}))


def annotation_gt(cycle, nom_fichier, projet_ls=3):
    (cycle["gt"] / nom_fichier).write_text(json.dumps({
        "id": 1, "project": projet_ls, "result": [],
        "task": {"project": projet_ls, "data": {"image": "ground_truth_images/f0.jpg"}}}))


def lire(chemin):
    return json.loads(chemin.read_text())


# --- move_correction_files_and_images -------------------------------------------------------------

def test_corrections_et_images_rejoignent_la_verite_terrain(cycle):
    annotation_gt(cycle, "1.json")
    (cycle["eval"] / "f1.jpg").write_bytes(b"image")
    correction(cycle, "1.json", "f1.jpg")

    gng.move_correction_files_and_images(cycle["projet"])

    assert (cycle["gt_images"] / "f1.jpg").read_bytes() == b"image"
    assert list(cycle["eval"].iterdir()) == []
    # Numéro 1 déjà pris : la correction devient 2.json, en gardant son extension
    assert sorted(p.name for p in cycle["gt"].iterdir()) == ["1.json", "2.json"]
    deplacee = lire(cycle["gt"] / "2.json")
    assert deplacee["id"] == 2
    assert deplacee["project"] == 3                     # projet Label Studio de la vérité terrain
    assert "ground_truth_images/f1.jpg" in deplacee["task"]["data"]["image"]


def test_corrections_sans_extension(cycle):
    # Si les fichiers n'ont pas d'extension, ils restent sans extension.
    annotation_gt(cycle, "4")
    correction(cycle, "4", "f1.jpg")
    gng.move_correction_files_and_images(cycle["projet"])
    assert sorted(p.name for p in cycle["gt"].iterdir()) == ["4", "5"]


def test_une_image_de_la_verite_terrain_n_est_jamais_ecrasee(cycle):
    # Une image de test portant le nom d'une image déjà annotée est renommée,
    # et sa correction pointe vers le nouveau nom.
    (cycle["gt_images"] / "f1.jpg").write_bytes(b"image-annotee")
    annotation_gt(cycle, "1.json")
    (cycle["eval"] / "f1.jpg").write_bytes(b"image-de-test")
    correction(cycle, "2.json", "f1.jpg")

    gng.move_correction_files_and_images(cycle["projet"])

    assert (cycle["gt_images"] / "f1.jpg").read_bytes() == b"image-annotee"
    assert (cycle["gt_images"] / "f1_1.jpg").read_bytes() == b"image-de-test"
    assert lire(cycle["gt"] / "2.json")["task"]["data"]["image"].endswith("ground_truth_images/f1_1.jpg")


def test_meme_image_deja_dans_la_verite_terrain(cycle):
    (cycle["gt_images"] / "f1.jpg").write_bytes(b"image")
    (cycle["eval"] / "f1.jpg").write_bytes(b"image")
    gng.move_correction_files_and_images(cycle["projet"])
    assert [p.name for p in cycle["gt_images"].iterdir()] == ["f1.jpg"]


# --- create_new_ground_truth et clean_data ---------------------------------------------------------

def test_create_new_ground_truth(cycle):
    corrigees = cycle["resultats"] / "correctedLabels"
    corrigees.mkdir(parents=True)
    (corrigees / "f1.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (cycle["resultats"] / "labels.txt").write_text("'0': 'miniature'\n'1': 'initiale'\n'2': 'bordure'\n")
    (cycle["eval"] / "f1.jpg").write_bytes(b"image")
    (cycle["eval"] / "._f1.jpg").write_bytes(b"")

    gng.create_new_ground_truth(cycle["projet"], cycle["modele"], create_groundtruth=True)

    assert (cycle["data"] / "labels" / "f1.txt").exists()
    assert [p.name for p in (cycle["data"] / "images").iterdir()] == ["f1.jpg"]
    assert "bordure" in (cycle["data"] / "labels.txt").read_text()


def test_clean_data_supprime_uniquement_les_copies_augmentees(cycle):
    for dossier, extension in [("images", ".jpg"), ("labels", ".txt")]:
        (cycle["data"] / dossier).mkdir(parents=True)
        for nom in ["f1", "f1_PT"]:
            (cycle["data"] / dossier / f"{nom}{extension}").write_bytes(b"")

    gng.clean_data(cycle["projet"])

    assert [p.name for p in (cycle["data"] / "images").iterdir()] == ["f1.jpg"]
    assert [p.name for p in (cycle["data"] / "labels").iterdir()] == ["f1.txt"]


def test_clean_data_sans_dossier_data(cycle):
    gng.clean_data(cycle["projet"])  # ne doit pas planter


# --- add_csv_data -------------------------------------------------------------------------------------

def test_add_csv_data_ignore_les_fichiers_caches(cycle):
    Image.new("RGB", (40, 20)).save(cycle["gt_images"] / "f1.jpg")
    (cycle["gt_images"] / "._f1.jpg").write_bytes(b"pas une image")  # plantait avant

    gng.add_csv_data(cycle["projet"], cycle["modele"])

    df = pd.read_csv(cycle["gt_images"] / "manuscrits_data.csv", sep=";")
    assert df["Image_name"].tolist() == ["f1"]