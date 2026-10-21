"""Tests des prédictions du modèle sur les images de test et de leur préparation pour Label Studio.

YOLO est remplacé par un faux modèle, qui renvoie des détections choisies par chaque test.
"""

import json
import os

import pandas as pd
import pytest
from PIL import Image

import app.src.scripts.predicting_and_checking_yolo_results as pred


class FauxTenseur:
    def __init__(self, valeur):
        self.valeur = valeur

    def item(self):
        return self.valeur

    def squeeze(self):
        return self

    def tolist(self):
        return self.valeur


class FausseBoite:
    """Une détection, avec les mêmes attributs que les boîtes de YOLO."""
    def __init__(self, classe, x, y, w, h, confiance):
        self.cls = FauxTenseur(classe)
        self.xywhn = FauxTenseur([x, y, w, h])
        self.conf = FauxTenseur(confiance)


class FauxResultat:
    def __init__(self, boites):
        self.boxes = boites


@pytest.fixture
def faux_yolo(mocker):
    """YOLO détecte, pour chaque image, les boîtes de detections[nom de l'image] (aucune par défaut)."""
    detections = {}

    def predire(source, **kwargs):
        nom = os.path.splitext(os.path.basename(source))[0]
        return [FauxResultat([FausseBoite(*boite) for boite in detections.get(nom, [])])]

    yolo = mocker.patch.object(pred, "YOLO")
    yolo.return_value.predict.side_effect = predire
    mocker.patch.object(pred, "which_device", return_value="cpu")
    return detections


@pytest.fixture
def etat_test(monkeypatch):
    monkeypatch.setenv("test_complete", "False")


def image_de_test(cycle, nom, taille=(200, 100)):
    Image.new("RGB", taille, "white").save(cycle["eval"] / nom)


# --- Prédictions ----------------------------------------------------------------------------

def test_predictions_enregistrees_au_format_yolo(cycle, faux_yolo, etat_test):
    image_de_test(cycle, "f1.jpg")
    faux_yolo["f1"] = [(1, 0.5, 0.5, 0.2, 0.4, 0.87)]

    pred.process_images_with_yolo(cycle["projet"], cycle["modele"])

    assert os.environ["test_complete"] == "True"
    assert (cycle["resultats"] / "labels" / "f1.txt").read_text() == "1 0.5000 0.5000 0.2000 0.4000 0.8700\n"


def test_le_modele_est_charge_une_seule_fois(cycle, faux_yolo, etat_test):
    for i in range(3):
        image_de_test(cycle, f"f{i}.jpg")
    pred.process_images_with_yolo(cycle["projet"], cycle["modele"])
    assert pred.YOLO.call_count == 1


def test_image_sans_detection_efface_une_ancienne_prediction(cycle, faux_yolo, etat_test):
    # Une prédiction restée d'un test précédent sur une image du même nom ne doit
    # pas être prise pour celle de la nouvelle image.
    image_de_test(cycle, "f1.jpg")
    (cycle["resultats"] / "labels").mkdir(parents=True)
    (cycle["resultats"] / "labels" / "f1.txt").write_text("0 0.5 0.5 0.1 0.1 0.9\n")

    pred.process_images_with_yolo(cycle["projet"], cycle["modele"])

    assert not (cycle["resultats"] / "labels" / "f1.txt").exists()


def test_erreur_pendant_les_predictions(cycle, faux_yolo, etat_test):
    # Sans cette gestion d'erreur, la page d'attente tournerait sans fin.
    image_de_test(cycle, "f1.jpg")
    pred.YOLO.return_value.predict.side_effect = RuntimeError("Fichier de poids invalide")

    pred.process_images_with_yolo(cycle["projet"], cycle["modele"])

    assert os.environ["test_complete"] == "Error"


def test_images_prises_en_compte(cycle, faux_yolo, etat_test):
    image_de_test(cycle, "f1.jpg")
    image_de_test(cycle, "f2.tif")
    (cycle["eval"] / ".DS_Store").write_bytes(b"")
    (cycle["eval"] / "._f3.jpg").write_bytes(b"")   # créé par macOS sur les disques externes
    faux_yolo.update({"f1": [(0, 0.5, 0.5, 0.1, 0.1, 0.9)], "f2": [(0, 0.5, 0.5, 0.1, 0.1, 0.9)]})

    pred.process_images_with_yolo(cycle["projet"], cycle["modele"])

    assert sorted(p.name for p in (cycle["resultats"] / "labels").iterdir()) == ["f1.txt", "f2.txt"]


def test_projet_dans_un_dossier_cache(tmp_path):
    # Seule la partie du chemin sous eval_images compte : un projet rangé dans un
    # dossier caché (~/.local/...) doit fonctionner normalement.
    eval_folder = tmp_path / ".local" / "manuscrits" / "image_inputs" / "eval_images"
    (eval_folder / ".cache").mkdir(parents=True)
    Image.new("RGB", (10, 10)).save(eval_folder / "f1.jpg")
    Image.new("RGB", (10, 10)).save(eval_folder / ".cache" / "f2.jpg")

    assert [p.name for p in pred._eval_images(eval_folder)] == ["f1.jpg"]


# --- Préparation pour Label Studio --------------------------------------------------------------

def test_fichier_d_import_label_studio(cycle):
    image_de_test(cycle, "f1.jpg", (200, 100))
    image_de_test(cycle, "f2.jpg")
    (cycle["resultats"] / "labels").mkdir(parents=True)
    (cycle["resultats"] / "labels" / "f1.txt").write_text("1 0.5 0.5 0.2 0.4 0.87\n")

    fichier = pred.get_ls_for_local_files(cycle["projet"], cycle["modele"])

    taches = {t["data"]["image"]: t for t in json.loads(fichier.read_text(encoding="utf-8"))}
    prefixe = "/data/local-files/?d=projects/manuscrits/image_inputs/eval_images/"
    assert set(taches) == {prefixe + "f1.jpg", prefixe + "f2.jpg"}

    boite = taches[prefixe + "f1.jpg"]["predictions"][0]["result"][0]
    # Label Studio attend le coin haut-gauche, en pourcentage de l'image
    assert boite["value"]["x"] == pytest.approx(40)
    assert boite["value"]["y"] == pytest.approx(30)
    assert boite["value"]["width"] == pytest.approx(20)
    assert boite["value"]["rectanglelabels"] == ["initiale"]
    assert boite["score"] == pytest.approx(0.87)
    # Une image sans détection est importée sans boîte, pour être annotée à la main
    assert taches[prefixe + "f2.jpg"]["predictions"][0]["result"] == []


def test_csv_des_predictions(cycle):
    image_de_test(cycle, "f1.jpg", (200, 100))
    image_de_test(cycle, "f2.jpg")
    (cycle["resultats"] / "labels").mkdir(parents=True)
    (cycle["resultats"] / "labels" / "f1.txt").write_text("0 0.5 0.5 0.2 0.4 0.9\n1 0.25 0.25 0.1 0.1 0.6\n")

    pred.yolo_to_csv(cycle["projet"], cycle["modele"])

    df = pd.read_csv(cycle["resultats"] / "results" / "manuscrits.csv", sep=";")
    assert len(df) == 3  # deux détections sur f1, une ligne vide pour f2
    lignes_f1 = df[df["Image_Path"].str.endswith("f1.jpg")]
    assert lignes_f1["Class_Name"].tolist() == ["miniature", "initiale"]
    assert lignes_f1["Absolute_coordinates"].tolist()[0] == "80 30 40 40"


def test_csv_des_metadonnees_porte_le_nom_du_projet(cycle):
    # add_csv_data cherche ce fichier sous le nom <projet>.csv
    image_de_test(cycle, "f1.jpg", (200, 100))
    pred.get_image_data(cycle["projet"])
    df = pd.read_csv(cycle["eval"] / "manuscrits.csv", sep=";")
    assert df[["Image_name", "Width", "Height"]].values.tolist() == [["f1", 200, 100]]