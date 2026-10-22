"""Tests des routes du test d'un modèle : import des images, prédictions, correction, évaluation, fin de cycle.

Les scripts appelés par les routes ont leurs propres tests (tests/scripts/) : ici, ils sont
remplacés par des mocks, pour vérifier uniquement l'enchaînement et les cas d'erreur.
"""

import io
import json
import os

import pandas as pd
import pytest

import app.routes.testing_routes as testing_routes
from app.services.project_service import create_project_structure

MODELE = "manuscrits_20260301_101500"


@pytest.fixture
def projet(espace_projets, monkeypatch):
    """Projet « manuscrits » sélectionné, avec un modèle entraîné comme dernier modèle."""
    create_project_structure("manuscrits")
    modele = espace_projets / "output" / "train" / MODELE
    (modele / "weights").mkdir(parents=True)
    (modele / "weights" / "best.pt").write_bytes(b"")
    pd.DataFrame({"model_name": [MODELE]}).to_csv(
        espace_projets / "projects" / "manuscrits" / "manuscrits.csv", index=False)
    (espace_projets / "config.json").write_text(
        json.dumps({"CURRENT_PROJECT_NAME": "manuscrits", "LAST_MODEL_PATH": str(modele)}))
    monkeypatch.delenv("test_complete", raising=False)
    return espace_projets


@pytest.fixture
def faux_thread(mocker):
    return mocker.patch.object(testing_routes, "Thread")


def images_de_test(*noms):
    return [(io.BytesIO(f"contenu-{nom}".encode()), nom) for nom in noms]


def dossier_eval(racine):
    return racine / "projects" / "manuscrits" / "image_inputs" / "eval_images"


# --- Import des images de test et prédictions ----------------------------------------------------

def test_test_du_dernier_modele(client, projet, faux_thread):
    reponse = client.post("/test_modele", content_type="multipart/form-data",
                          data={"fichiers": images_de_test("Épître 01.jpg")})

    assert reponse.status_code == 200
    # Même règle de nommage que pour les images de vérité terrain
    assert [p.name for p in dossier_eval(projet).iterdir()] == ["epitre_01.jpg"]
    args = faux_thread.call_args.kwargs["args"]
    assert args == (projet / "projects" / "manuscrits", projet / "output" / "train" / MODELE)
    faux_thread.return_value.start.assert_called_once()
    assert os.environ["test_complete"] == "False"


def test_test_sans_image(client, projet, faux_thread):
    # Avant, les prédictions démarraient même si le formulaire était invalide.
    reponse = client.post("/test_modele", data={})
    assert reponse.status_code == 302
    faux_thread.assert_not_called()


def test_test_sans_modele(client, projet, faux_thread):
    (projet / "config.json").write_text(json.dumps({"CURRENT_PROJECT_NAME": "manuscrits", "LAST_MODEL_PATH": ""}))
    reponse = client.post("/test_modele", content_type="multipart/form-data",
                          data={"fichiers": images_de_test("f1.jpg")})
    assert reponse.headers["Location"] == "/"
    faux_thread.assert_not_called()


@pytest.mark.parametrize(
    "etat, destination",
    [("True", "/test_end"), ("Error", "/accueil_projet"), (None, "/accueil_projet")],
    ids=["terminees", "erreur", "aucune_en_cours"],
)
def test_suivi_des_predictions(client, projet, monkeypatch, etat, destination):
    if etat is not None:
        monkeypatch.setenv("test_complete", etat)
    assert client.get("/check_inference").headers["Location"] == destination


def test_suivi_des_predictions_en_cours(client, projet, monkeypatch):
    monkeypatch.setenv("test_complete", "False")
    assert client.get("/check_inference").status_code == 200


# --- Test d'un modèle choisi dans la liste ------------------------------------------------------------

def test_liste_des_modeles_du_projet(client, projet):
    page = client.get("/test_pre_trained_model").get_data(as_text=True)
    assert MODELE in page


def test_test_d_un_modele_choisi(client, projet, faux_thread):
    reponse = client.post("/test_images_pretrained", content_type="multipart/form-data",
                          data={"modele": MODELE, "fichiers": images_de_test("f1.jpg")})

    assert reponse.status_code == 200
    config = json.loads((projet / "config.json").read_text())
    assert config["LAST_MODEL_PATH"].endswith(f"output/train/{MODELE}")
    faux_thread.return_value.start.assert_called_once()


@pytest.mark.parametrize("modele", ["incunables_20260502_090000", "------"], ids=["autre_projet", "choix_vide"])
def test_test_refuse_un_modele_hors_du_projet(client, projet, faux_thread, modele):
    # La page ne propose que les modèles du projet : le formulaire doit refuser les autres.
    (projet / "output" / "train" / "incunables_20260502_090000" / "weights").mkdir(parents=True)

    reponse = client.post("/test_images_pretrained", content_type="multipart/form-data",
                          data={"modele": modele, "fichiers": images_de_test("f1.jpg")})

    assert reponse.headers["Location"] == "/test_pre_trained_model"
    faux_thread.assert_not_called()


# --- Correction dans Label Studio ---------------------------------------------------------------------

@pytest.fixture
def faux_label_studio(mocker):
    for nom in ["configure_label_studio_root", "launch_label_studio_async", "yolo_to_csv"]:
        mocker.patch.object(testing_routes, nom)
    mocker.patch.object(testing_routes, "get_ls_for_local_files", return_value="taches.json")
    mocker.patch.object(testing_routes, "get_labeling_code", return_value=("<View/>", "config.txt"))
    return mocker.patch.object(testing_routes, "setup_ls_correction_project",
                               return_value=(2, "http://localhost:8080/projects/2"))


def test_correction_dans_label_studio(client, projet, faux_label_studio):
    page = client.get("/test_end").get_data(as_text=True)
    assert "http://localhost:8080/projects/2" in page
    assert faux_label_studio.call_args.kwargs["ls_result_file"] == "taches.json"


def test_correction_label_studio_injoignable(client, projet, faux_label_studio):
    faux_label_studio.side_effect = ConnectionError("Label Studio ne répond pas")
    assert client.get("/test_end").headers["Location"] == "/accueil_projet"


# --- Évaluation et fin de cycle -------------------------------------------------------------------------

@pytest.fixture
def faux_evaluation(mocker, projet):
    for nom in ["add_new_labels", "get_corrected_label_files", "get_csv_results"]:
        mocker.patch.object(testing_routes, nom)
    resultats = projet / "output" / "predict" / f"manuscrits_{MODELE}" / "results"
    mocker.patch.object(testing_routes, "get_txt_results", return_value=resultats / "results_for_evaluation.png")
    return mocker.patch.object(testing_routes, "create_confusion_matrix", return_value=resultats / "confusion_matrix.png")


def test_evaluation(client, projet, faux_evaluation, mocker):
    omises = [{"image": "f5.jpg", "raison": "image non corrigée (non validée) dans Label Studio"}]
    mocker.patch.object(testing_routes, "read_omitted_images", return_value=omises)
    rendu = mocker.spy(testing_routes, "render_template")

    flash = mocker.spy(testing_routes, "flash")

    page = client.get("/evaluate_model").get_data(as_text=True)

    assert f"output/predict/manuscrits_{MODELE}/results/confusion_matrix.png" in page
    # La liste des images omises est transmise à la page
    assert rendu.call_args.kwargs["images_omises"] == omises
    assert "1 image(s) de test omise(s)" in flash.call_args.args[0]


def test_evaluation_sans_correction(client, projet, faux_evaluation, mocker):
    faux_evaluation.return_value = None   # pas de matrice : aucune correction à évaluer
    rendu = mocker.spy(testing_routes, "render_template")

    assert client.get("/evaluate_model").status_code == 200
    assert rendu.call_args.args[0] == "/pages/no_evaluation.html"


def test_evaluation_classe_inconnue(client, projet, faux_evaluation, mocker):
    mocker.patch.object(testing_routes, "get_corrected_label_files",
                        side_effect=ValueError("La classe 'enluminure' est absente de labels.txt."))
    assert client.get("/evaluate_model").headers["Location"] == "/accueil_projet"


def test_fin_de_cycle(client, projet, mocker):
    etapes = {nom: mocker.patch.object(testing_routes, nom) for nom in
              ["create_new_ground_truth", "move_correction_files_and_images", "add_csv_data", "clean_data"]}

    reponse = client.get("/dispatch_corrections")

    assert reponse.status_code == 200
    for etape in etapes.values():
        etape.assert_called_once()