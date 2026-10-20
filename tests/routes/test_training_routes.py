"""Tests des routes d'entraînement : paramètres, lancement, suivi et résultats.

Le thread d'entraînement et YOLO sont remplacés par des mocks : aucun modèle
n'est réellement entraîné.
"""

import json
import os

import pandas as pd
import pytest

import app.routes.training_routes as training_routes
from app.services.project_service import create_project_structure

MODELE = "manuscrits_20260301_101500"


@pytest.fixture
def projet(espace_projets, monkeypatch):
    """Projet « manuscrits » sélectionné, sans entraînement en cours."""
    create_project_structure("manuscrits")
    (espace_projets / "config.json").write_text(
        json.dumps({"CURRENT_PROJECT_NAME": "manuscrits", "LAST_MODEL_PATH": ""}))
    monkeypatch.delenv("complete", raising=False)
    return espace_projets


@pytest.fixture
def faux_thread(mocker):
    return mocker.patch.object(training_routes, "Thread")


@pytest.fixture
def faux_yolo(mocker):
    yolo = mocker.patch.object(training_routes, "YOLO")
    yolo.return_value.names = {0: "miniature"}
    return yolo


def ajouter_modele_au_projet(racine, nom=MODELE):
    """Un modèle déjà entraîné, enregistré dans le CSV du projet."""
    (racine / "output" / "train" / nom / "weights").mkdir(parents=True)
    (racine / "output" / "train" / nom / "weights" / "best.pt").write_bytes(b"")
    pd.DataFrame({"model_name": [nom]}).to_csv(racine / "projects" / "manuscrits" / "manuscrits.csv", index=False)


def lancer(client, epochs="10", dropout="0.1", modele="------"):
    return client.post("/start_training", data={"epochs": epochs, "dropout": dropout, "modele": modele})


def ecrire_config(racine, model_name=MODELE, epochs=10):
    (racine / "training_config.json").write_text(json.dumps({
        "use_model": "yolo26s.pt", "model_name": model_name, "img_size": 640,
        "epochs": epochs, "batch": -1, "workers": 8, "dropout": 0.1}))


# --- Paramètres --------------------------------------------------------------------------

def test_page_de_parametres_liste_les_modeles(client, projet):
    ajouter_modele_au_projet(projet)
    page = client.get("/training_setup").get_data(as_text=True)
    assert "------" in page
    assert MODELE in page


# --- Lancement ------------------------------------------------------------------------------

def test_lancement_depuis_zero(client, projet, faux_thread):
    reponse = lancer(client)

    assert reponse.status_code == 200
    args = faux_thread.call_args.kwargs["args"]
    assert args[0] == projet / "projects" / "manuscrits"  # chemin complet du projet
    assert args[1:4] == (10, 0.1, "yolo26s.pt")
    assert args[4].startswith("manuscrits_")
    faux_thread.return_value.start.assert_called_once()
    assert os.environ["complete"] == "False"
    assert json.loads((projet / "training_config.json").read_text())["epochs"] == 10


def test_lancement_a_partir_d_un_modele_existant(client, projet, faux_thread):
    ajouter_modele_au_projet(projet)
    lancer(client, modele=MODELE)
    modele_de_depart = faux_thread.call_args.kwargs["args"][3]
    assert modele_de_depart.endswith(f"{MODELE}/weights/best.pt")


def test_lancement_en_visitant_l_url(client, projet, faux_thread):
    assert client.get("/start_training").status_code == 302
    faux_thread.assert_not_called()


@pytest.mark.parametrize(
    "epochs, dropout",
    [("", "0.1"), ("dix", "0.1"), ("0", "0.1"), ("10", ""), ("10", "1.5"), ("10", "-0.1")],
    ids=["epochs_vide", "epochs_texte", "zero_epoch", "dropout_vide", "dropout_trop_grand", "dropout_negatif"],
)
def test_parametres_invalides(client, projet, faux_thread, epochs, dropout):
    reponse = lancer(client, epochs=epochs, dropout=dropout)
    assert reponse.headers["Location"] == "/training_setup"
    faux_thread.assert_not_called()


def test_modele_de_depart_inconnu(client, projet, faux_thread):
    # La valeur reçue sert à construire un chemin : seule la liste des modèles du projet est acceptée.
    reponse = lancer(client, modele="../../ailleurs")
    assert reponse.headers["Location"] == "/training_setup"
    faux_thread.assert_not_called()


def test_un_seul_entrainement_a_la_fois(client, projet, faux_thread, monkeypatch):
    monkeypatch.setenv("complete", "False")
    reponse = lancer(client)
    assert reponse.headers["Location"] == "/check_training_status"
    faux_thread.assert_not_called()


# --- Suivi de l'entraînement -------------------------------------------------------------------

def test_suivi_affiche_l_epoque_en_cours(client, projet, monkeypatch):
    monkeypatch.setenv("complete", "False")
    ecrire_config(projet)
    dossier = projet / "output" / "train" / MODELE
    dossier.mkdir(parents=True)
    # Anciennes versions de YOLO : noms de colonnes précédés d'espaces
    (dossier / "results.csv").write_text("               epoch,  train/box_loss\n1,0.9\n2,0.8\n3,0.7\n")

    assert "3/10" in client.get("/check_training_status").get_data(as_text=True)


def test_suivi_results_csv_encore_vide(client, projet, monkeypatch):
    monkeypatch.setenv("complete", "False")
    ecrire_config(projet)
    dossier = projet / "output" / "train" / MODELE
    dossier.mkdir(parents=True)
    (dossier / "results.csv").write_text("epoch,train/box_loss\n")

    assert "0/10" in client.get("/check_training_status").get_data(as_text=True)


def test_suivi_entrainement_termine(client, projet, monkeypatch):
    monkeypatch.setenv("complete", "True")
    assert client.get("/check_training_status").headers["Location"] == "/training_result"


def test_suivi_entrainement_en_echec(client, projet, monkeypatch):
    monkeypatch.setenv("complete", "Error")
    assert client.get("/check_training_status").headers["Location"] == "/training_setup"


def test_suivi_sans_entrainement_en_cours(client, projet):
    # Par exemple après un redémarrage de l'app : la variable « complete » n'existe plus.
    assert client.get("/check_training_status").headers["Location"] == "/training_setup"


# --- Résultats ----------------------------------------------------------------------------------

def test_resultats_enregistrent_le_modele(client, projet, faux_yolo):
    ecrire_config(projet)
    (projet / "output" / "train" / MODELE / "weights").mkdir(parents=True)
    (projet / "output" / "train" / MODELE / "weights" / "best.pt").write_bytes(b"")

    reponse = client.get("/training_result")

    assert reponse.status_code == 200
    df = pd.read_csv(projet / "projects" / "manuscrits" / "manuscrits.csv")
    assert df["model_name"].tolist() == [MODELE]
    assert df["origine"].tolist() == ["TiamaT"]
    config = json.loads((projet / "config.json").read_text())
    assert config["LAST_MODEL_PATH"].endswith(f"output/train/{MODELE}")


def test_recharger_les_resultats_ne_cree_pas_de_doublon(client, projet, faux_yolo):
    ecrire_config(projet)
    (projet / "output" / "train" / MODELE / "weights").mkdir(parents=True)
    (projet / "output" / "train" / MODELE / "weights" / "best.pt").write_bytes(b"")

    client.get("/training_result")
    client.get("/training_result")  # F5

    df = pd.read_csv(projet / "projects" / "manuscrits" / "manuscrits.csv")
    assert df["model_name"].tolist() == [MODELE]


def test_resultats_sans_modele_entraine(client, projet, faux_yolo):
    ecrire_config(projet)
    assert client.get("/training_result").headers["Location"] == "/training_setup"