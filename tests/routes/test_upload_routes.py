"""Tests de l'import d'images dans un projet et de l'import d'un modèle YOLO déjà entraîné.

YOLO est remplacé par un mock : les tests n'ont besoin ni d'un vrai modèle,
ni de charger PyTorch.
"""

import io
import json

import pandas as pd
import pytest

import app.routes.upload_routes as upload_routes
from app.services.project_service import create_project_structure


@pytest.fixture
def projet(espace_projets):
    """Un projet « manuscrits » créé et sélectionné comme projet courant."""
    create_project_structure("manuscrits")
    (espace_projets / "config.json").write_text(
        json.dumps({"CURRENT_PROJECT_NAME": "manuscrits", "LAST_MODEL_PATH": ""}))
    return espace_projets


@pytest.fixture
def faux_yolo(mocker):
    """Remplace YOLO : tout fichier .pt est lu comme un modèle à deux classes."""
    yolo = mocker.patch("app.routes.upload_routes.YOLO")
    yolo.return_value.names = {0: "miniature", 1: "lettrine ornée"}
    return yolo


def fichier(nom, contenu=b"contenu"):
    return (io.BytesIO(contenu), nom)


def dossier_images(racine):
    return racine / "projects" / "manuscrits" / "image_inputs" / "ground_truth_images"


def dossier_modele(racine, nom="manuscrits_v1"):
    return racine / "output" / "train" / nom


def envoyer_modele(client, nom="v1", contenu=b"poids", nom_fichier="best.pt"):
    return client.post("/upload_modele", content_type="multipart/form-data",
                       data={"nom": nom, "model": fichier(nom_fichier, contenu)})


# --- Import d'images -------------------------------------------------------------------

def test_upload_d_images(client, projet):
    reponse = client.post("/upload", content_type="multipart/form-data",
                          data={"fichiers": [fichier("f1.jpg"), fichier("f2.png")]})

    assert reponse.status_code == 200
    assert sorted(p.name for p in dossier_images(projet).iterdir()) == ["f1.jpg", "f2.png"]


def test_upload_refuse_un_fichier_qui_n_est_pas_une_image(client, projet):
    reponse = client.post("/upload", content_type="multipart/form-data",
                          data={"fichiers": [fichier("notes.txt")]})

    assert reponse.status_code == 302
    assert list(dossier_images(projet).iterdir()) == []


def test_upload_sans_fichier_redirige(client, projet):
    assert client.get("/upload").status_code == 302


def test_upload_d_une_image_deja_presente_ne_cree_pas_de_doublon(client, projet):
    # Sans précaution, la même image serait enregistrée une 2e fois sous le nom f1_1.jpg,
    # et se retrouverait en double dans le dataset d'entraînement.
    (dossier_images(projet) / "f1.jpg").write_bytes(b"meme-image")

    client.post("/upload", content_type="multipart/form-data",
                data={"fichiers": [fichier("f1.jpg", b"meme-image"), fichier("f2.jpg")]})

    assert sorted(p.name for p in dossier_images(projet).iterdir()) == ["f1.jpg", "f2.jpg"]


def test_upload_de_la_meme_image_deux_fois_dans_le_meme_envoi(client, projet):
    client.post("/upload", content_type="multipart/form-data",
                data={"fichiers": [fichier("f1.jpg"), fichier("f1.jpg")]})
    assert [p.name for p in dossier_images(projet).iterdir()] == ["f1.jpg"]


def test_upload_conserve_les_lettres_accentuees(client, projet):
    client.post("/upload", content_type="multipart/form-data",
                data={"fichiers": [fichier("épître_01.jpg")]})
    assert [p.name for p in dossier_images(projet).iterdir()] == ["epitre_01.jpg"]


def test_upload_deux_images_differentes_au_meme_nom_ascii(client, projet, mocker):
    # folio_é.jpg et folio_è.jpg deviennent toutes deux folio_e.jpg :
    # la seconde est conservée sous un autre nom, et l'utilisateur est prévenu.
    # mocker.spy laisse flash fonctionner normalement, tout en enregistrant ses appels.
    flash = mocker.spy(upload_routes, "flash")

    client.post("/upload", content_type="multipart/form-data",
                data={"fichiers": [fichier("folio_é.jpg", b"image-1"), fichier("folio_è.jpg", b"image-2")]})

    images = dossier_images(projet)
    assert (images / "folio_e.jpg").read_bytes() == b"image-1"
    assert (images / "folio_e_1.jpg").read_bytes() == b"image-2"
    messages = " ".join(str(appel.args[0]) for appel in flash.call_args_list)
    assert "folio_è.jpg enregistrée sous folio_e_1.jpg" in messages


def test_upload_renvoi_d_une_image_renommee_ne_cree_pas_de_doublon(client, projet):
    images = dossier_images(projet)
    (images / "folio_e.jpg").write_bytes(b"image-1")
    (images / "folio_e_1.jpg").write_bytes(b"image-2")

    client.post("/upload", content_type="multipart/form-data",
                data={"fichiers": [fichier("folio_è.jpg", b"image-2")]})

    assert sorted(p.name for p in images.iterdir()) == ["folio_e.jpg", "folio_e_1.jpg"]


# --- Import d'un modèle ------------------------------------------------------------------

def test_import_d_un_modele(client, projet, faux_yolo):
    reponse = envoyer_modele(client)

    # Après l'import, on revient sur la page de gestion des modèles.
    assert reponse.status_code == 302
    assert reponse.headers["Location"] == "/gestion_modeles"

    modele = dossier_modele(projet)
    assert (modele / "weights" / "best.pt").read_bytes() == b"poids"
    assert (modele / "labels.txt").read_text(encoding="utf-8") == \
        "'0': 'miniature'\n'1': 'lettrine ornée'\n"

    df = pd.read_csv(projet / "projects" / "manuscrits" / "manuscrits.csv")
    assert df["model_name"].tolist() == ["manuscrits_v1"]
    assert df["origine"].tolist() == ["import"]


def test_import_ajoute_le_modele_a_la_suite_des_autres(client, projet, faux_yolo):
    csv = projet / "projects" / "manuscrits" / "manuscrits.csv"
    pd.DataFrame({"use_model": ["yolo26s.pt"], "model_name": ["manuscrits_20260301_101500"],
                  "img_size": [640], "epochs": [50], "batch": [-1], "workers": [8],
                  "dropout": [0.0], "origine": ["TiamaT"], "classes": ["{0: 'miniature'}"]}).to_csv(csv, index=False)

    envoyer_modele(client)

    assert pd.read_csv(csv)["model_name"].tolist() == ["manuscrits_20260301_101500", "manuscrits_v1"]


def test_import_charge_le_modele_une_seule_fois(client, projet, faux_yolo):
    envoyer_modele(client)
    assert faux_yolo.call_count == 1


def test_import_d_un_modele_deja_existant(client, projet, faux_yolo):
    (dossier_modele(projet) / "weights").mkdir(parents=True)
    (dossier_modele(projet) / "weights" / "best.pt").write_bytes(b"ancien")

    reponse = envoyer_modele(client, contenu=b"nouveau")

    assert reponse.status_code == 302
    assert (dossier_modele(projet) / "weights" / "best.pt").read_bytes() == b"ancien"


def test_import_d_un_fichier_pt_invalide(client, projet, faux_yolo):
    # Un fichier .pt qui n'est pas un modèle YOLO : le chargement échoue.
    faux_yolo.side_effect = RuntimeError("Fichier de poids invalide")

    reponse = envoyer_modele(client)

    assert reponse.status_code == 302
    # Rien ne doit rester : sinon le nom serait bloqué (« ce modèle existe déjà »).
    assert not dossier_modele(projet).exists()
    assert not (projet / "projects" / "manuscrits" / "manuscrits.csv").exists()


def test_import_refuse_un_fichier_qui_n_est_pas_un_pt(client, projet, faux_yolo):
    reponse = envoyer_modele(client, nom_fichier="modele.zip")
    assert reponse.status_code == 302
    assert not dossier_modele(projet).exists()


@pytest.mark.parametrize("nom", ["", "../evasion", "nom avec espaces"],
                         ids=["vide", "remontee", "espaces"])
def test_import_refuse_un_nom_de_modele_invalide(client, projet, faux_yolo, nom):
    reponse = envoyer_modele(client, nom=nom)
    assert reponse.status_code == 302
    assert not (projet / "output").exists()