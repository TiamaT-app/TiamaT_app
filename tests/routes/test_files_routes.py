"""Tests des routes qui servent des fichiers : images affichées dans les pages et téléchargement des modèles.

Ces routes reçoivent un chemin dans l'URL. Elles doivent servir les fichiers
dont l'app a besoin, et refuser (erreur 404) tout le reste : fichiers de
configuration, secrets du .env, fichiers hors du dossier de l'app…
"""

import re

import pandas as pd
import pytest

from app.services.project_service import create_project_structure


@pytest.fixture
def fichiers(espace_projets):
    """Crée les fichiers qu'un projet TiamaT produit, plus quelques fichiers sensibles."""
    racine = espace_projets
    create_project_structure("manuscrits")
    images = racine / "projects" / "manuscrits" / "image_inputs" / "ground_truth_images"
    (images / "f1.jpg").write_bytes(b"image-corpus")
    (images / "f2.JPG").write_bytes(b"image-majuscules")

    modele = racine / "output" / "train" / "manuscrits_20260301_101500"
    (modele / "weights").mkdir(parents=True)
    (modele / "confusion_matrix.png").write_bytes(b"matrice")
    (modele / "weights" / "best.pt").write_bytes(b"poids-du-modele")

    resultats = racine / "output" / "predict" / "manuscrits_manuscrits_20260301_101500" / "results"
    resultats.mkdir(parents=True)
    (resultats / "metrics.png").write_bytes(b"metriques")

    # Fichiers qui ne doivent jamais être servis
    (racine / ".env").write_text("LABEL_STUDIO_API_TOKEN=secret")
    (racine / "config.json").write_text("{}")
    (racine / "secret.pt").write_bytes(b"hors-de-output-train")
    (racine / "projects" / "manuscrits" / "annotations" / "ground_truth" / "1.json").write_text("{}")
    return racine


# --- Images affichées dans les pages -------------------------------------------------

def test_image_du_corpus(client, fichiers):
    reponse = client.get("/images/projects/manuscrits/image_inputs/ground_truth_images/f1.jpg")
    assert reponse.status_code == 200
    assert reponse.data == b"image-corpus"


def test_image_extension_majuscules(client, fichiers):
    reponse = client.get("/images/projects/manuscrits/image_inputs/ground_truth_images/f2.JPG")
    assert reponse.status_code == 200


def test_matrice_de_confusion_apres_entrainement(client, fichiers):
    reponse = client.get("/images/output/train/manuscrits_20260301_101500/confusion_matrix.png")
    assert reponse.data == b"matrice"


def test_serve_images_resultats_d_evaluation(client, fichiers):
    reponse = client.get("/serve_images/output/predict/manuscrits_manuscrits_20260301_101500/results/metrics.png")
    assert reponse.data == b"metriques"


@pytest.mark.parametrize("route", ["/images/", "/serve_images/"])
@pytest.mark.parametrize(
    "chemin",
    [
        ".env",                                                   # le token Label Studio
        "config.json",                                            # pas une image
        "projects/manuscrits/annotations/ground_truth/1.json",    # pas une image
        "projects/../.env",                                       # remonte d'un dossier
        "projects/%2e%2e/.env",                                   # idem, encodé dans l'URL
        "projects/manuscrits/../../../../../../etc/passwd",       # sort du dossier de l'app
    ],
    ids=["env", "config", "json_annotation", "remontee", "remontee_encodee", "hors_app"],
)
def test_images_refuse_les_fichiers_non_autorises(client, fichiers, route, chemin):
    assert client.get(route + chemin).status_code == 404


def test_images_refuse_les_fichiers_caches(client, fichiers):
    cache = fichiers / "projects" / "manuscrits" / ".cache"
    cache.mkdir()
    (cache / "vignette.png").write_bytes(b"")
    assert client.get("/images/projects/manuscrits/.cache/vignette.png").status_code == 404


# --- Téléchargement des modèles -------------------------------------------------------------

def test_telechargement_d_un_modele(client, fichiers):
    chemin = fichiers / "output" / "train" / "manuscrits_20260301_101500" / "weights" / "best.pt"
    # Même forme d'URL que les liens générés par gestion_modeles : /download/<chemin absolu>
    reponse = client.get(f"/download/{chemin}", follow_redirects=True)
    assert reponse.status_code == 200
    assert reponse.data == b"poids-du-modele"
    assert "attachment" in reponse.headers["Content-Disposition"]


@pytest.mark.parametrize(
    "nom_fichier",
    [".env", "config.json", "secret.pt"],
    ids=["env", "config", "pt_hors_de_output_train"],
)
def test_telechargement_refuse_les_fichiers_non_autorises(client, fichiers, nom_fichier):
    reponse = client.get(f"/download/{fichiers / nom_fichier}", follow_redirects=True)
    assert reponse.status_code == 404


def test_telechargement_refuse_un_fichier_systeme(client, fichiers):
    assert client.get("/download//etc/passwd", follow_redirects=True).status_code == 404


def test_lien_de_gestion_modeles_fonctionne(client, fichiers):
    # De bout en bout : le lien affiché sur la page de gestion permet bien de télécharger le modèle.
    (fichiers / "config.json").write_text('{"CURRENT_PROJECT_NAME": "manuscrits", "LAST_MODEL_PATH": ""}')
    pd.DataFrame({"model_name": ["manuscrits_20260301_101500"]}).to_csv(
        fichiers / "projects" / "manuscrits" / "manuscrits.csv", index=False)

    page = client.get("/gestion_modeles").get_data(as_text=True)
    lien = re.search(r'href="(/download/[^"]+)"', page).group(1)

    reponse = client.get(lien, follow_redirects=True)
    assert reponse.data == b"poids-du-modele"