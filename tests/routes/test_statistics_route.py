"""Tests de la page des statistiques du dataset : de l'annotation Label Studio au graphique."""

import json

import pytest
from PIL import Image

import app.routes.statistics_route as statistics_route
from app.services.project_service import create_project_structure


@pytest.fixture
def projet(espace_projets):
    """Projet « manuscrits » sélectionné, avec deux images annotées dans Label Studio."""
    projet = create_project_structure("manuscrits")
    (espace_projets / "config.json").write_text(
        json.dumps({"CURRENT_PROJECT_NAME": "manuscrits", "LAST_MODEL_PATH": ""}))
    for numero, (image, classe) in enumerate([("f1.jpg", "miniature"), ("f2.jpg", "initiale")], start=1):
        Image.new("RGB", (40, 20), "white").save(projet / "image_inputs" / "ground_truth_images" / image)
        annotation = {"id": numero,
                      "result": [{"value": {"x": 10, "y": 10, "width": 20, "height": 20, "rectanglelabels": [classe]}}],
                      "task": {"data": {"image": f"/data/local-files/?d=manuscrits/image_inputs/ground_truth_images/{image}"}}}
        (projet / "annotations" / "ground_truth" / f"{numero}.json").write_text(json.dumps(annotation))
    return espace_projets


def test_statistiques_completes(client, projet):
    reponse = client.get("/dataset_statistics")

    assert reponse.status_code == 200
    stats = projet / "data" / "manuscrits" / "dataset_statistics"
    for fichier in ["class_distribution.csv", "class_distribution.png", "annotations_per_img.csv", "global_data.csv"]:
        assert (stats / fichier).exists()

    image = client.get("/dataset_stats")
    assert image.status_code == 200
    assert image.mimetype == "image/png"


def test_le_graphique_n_est_pas_mis_en_cache(client, projet):
    # Sinon, après un nouveau calcul, le navigateur réafficherait l'ancien graphique.
    client.get("/dataset_statistics")
    assert "no-cache" in client.get("/dataset_stats").headers["Cache-Control"]


def test_statistiques_sans_projet_courant(client, espace_projets):
    reponse = client.get("/dataset_statistics")
    assert reponse.headers["Location"] == "/"
    assert not (espace_projets.parent / "data").exists()


def test_erreur_de_preparation_affichee(client, projet, mocker):
    erreur = "La classe 'enluminure' (annotation 3.json) est absente de labels.txt."
    mocker.patch.object(statistics_route, "create_dataset", side_effect=ValueError(erreur))
    flash = mocker.spy(statistics_route, "flash")

    reponse = client.get("/dataset_statistics")

    assert reponse.headers["Location"] == "/accueil_projet"
    assert erreur in flash.call_args.args[0]


def test_graphique_absent(client, projet):
    assert client.get("/dataset_stats").status_code == 404