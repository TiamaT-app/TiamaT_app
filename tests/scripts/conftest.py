"""Fixtures partagées par les tests des scripts du cycle de test d'un modèle."""

import pytest

from app.services.project_service import create_project_structure

MODELE = "manuscrits_20260301_101500"


@pytest.fixture
def cycle(espace_projets):
    """Un projet et un modèle entraîné, avec les dossiers utilisés pendant le test du modèle."""
    projet = create_project_structure("manuscrits")
    modele = espace_projets / "output" / "train" / MODELE
    (modele / "weights").mkdir(parents=True)
    (modele / "weights" / "best.pt").write_bytes(b"")
    (modele / "labels.txt").write_text("'0': 'miniature'\n'1': 'initiale'\n", encoding="utf-8")
    resultats = espace_projets / "output" / "predict" / f"manuscrits_{MODELE}"
    return {
        "racine": espace_projets,
        "projet": projet,
        "modele": modele,
        "resultats": resultats,
        "eval": projet / "image_inputs" / "eval_images",
        "corrections": projet / "annotations" / "prediction_corrections",
        "gt_images": projet / "image_inputs" / "ground_truth_images",
        "gt": projet / "annotations" / "ground_truth",
        "data": espace_projets / "data" / "manuscrits",
    }