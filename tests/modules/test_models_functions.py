"""Tests de la liste des modèles entraînés d'un projet.

get_model_list cherche le fichier projects/<projet>/<projet>.csv à partir du
dossier courant (là où l'app est lancée). Les tests se placent donc dans un
dossier temporaire avec monkeypatch.chdir, pour ne jamais lire tes vrais projets.
"""

import pandas as pd
import pytest

from app.src.modules.models_functions import get_model_list


@pytest.fixture
def dossier_app(tmp_path, monkeypatch):
    """Fait de tmp_path le dossier courant, comme si l'app était lancée depuis là."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def ecrire_csv_modeles(dossier_app, projet, modeles):
    dossier = dossier_app / "projects" / projet
    dossier.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"model_name": modeles}).to_csv(dossier / f"{projet}.csv", index=False)


def test_sans_csv_seulement_le_choix_vide(dossier_app):
    assert get_model_list("manuscrits") == ["------"]


def test_liste_les_modeles_du_projet(dossier_app):
    ecrire_csv_modeles(dossier_app, "manuscrits",
                       ["manuscrits_20260301_101500", "manuscrits_20260415_143000"])
    assert get_model_list("manuscrits") == [
        "------",                      # toujours en premier : « partir de zéro »
        "manuscrits_20260301_101500",
        "manuscrits_20260415_143000",
    ]


def test_ne_melange_pas_les_projets(dossier_app):
    ecrire_csv_modeles(dossier_app, "manuscrits", ["manuscrits_20260301_101500"])
    ecrire_csv_modeles(dossier_app, "incunables", ["incunables_20260502_090000"])
    assert get_model_list("incunables") == ["------", "incunables_20260502_090000"]


def test_csv_sans_modele(dossier_app):
    # Le CSV existe (créé avec le projet) mais aucun modèle n'a encore été entraîné.
    ecrire_csv_modeles(dossier_app, "manuscrits", [])
    assert get_model_list("manuscrits") == ["------"]