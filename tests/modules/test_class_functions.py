"""Tests des correspondances entre identifiants et noms de classes."""

import pytest

from app.src.modules.class_functions import get_class_code, get_class_name


@pytest.fixture
def labels():
    # Même format que le dictionnaire de classes d'un projet TiamaT :
    # identifiant (en chaîne) -> nom de la classe.
    return {"0": "miniature", "1": "initiale", "2": "bordure"}


# --- get_class_name ----------------------------------------------------------

def test_get_class_name_avec_un_entier(labels):
    assert get_class_name(1, labels) == "initiale"


@pytest.mark.parametrize("class_id", [1.0, "1", "1.0"], ids=["float", "chaine", "chaine_float"])
def test_get_class_name_accepte_float_et_chaine(labels, class_id):
    # YOLO et les CSV renvoient souvent l'identifiant sous forme 1.0 ou "1".
    assert get_class_name(class_id, labels) == "initiale"


def test_get_class_name_identifiant_inconnu(labels):
    assert get_class_name(99, labels) == "unknown-class"


# --- get_class_code ----------------------------------------------------------

def test_get_class_code_nom_connu(labels):
    # Attention : la fonction prend le MÊME dictionnaire (id -> nom) que
    # get_class_name et l'inverse elle-même. Elle renvoie l'identifiant en chaîne.
    assert get_class_code("bordure", labels) == "2"


def test_get_class_code_nom_inconnu(labels):
    assert get_class_code("enluminure", labels) == "unknown-class"


def test_get_class_code_ne_modifie_pas_le_dictionnaire(labels):
    copie = dict(labels)
    get_class_code("miniature", labels)
    assert labels == copie


# --- Aller-retour ------------------------------------------------------------

def test_aller_retour_nom_code_nom(labels):
    for class_id, class_name in labels.items():
        assert get_class_code(class_name, labels) == class_id
        assert get_class_name(class_id, labels) == class_name