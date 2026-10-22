import sys

import pytest

from app.app import app as flask_app
from app.services import project_service


@pytest.fixture
def app():
    flask_app.config.update(
        TESTING=True,
        # Les formulaires Flask-WTF exigent un jeton CSRF caché, que le client
        # de test ne fournit pas. On désactive cette vérification pendant les tests.
        WTF_CSRF_ENABLED=False,
    )
    yield flask_app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def espace_projets(tmp_path, monkeypatch):
    """Isole les tests dans un dossier temporaire, comme si l'app était lancée depuis là.

    - le dossier courant devient tmp_path : config.json, data/ et output/ y sont créés ;
    - projects_folder est redirigé vers tmp_path/projects.

    projects_folder est calculé une seule fois, au chargement de project_service,
    et plusieurs routes l'importent avec « from ... import projects_folder ».
    Chacune garde alors sa propre copie de la valeur : il faut la remplacer dans
    chaque module de l'app qui la possède, pas seulement dans project_service.

    Aucun test ne peut ainsi toucher à tes vrais projets.
    """
    monkeypatch.chdir(tmp_path)
    dossier_projets = tmp_path / "projects"
    dossier_projets.mkdir()

    vrai_dossier = project_service.projects_folder
    for module in list(sys.modules.values()):
        nom = getattr(module, "__name__", "") or ""
        if nom.startswith("app.") and getattr(module, "projects_folder", None) == vrai_dossier:
            monkeypatch.setattr(module, "projects_folder", dossier_projets)
    return tmp_path


@pytest.fixture(autouse=True)
def sans_vrai_label_studio(monkeypatch):
    """Garantit qu'aucun test ne contacte le vrai Label Studio, même s'il tourne sur l'ordinateur.

    autouse=True : cette fixture s'applique à tous les tests, sans qu'ils aient à la demander.
    Le token est retiré de l'environnement, et Label Studio est considéré comme non lancé.
    Sans elle, un test de suppression de projet pourrait supprimer de vrais projets Label Studio
    portant le même nom que les projets de test. Les tests qui ont besoin d'un Label Studio
    « lancé » le simulent eux-mêmes, avec un faux client.
    """
    from app.services import label_studio_api_service
    monkeypatch.delenv("LABEL_STUDIO_API_TOKEN", raising=False)
    monkeypatch.setattr(label_studio_api_service, "is_label_studio_running", lambda port=None: False)