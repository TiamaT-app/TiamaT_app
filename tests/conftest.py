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
    - projects_folder (calculé une seule fois à l'import de project_service)
      est redirigé vers tmp_path/projects.

    Aucun test ne peut ainsi toucher à tes vrais projets.
    """
    monkeypatch.chdir(tmp_path)
    dossier_projets = tmp_path / "projects"
    dossier_projets.mkdir()
    monkeypatch.setattr(project_service, "projects_folder", dossier_projets)
    return tmp_path