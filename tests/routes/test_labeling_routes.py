"""Tests du lancement de Label Studio et de l'enregistrement du token d'API.

Label Studio n'est jamais réellement lancé : ses fonctions de lancement et de
configuration sont remplacées par des mocks.
"""

import json
import os

import pytest

import app.routes.labeling_routes as labeling_routes
from app.services.project_service import create_project_structure

TOKEN = "abc123def456"


@pytest.fixture
def variables_ls(monkeypatch):
    """Part d'un état propre pour les variables d'environnement utilisées par les routes.

    Les routes écrivent directement dans os.environ. En déclarant ces variables via
    monkeypatch, on s'assure qu'elles retrouvent leur valeur d'origine après chaque test.
    """
    monkeypatch.delenv("LABEL_STUDIO_API_TOKEN", raising=False)
    monkeypatch.setenv("LS_READY", "False")
    monkeypatch.delenv("LS_URL_FOUND", raising=False)


@pytest.fixture
def projet(espace_projets, variables_ls):
    """Un projet « manuscrits » sélectionné comme projet courant."""
    create_project_structure("manuscrits")
    (espace_projets / "config.json").write_text(
        json.dumps({"CURRENT_PROJECT_NAME": "manuscrits", "LAST_MODEL_PATH": ""}))
    return espace_projets


@pytest.fixture
def faux_label_studio(mocker):
    """Remplace tout ce qui lancerait réellement Label Studio."""
    return {
        "thread": mocker.patch.object(labeling_routes, "Thread"),
        "configure": mocker.patch.object(labeling_routes, "configure_label_studio_root"),
        "launch": mocker.patch.object(labeling_routes, "launch_label_studio_async"),
        "setup": mocker.patch("app.services.label_studio_api_service.setup_label_studio_project",
                              return_value=(1, "http://localhost:8080/projects/1")),
    }


# --- Lancement de Label Studio ---------------------------------------------------------

def test_sans_token_affiche_le_formulaire_du_token(client, projet, faux_label_studio):
    reponse = client.get("/label_lancement")

    assert reponse.status_code == 200
    faux_label_studio["thread"].return_value.start.assert_called_once()  # Label Studio démarre en fond
    faux_label_studio["launch"].assert_not_called()


def test_avec_token_lance_label_studio(client, projet, faux_label_studio, monkeypatch):
    monkeypatch.setenv("LABEL_STUDIO_API_TOKEN", TOKEN)

    reponse = client.get("/label_lancement")

    assert reponse.status_code == 200
    faux_label_studio["configure"].assert_called_once()
    faux_label_studio["launch"].assert_called_once()
    assert os.environ["LS_READY"] == "False"


def test_lancement_normalise_les_noms_d_images(client, projet, faux_label_studio, monkeypatch):
    monkeypatch.setenv("LABEL_STUDIO_API_TOKEN", TOKEN)
    images = projet / "projects" / "manuscrits" / "image_inputs" / "ground_truth_images"
    (images / "Épître 01.JPG").write_bytes(b"")

    client.get("/label_lancement")

    assert [p.name for p in images.iterdir()] == ["epitre_01.jpg"]


def test_label_studio_pret_configure_le_projet(client, projet, faux_label_studio, monkeypatch):
    # Quand Label Studio est prêt, la fonction on_ready transmise au lancement configure le projet.
    monkeypatch.setenv("LABEL_STUDIO_API_TOKEN", TOKEN)
    client.get("/label_lancement")
    on_ready = faux_label_studio["launch"].call_args.kwargs["on_ready"]

    on_ready()

    assert os.environ["LS_READY"] == "True"
    assert os.environ["LS_URL_FOUND"] == "http://localhost:8080/projects/1"


def test_erreur_de_configuration_sans_afficher_le_token(client, projet, faux_label_studio, monkeypatch, capsys):
    monkeypatch.setenv("LABEL_STUDIO_API_TOKEN", TOKEN)
    faux_label_studio["setup"].side_effect = RuntimeError("Label Studio injoignable")
    client.get("/label_lancement")

    faux_label_studio["launch"].call_args.kwargs["on_ready"]()

    assert os.environ["LS_READY"] == "Error"
    # Le token est un secret : il ne doit jamais apparaître dans le terminal ou les logs.
    assert TOKEN not in capsys.readouterr().out


def test_lancement_sans_projet_courant(client, espace_projets, variables_ls, faux_label_studio, monkeypatch):
    monkeypatch.setenv("LABEL_STUDIO_API_TOKEN", TOKEN)
    reponse = client.get("/label_lancement")
    assert reponse.status_code == 302
    faux_label_studio["launch"].assert_not_called()


# --- Suivi du lancement ------------------------------------------------------------------------

def test_statut_en_cours(client, projet):
    assert client.get("/check_ls_status").status_code == 200


def test_statut_pret_redirige(client, projet, monkeypatch):
    monkeypatch.setenv("LS_READY", "True")
    reponse = client.get("/check_ls_status")
    assert reponse.headers["Location"] == "/ls_ready"


def test_statut_erreur_redirige_vers_le_projet(client, projet, monkeypatch):
    monkeypatch.setenv("LS_READY", "Error")
    reponse = client.get("/check_ls_status")
    assert reponse.headers["Location"] == "/accueil_projet"


def test_page_label_studio_pret(client, projet, monkeypatch):
    monkeypatch.setenv("LS_URL_FOUND", "http://localhost:8080/projects/1")
    page = client.get("/ls_ready").get_data(as_text=True)
    assert "http://localhost:8080/projects/1" in page


# --- Enregistrement du token -------------------------------------------------------------------

def lire_env(racine):
    return (racine / ".env").read_text(encoding="utf-8")


def test_ajout_du_token(client, projet):
    reponse = client.post("/token_added", data={"token": TOKEN})

    assert reponse.headers["Location"] == "/label_lancement"
    assert f"LABEL_STUDIO_API_TOKEN={TOKEN}" in lire_env(projet)
    assert os.environ["LABEL_STUDIO_API_TOKEN"] == TOKEN


def test_ajout_du_token_conserve_le_reste_du_env(client, projet):
    (projet / ".env").write_text("AUTRE_VARIABLE=1\n", encoding="utf-8")
    client.post("/token_added", data={"token": TOKEN})
    assert "AUTRE_VARIABLE=1" in lire_env(projet)


def test_nouveau_token_remplace_l_ancien(client, projet):
    client.post("/token_added", data={"token": "ancien"})
    client.post("/token_added", data={"token": TOKEN})

    assert lire_env(projet).count("LABEL_STUDIO_API_TOKEN=") == 1
    assert f"LABEL_STUDIO_API_TOKEN={TOKEN}" in lire_env(projet)


def test_visiter_la_page_n_ecrit_rien(client, projet):
    # Sans formulaire envoyé, rien ne doit être ajouté au .env.
    reponse = client.get("/token_added")
    assert reponse.status_code == 302
    assert not (projet / ".env").exists()


@pytest.mark.parametrize("token", ["", "abc\nAUTRE=pirate", "abc def"], ids=["vide", "retour_ligne", "espace"])
def test_token_invalide_refuse(client, projet, token):
    client.post("/token_added", data={"token": token})
    assert not (projet / ".env").exists()
    assert "LABEL_STUDIO_API_TOKEN" not in os.environ