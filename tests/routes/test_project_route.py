"""Tests des routes de gestion des projets : accueil, création, sélection, suppression."""

import json

import pandas as pd
import pytest

from app.services.project_service import create_project_structure
import app.routes.project_route as project_route


def projet_courant(racine):
    return json.loads((racine / "config.json").read_text())["CURRENT_PROJECT_NAME"]


def definir_projet_courant(racine, nom):
    (racine / "config.json").write_text(
        json.dumps({"CURRENT_PROJECT_NAME": nom, "LAST_MODEL_PATH": ""}))


# --- Accueil -------------------------------------------------------------------------

def test_accueil_affiche_les_projets_existants(client, espace_projets):
    create_project_structure("manuscrits")
    reponse = client.get("/")
    assert reponse.status_code == 200
    assert "manuscrits" in reponse.get_data(as_text=True)


# --- Création et sélection d'un projet ---------------------------------------------

def test_creer_un_projet(client, espace_projets):
    reponse = client.post("/accueil_projet", data={"nom": "manuscrits"})

    assert reponse.status_code == 200
    assert (espace_projets / "projects" / "manuscrits" / "image_inputs" / "ground_truth_images").is_dir()
    assert projet_courant(espace_projets) == "manuscrits"


def test_creer_un_projet_avec_accents_refuse(client, espace_projets):
    reponse = client.post("/accueil_projet", data={"nom": "Lettrines_médiévales"})
    assert reponse.status_code == 302
    assert list((espace_projets / "projects").iterdir()) == []


def test_creer_un_projet_deja_existant(client, espace_projets):
    create_project_structure("manuscrits")
    definir_projet_courant(espace_projets, "incunables")

    reponse = client.post("/accueil_projet", data={"nom": "manuscrits"})

    assert reponse.status_code == 200
    assert projet_courant(espace_projets) == "incunables"  # inchangé


def test_choisir_un_projet_existant(client, espace_projets):
    create_project_structure("manuscrits")
    reponse = client.post("/accueil_projet", data={"projet_existant": "manuscrits"})
    assert reponse.status_code == 200
    assert projet_courant(espace_projets) == "manuscrits"


def test_choisir_un_projet_qui_n_existe_pas(client, espace_projets):
    # La liste déroulante n'accepte que les projets existants.
    reponse = client.post("/accueil_projet", data={"projet_existant": "inexistant"})
    assert reponse.status_code == 302
    assert not (espace_projets / "projects" / "inexistant").exists()


@pytest.mark.parametrize(
    "nom",
    ["../evasion", "sous/dossier", ".", "..", "nom avec espaces", "nom\nretour"],
    ids=["remonte_d_un_niveau", "slash", "point", "point_point", "espaces", "retour_ligne"],
)
def test_creer_un_projet_nom_invalide(client, espace_projets, nom):
    reponse = client.post("/accueil_projet", data={"nom": nom})

    assert reponse.status_code == 302  # renvoyé vers l'accueil avec un message
    assert not (espace_projets / "evasion").exists()
    assert list((espace_projets / "projects").iterdir()) == []
    assert not (espace_projets / "config.json").exists()


def test_accueil_projet_sans_projet_courant_redirige(client, espace_projets):
    reponse = client.get("/accueil_projet")
    assert reponse.status_code == 302
    assert reponse.headers["Location"] == "/"


def test_retour_au_projet_affiche_le_projet_courant(client, espace_projets):
    # Lien « Retour au projet » (simple GET) : on retrouve la page du projet courant.
    create_project_structure("test_pipeline")
    definir_projet_courant(espace_projets, "test_pipeline")

    reponse = client.get("/accueil_projet")

    assert reponse.status_code == 200
    assert "test_pipeline" in reponse.get_data(as_text=True)


def test_retour_au_projet_supprime_redirige(client, espace_projets):
    # Le projet courant n'existe plus (supprimé à la main) : retour à l'accueil.
    definir_projet_courant(espace_projets, "projet_disparu")
    reponse = client.get("/accueil_projet")
    assert reponse.status_code == 302
    assert not (espace_projets / "projects" / "projet_disparu").exists()


# --- Gestion des modèles --------------------------------------------------------------

def test_gestion_modeles_sans_modele(client, espace_projets):
    create_project_structure("manuscrits")
    definir_projet_courant(espace_projets, "manuscrits")
    assert client.get("/gestion_modeles").status_code == 200


def test_gestion_modeles_affiche_le_lien_de_telechargement(client, espace_projets):
    create_project_structure("manuscrits")
    definir_projet_courant(espace_projets, "manuscrits")
    poids = espace_projets / "output" / "train" / "manuscrits_20260301_101500" / "weights"
    poids.mkdir(parents=True)
    (poids / "best.pt").write_bytes(b"")
    pd.DataFrame({"model_name": ["manuscrits_20260301_101500"]}).to_csv(
        espace_projets / "projects" / "manuscrits" / "manuscrits.csv", index=False)

    page = client.get("/gestion_modeles").get_data(as_text=True)

    assert "Download" in page
    assert "Erreur avec le modèle" not in page


# --- Suppression ----------------------------------------------------------------------

def test_suppression_refuse_un_simple_lien(client, espace_projets):
    # Visiter l'URL (lien, historique, préchargement) ne doit rien supprimer.
    create_project_structure("manuscrits")
    definir_projet_courant(espace_projets, "manuscrits")

    reponse = client.get("/suppression_projet")

    assert reponse.status_code == 405  # « méthode non autorisée »
    assert (espace_projets / "projects" / "manuscrits").is_dir()


def test_suppression_supprime_le_projet_courant(client, espace_projets):
    create_project_structure("manuscrits")
    create_project_structure("incunables")
    definir_projet_courant(espace_projets, "manuscrits")

    reponse = client.post("/suppression_projet")

    assert reponse.status_code == 302
    assert not (espace_projets / "projects" / "manuscrits").exists()
    assert (espace_projets / "projects" / "incunables").is_dir()
    # Le projet supprimé n'est plus le projet courant.
    assert projet_courant(espace_projets) == ""


def test_suppression_sans_projet_courant_ne_supprime_rien(client, espace_projets):
    # Cas du config.json absent ou vide : load_current_project() renvoie "".
    create_project_structure("manuscrits")
    (espace_projets / "data" / "manuscrits").mkdir(parents=True)

    reponse = client.post("/suppression_projet")

    assert reponse.status_code == 302
    assert (espace_projets / "projects" / "manuscrits").is_dir()
    assert (espace_projets / "data" / "manuscrits").is_dir()


def test_suppression_exige_le_jeton_csrf(app, client, espace_projets):
    # Avec la protection CSRF active (comme dans l'app réelle), une requête POST
    # venue d'ailleurs que du formulaire de confirmation ne supprime rien.
    app.config["WTF_CSRF_ENABLED"] = True
    create_project_structure("manuscrits")
    definir_projet_courant(espace_projets, "manuscrits")

    client.post("/suppression_projet")

    assert (espace_projets / "projects" / "manuscrits").is_dir()


def test_bouton_de_suppression_envoie_un_formulaire(client, espace_projets):
    # Le bouton « Supprimer définitivement » (partials/corpus.html) doit envoyer un POST, pas suivre un lien.
    page = client.post("/accueil_projet", data={"nom": "manuscrits"}).get_data(as_text=True)
    assert 'action="/suppression_projet"' in page
    assert 'href="/suppression_projet"' not in page

def test_suppression_supprime_aussi_les_projets_label_studio(client, espace_projets, mocker):
    suppression_ls = mocker.patch.object(project_route, "delete_ls_projects", return_value=["manuscrits"])
    create_project_structure("manuscrits")
    definir_projet_courant(espace_projets, "manuscrits")

    client.post("/suppression_projet")

    suppression_ls.assert_called_once_with("manuscrits")

def test_suppression_si_label_studio_n_est_pas_lance(client, espace_projets, mocker):
    # Le projet TiamaT est supprimé quand même, et l'utilisateur est prévenu.
    mocker.patch.object(project_route, "delete_ls_projects", side_effect=RuntimeError("Label Studio n'est pas lancé"))
    flash = mocker.spy(project_route, "flash")
    create_project_structure("manuscrits")
    definir_projet_courant(espace_projets, "manuscrits")

    reponse = client.post("/suppression_projet")

    assert reponse.status_code == 302
    assert not (espace_projets / "projects" / "manuscrits").exists()
    assert "Label Studio n'est pas lancé" in flash.call_args.args[0]


def test_suppression_refusee_ne_touche_pas_label_studio(client, espace_projets, mocker):
    # Sans projet courant valide, rien n'est supprimé, ni dans TiamaT ni dans Label Studio.
    suppression_ls = mocker.patch.object(project_route, "delete_ls_projects")
    client.post("/suppression_projet")
    suppression_ls.assert_not_called()