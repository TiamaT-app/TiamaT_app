"""
Route de lancement de Label Studio pour le projet courant.

LOCAL_FILES_DOCUMENT_ROOT est fixe sur projects_folder (le parent de TOUS les
projets), pas sur le dossier du projet courant -- cette racine commune est ce
qui permettra plus tard de piloter Label Studio via son API (creation de
projet, configuration du storage) sans avoir a relancer le serveur a chaque
changement de projet.

L'etat du lancement est transmis entre le thread de fond et les routes par
des variables d'environnement : LS_READY ("False", "True" ou "Error") et LS_URL_FOUND.
"""
import os
from pathlib import Path
from threading import Thread
from flask import Blueprint, render_template, redirect, url_for, flash

from ..src.scripts.get_training_data import clean_image_name
from ..src.scripts.run_label_studio import launch_LS2
from ..src.models.formulaires import LsToken


from app.services.config_service import load_current_project
from app.services.project_service import projects_folder, list_existing_projects
from app.services.label_studio_service import configure_label_studio_root, launch_label_studio_async
from app.services.label_studio_api_service import get_ls_url


labeling_bp = Blueprint("labeling", __name__)

ENV_FILE = Path(".env")
TOKEN_VARIABLE = "LABEL_STUDIO_API_TOKEN"


def _project_paths(project_name: str) -> tuple[Path, Path]:
    """Dossiers des images et des annotations de verite terrain du projet."""
    project_folder = projects_folder / project_name
    return (project_folder / "image_inputs" / "ground_truth_images",
            project_folder / "annotations" / "ground_truth")


def _setup_ls_in_background(project_name: str, chemin_images, chemin_labels) -> None:
    """
    Orchestre le lancement de LS et la configuration du projet en arriere-plan.
    Met a jour la variable d'etat LS_READY une fois termine.
    """

    from app.services.label_studio_api_service import setup_label_studio_project
    labels_file = Path.cwd() / "data" / project_name / "labels.txt"

    try:
        _, ls_url = setup_label_studio_project(
            project_name=project_name,
            chemin_images=chemin_images,
            chemin_labels=chemin_labels,
            labels_file=labels_file,
        )
        os.environ["LS_URL_FOUND"] = ls_url
        os.environ["LS_READY"] = "True"

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Erreur lors du setup Label Studio : {e}")
        os.environ["LS_READY"] = "Error"


@labeling_bp.route("/label_lancement", methods=['GET', 'POST'])
def consignes():
    if os.environ.get(TOKEN_VARIABLE) is None:
        thread = Thread(target=launch_LS2)
        thread.daemon = True
        thread.start()
        form = LsToken()
        return render_template("/pages/ajout_ls_token.html", form=form)

    project_name = load_current_project()
    # Sans projet valide, les chemins pointeraient sur projects/ lui-même
    if project_name not in list_existing_projects():
        flash("Merci de choisir un projet avant de lancer Label Studio.")
        return redirect(url_for('project.accueil'))

    chemin_images, chemin_labels = _project_paths(project_name)

    clean_image_name(str(projects_folder / project_name))
    configure_label_studio_root()

    os.environ["LS_READY"] = "False"

    def on_ls_ready():
        _setup_ls_in_background(project_name, chemin_images, chemin_labels)

    launch_label_studio_async(on_ready=on_ls_ready)

    #pour la future extinction automatique de LS voir cette page : https://stackoverflow.com/questions/31712056/how-do-i-get-a-threads-pid
    return render_template(
        "/pages/lancement_label_studio.html",
        project_name=project_name,
        chemin_images=chemin_images,
        chemin_labels=chemin_labels)


@labeling_bp.route("/check_ls_status")
def check_ls_status():
    status = os.environ.get("LS_READY", "False")
    if status == "True":
        return redirect(url_for('labeling.ls_ready'))
    elif status == "Error":
        flash("Une erreur est survenue lors du lancement de Label Studio. Réessayez.")
        return redirect(url_for('project.accueil_projet'))

    project_name = load_current_project()
    chemin_images, chemin_labels = _project_paths(project_name)
    return render_template("/pages/lancement_label_studio.html",
                           project_name=project_name,
                           chemin_images=chemin_images,
                           chemin_labels=chemin_labels)


@labeling_bp.route("/ls_ready")
def ls_ready():
    project_name = load_current_project()
    # « or » plutôt qu'une valeur par défaut de .get() : get_ls_url() n'est appelée qu'en cas de besoin
    ls_url = os.environ.get("LS_URL_FOUND") or get_ls_url()
    chemin_images, chemin_labels = _project_paths(project_name)

    return render_template("/pages/label_studio_pret.html",
                           project_name=project_name,
                           chemin_images=chemin_images,
                           chemin_labels=chemin_labels,
                           ls_url=ls_url)


def _save_token(token: str) -> None:
    """
    Enregistre le token dans le .env, en remplacant un eventuel ancien token
    et en conservant les autres lignes du fichier.
    """
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    lines = [line for line in lines if not line.startswith(f"{TOKEN_VARIABLE}=")]
    lines.append(f"{TOKEN_VARIABLE}={token}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


@labeling_bp.route("/token_added", methods=['GET', 'POST'])
def add_token():
    form = LsToken()
    if not form.validate_on_submit():
        flash("Le token n'a pas pu être enregistré : vérifiez qu'il est complet, sans espace ni retour à la ligne.")
        return redirect(url_for('labeling.consignes'))

    _save_token(form.token.data)
    os.environ[TOKEN_VARIABLE] = form.token.data
    flash("Personnal API Token bien ajouté.")
    return redirect(url_for('labeling.consignes'))