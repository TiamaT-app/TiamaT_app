"""
Route de lancement de Label Studio pour le projet courant.
 
LOCAL_FILES_DOCUMENT_ROOT est fixe sur projects_folder (le parent de TOUS les
projets), pas sur le dossier du projet courant -- cette racine commune est ce
qui permettra plus tard de piloter Label Studio via son API (creation de
projet, configuration du storage) sans avoir a relancer le serveur a chaque
changement de projet.
"""
import os
from pathlib import Path
from threading import Thread
from flask import Blueprint, render_template, redirect, url_for, flash

from ..src.scripts.get_training_data import clean_image_name
from ..src.scripts.run_label_studio import launch_LS
from ..src.models.formulaires import LsToken


from app.services.config_service import load_current_project
from app.services.project_service import projects_folder
from app.services.label_studio_service import configure_label_studio_root, launch_label_studio_async
from app.services.label_studio_api_service import get_ls_url
 

labeling_bp = Blueprint("labeling", __name__)

def _setup_ls_in_background(project_name: str, chemin_images, chemin_labels) -> None:
    """
    Orchestre le lancement de LS et la configuration du projet en arriere-plan.
    Met a jour la variable d'etat LS_READY une fois termine.
    """

    from app.services.label_studio_api_service import setup_label_studio_project
    labels_file = projects_folder / project_name / "data" / "labels.txt"

    try:
        print(f"DEBUG token : '{os.environ.get('LABEL_STUDIO_API_TOKEN')}'")
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

@labeling_bp.route("/label_lancement",methods=['GET', 'POST'])
def consignes() :
    if os.environ.get('LABEL_STUDIO_API_TOKEN') is None:
        launch_LS()
        form = LsToken()
        return render_template("/pages/ajout_ls_token.html", form=form)
    else:
        project_name = load_current_project()
        chemin_images = projects_folder / project_name / "image_inputs" / "ground_truth_images"
        chemin_labels = projects_folder / project_name / "annotations" / "ground_truth"

        clean_image_name(str(projects_folder / project_name))
        configure_label_studio_root()
        
        os.environ["LS_READY"] = "False"

        def on_ls_ready():
            _setup_ls_in_background(project_name, chemin_images, chemin_labels)
        
        launch_label_studio_async(on_ready=on_ls_ready)

        os.environ["LS_READY"] = "False"
        thread = Thread(
            target=_setup_ls_in_background,
            args=(project_name, chemin_images, chemin_labels)
        )
        thread.daemon = True
        thread.start()
    
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
    chemin_images = projects_folder / project_name / "image_inputs" / "ground_truth_images"
    chemin_labels = projects_folder / project_name / "annotations" / "ground_truth"
    return render_template("/pages/lancement_label_studio.html", 
                           project_name = project_name,
                           chemin_images = chemin_images,
                           chemin_labels = chemin_labels)


@labeling_bp.route("/ls_ready")
def ls_ready():
    project_name = load_current_project()
    ls_url = os.environ.get("LS_URL_FOUND", get_ls_url())
    chemin_images = projects_folder / project_name / "image_inputs" / "ground_truth_images"
    chemin_labels = projects_folder / project_name / "annotations" / "ground_truth"
    
    return render_template("/pages/label_studio_pret.html",
                           project_name=project_name,
                           chemin_images=chemin_images,
                           chemin_labels=chemin_labels,
                           ls_url=ls_url)

@labeling_bp.route("/token_added", methods=['GET', 'POST'])
def add_token():
    form = LsToken()
    if form.validate_on_submit:
        with open(Path(".env"), 'a') as f:
            f.write(f'\nLABEL_STUDIO_API_TOKEN={form.token.data}')
            flash("Personnal API Token bien ajouté.")
            os.environ["LABEL_STUDIO_API_TOKEN"] = form.token.data 
            return redirect(url_for('labeling.consignes'))
    else:
        flash("Erreur du formulaire.")
        return redirect(url_for('labeling.consgines'))
