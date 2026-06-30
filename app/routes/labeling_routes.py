"""
Route de lancement de Label Studio pour le projet courant.
 
LOCAL_FILES_DOCUMENT_ROOT est fixe sur projects_folder (le parent de TOUS les
projets), pas sur le dossier du projet courant -- cette racine commune est ce
qui permettra plus tard de piloter Label Studio via son API (creation de
projet, configuration du storage) sans avoir a relancer le serveur a chaque
changement de projet.
"""

from flask import Blueprint, render_template

from app.app import app

from ..src.scripts.run_label_studio import launch_LS
from ..src.scripts.get_training_data import clean_image_name

from app.services.config_service import load_current_project
from app.services.project_service import projects_folder
from app.services.label_studio_service import configure_label_studio_root, launch_label_studio_async

labeling_bp = Blueprint("labeling", __name__)

@labeling_bp.route("/label_lancement",methods=['GET', 'POST'])
def consignes() :
    
    project_name = load_current_project()
    
    chemin_images = projects_folder / project_name / "image_inputs" / "ground_truth_images"
    chemin_labels = projects_folder / project_name / "annotations" / "ground_truth"

    clean_image_name(str(projects_folder / project_name))
    
    configure_label_studio_root()
    launch_label_studio_async()
   
    #pour la future extinction automatique de LS voir cette page : https://stackoverflow.com/questions/31712056/how-do-i-get-a-threads-pid
    return render_template("/pages/lancement_label_studio.html", project_name=project_name, chemin_images=chemin_images, chemin_labels=chemin_labels)