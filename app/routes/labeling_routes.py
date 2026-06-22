import os
import json
from pathlib import Path
from threading import Thread

from app.app import app

from flask import Blueprint, render_template

from ..src.scripts.run_label_studio import launch_LS
from ..src.scripts.get_training_data import clean_image_name

labeling_bp = Blueprint("labeling", __name__)

@labeling_bp.route("/label_lancement",methods=['GET', 'POST'])
def consignes() :
    root = Path.cwd()
    with open("config.json","r") as f:
        app.config["CURRENT_PROJECT_NAME"]=json.load(f).get("CURRENT_PROJECT_NAME")
    project_name = app.config['CURRENT_PROJECT_NAME']

    chemin_images = os.path.join(root,project_name, "image_inputs", "ground_truth_images")
    chemin_labels =os.path.join(root,project_name, "annotations", "ground_truth")
    

    clean_image_name(os.path.join(root, project_name))
    os.environ['LOCAL_FILES_DOCUMENT_ROOT'] = f'{root.as_posix()}'
    os.environ['LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED'] = 'true'
    #ouverture de label studio en fond (thread), affichage de la page en même temps. 
    thread = Thread(target=launch_LS)
    thread.daemon = True
    thread.start()
   
    #pour la future extinction automatique de LS voir cette page : https://stackoverflow.com/questions/31712056/how-do-i-get-a-threads-pid
    return render_template("/pages/lancement_label_studio.html", project_name=project_name, chemin_images=chemin_images, chemin_labels=chemin_labels)

    