"""
Routes de gestion des projets TiamaT.
 
Ce fichier regroupe les deux pages liees au choix/creation d'un projet :
- accueil()        : page d'accueil, choix d'un projet existant ou creation
                      d'un nouveau projet (route "/")
- accueil_projet()  : page d'import d'images pour le projet selectionne/cree
                      (route "/accueil_projet"), qui cree aussi l'arborescence
                      de dossiers du projet si elle n'existe pas encore.
 
Tous les projets vivent sous projects_folder (projects/{nom_du_projet}/),
distinct du dossier "project" (singulier) qui est le squelette/template
historique, non utilise dans ce fichier.
"""

import os
import json
import random
from pathlib import Path

from flask import Blueprint , render_template

from app.app import app
from app.config import Config
from ..src.models.formulaires import NomDuProjet, ImportImages

project_bp = Blueprint("project", __name__)


# NB : generales.py avait UNE SEULE copie de ce dict, partagee par toutes les
# routes (training, testing, etc.). Ce fichier en a sa propre copie pour
# l'instant -- on la centralisera dans config_service.py.
config_dict = {"CURRENT_PROJECT_NAME": "", "LAST_MODEL_PATH": ""}

projects_folder = Path.cwd() / "projects"
projects_folder.mkdir(exist_ok=True)

def list_existing_projects():
    """Renvoie les noms (str) des projets existants sous projects_folder."""
    return [
          x.name for x in projects_folder.iterdir()
          if x.is_dir() and not x.name.startswith('.')
          ]

# Route de l'accueil  pour le choix du projet
@project_bp.route("/",methods=['GET', 'POST'])
def accueil():
    # pour marquer qu'on est pas en suite d'une passe, utile dans la route suivante
    app.config["SECOND_PASS"] = False
    form = NomDuProjet()

    liste_projets = list_existing_projects()

    # Affiche le nom des dossiers existants 
    choices = [('', '-- Sélectionner un projet --')]+[(projet,projet) for projet in liste_projets]
    form.projet_existant.choices = choices
    return render_template("/pages/accueil.html" , form = form, liste_projets=liste_projets)
   

# Route qui permet le chargement des images + affichages de celles déjà chargées
@project_bp.route("/accueil_projet",methods=['GET', 'POST'])
def accueil_projet():
    form = NomDuProjet()
    # Liste des dossiers qui sont dans l'application de base
    liste_projets = list_existing_projects()
    
    # Affiche le nom des dossiers existants 
    choices = [('', '-- Sélectionner un projet --')]+[(projet,projet) for projet in liste_projets]  
    form.projet_existant.choices = choices
    
    if not app.config["SECOND_PASS"]:
        # Si il y a une donnée dans "nom", on check que ce dossier existe pas déjà : 
        if form.validate_on_submit() and form.nom.data:
                project_name = form.nom.data
                liste_noms = []
                if project_name in liste_projets:
                    return render_template("/pages/erreur_nom_projet.html", project_name=project_name)
                else:
                    config_dict['CURRENT_PROJECT_NAME'] = project_name
                    jsonstr = json.dumps(config_dict)
                    with open ("config.json","w") as f:
                        f.write(jsonstr)
                    app.config['CURRENT_PROJECT_NAME'] = project_name
                
        #s'il y a une donnée mais pas dans nom, on prend le nom de projet déjà existant du formulaire
        elif form.validate_on_submit():
                project_name = form.projet_existant.data
                config_dict['CURRENT_PROJECT_NAME'] = project_name
                jsonstr = json.dumps(config_dict)
                with open("config.json","w") as f:
                        f.write(jsonstr)
                app.config['CURRENT_PROJECT_NAME'] = project_name     
        else:
            print(form.errors)
            return render_template("/pages/erreur_nom_projet2.html", project_name="erreur")
  
        
    project_name = app.config['CURRENT_PROJECT_NAME']
    new_project = projects_folder / project_name
    
    # on check si le projet existe déjà 
    if not new_project.is_dir():
        gt_img_folder = new_project / "image_inputs" / "ground_truth_images"
        eval_img_folder = new_project / "image_inputs" / "eval_images"
        gt_ann_folder = new_project / "annotations" / "ground_truth"
        eval_ann_foler = new_project / "annotations" / "prediction_corrections"
        
        gt_img_folder.mkdir(parents=True, exist_ok=True)
        eval_img_folder.mkdir(parents=True, exist_ok=True)
        gt_ann_folder.mkdir(parents=True, exist_ok=True)
        eval_ann_foler.mkdir(parents=True, exist_ok=True)
        
    
    #ci desssous si le projet existe déjà on sort ses images pour les afficher.
    img_ext = {'.jpg', '.jpeg', '.png'}
    gt_img_folder = new_project / "image_inputs" / "ground_truth_images"
    liste_images = [i for i in gt_img_folder.iterdir() if i.is_file() 
                    and i.suffix.lower()in img_ext
                    and not i.name.startswith('.')]


    random.shuffle(liste_images)    
    nbre_images =(len(liste_images))
    
    
    if len(liste_images)>= 8 :
        liste_images= liste_images[:8]
        liste_noms = [Path(img).name for img in liste_images]
        liste_images=[i.relative_to(Path.cwd()).as_posix() for i in liste_images]

    elif len(liste_images)== 0:
        liste_images = "Pas encore d'images uploadées"
        liste_noms=[]
    else :
        liste_noms = [Path(img).name for img in liste_images]
        liste_images=[Path(i).relative_to(Path.cwd()).as_posix() for i in liste_images]
        pass
    
    chemin_images = gt_img_folder
    chemin_labels = new_project / "annotations" / "ground_truth"
    form2=ImportImages()
        
    return render_template(
        "/pages/import_images.html", 
        project_name=project_name,
        form2 = form2,
        form= form,
        liste_images= liste_images,
        liste_noms=liste_noms,
        nbre_images=nbre_images,
        chemin_images= chemin_images,
        chemin_labels=chemin_labels)
