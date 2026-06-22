"""
Routes de gestion des projets TiamaT : choix/creation d'un projet (page
d'accueil) et page d'import d'images pour ce projet.
"""

import os
import json
import shutil
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


# Dossiers internes a l'application, a exclure de la liste des projets utilisateur
liste_dossiers = ["data", "output", "project", "app", "docs", "tiamat_env"]

# Route de l'accueil  pour le choix du projet
@project_bp.route("/",methods=['GET', 'POST'])
def accueil():
    print("ROUTE PROJECT_BP")
    # pour marquer qu'on est pas en suite d'une passe, utile dans la route suivante 
   
    app.config["SECOND_PASS"] = False
    p = Path.cwd()
    form = NomDuProjet()
    
    # Liste des dossiers qui sont dans l'application de base
    liste_dossiers = ["data", "output", "project", "app", "docs", "tiamat_env"]
    liste_projets = [
        x for x in os.listdir(p)
        if os.path.isdir(os.path.join(p, x)) and not x.startswith('.') and x not in liste_dossiers
    ]

    # IL FAUT UN TUPLE NOM AFFICHE + VALEUR dans les choices
    choices = [('', '-- Sélectionner un projet --')]+[(projet,projet) for projet in liste_projets]
    form.projet_existant.choices = choices
    return render_template("/pages/accueil.html" , form = form, liste_projets=liste_projets)
   
# Route qui permet le chargement des images + affichages de celles déjà chargées
@project_bp.route("/accueil_projet",methods=['GET', 'POST'])

def accueil_projet():
    p = Path.cwd()
    form = NomDuProjet()
    # Cf. route précédente
    liste_dossiers = ["data", "output", "project", "app", "docs", "tiamat_env"]
    liste_projets = [
        x for x in os.listdir(p)
        if os.path.isdir(os.path.join(p, x)) and not x.startswith('.') and x not in liste_dossiers
    ]
    
    # IL FAUT UN TUPLE NOM AFFICHE + VALEUR dans les choices
    choices = [('', '-- Sélectionner un projet --')]+[(projet,projet) for projet in liste_projets]  
    form.projet_existant.choices = choices
    
    if not app.config ["SECOND_PASS"]:
        # Si il y a une donnée dans "nom", on check que ce dossier existe pas déjà : 
        if form.validate_on_submit() and form.nom.data:
                project_name= form.nom.data
                liste_noms=[]
                if project_name in os.listdir(p):
                    return render_template("/pages/erreur_nom_projet.html", project_name=project_name)
                else:
                    config_dict['CURRENT_PROJECT_NAME']=project_name
                    jsonstr = json.dumps(config_dict)
                    with open ("config.json","w") as f:
                        f.write(jsonstr)
                    app.config['CURRENT_PROJECT_NAME']= project_name
                    pass
                
        #s'il y a une donnée mais pas dans nom, on prend le nom de projet déjà existant du formulaire
        elif form.validate_on_submit():
                project_name= form.projet_existant.data
                config_dict['CURRENT_PROJECT_NAME']=project_name
                jsonstr = json.dumps(config_dict)
                with open ("config.json","w") as f:
                        f.write(jsonstr)
                app.config['CURRENT_PROJECT_NAME']= project_name     
        else:
            print(form.errors)
            return render_template("/pages/erreur_nom_projet2.html", project_name="erreur")
    else:
        pass    
        
    project_name = app.config['CURRENT_PROJECT_NAME']
    
    
    #on check si le projet existe déjà 
    if (Path.cwd() / project_name).is_dir():
        liste_images=[]
        pass
    else:
        shutil.copytree(Path.cwd() / 'project', project_name)
    
    #ci desssous si le projet existe déjà on sort ses images pour les afficher.
    p_img_gt = Path.cwd()/project_name/"image_inputs"/"ground_truth_images"
    liste_images = [i for i in p_img_gt.iterdir()]
    
    img_ext = {'.jpg', '.jpeg', '.png'}
    
    liste_images = [i for i in liste_images if i.is_file() and i.suffix.lower() in img_ext]
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
    
    root = os.getcwd()
    chemin_images = os.path.join(root,project_name, "image_inputs", "ground_truth_images")
    chemin_labels =os.path.join(root,project_name, "annotations", "ground_truth")
    form2=ImportImages()
        
    return render_template("/pages/import_images.html", project_name=project_name, form2 = form2, form= form, liste_images= liste_images, liste_noms=liste_noms, nbre_images=nbre_images, chemin_images= chemin_images, chemin_labels=chemin_labels)
