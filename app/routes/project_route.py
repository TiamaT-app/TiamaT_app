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

from pathlib import Path

from flask import Blueprint , render_template, flash, redirect, url_for

from app.app import app
from ..src.models.formulaires import NomDuProjet, ImportImages

from app.services.config_service import load_current_project, save_config
from app.services.project_service import list_existing_projects, create_project_structure
from app.services.image_service import get_preview_images

project_bp = Blueprint("project", __name__)

# Route de l'accueil  pour le choix du projet
@project_bp.route("/",methods=['GET', 'POST'])
def accueil():
    # pour marquer qu'on est pas en suite d'une passe, utile dans la route suivante
    app.config["SECOND_PASS"] = False
    form = NomDuProjet()

    liste_projets = list_existing_projects()

    # Affiche le nom des dossiers existants 
    choices = [('', '-- Sélectionner un projet --')]+[(projet,projet) for projet in liste_projets]
    form.projet_existant.choices = choices # type: ignore[assignment]
    return render_template("/pages/accueil.html" , form = form, liste_projets=liste_projets)
   

# Route qui permet le chargement des images + affichages de celles déjà chargées
@project_bp.route("/accueil_projet",methods=['GET', 'POST'])
def accueil_projet():
    form = NomDuProjet()
    # Liste des dossiers qui sont dans l'application de base
    liste_projets = list_existing_projects()
    
    # Affiche le nom des dossiers existants => pourquoi on fait ça ici?
    choices = [('', '-- Sélectionner un projet --')]+[(projet,projet) for projet in liste_projets]  
    form.projet_existant.choices = choices # type: ignore[assignment]
    
    if not app.config["SECOND_PASS"]:
        # Si il y a une donnée dans "nom", on check que ce dossier existe pas déjà : 
        if form.validate_on_submit() and len(str(form.nom.data)) > 0:
                project_name = form.nom.data
                if project_name in liste_projets:
                    return render_template("/pages/erreur_nom_projet.html", project_name=project_name)
                else:
                    save_config(project_name=project_name)
        
        #s'il y a une donnée mais pas dans nom, on prend le nom de projet déjà existant du formulaire
        elif form.validate_on_submit() and form.projet_existant.data:
                project_name = form.projet_existant.data
                save_config(project_name=project_name)
        else:
            print(form.errors)
            flash("Merci de renseigner un nom de projet ou de choisir un projet existant.")
            return redirect(url_for('project.accueil'))
  
        
    if app.config["SECOND_PASS"]:
        project_name = load_current_project()
    
    project_path = create_project_structure(project_name)
    
    #ci desssous si le projet existe déjà on sort ses images pour les afficher.
    chemin_images = project_path / "image_inputs" / "ground_truth_images"
    liste_images, liste_noms, nbre_images = get_preview_images(chemin_images)
    

    chemin_labels = project_path / "annotations" / "ground_truth"
    form2 = ImportImages()
        
    return render_template(
        "/pages/import_images.html", 
        project_name=project_name,
        form2=form2,
        form=form,
        liste_images=liste_images,
        liste_noms=liste_noms,
        nbre_images=nbre_images,
        chemin_images=chemin_images,
        chemin_labels=chemin_labels)
