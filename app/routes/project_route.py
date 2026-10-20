"""
Routes de gestion des projets TiamaT.
 
- accueil()            : page d'accueil, choix d'un projet existant ou creation
                         d'un nouveau projet (route "/")
- accueil_projet()     : page d'import d'images pour le projet selectionne/cree
                         (route "/accueil_projet"), qui cree aussi l'arborescence
                         de dossiers du projet si elle n'existe pas encore.
- gestion_modeles()    : liste des modeles du projet courant, avec lien de telechargement
- suppression_projet() : suppression du projet courant (POST uniquement)
 
Tous les projets vivent sous projects_folder (projects/{nom_du_projet}/).
"""


from pathlib import Path
import os
import shutil
import re

from flask import Blueprint , render_template, flash, redirect, url_for
import pandas as pd

from app.app import app
from ..src.models.formulaires import NomDuProjet, ImportImages, ImportModel, ConfirmationSuppression

from app.services.config_service import load_current_project, save_config
from app.services.project_service import list_existing_projects, create_project_structure, delete_project
from app.services.image_service import get_preview_images


project_bp = Blueprint("project", __name__)

def _project_form() -> NomDuProjet:
    """Formulaire de choix/creation de projet, avec la liste des projets existants."""
    form = NomDuProjet()

    # Affiche le nom des dossiers existants 
    choices = [('', '-- Sélectionner un projet --')] + [(projet, projet) for projet in list_existing_projects()]
    form.projet_existant.choices = choices  # type: ignore[assignment]
    
    return form


# Route de l'accueil  pour le choix du projet
@project_bp.route("/",methods=['GET', 'POST'])
def accueil():
    # pour marquer qu'on est pas en suite d'une passe, utile dans la route suivante
    app.config["SECOND_PASS"] = False
    form = _project_form()
    return render_template("/pages/accueil.html" , form = form, liste_projets=list_existing_projects())
   

# Route qui permet le chargement des images + affichages de celles déjà chargées
@project_bp.route("/accueil_projet",methods=['GET', 'POST'])
def accueil_projet():
    form = _project_form()
    
    if not form.validate_on_submit():
        for errors in form.errors.values():
            for error in errors:
                flash(error)
        if not form.errors:
            flash("Merci de renseigner un nom de projet ou de choisir un projet existant.")
        return redirect(url_for('project.accueil'))
 
    # Création d'un nouveau projet : on vérifie que ce nom n'est pas déjà pris
    if form.nom.data:
        project_name = form.nom.data
        if project_name in list_existing_projects():
            return render_template("/pages/erreur_nom_projet.html", project_name=project_name)
    # Sinon, sélection d'un projet existant dans la liste déroulante
    elif form.projet_existant.data:
        project_name = form.projet_existant.data
    else:
        flash("Merci de renseigner un nom de projet ou de choisir un projet existant.")
        return redirect(url_for('project.accueil'))
 
    save_config(project_name=project_name)
    project_path = create_project_structure(project_name)
 
    # Si le projet existe déjà, on affiche un aperçu de ses images
    chemin_images = project_path / "image_inputs" / "ground_truth_images"
    liste_images, liste_noms, nbre_images = get_preview_images(chemin_images)
    chemin_labels = project_path / "annotations" / "ground_truth"
 
    return render_template(
        "/pages/import_images.html",
        project_name=project_name,
        form2=ImportImages(),
        form=form,
        liste_images=liste_images,
        liste_noms=liste_noms,
        nbre_images=nbre_images,
        chemin_images=chemin_images,
        chemin_labels=chemin_labels)

@project_bp.route("/gestion_modeles", methods=['GET', 'POST'])
def gestion_modeles():
    project_name = load_current_project()
    form = ImportModel()
    csv_file = Path("projects") / project_name / f"{project_name}.csv"
 
    if not csv_file.is_file():
        return render_template("pages/gestion_modele2.html", form=form)
 
    df = pd.read_csv(csv_file)
    liste_urls = []
    for _, row in df.iterrows():
        path_model = Path.cwd() / "output" / "train" / row["model_name"] / "weights" / "best.pt"
        if os.path.isfile(path_model):
            liste_urls.append(f'<a href="/download/{path_model}" class="btn btn-primary btn-sm">Download</a>')
        else:
            liste_urls.append(f"Erreur avec le modèle {row['model_name']}")
    df["urls"] = liste_urls
    return render_template("pages/gestion_modeles.html", tables=[df.to_html(classes='data', escape=False)],
                           titles=df.columns.values, nom_projet=project_name, form=form)
 
 
 
# POST uniquement : une suppression ne doit jamais pouvoir se déclencher
# en visitant simplement l'URL (lien, historique, préchargement du navigateur).
@project_bp.route("/suppression_projet", methods=["POST"])
def suppression_projet():
    form = ConfirmationSuppression()
    if not form.validate_on_submit():
        flash("La demande de suppression n'a pas pu être vérifiée, merci de réessayer.")
        return redirect(url_for('project.accueil'))
 
    project_name = load_current_project()
    try:
        delete_project(project_name)
    except ValueError:
        flash("Aucun projet valide à supprimer.")
        return redirect(url_for('project.accueil'))
 
    # Le projet supprimé ne doit plus être le projet courant
    save_config(project_name="")
    flash(f"Projet {project_name} supprimé.")
    return redirect(url_for('project.accueil'))
