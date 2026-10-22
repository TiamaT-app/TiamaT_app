"""
Routes des statistiques du dataset du projet courant.

- dataset_statistics()  : prepare le dataset (data/) a partir des annotations, calcule
                          les statistiques et affiche la distribution des classes
- serve_dataset_image() : renvoie le graphique de distribution des classes
"""
from pathlib import Path

from flask import Blueprint, render_template, send_file, url_for, flash, redirect

from ..src.scripts.get_training_data import create_dataset
from ..src.scripts.statistics_for_training import (create_stats_folder, encoding, annotations_per_img,
                                                    classes_distribution, get_global_results)

from ..services.config_service import load_current_project
from ..services.project_service import projects_folder, list_existing_projects

statistics_bp = Blueprint("statistics", __name__)


@statistics_bp.route("/dataset_statistics", methods=['GET', 'POST'])
def dataset_statistics():
    project_name = load_current_project()
    # Sans projet valide, le dossier data/ serait calculé à partir de projects/ lui-même
    if project_name not in list_existing_projects():
        flash("Merci de choisir un projet avant de calculer ses statistiques.")
        return redirect(url_for('project.accueil'))
    project_folder = projects_folder / project_name

    try:
        create_dataset(project_folder, manually_downloaded=False)
        print("dataset created")

        create_stats_folder(project_folder)
        encoding(project_folder)
        annotations_per_img(project_folder)
        classes_distribution(project_folder)
        print("class distributon done")

        get_global_results(project_folder)
        print("got global results")
    except (ValueError, FileNotFoundError) as e:
        # Par exemple une classe absente de labels.txt : le message indique laquelle et dans quel fichier
        print(f"Erreur lors de la préparation du dataset : {e}")
        flash(f"Impossible de préparer le dataset : {e}")
        return redirect(url_for('project.accueil_projet'))

    return render_template(
        "/pages/class_distrib.html",
        class_distibution_path=url_for('statistics.serve_dataset_image'),
        project_name=project_name)


@statistics_bp.route('/dataset_stats')
def serve_dataset_image():
    project_name = load_current_project()
    file_path = Path.cwd() / "data" / project_name / "dataset_statistics" / 'class_distribution.png'

    if project_name and file_path.exists():
        # max_age=0 : le navigateur ne doit pas réafficher le graphique d'un calcul précédent
        return send_file(file_path, mimetype='image/png', max_age=0)
    return "File not found", 404