from pathlib import Path

from flask import Blueprint, render_template, send_file, url_for

from ..src.scripts.get_training_data import *
from ..src.scripts.statistics_for_training import *

from ..services.config_service import load_current_project

statistics_bp = Blueprint("statistics", __name__)

projects_folder = Path.cwd() / "projects"    

@statistics_bp.route("/dataset_statistics", methods=['GET', 'POST'])
def dataset_statistics():
    project_name = load_current_project()
    project_folder = projects_folder / project_name

    create_dataset(project_folder, manually_downloaded=False)
    print("dataset created")
    
    # Create the statistic folder
    create_stats_folder(project_folder)
    clean_LS(project_folder, annotated_with_LS=False)
    encoding(project_folder)
    annotations_per_img(project_folder)
    classes_distribution(project_folder)
    print("class distributon done")
    
    get_global_results(project_folder)
    print("got global results")
    
    return render_template(
        "/pages/class_distrib.html",
        class_distibution_path=url_for('statistics.serve_dataset_image'), 
        project_name = project_name)


@statistics_bp.route('/dataset_stats')
def serve_dataset_image():
    project_name = load_current_project()
    filename = 'class_distribution.png'
    file_path = Path.cwd() / "data" / project_name / "dataset_statistics" / filename
    
    if file_path.exists():
        return send_file(file_path, mimetype='image/png')
    return "File not found", 404