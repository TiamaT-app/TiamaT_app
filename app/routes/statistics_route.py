import os
import json
from pathlib import Path

from app.app import app

from flask import Blueprint, render_template, send_file, url_for

from ..src.scripts.get_training_data import *
from ..src.scripts.statistics_for_training import *

statistics_bp = Blueprint("statistics", __name__)


@statistics_bp.route("/dataset_statistics", methods=['GET', 'POST'])
def dataset_statistics():
    with open("config.json","r") as f:
        app.config["CURRENT_PROJECT_NAME"]=json.load(f).get("CURRENT_PROJECT_NAME")
    project_folder = os.path.join(os.getcwd(), app.config['CURRENT_PROJECT_NAME'])
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
    
    return render_template("/pages/class_distrib.html",class_distibution_path=url_for('statistics.serve_dataset_image'), project_name=app.config['CURRENT_PROJECT_NAME'])


@statistics_bp.route('/dataset_stats')
def serve_dataset_image():
    with open("config.json","r") as f:
        app.config["CURRENT_PROJECT_NAME"]=json.load(f).get("CURRENT_PROJECT_NAME")
    project_name=app.config['CURRENT_PROJECT_NAME']
    filename='class_distribution.png'
    file_path = Path.cwd() / "data" / project_name / "dataset_statistics" / filename
    if file_path.exists():
        return send_file(file_path, mimetype='image/png')
    return "File not found", 404