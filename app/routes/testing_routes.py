from app.app import app

import os
from datetime import datetime
from threading import Thread

from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_uploads import configure_uploads

from ..src.scripts.model_evaluation import *
from ..src.scripts.generate_new_ground_truth import *
from ..src.scripts.predicting_and_checking_yolo_results import *

from ..src.modules.models_functions import get_model_list
from ..src.modules.generate_labels import add_new_labels, get_labeling_code

from ..src.models.formulaires import images as images_uploadees, images2 as images_uploadees2, ImportImages, ImportImages2

from app.services.config_service import load_current_project,load_last_model_path, save_config
from app.services.project_service import projects_folder
from app.services.label_studio_service import launch_label_studio_async

testing_bp = Blueprint("testing", __name__)

@testing_bp.route("/check_inference", methods=['GET', 'POST'])
def check_inference():
    if os.environ["test_complete"]== "True":
        return redirect(url_for('testing.test_end'))
    return render_template('/pages/loading2.html')


@testing_bp.route("/test_modele", methods=['GET', 'POST'])
def test_upload():
    os.environ["test_complete"]= "False"
    project_name = load_current_project()
    last_model = load_last_model_path()
    app.config['UPLOADED_IMAGES_DEST'] = str(projects_folder / project_name / "image_inputs" / "eval_images")
    configure_uploads(app, images_uploadees)
    form2 = ImportImages()
    
    abspathtoimages = projects_folder / project_name
    if form2.validate_on_submit():
        files2 = []
        for fichier in form2.fichiers.data:
            images_uploadees.save(fichier)
            files2.append(fichier.filename)
    thread = Thread(target=process_images_with_yolo, args=(abspathtoimages, last_model) )
    thread.start()
    return render_template("/pages/loading2.html")




#la route ci dessous récupère des images pour tester le modèle et fait passer le modèle dessus
@testing_bp.route("/test_end", methods=['GET', 'POST'])
def test_end():
    project_name = load_current_project()
    abspathtoimages = projects_folder / project_name
    last_model = load_last_model_path()


    
    launch_label_studio_async()
    
    yolo_to_csv(abspathtoimages, last_model)
    ls_result = get_ls_for_local_files(abspathtoimages, last_model)
    html_template = get_labeling_code(abspathtoimages, last_model)
    chemin_images = projects_folder / project_name / "image_inputs" / "eval_images"
    chemin_labels = projects_folder / project_name / "annotations" / "prediction_corrections"
         
   
    return render_template(
        "/pages/checking_results.html", 
        html_template=html_template, 
        ls_result=ls_result,
        project_name=project_name, 
        current_folder = Path.cwd(), 
        chemin_images=chemin_images, 
        chemin_labels=chemin_labels)

@testing_bp.route("/dispatch_corrections",methods=['GET', 'POST'])
def dispatch_corrections():
    project_name = load_current_project()
    
    project_folder = Path(projects_folder / project_name)
    yolo_model_folder = load_last_model_path()
    
    create_new_ground_truth(project_folder, yolo_model_folder, create_groundtruth=True)
    move_correction_files_and_images(project_folder)
    add_csv_data(project_folder, yolo_model_folder)
    clean_data(project_folder)
    app.config["SECOND_PASS"]= "true"
    
    return render_template(
        "/pages/end_cycle.html",
        nom_du_projet = project_name)


@testing_bp.route("/evaluate_model",methods=['GET', 'POST'])
def evaluate_model():
    project_name = load_current_project()
    yolo_model_folder = load_last_model_path()
    project_folder = Path(projects_folder, project_name)
    
    # Update YOLO label definitions with new classes found in prediction correction files 
    add_new_labels(project_folder, yolo_model_folder)
    # Generate the corrected files in YOLO format
    get_corrected_label_files(project_folder, yolo_model_folder)
    # Generate a CSV with the corrected data
    get_csv_results(project_folder, yolo_model_folder, all_results=False)
    # Generate the file with metrics
    png_path = get_txt_results(project_folder, yolo_model_folder)
    png_path = png_path.relative_to(Path.cwd())

    # Generate the confusion matrix
    matrix_path = create_confusion_matrix(project_folder, yolo_model_folder)
    matrix_path = matrix_path.relative_to(Path.cwd())
    return render_template(
        "/pages/model_evaluation.html", 
        project_name=project_name, 
        png_path=Path(png_path).as_posix(),
        matrix_path=Path(matrix_path).as_posix())

@testing_bp.route("/test_pre_trained_model",methods=['GET', 'POST'])
def test_pretrained():

    project_name = load_current_project()
    form = ImportImages2()

    liste_modeles = get_model_list(project_name)
    choices_modeles = [(m, m) for m in liste_modeles]
    form.modele.choices = choices_modeles  # type: ignore[assignment]

    return render_template("/pages/test_pre_trained.html", liste_modeles=choices_modeles, form= form, project_name = project_name)

@testing_bp.route("/test_images_pretrained", methods=["GET","POST"])
def test_images_pretrained():
    os.environ["test_complete"]= "False"
    project_name = load_current_project()
    app.config['UPLOADED_IMAGES_DEST'] = str(projects_folder / project_name / "image_inputs" / "eval_images")
    form = ImportImages2()

    modeles_dispo = [m.name for m in (Path("output") / "train").iterdir() if m.is_dir()]
    form.modele.choices = [(m, m) for m in modeles_dispo]
    
    configure_uploads(app, images_uploadees2)
    abspathtoimages = projects_folder / project_name
    files2 = []
    if form.validate_on_submit():
        model_name = request.form.get("modele")
        print(f"Le modèle sélectionnée est :{model_name}.")
        model_path = Path.cwd() / "output" / "train" / str(model_name)
        
        for fichier in form.fichiers.data:
            images_uploadees2.save(fichier)
            files2.append(fichier.filename)
    else:
        print("erreur de formulaire")
        print(form.errors)
        flash("Merci de sélectionner un modèle et au moins une image.")
        return redirect(url_for('testing.test_pretrained'))
    save_config(project_name=project_name, last_model_path=model_path)
    os.environ["test_complete"]= "False"

    thread = Thread(target=process_images_with_yolo, args=(abspathtoimages, model_path) )
    thread.start()
    return render_template("/pages/loading2.html")

    

    '''launch_label_studio_async()
    
    yolo_to_csv(abspathtoimages, model_path)
    ls_result = get_ls_for_local_files(abspathtoimages, str(model_path))
    html_template = get_labeling_code(abspathtoimages, str(model_path))
    
    chemin_images = projects_folder / project_name / "image_inputs" / "eval_images"
    chemin_labels = projects_folder / project_name / "annotations" / "prediction_corrections"
         
    return render_template("/pages/checking_results.html", 
                           html_template=html_template, 
                           ls_result=ls_result, 
                           project_name=project_name, 
                           current_folder = Path.cwd(), 
                           chemin_images=chemin_images, 
                           chemin_labels=chemin_labels)'''    
