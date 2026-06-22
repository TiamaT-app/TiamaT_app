import os
import json
from pathlib import Path

from threading import Thread

from app.app import app

from flask import Blueprint, render_template, request
from flask_uploads import configure_uploads

from ..src.scripts.model_evaluation import *
from ..src.scripts.run_label_studio import launch_LS
from ..src.scripts.generate_new_ground_truth import *
from ..src.scripts.predicting_and_checking_yolo_results import *

from ..src.models.formulaires import images as images_uploadees, images2 as images_uploadees2, ImportImages, ImportImages2

testing_bp = Blueprint("testing", __name__)

#la route ci dessous récupère des images pour tester le modèle et fait passer le modèle dessus
@testing_bp.route("/test_modele", methods=['GET', 'POST'])
def test_upload():
    with open('config.json', 'r') as f:
        config = json.load(f)
        project_name = config.get('CURRENT_PROJECT_NAME')
        last_model = config.get("LAST_MODEL_PATH")
        app.config["CURRENT_PROJECT_NAME"]=project_name
        app.config['LAST_MODEL_PATH']=last_model
        
    project_name = app.config['CURRENT_PROJECT_NAME']
    app.config['UPLOADED_IMAGES_DEST'] = os.path.join(project_name, "image_inputs", "eval_images")
    configure_uploads(app, images_uploadees)
    form2 = ImportImages()
    root = Path.cwd()
    
    abspathtoimages = root / project_name
    if form2.validate_on_submit():
        files2=[]
        for fichier in form2.fichiers.data:
            images_uploadees.save(fichier)
            files2.append(fichier.filename)
    
    process_images_with_yolo(abspathtoimages, app.config['LAST_MODEL_PATH'])
    
    thread = Thread(target=launch_LS)
    thread.daemon = True
    thread.start()
    yolo_to_csv(abspathtoimages, app.config['LAST_MODEL_PATH'])
    ls_result = get_ls_for_local_files(abspathtoimages, app.config['LAST_MODEL_PATH'])
    html_template = get_labeling_code(abspathtoimages, app.config['LAST_MODEL_PATH'])
    chemin_images = root / project_name / "image_inputs" / "eval_images"
    chemin_labels = root / project_name / "annotations" / "prediction_corrections"
         
   
    return render_template("/pages/checking_results.html", html_template=html_template, ls_result=ls_result, project_name=project_name, current_folder = Path.cwd(), chemin_images=chemin_images, chemin_labels=chemin_labels)

@testing_bp.route("/dispatch_corrections",methods=['GET', 'POST'])
def dispatch_corrections():
    with open("config.json","r") as f:
        config= json.load(f)
        app.config["CURRENT_PROJECT_NAME"]=config.get("CURRENT_PROJECT_NAME")
        app.config['LAST_MODEL_PATH']=config.get("LAST_MODEL_PATH")
    project_folder=Path(Path.cwd()/app.config["CURRENT_PROJECT_NAME"])
    yolo_model_folder=app.config["LAST_MODEL_PATH"]
    create_new_ground_truth(project_folder, yolo_model_folder, create_groundtruth=True)
    move_correction_files_and_images(project_folder)
    add_csv_data(project_folder, yolo_model_folder)
    clean_data(project_folder)
    app.config["SECOND_PASS"]= "true"
    return render_template("/pages/end_cycle.html", nom_du_projet = app.config['CURRENT_PROJECT_NAME'])


@testing_bp.route("/evaluate_model",methods=['GET', 'POST'])
def evaluate_model():
    with open("config.json","r") as f:
        config= json.load(f)
        app.config["CURRENT_PROJECT_NAME"]=config.get("CURRENT_PROJECT_NAME")
        app.config['LAST_MODEL_PATH']=config.get("LAST_MODEL_PATH")
    root = Path.cwd()
    project_name = app.config["CURRENT_PROJECT_NAME"]
    yolo_model_folder = app.config['LAST_MODEL_PATH']
    project_folder=Path(root, project_name)
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
    return render_template("/pages/model_evaluation.html", project_name=project_name, png_path=Path(png_path).as_posix(), matrix_path=Path(matrix_path).as_posix())

@testing_bp.route("/test_pre_trained_model",methods=['GET', 'POST'])
def test_pretrained():
    
    with open("config.json","r") as f:
        config= json.load(f)
        app.config["CURRENT_PROJECT_NAME"]=config.get("CURRENT_PROJECT_NAME")
        app.config['LAST_MODEL_PATH']=config.get("LAST_MODEL_PATH")
    form=ImportImages2()

    project_folder=Path(Path.cwd()/app.config["CURRENT_PROJECT_NAME"])
    project_name =config.get("CURRENT_PROJECT_NAME")
    liste_modeles = get_model_list(project_name)
    form.modele.choices = liste_modeles
    liste_modeles2 =[]
    for i in liste_modeles:
        i2 = (i, i)
        liste_modeles2.append(i2) 
    return render_template("/pages/test_pre_trained.html", liste_modeles=liste_modeles2, form= form, project_name = project_name)

@testing_bp.route("/test_images_pretrained", methods=["GET","POST"])
def test_images_pretrained():
    with open("config.json","r") as f:
        config= json.load(f)
        app.config["CURRENT_PROJECT_NAME"]=config.get("CURRENT_PROJECT_NAME")
        app.config['LAST_MODEL_PATH']=config.get("LAST_MODEL_PATH")
    project_name =config.get("CURRENT_PROJECT_NAME")
    app.config['UPLOADED_IMAGES_DEST'] = os.path.join(project_name, "image_inputs", "eval_images")
    app.config['UPLOADED_IMAGES_DEST'] = os.path.join(project_name, "image_inputs", "eval_images")
    project_folder=Path(Path.cwd()/app.config["CURRENT_PROJECT_NAME"])
    
    form=ImportImages2()

    modeles_dispo = [m.name for m in (Path("output") / "train").iterdir() if m.is_dir()]
    form.modele.choices = [(m, m) for m in modeles_dispo]
    root = Path.cwd()
    
    configure_uploads(app, images_uploadees2)
    abspathtoimages = root / project_name
    if form.validate_on_submit():
        model_name = request.form.get("modele")
        print(f"Le modèle sélectionnée est :{model_name}.")
        model = str(Path("output") /"train"/ model_name /"weights"/ "best.pt")
        model_path = (Path.cwd()/"output"/"train"/model_name)
        
        
        files2=[]
        for fichier in form.fichiers.data:
            images_uploadees2.save(fichier)
            files2.append(fichier.filename)
    else:
        print("erreur de formulaire")
        print(form.errors)
    process_images_with_yolo(abspathtoimages, model_path)
    config_dict={"CURRENT_PROJECT_NAME":project_name, "LAST_MODEL_PATH":model_path.as_posix()}
    jsonstr = json.dumps(config_dict)
    with open ("config.json","w") as f:
                    f.write(jsonstr)
    thread = Thread(target=launch_LS)
    thread.daemon = True
    thread.start()
    yolo_to_csv(abspathtoimages, model_path)
    ls_result = get_ls_for_local_files(abspathtoimages, model_path)
    html_template = get_labeling_code(abspathtoimages, model_path)
    chemin_images = root / project_name / "image_inputs" / "eval_images"
    chemin_labels = root / project_name / "annotations" / "prediction_corrections"
         
   
    return render_template("/pages/checking_results.html", html_template=html_template, ls_result=ls_result, project_name=project_name, current_folder = Path.cwd(), chemin_images=chemin_images, chemin_labels=chemin_labels)    
