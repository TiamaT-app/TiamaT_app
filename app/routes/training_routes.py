import os
import json
from pathlib import Path

from threading import Thread

from app.app import app

from flask import Blueprint, render_template, request, redirect, url_for

from ..src.scripts.data_preparation_and_training import *
from ..src.scripts.predicting_and_checking_yolo_results import get_model_list
from ..src.models.formulaires import ImportImages

training_bp = Blueprint("training", __name__)

@training_bp.route("/training_setup", methods=['GET', 'POST'])
def training_setup():
    with open("config.json","r") as f:
        app.config["CURRENT_PROJECT_NAME"]=json.load(f).get("CURRENT_PROJECT_NAME")
    project_name = app.config["CURRENT_PROJECT_NAME"]
    if Path("output").is_dir():
        liste_modeles = get_model_list(project_name)
    else:
         liste_modeles=["------"]
    return render_template("/pages/training_setup.html", liste_modeles=liste_modeles)



@training_bp.route("/start_training", methods=['GET', 'POST'])
# On met à part la fonction d'entraînement pour pouvoir la multithreader
def training():
    with open("config.json","r") as f:
        app.config["CURRENT_PROJECT_NAME"]=json.load(f).get("CURRENT_PROJECT_NAME")
    project_name = app.config['CURRENT_PROJECT_NAME']
    
    if request.method == 'POST':
        nombre_epoch = int(request.form.get("epochs"))
        dropout = float(request.form.get("dropout"))
        model_name = request.form.get("modele")
        if model_name != "------":
            model = str(Path("output") /"train"/ model_name /"weights"/ "best.pt")
            model_name = f'{str(project_name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}'
            
            
        else :
            model = "yolo11n.pt"
            model_name = f'{str(project_name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}'
    os.environ["complete"]= "False"
    thread = Thread(target=entrainement, args=(project_name,nombre_epoch, dropout, model, model_name) )
    thread.start()
    training_config = {"use_model" : model, # to be changed as needed, by default use 'yolov11n.pt'
        "model_name" : model_name, # model name composed with the project name and the date (YMD_HMS)
        "img_size" : 640, # to be changed as needed, by default use 640
        "epochs" : nombre_epoch, # to be changed as needed
        "batch" : -1, # to be changed as needed, by default use 8 or or -1 for AutoBatch
        "workers" : 8, # to be changed as needed, by default 24, or 8 (https://docs.ultralytics.com/modes/train/#train-settings)
        "dropout" : dropout}
    
    with open("training_config.json", "w") as f:
        json.dump(training_config,f)
    return render_template('pages/loading.html')

@training_bp.route("/check_training_status")
def check_training_status():
    if os.environ["complete"]== "True":
        return redirect(url_for('training.result'))
    return render_template('/pages/loading.html')

    
@training_bp.route("/training_result") #Config du réentraînement
def result():
    with open("config.json","r") as f:
        app.config["CURRENT_PROJECT_NAME"]=json.load(f).get("CURRENT_PROJECT_NAME")
    project_folder=app.config["CURRENT_PROJECT_NAME"]
    with open("training_config.json","r") as f:
        dico_config = json.load(f)
    use_model = dico_config["use_model"]
    img_size = dico_config["img_size"]
    epochs=dico_config["epochs"]
    batch=dico_config["batch"]
    workers=dico_config["workers"]
    dropout = dico_config["dropout"]
    model_name = dico_config["model_name"]
    
    dispatch_data(project_folder, use_model, img_size, 
                  epochs, batch, workers, dropout, model_name,
                  pretrained_model=None, interrupted_model_folder=False)
    
    
    cwd = Path.cwd()
    model_path= cwd / 'output' / 'train' / model_name
    app.config['LAST_MODEL_PATH'] = model_path
    dossier_images = Path('output') / 'train' / model_name
    config_dict={"CURRENT_PROJECT_NAME":project_folder, "LAST_MODEL_PATH":model_path.as_posix()}
    jsonstr = json.dumps(config_dict)
    with open ("config.json","w") as f:
                    f.write(jsonstr)
    chemin_matrice = Path(dossier_images / "confusion_matrix.png")
    chemin_val_batch= Path(dossier_images / "val_batch0_pred.jpg")
    form2=ImportImages()
    return render_template("/pages/entrainement_fini.html", model_path = model_path, form2= form2, chemin_matrice=chemin_matrice.as_posix(), chemin_val_batch=chemin_val_batch.as_posix())

