import os
import json
from pathlib import Path
from datetime import datetime
from threading import Thread
import pandas as pd

from flask import Blueprint, render_template, request, redirect, url_for

from ..src.scripts.data_preparation_and_training import training, dispatch_data
from ..src.modules.models_functions import get_model_list
from ..src.models.formulaires import ImportImages

from app.services.config_service import load_current_project, save_config
from app.services.project_service import projects_folder

training_bp = Blueprint("training", __name__)


@training_bp.route("/training_setup", methods=['GET', 'POST'])
def training_setup():
    project_name = load_current_project()
    if (Path("projects")/project_name/f"{project_name}.csv").is_file():
        liste_modeles = get_model_list(project_name)
    else:
         liste_modeles =[]
         liste_modeles.append("------")

         print(liste_modeles)
         print(type(liste_modeles))
         
    return render_template("/pages/training_setup.html", liste_modeles=liste_modeles)

@training_bp.route("/start_training", methods=['GET', 'POST'])
def launch_training():
    project_name = load_current_project()
    
    if request.method != 'POST':
        return redirect(url_for('training.training_setup'))
    
    nombre_epoch = int(request.form.get("epochs")) # type: ignore[assignment]
    dropout = float(request.form.get("dropout")) # type: ignore[assignment]
    model_name = request.form.get("modele")

    if model_name != "------":
        model = str(Path.cwd() / "output" / "train" / str(model_name) / "weights" / "best.pt")
        model_name = f'{str(project_name)}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    else:
        model = "yolo26s.pt"
        model_name = f'{str(project_name)}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'


    os.environ["complete"]= "False"
    thread = Thread(target=training, args=(project_name,nombre_epoch, dropout, model, model_name) )
    thread.start()
    
    training_config = {
        "use_model" : model, # to be changed as needed, by default use 'yolov11n.pt'
        "model_name" : model_name, # model name composed with the project name and the date (YMD_HMS)
        "img_size" : 640, # to be changed as needed, by default use 640
        "epochs" : nombre_epoch, # to be changed as needed
        "batch" : -1, # to be changed as needed, by default use 8 or or -1 for AutoBatch
        "workers" : 8, # to be changed as needed, by default 24, or 8 (https://docs.ultralytics.com/modes/train/#train-settings)
        "dropout" : dropout
        }
    
    with open("training_config.json", "w") as f:
        json.dump(training_config,f, indent=4)

    return render_template('/pages/loading.html',epoch = f"0/{str(nombre_epoch)}")

@training_bp.route("/check_training_status")
def check_training_status():
    if os.environ["complete"]== "True":
        return redirect(url_for('training.result'))
    else:
        with open("training_config.json","r") as f:
            config = json.load(f)
        model_name = config["model_name"]
        epochs = config["epochs"]
        if (Path("output")/"train"/model_name/"results.csv").is_file():
            epoch_actuelle = (pd.read_csv(Path("output")/"train"/model_name/"results.csv")["epoch"]).to_list()[-1]
            return render_template('/pages/loading.html', epoch = f"{epoch_actuelle}/{epochs}")
        else:
            return render_template('/pages/loading.html', epoch = f"0/{epochs}")
  

        

    
@training_bp.route("/training_result") #Config du réentraînement
def result():
    project_name = load_current_project()
    project_folder = projects_folder / project_name
    with open("training_config.json","r") as f:
        dico_config = json.load(f)
        use_model = dico_config["use_model"]
        img_size = dico_config["img_size"]
        epochs = dico_config["epochs"]
        batch = dico_config["batch"]
        workers = dico_config["workers"]
        dropout = dico_config["dropout"]
        model_name = dico_config["model_name"]
    dico_config["use_model"] = use_model.replace(str(Path.cwd()), "") 
    if (Path(project_folder)/f"{project_name}.csv").is_file():
        df = pd.read_csv(Path(project_folder)/f"{project_name}.csv")
        dico_config["origine"] = "TiamaT" 
        df.loc[len(df)] = dico_config
        df.to_csv(Path(project_folder)/f"{project_name}.csv", index = False)
    else :
        dico_config["origine"] = "TiamaT"
        df = pd.DataFrame(dico_config, index=[0])
        df.to_csv(Path(project_folder)/f"{project_name}.csv", index = False)




    dispatch_data(project_folder, use_model, img_size, 
                  epochs, batch, workers, dropout, model_name,
                  pretrained_model = False, interrupted_model_folder = False)
    
    
    abs_model_path = Path.cwd() / 'output' / 'train' / model_name
    save_config(project_name=project_name, last_model_path=abs_model_path)
    
    model_path = Path('output') / 'train' / model_name
    chemin_matrice = model_path / "confusion_matrix.png"
    chemin_val_batch = model_path / "val_batch0_pred.jpg"
    form2 = ImportImages()
    return render_template("/pages/entrainement_fini.html", model_path = abs_model_path, form2= form2, chemin_matrice=chemin_matrice.as_posix(), chemin_val_batch=chemin_val_batch.as_posix())

