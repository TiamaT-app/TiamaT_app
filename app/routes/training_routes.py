"""
Routes de l'entrainement d'un modele YOLO sur le projet courant.

- training_setup()        : choix des parametres (epoques, dropout, modele de depart)
- launch_training()       : lance l'entrainement dans un thread de fond
- check_training_status() : page d'attente, rechargee regulierement, avec l'epoque en cours
- result()                : resultats, une fois l'entrainement termine

L'etat de l'entrainement est transmis par os.environ["complete"] : "False" pendant
l'entrainement, puis "True" ou "Error" (voir training() dans data_preparation_and_training).
Les parametres de l'entrainement en cours sont enregistres dans training_config.json.
"""
import os
import json
from pathlib import Path
from datetime import datetime
from threading import Thread
import pandas as pd


from flask import Blueprint, render_template, request, redirect, url_for, flash
from ultralytics import YOLO

from ..src.scripts.data_preparation_and_training import training
from ..src.modules.models_functions import get_model_list
from ..src.models.formulaires import ImportImages

from app.services.config_service import load_current_project, save_config
from app.services.project_service import projects_folder, list_existing_projects

training_bp = Blueprint("training", __name__)

TRAINING_CONFIG = Path("training_config.json")
DEFAULT_MODEL = "yolo26s.pt"
NO_MODEL = "------"


@training_bp.route("/training_setup", methods=['GET', 'POST'])
def training_setup():
    # get_model_list renvoie ["------"] si le projet n'a pas encore de modele
    liste_modeles = get_model_list(load_current_project())
    return render_template("/pages/training_setup.html", liste_modeles=liste_modeles)


@training_bp.route("/start_training", methods=['GET', 'POST'])
def launch_training():
    if request.method != 'POST':
        return redirect(url_for('training.training_setup'))

    project_name = load_current_project()
    if project_name not in list_existing_projects():
        flash("Merci de choisir un projet avant de lancer un entraînement.")
        return redirect(url_for('project.accueil'))

    # Un seul entraînement à la fois : un 2e écraserait training_config.json
    if os.environ.get("complete") == "False":
        flash("Un entraînement est déjà en cours.")
        return redirect(url_for('training.check_training_status'))

    try:
        nombre_epoch = int(request.form.get("epochs", ""))
        dropout = float(request.form.get("dropout", ""))
    except ValueError:
        flash("Le nombre d'époques et le dropout doivent être des nombres.")
        return redirect(url_for('training.training_setup'))
    if nombre_epoch < 1 or not 0 <= dropout < 1:
        flash("Le nombre d'époques doit être au moins 1, et le dropout compris entre 0 et 1.")
        return redirect(url_for('training.training_setup'))

    # Le modèle de départ doit faire partie des modèles du projet
    model_choice = request.form.get("modele", NO_MODEL)
    if model_choice not in get_model_list(project_name):
        flash("Modèle de départ inconnu, merci d'en choisir un dans la liste.")
        return redirect(url_for('training.training_setup'))

    if model_choice == NO_MODEL:
        model = DEFAULT_MODEL
    else:
        model = str(Path.cwd() / "output" / "train" / model_choice / "weights" / "best.pt")
    model_name = f'{project_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    training_config = {
        "use_model" : model, # starting model: yolo26s.pt by default, or the best.pt of a previous model
        "model_name" : model_name, # model name composed with the project name and the date (YMD_HMS)
        "img_size" : 640, # to be changed as needed, by default use 640
        "epochs" : nombre_epoch,
        "batch" : -1, # -1 for AutoBatch
        "workers" : 8, # https://docs.ultralytics.com/modes/train/#train-settings
        "dropout" : dropout
        }
    # Écrit avant de lancer le thread : la page d'attente en a besoin dès son premier affichage
    TRAINING_CONFIG.write_text(json.dumps(training_config, indent=4), encoding="utf-8")

    os.environ["complete"] = "False"
    thread = Thread(target=training, args=(projects_folder / project_name, nombre_epoch, dropout, model, model_name))
    thread.start()

    return render_template('/pages/loading.html', epoch=f"0/{nombre_epoch}")


def _current_epoch(model_name: str) -> int:
    """Dernière époque terminée, lue dans le results.csv que YOLO complète à chaque époque."""
    results_file = Path("output") / "train" / model_name / "results.csv"
    if not results_file.is_file():
        return 0
    try:
        df = pd.read_csv(results_file)
    except pd.errors.EmptyDataError:
        return 0
    # Selon les versions de YOLO, les noms de colonnes sont précédés d'espaces
    df.columns = df.columns.str.strip()
    if df.empty or "epoch" not in df.columns:
        return 0
    return int(df["epoch"].iloc[-1])


@training_bp.route("/check_training_status")
def check_training_status():
    status = os.environ.get("complete")
    if status == "True":
        return redirect(url_for('training.result'))
    if status == "Error":
        flash("L'entraînement a échoué. Le détail de l'erreur est affiché dans le terminal.")
        return redirect(url_for('training.training_setup'))
    # Aucun entraînement en cours (par exemple après un redémarrage de l'app)
    if status != "False" or not TRAINING_CONFIG.is_file():
        return redirect(url_for('training.training_setup'))

    config = json.loads(TRAINING_CONFIG.read_text(encoding="utf-8"))
    return render_template('/pages/loading.html',
                           epoch=f"{_current_epoch(config['model_name'])}/{config['epochs']}")


def _record_model(project_name: str, training_config: dict, weights_file: Path) -> None:
    """
    Ajoute le modèle entraîné au CSV des modèles du projet, une seule fois :
    recharger la page des résultats ne crée pas de ligne en double.
    """
    csv_file = projects_folder / project_name / f"{project_name}.csv"
    existing = pd.read_csv(csv_file) if csv_file.is_file() else None
    if existing is not None and training_config["model_name"] in existing["model_name"].astype(str).tolist():
        return

    row = dict(training_config)
    row["use_model"] = str(row["use_model"]).replace(str(Path.cwd()), "")
    row["origine"] = "TiamaT"
    row["classes"] = YOLO(weights_file).names
    new_row = pd.DataFrame([row])
    df = new_row if existing is None else pd.concat([existing, new_row], ignore_index=True)
    df.to_csv(csv_file, index=False)


@training_bp.route("/training_result")
def result():
    project_name = load_current_project()
    if not TRAINING_CONFIG.is_file():
        return redirect(url_for('training.training_setup'))

    training_config = json.loads(TRAINING_CONFIG.read_text(encoding="utf-8"))
    model_name = training_config["model_name"]
    model_path = Path('output') / 'train' / model_name
    weights_file = model_path / "weights" / "best.pt"
    if not weights_file.is_file():
        flash("Aucun modèle entraîné n'a été trouvé pour cet entraînement.")
        return redirect(url_for('training.training_setup'))

    _record_model(project_name, training_config, weights_file)

    abs_model_path = Path.cwd() / 'output' / 'train' / model_name
    save_config(project_name=project_name, last_model_path=abs_model_path)

    return render_template("/pages/entrainement_fini.html",
                           model_path=abs_model_path,
                           form2=ImportImages(),
                           chemin_matrice=(model_path / "confusion_matrix.png").as_posix(),
                           chemin_val_batch=(model_path / "val_batch0_pred.jpg").as_posix())