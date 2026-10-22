"""
Routes du test et de l'evaluation d'un modele sur de nouvelles images.

Parcours d'un cycle :
1. test_upload() ou test_images_pretrained() : import des images de test et predictions en fond
2. check_inference() : page d'attente, jusqu'a la fin des predictions
3. test_end() : les predictions sont envoyees a Label Studio pour correction
4. evaluate_model() : comparaison des predictions et des corrections (metriques, matrice de confusion)
5. dispatch_corrections() : les images corrigees rejoignent la verite terrain du projet

L'etat des predictions est transmis par os.environ["test_complete"] : "False" pendant
les predictions, puis "True" ou "Error" (voir process_images_with_yolo).
"""
from app.app import app

import os
from pathlib import Path
from threading import Thread

from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_uploads import configure_uploads

from ..src.scripts.model_evaluation import (get_corrected_label_files, get_csv_results, get_txt_results,
                                            create_confusion_matrix, read_omitted_images)
from ..src.scripts.generate_new_ground_truth import (create_new_ground_truth, move_correction_files_and_images,
                                                     add_csv_data, clean_data)
from ..src.scripts.predicting_and_checking_yolo_results import (process_images_with_yolo, yolo_to_csv,
                                                                get_ls_for_local_files)

from ..src.modules.models_functions import get_model_list
from ..src.modules.generate_labels import add_new_labels, get_labeling_code

from ..src.models.formulaires import images as images_uploadees, images2 as images_uploadees2, ImportImages, ImportImages2
from .upload_routes import save_without_overwriting

from app.services.config_service import load_current_project, load_last_model_path, save_config
from app.services.project_service import projects_folder, list_existing_projects
from app.services.label_studio_service import launch_label_studio_async, configure_label_studio_root
from app.services.label_studio_api_service import setup_ls_correction_project

testing_bp = Blueprint("testing", __name__)

NO_MODEL = "------"


def _current_project_and_model():
    """
    Projet courant et dernier modele utilise, s'ils existent encore.
    Renvoie (project_name, model_folder), ou (None, None) apres avoir prepare un message d'erreur.
    """
    project_name = load_current_project()
    if project_name not in list_existing_projects():
        flash("Merci de choisir un projet.")
        return None, None
    model_folder = load_last_model_path()
    if not model_folder or not (Path(model_folder) / "weights" / "best.pt").is_file():
        flash("Aucun modèle à tester : entraînez ou importez d'abord un modèle.")
        return None, None
    return project_name, Path(model_folder)


def _eval_folder(project_name: str) -> Path:
    return projects_folder / project_name / "image_inputs" / "eval_images"


def _save_test_images(form, project_name: str) -> int:
    """
    Enregistre les images de test envoyées dans eval_images, avec la même règle de nommage
    que les images de vérité terrain, sans écraser ni dupliquer. Renvoie le nombre d'images enregistrées.
    """
    eval_folder = _eval_folder(project_name)
    eval_folder.mkdir(parents=True, exist_ok=True)
    saved = 0
    for fichier in form.fichiers.data:
        nom_enregistre, renommee = save_without_overwriting(fichier, eval_folder)
        if nom_enregistre is not None:
            saved += 1
            if renommee:
                flash(f"{fichier.filename} enregistrée sous {nom_enregistre} : "
                      f"une autre image de test porte déjà ce nom.")
    return saved


def _start_predictions(project_name: str, model_folder: Path):
    """Lance les prédictions en fond et affiche la page d'attente."""
    os.environ["test_complete"] = "False"
    thread = Thread(target=process_images_with_yolo, args=(projects_folder / project_name, model_folder))
    thread.start()
    return render_template("/pages/loading2.html")


@testing_bp.route("/check_inference", methods=['GET', 'POST'])
def check_inference():
    status = os.environ.get("test_complete")
    if status == "True":
        return redirect(url_for('testing.test_end'))
    if status == "Error":
        flash("Les prédictions ont échoué. Le détail de l'erreur est affiché dans le terminal.")
        return redirect(url_for('project.accueil_projet'))
    if status != "False":
        # Aucune prédiction en cours, par exemple après un redémarrage de l'app
        return redirect(url_for('project.accueil_projet'))
    return render_template('/pages/loading2.html')


@testing_bp.route("/test_modele", methods=['GET', 'POST'])
def test_upload():
    project_name, model_folder = _current_project_and_model()
    if project_name is None:
        return redirect(url_for('project.accueil'))

    # flask_uploads sert ici à vérifier que les fichiers envoyés sont des images
    app.config['UPLOADED_IMAGES_DEST'] = str(_eval_folder(project_name))
    configure_uploads(app, images_uploadees)
    form2 = ImportImages()

    if not form2.validate_on_submit():
        flash("Merci de choisir au moins une image à tester.")
        return redirect(url_for('project.accueil_projet'))

    _save_test_images(form2, project_name)
    return _start_predictions(project_name, model_folder)


#la route ci dessous récupère des images pour tester le modèle et fait passer le modèle dessus
@testing_bp.route("/test_end", methods=['GET', 'POST'])
def test_end():
    project_name, model_folder = _current_project_and_model()
    if project_name is None:
        return redirect(url_for('project.accueil'))
    abspathtoimages = projects_folder / project_name

    try:
        configure_label_studio_root()
        launch_label_studio_async()

        yolo_to_csv(abspathtoimages, model_folder)
        ls_result_file = get_ls_for_local_files(abspathtoimages, model_folder)
        _, label_config_file = get_labeling_code(abspathtoimages, model_folder)

        project_id, ls_url = setup_ls_correction_project(
            project_name=project_name,
            labels_folder_path=projects_folder / project_name / "annotations" / "prediction_corrections",
            img_folder_path=_eval_folder(project_name),
            ls_result_file=ls_result_file,
            label_config_file=label_config_file,
        )
    except Exception as e:
        print(f"Erreur lors de la préparation des corrections : {e}")
        flash(f"Impossible de préparer la correction dans Label Studio : {e}")
        return redirect(url_for('project.accueil_projet'))

    return render_template(
        "/pages/checking_results.html",
        project_name=project_name,
        ls_url=ls_url
        )


@testing_bp.route("/dispatch_corrections", methods=['GET', 'POST'])
def dispatch_corrections():
    project_name, model_folder = _current_project_and_model()
    if project_name is None:
        return redirect(url_for('project.accueil'))
    project_folder = projects_folder / project_name

    create_new_ground_truth(project_folder, model_folder, create_groundtruth=True)
    move_correction_files_and_images(project_folder)
    add_csv_data(project_folder, model_folder)
    # Fin du cycle : suppression des images augmentées (_PT) et de leurs annotations
    clean_data(project_folder)

    return render_template(
        "/pages/end_cycle.html",
        nom_du_projet=project_name)


@testing_bp.route("/evaluate_model", methods=['GET', 'POST'])
def evaluate_model():
    project_name, model_folder = _current_project_and_model()
    if project_name is None:
        return redirect(url_for('project.accueil'))
    project_folder = projects_folder / project_name

    try:
        # Update YOLO label definitions with new classes found in prediction correction files
        add_new_labels(project_folder, model_folder)

        # Generate the corrected files in YOLO format
        get_corrected_label_files(project_folder, model_folder)

        # Generate a CSV with the corrected data
        get_csv_results(project_folder, model_folder, all_results=False)

        # Generate the file with metrics
        png_path = get_txt_results(project_folder, model_folder)

        # Generate the confusion matrix
        matrix_path = create_confusion_matrix(project_folder, model_folder)
    except (ValueError, FileNotFoundError) as e:
        print(f"Erreur lors de l'évaluation : {e}")
        flash(f"Impossible d'évaluer le modèle : {e}")
        return redirect(url_for('project.accueil_projet'))

    # Images de test omises de l'évaluation (utilisées pour l'entraînement, ou non corrigées),
    # à afficher pour que les scores puissent être interprétés : [{"image": ..., "raison": ...}]
    images_omises = read_omitted_images(project_folder, model_folder)
    if images_omises:
        flash(f"{len(images_omises)} image(s) de test omise(s) de l'évaluation : "
                f"le détail est indiqué sous les résultats.")

    if matrix_path is None:
        return render_template("/pages/no_evaluation.html", project_name=project_name,
                               images_omises=images_omises)

    return render_template(
        "/pages/model_evaluation.html",
        project_name=project_name,
        png_path=Path(png_path).relative_to(Path.cwd()).as_posix(),
        matrix_path=Path(matrix_path).relative_to(Path.cwd()).as_posix(),
        images_omises=images_omises)


def _testable_models(project_name: str) -> list[str]:
    """Modèles du projet (entraînés ou importés), sans le choix « partir de zéro »."""
    return [m for m in get_model_list(project_name) if m != NO_MODEL]


@testing_bp.route("/test_pre_trained_model", methods=['GET', 'POST'])
def test_pretrained():
    project_name = load_current_project()
    form = ImportImages2()

    choices_modeles = [(m, m) for m in get_model_list(project_name)]
    form.modele.choices = choices_modeles  # type: ignore[assignment]

    return render_template("/pages/test_pre_trained.html", liste_modeles=choices_modeles, form=form, project_name=project_name)


@testing_bp.route("/test_images_pretrained", methods=["GET", "POST"])
def test_images_pretrained():
    project_name = load_current_project()
    if project_name not in list_existing_projects():
        flash("Merci de choisir un projet.")
        return redirect(url_for('project.accueil'))

    app.config['UPLOADED_IMAGES_DEST'] = str(_eval_folder(project_name))
    configure_uploads(app, images_uploadees2)
    form = ImportImages2()

    # Mêmes modèles que ceux proposés sur la page : ceux du projet, et pas ceux des autres projets
    form.modele.choices = [(m, m) for m in _testable_models(project_name)]  # type: ignore[assignment]

    if not form.validate_on_submit() or not form.modele.data:
        print(form.errors)
        flash("Merci de sélectionner un modèle et au moins une image.")
        return redirect(url_for('testing.test_pretrained'))

    model_folder = Path.cwd() / "output" / "train" / form.modele.data
    print(f"Le modèle sélectionné est : {form.modele.data}.")

    _save_test_images(form, project_name)
    save_config(project_name=project_name, last_model_path=model_folder)
    return _start_predictions(project_name, model_folder)