"""
Routes d'import de fichiers dans le projet courant.

- upload()        : import d'images dans projects/{projet}/image_inputs/ground_truth_images/
- upload_modele() : import d'un modele YOLO deja entraine (.pt) dans output/train/{projet}_{nom}/
"""

from pathlib import Path
import os
import shutil

import pandas as pd
from ultralytics import YOLO
from app.app import app

from flask import Blueprint, render_template, flash, redirect, url_for
from flask_uploads import configure_uploads

from ..src.models.formulaires import ImportModel, ImportImages, images as images_uploadees
from ..src.scripts.get_training_data import normalized_image_name

from app.services.config_service import load_current_project
from app.services.project_service import projects_folder
from app.services.image_service import get_preview_images


upload_bp = Blueprint("upload", __name__)

def save_without_overwriting(fichier, folder: Path) -> tuple[str | None, bool]:
    """
    Enregistre l'image envoyée dans folder sous un nom normalisé (normalized_image_name,
    la même règle que clean_image_name), sans jamais écraser ni dupliquer.

    - si une image identique (même contenu) existe déjà sous ce nom ou une variante
      (nom_1.jpg...), rien n'est enregistré : renvoie (None, False) ;
    - si le nom est pris par une autre image (ex : folio_é.jpg et folio_è.jpg, qui
      deviennent toutes deux folio_e.jpg), l'image est enregistrée sous nom_1.jpg,
      nom_2.jpg... : renvoie (nom_enregistre, True) ;
    - sinon : renvoie (nom_enregistre, False).
    """
    contenu = fichier.read()
    nom = normalized_image_name(fichier.filename)
    stem, extension = os.path.splitext(nom)

    candidat, numero = nom, 0
    while (folder / candidat).exists():
        if (folder / candidat).read_bytes() == contenu:
            return None, False
        numero += 1
        candidat = f"{stem}_{numero}{extension}"

    (folder / candidat).write_bytes(contenu)
    return candidat, numero > 0


@upload_bp.route("/upload", methods=['GET', 'POST'])
def upload():
    project_name = load_current_project()
    chemin_images = projects_folder / project_name / "image_inputs" / "ground_truth_images"
    chemin_labels = projects_folder / project_name / "annotations" / "ground_truth"

    # NB : doit pointer sous projects/, comme chemin_images ci-dessus
    app.config['UPLOADED_IMAGES_DEST'] = str(chemin_images)

    form = ImportImages()
    configure_uploads(app, images_uploadees)

    if not form.validate_on_submit():
        print(form.errors)
        flash("Erreur lors de l'upload des images, merci de réessayer.")
        return redirect(url_for('project.accueil_projet'))

    list_img = []
    deja_presentes = 0
    for fichier in form.fichiers.data:
        nom_enregistre, renommee = save_without_overwriting(fichier, chemin_images)
        if nom_enregistre is None:
            deja_presentes += 1
            continue
        list_img.append(nom_enregistre)
        if renommee:
            flash(f"{fichier.filename} enregistrée sous {nom_enregistre} : "
                  f"une autre image du projet porte déjà le nom {normalized_image_name(fichier.filename)}.")

    if deja_presentes:
        flash(f"{deja_presentes} image(s) déjà présente(s) dans le projet, non importée(s) à nouveau.")

    files2 = f"Vos {len(list_img)} images"
    liste_images, liste_noms, _ = get_preview_images(chemin_images)

    return render_template(
        "/pages/upload_full.html",
        project_name=project_name,
        files2=files2,
        form=form,
        form2=ImportImages(),
        chemin_images=str(chemin_images),
        chemin_labels=str(chemin_labels),
        liste_images=liste_images,
        liste_noms=liste_noms)


@upload_bp.route("/upload_modele", methods=['GET', 'POST'])
def upload_modele():
    project_name = load_current_project()
    form = ImportModel()

    if not form.validate_on_submit():
        for errors in form.errors.values():
            for error in errors:
                flash(error)
        if not form.errors:
            flash("Problème avec le formulaire.")
        return redirect(url_for('project.gestion_modeles'))

    model_name = f"{project_name}_{form.nom.data}"
    model_folder = Path("output") / "train" / model_name
    weights_file = model_folder / "weights" / "best.pt"

    if model_folder.is_dir():
        flash("Ce modèle existe déjà, merci de choisir un autre nom.")
        return redirect(url_for('project.gestion_modeles'))

    weights_file.parent.mkdir(parents=True)
    form.model.data.save(weights_file)

    # On vérifie que le fichier est bien un modèle YOLO, et on lit ses classes une seule fois.
    try:
        class_names = YOLO(weights_file).names
    except Exception as e:
        # Sans ce nettoyage, le dossier resterait et le nom serait bloqué (« ce modèle existe déjà »).
        shutil.rmtree(model_folder)
        print(f"Import du modèle impossible : {e}")
        flash("Ce fichier n'est pas un modèle YOLO valide.")
        return redirect(url_for('project.gestion_modeles'))

    # Ajout du modèle au CSV des modèles du projet (créé s'il n'existe pas encore)
    csv_file = Path("projects") / project_name / f"{project_name}.csv"
    nouvelle_ligne = pd.DataFrame([{
        "model_name": model_name, "use_model": "?", "img_size": "?", "epochs": "?", "batch": "?",
        "workers": "?", "dropout": "?", "origine": "import", "classes": class_names}])
    if csv_file.is_file():
        df = pd.concat([pd.read_csv(csv_file), nouvelle_ligne], ignore_index=True)
    else:
        df = nouvelle_ligne
    df.to_csv(csv_file, index=False)

    # labels.txt au format lu par get_labels : 'id': 'nom' sur chaque ligne
    with open(model_folder / "labels.txt", "w", encoding="utf-8") as f:
        for key, val in class_names.items():
            f.write(f"'{key}': '{val}'\n")

    flash(f"Modèle {model_name} importé.")
    # Redirection plutôt qu'affichage direct : la page de gestion construit déjà le tableau
    # des modèles, et actualiser la page ne renverra pas le formulaire une 2e fois.
    return redirect(url_for('project.gestion_modeles'))