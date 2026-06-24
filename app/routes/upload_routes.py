#route qui se lance au clic sur le bouton d'upload d'images dans un projet, place les images dans groundtruth

import os
import json
import random
from pathlib import Path

from app.app import app
from app.config import Config

from flask import Blueprint, render_template, flash, redirect, url_for
from flask_uploads import configure_uploads

from ..src.models.formulaires import ImportImages, images as images_uploadees

upload_bp = Blueprint("upload", __name__)

projects_folder = Path.cwd() / "projects"

@upload_bp.route("/upload", methods = ['GET', 'POST'])
def upload():
    with open("config.json","r") as f:
        app.config["CURRENT_PROJECT_NAME"]=json.load(f).get("CURRENT_PROJECT_NAME")
    print(app.config['CURRENT_PROJECT_NAME'])
    
    project_name = app.config['CURRENT_PROJECT_NAME']
    chemin_images = projects_folder / project_name / "image_inputs" / "ground_truth_images"
    chemin_labels = projects_folder / project_name / "annotations" / "ground_truth"
    
    # NB : doit pointer sous projects/, comme chemin_images ci-dessus
    app.config['UPLOADED_IMAGES_DEST'] = str(chemin_images)
    
    form = ImportImages()
    form2= ImportImages()
    configure_uploads(app, images_uploadees)
    
    if form.validate_on_submit():
        files2=[]
        for fichier in form.fichiers.data:
            images_uploadees.save(fichier)
            files2.append(fichier.filename)
        # Get the list of files from webpage
        files2 = str(files2[0] + f" et {len(files2)-1} autre images")
        
        IMG_EXT = {'.jpg', '.jpeg', '.png'}
        liste_images = [
            i for i in chemin_images.iterdir() 
            if i.is_file() and i.suffix.lower() in IMG_EXT and not i.name.startswith('.')
            ]
        random.shuffle(liste_images)
        nb_images = len(liste_images)
        
        if nb_images >= 8 :
            liste_images = liste_images[:8]
            liste_noms = [Path(img).name for img in liste_images]
            liste_images = [i.relative_to(Path.cwd()).as_posix() for i in liste_images]
        elif nb_images == 0:
            liste_images = "Pas encore d'images uploadées"
            liste_noms = []
        else :
            liste_noms = [Path(img).name for img in liste_images]
            liste_images=[Path(i).relative_to(Path.cwd()).as_posix() for i in liste_images]

        return render_template(
            "/pages/upload_full.html", 
            project_name= project_name, 
            files2 = files2, 
            form=form, 
            form2=form2, 
            chemin_images=str(chemin_images), 
            chemin_labels=str(chemin_labels), 
            liste_images=liste_images, 
            liste_noms=liste_noms)
    else:
        print(form.errors)
        print("erreur de formulaire")
        flash("Erreur lors de l'upload des images, merci de réessayer.")
        return redirect(url_for('project.accueil_projet'))