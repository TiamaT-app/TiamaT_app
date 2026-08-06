#route qui se lance au clic sur le bouton d'upload d'images dans un projet, place les images dans groundtruth
from pathlib import Path
import os
import pandas as pd
from ultralytics import YOLO
from app.app import app

from flask import Blueprint, render_template, flash, redirect, url_for
from flask_uploads import configure_uploads

from ..src.models.formulaires import ImportModel, ImportImages, images as images_uploadees

from app.services.config_service import load_current_project
from app.services.project_service import projects_folder
from app.services.image_service import get_preview_images


upload_bp = Blueprint("upload", __name__)


@upload_bp.route("/upload", methods = ['GET', 'POST'])
def upload():
    
    project_name = load_current_project()
    chemin_images = projects_folder / project_name / "image_inputs" / "ground_truth_images"
    chemin_labels = projects_folder / project_name / "annotations" / "ground_truth"
    
    # NB : doit pointer sous projects/, comme chemin_images ci-dessus
    app.config['UPLOADED_IMAGES_DEST'] = str(chemin_images)
    
    form = ImportImages()
    form2= ImportImages()
    configure_uploads(app, images_uploadees)
    
    if form.validate_on_submit():
        list_img = []
        for fichier in form.fichiers.data:
            images_uploadees.save(fichier)
            list_img.append(fichier.filename)
        # Get the list of files from webpage
        files2 = f"Vos {len(list_img)} images"
        
        liste_images, liste_noms, _ = get_preview_images(chemin_images)

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
    

@upload_bp.route("/upload_modele", methods = ['GET', 'POST'])
def upload_modele():
    project_name = load_current_project()
    form = ImportModel()
    if form.validate_on_submit():
        model_name= f"{project_name}_{form.nom.data}"
        fichier = form.model.data
        if (Path("output")/"train"/model_name).is_dir():
            flash("Ce modèle existe déjà, merci de choisir un autre nom.")
            return redirect(url_for('project.gestion_modeles'))
        else:
            os.makedirs((Path("output")/"train"/model_name))
            os.makedirs((Path("output")/"train"/model_name/"weights"))

            if (Path("projects")/project_name/f"{project_name}.csv").is_file():
                df = pd.read_csv(Path("projects")/project_name/f"{project_name}.csv")
                dico = {"use_model": "?", "model_name": model_name, "img_size": "?" , "epochs": "?", "batch": "?", "workers": "?", "dropout": "?", "origine": "import"}
                df.loc[len(df)] = dico
                df.to_csv(Path("projects")/project_name/f"{project_name}.csv", index =False)
            else :
                dico = {"use_model": "?", "model_name": model_name, "img_size": "?" , "epochs": "?", "batch": "?", "workers": "?", "dropout": "?", "origine": "import"}
                df = pd.DataFrame(dico, index=[0])
                df.to_csv(Path("projects")/project_name/f"{project_name}.csv", index = False)
            fichier = form.model.data
            fichier.save((Path("output")/"train"/model_name/"weights"/"best.pt"))
            
            with open (Path("output")/"train"/model_name/ "labels.txt", "w") as f:
                for key, val in (YOLO((Path("output")/"train"/model_name/"weights"/"best.pt")).names).items() :
                    f.write(f"'{key}': '{val}'\n")
            liste_urls =[] 
            for index, row in df.iterrows():
                path_model = Path.cwd()/"output"/"train"/row["model_name"]/ "weights"/"best.pt"
                if os.path.isfile(path_model):
                    liste_urls.append(f'<a href="/download/{path_model}" class="btn btn-primary btn-sm">Download</a>')
                else:
                    liste_urls.append(f"Erreur avec le modèle {row['model_name']}")
            df["urls"]= liste_urls 
            
            return render_template("pages/gestion_modeles.html", tables=[df.to_html(classes='data', escape=False)], titles=df.columns.values, nom_projet = project_name, form = form)
    else:
        flash("Problème avec le formulaire.")
        return redirect(url_for('project.gestion_modeles'))



        
        
        
