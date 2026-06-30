#route qui se lance au clic sur le bouton d'upload d'images dans un projet, place les images dans groundtruth

from app.app import app

from flask import Blueprint, render_template, flash, redirect, url_for
from flask_uploads import configure_uploads

from ..src.models.formulaires import ImportImages, images as images_uploadees

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
        files2=[]
        for fichier in form.fichiers.data:
            images_uploadees.save(fichier)
            files2.append(fichier.filename)
        # Get the list of files from webpage
        files2 = str(files2[0] + f" et {len(files2)-1} autre images")
        
        liste_images, liste_noms, nbre_images = get_preview_images(chemin_images)

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