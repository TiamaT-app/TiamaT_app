from app.app import app, Config  # Importation de l'application Flask
from flask import render_template, request, flash, redirect, url_for, abort, send_file, send_from_directory  # Fonctions Flask pour gérer les requêtes et réponses
from flask_wtf import FlaskForm
from flask_uploads import configure_uploads

from threading import Thread
import os
import sys
from pathlib import Path
import shutil
sys.path.append(str(Path(__file__).resolve().parent/"src"))
import random
import json
from datetime import datetime

# from scripts import *
from ..src.scripts.get_training_data import *
# data_preparation_and_training, statistics_for_training
from ..src.scripts.model_evaluation import *
from ..src.scripts.statistics_for_training import *
from ..src.scripts.data_preparation_and_training import *
from ..src.scripts.predicting_and_checking_yolo_results import *
from ..src.models.formulaires import images as images_uploadees
from ..src.models.formulaires import images2 as images_uploadees2
from ..src.models.formulaires import *
from ..src.scripts.generate_new_ground_truth import *

# initialisation de la variable pour le futur json de configuration
config_dict={"CURRENT_PROJECT_NAME":"", "LAST_MODEL_PATH":""}

# Route de l'accueil  pour le choix du projet
@app.route("/",methods=['GET', 'POST'])
def accueil():
    # pour marquer qu'on est pas en suite d'une passe, utile dans la route suivante 
   
    app.config["SECOND_PASS"]=False
    p = Path.cwd()
    form = NomDuProjet()
    # Liste des dossiers qui sont dans l'application de base
    liste_dossiers=["data","output","project","app"]

    # Liste de tous les dossiers qui sont actuellement dans l'application, projets inclus
    liste_projets = [x for x in os.listdir(p) if os.path.isdir(x)]
    
    # Tri des dossiers pour faire remonter ceux qui sont des projets
    for i in liste_dossiers:
        if i in liste_projets:
            liste_projets.remove(i)

    # IL FAUT UN TUPLE NOM AFFICHE + VALEUR dans les choices
    choices = [('', '-- Sélectionner un projet --')]+[(projet,projet) for projet in liste_projets]
    form.projet_existant.choices = choices
    return render_template("/pages/accueil.html" , form = form, liste_projets=liste_projets)
   
# Route qui permet le chargement des images + affichages de celles déjà chargées
@app.route("/accueil_projet",methods=['GET', 'POST'])

def accueil_projet():
    p = Path.cwd()
    form = NomDuProjet()
    # Cf. route précédente
    liste_dossiers=["data","output","project","app"]
    liste_projets = [x for x in os.listdir(p) if os.path.isdir(x)]
    for i in liste_dossiers:
        if i in liste_projets:
            liste_projets.remove(i)
    
    # IL FAUT UN TUPLE NOM AFFICHE + VALEUR dans les choices
    choices = [('', '-- Sélectionner un projet --')]+[(projet,projet) for projet in liste_projets]  
    form.projet_existant.choices = choices
    if not app.config ["SECOND_PASS"]:
        #Si il y a une donnée dans "nom", on check que ce dossier existe pas déjà : 
        if form.validate_on_submit() and form.nom.data:
                project_name= form.nom.data
                liste_noms=[]
                if project_name in os.listdir(Path.cwd()):
                    return render_template("/pages/erreur_nom_projet.html", project_name=project_name)
                else:
                    config_dict['CURRENT_PROJECT_NAME']=project_name
                    jsonstr = json.dumps(config_dict)
                    with open ("config.json","w") as f:
                        f.write(jsonstr)
                    app.config['CURRENT_PROJECT_NAME']= project_name
                    pass
                
        #s'il y a une donnée mais pas dans nom, on prend le nom de projet déjà existant du formulaire
        elif form.validate_on_submit():
                project_name= form.projet_existant.data
                config_dict['CURRENT_PROJECT_NAME']=project_name
                jsonstr = json.dumps(config_dict)
                with open ("config.json","w") as f:
                        f.write(jsonstr)
                app.config['CURRENT_PROJECT_NAME']= project_name     
        else:
            print(form.errors)
            return render_template("/pages/erreur_nom_projet2.html", project_name="erreur")
    else:
        pass    
        
    project_name = app.config['CURRENT_PROJECT_NAME']
    
    
    #on check si le projet existe déjà 
    if (Path.cwd() / project_name).is_dir():
        liste_images=[]
        pass
    else:
        shutil.copytree(Path.cwd() / 'project', project_name)
    
    #ci desssous si le projet existe déjà on sort ses images pour les afficher.
    # p1 = Path((Path.cwd()/project_name/"image_inputs"/"eval_images"))
    p_img_gt = Path.cwd()/project_name/"image_inputs"/"ground_truth_images"
    liste_images = [i for i in p_img_gt.iterdir()]
    
    
    img_ext = {'.jpg', '.jpeg', '.png'}
    
    liste_images = [i for i in liste_images if i.is_file() and i.suffix.lower() in img_ext]
    random.shuffle(liste_images)
    
    nbre_images =(len(liste_images))
    
    
    if len(liste_images)>= 8 :
        liste_images= liste_images[:8]
        liste_noms = [Path(img).name for img in liste_images]
        liste_images=[i.relative_to(Path.cwd()).as_posix() for i in liste_images]

        
    elif len(liste_images)== 0:
        liste_images = "Pas encore d'images uploadées"
        liste_noms=[]
    else :
        liste_noms = [Path(img).name for img in liste_images]
        liste_images=[Path(i).relative_to(Path.cwd()).as_posix() for i in liste_images]
        pass
    
    root = os.getcwd()
    chemin_images = os.path.join(root,project_name, "image_inputs", "ground_truth_images")
    chemin_labels =os.path.join(root,project_name, "annotations", "ground_truth")
    form2=ImportImages()
        
    return render_template("/pages/import_images.html", project_name=project_name, form2 = form2, form= form, liste_images= liste_images, liste_noms=liste_noms, nbre_images=nbre_images, chemin_images= chemin_images, chemin_labels=chemin_labels)

#la route suivante sert juste à afficher une image
@app.route('/images/<path:filename>')
def serve_image(filename):
    print((Path.cwd()))
    print(Path(filename))
    
    return send_from_directory(os.getcwd(), filename)

@app.route('/serve_images/<path:relpath>')
def serve_image2(relpath):
    filename= Path(relpath).name
    directory = Path.cwd() / relpath
    directory = directory.parents[0]
    return send_from_directory(directory, filename)

@app.route('/download/<path:filepath>')
def download(filepath):
    if not Path.is_absolute(Path(filepath)):
        filepath='/'+filepath
    return send_file(filepath, as_attachment=True)


# route upload les images quand on clique sur 'upload'
@app.route("/upload", methods = ['GET', 'POST'])
#route qui se lance au clic sur le bouton d'upload d'images dans un projet, place les images dans groundtruth
def upload():
    with open("config.json","r") as f:
        app.config["CURRENT_PROJECT_NAME"]=json.load(f).get("CURRENT_PROJECT_NAME")
    print(app.config['CURRENT_PROJECT_NAME'])
    project_name = app.config['CURRENT_PROJECT_NAME']
    app.config['UPLOADED_IMAGES_DEST'] = os.path.join(project_name, "image_inputs", "ground_truth_images")
    form = ImportImages()
    form2= ImportImages()
    configure_uploads(app, images_uploadees)
    root=os.getcwd()
    chemin_images = os.path.join(root,project_name, "image_inputs", "ground_truth_images")
    chemin_labels =os.path.join(root,project_name, "annotations", "ground_truth")
    if form.validate_on_submit():
        files2=[]
        for fichier in form.fichiers.data:
            images_uploadees.save(fichier)
            files2.append(fichier.filename)
        # Get the list of files from webpage
        files2 = str(files2[0] + f" et {len(files2)-1} autre images")
    
       
        p_img_gt = Path.cwd()/project_name/"image_inputs"/"ground_truth_images"
        liste_images = [i for i in p_img_gt.iterdir()]

        img_ext = {'.jpg', 'jpeg', '.png'}
        
        liste_images = [i for i in liste_images if i.is_file() and i.suffix.lower() in img_ext]
        random.shuffle(liste_images)
        nbre_images= len(liste_images)
        
        if len(liste_images)>= 8 :
            liste_images= liste_images[:8]
            liste_noms = [Path(img).name for img in liste_images]
            liste_images=[i.relative_to(Path.cwd()).as_posix() for i in liste_images]

            
        elif len(liste_images)== 0:
            liste_images = "Pas encore d'images uploadées"
            liste_noms=[]
        else :
            liste_noms = [Path(img).name for img in liste_images]
            liste_images=[Path(i).relative_to(Path.cwd()).as_posix() for i in liste_images]
            pass
        
        root = os.getcwd()
        chemin_images = os.path.join(root,project_name, "image_inputs", "ground_truth_images")
        chemin_labels =os.path.join(root,project_name, "annotations", "ground_truth")
        return render_template ("/pages/upload_full.html", project_name= project_name, files2 = files2, form=form, form2=form2, chemin_images=str(chemin_images), chemin_labels=str(chemin_labels), liste_images=liste_images, liste_noms=liste_noms )
    else:
        print(form.errors)
        print("erreur de formulaire")


@app.route("/label_lancement",methods=['GET', 'POST'])
def consignes() :
    root = Path.cwd()
    with open("config.json","r") as f:
        app.config["CURRENT_PROJECT_NAME"]=json.load(f).get("CURRENT_PROJECT_NAME")
    project_name = app.config['CURRENT_PROJECT_NAME']

    chemin_images = os.path.join(root,project_name, "image_inputs", "ground_truth_images")
    chemin_labels =os.path.join(root,project_name, "annotations", "ground_truth")
    

    clean_image_name(os.path.join(root, project_name))
    os.environ['LOCAL_FILES_DOCUMENT_ROOT'] = f'{root.as_posix()}'
    os.environ['LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED'] = 'true'
    #ouverture de label studio en fond (thread), affichage de la page en même temps. 
    thread = Thread(target=launch_LS)
    thread.daemon = True
    thread.start()
   
    #pour la future extinction automatique de LS voir cette page : https://stackoverflow.com/questions/31712056/how-do-i-get-a-threads-pid
    return render_template("/pages/lancement_label_studio.html", project_name=project_name, chemin_images=chemin_images, chemin_labels=chemin_labels)

    


@app.route("/dataset_statistics", methods=['GET', 'POST'])
def dataset_statistics():
    with open("config.json","r") as f:
        app.config["CURRENT_PROJECT_NAME"]=json.load(f).get("CURRENT_PROJECT_NAME")
    project_folder =os.path.join(os.getcwd(), app.config['CURRENT_PROJECT_NAME'])
    create_dataset(project_folder, manually_downloaded=False)
    print("dataset created")
    # Create the statistic folder
    create_stats_folder(project_folder)
    
    clean_LS(project_folder, annotated_with_LS=False)
    
    encoding(project_folder)
    
    annotations_per_img(project_folder)
    
    classes_distribution(project_folder)
    print("class distributon done")
    get_global_results(project_folder)
    print("got global results")
    
    return render_template("/pages/class_distrib.html",class_distibution_path=url_for('serve_dataset_image'), project_name=app.config['CURRENT_PROJECT_NAME'])


@app.route('/dataset_stats')
def serve_dataset_image():
    with open("config.json","r") as f:
        app.config["CURRENT_PROJECT_NAME"]=json.load(f).get("CURRENT_PROJECT_NAME")
    project_name=app.config['CURRENT_PROJECT_NAME']
    filename='class_distribution.png'
    file_path = Path.cwd() / "data" / project_name / "dataset_statistics" / filename
    if file_path.exists():
        return send_file(file_path, mimetype='image/png')
    return "File not found", 404


@app.route("/training_setup", methods=['GET', 'POST'])
def training_setup():
    with open("config.json","r") as f:
        app.config["CURRENT_PROJECT_NAME"]=json.load(f).get("CURRENT_PROJECT_NAME")
    project_name = app.config["CURRENT_PROJECT_NAME"]
    if Path("output").is_dir():
        liste_modeles = get_model_list(project_name)
    else:
         liste_modeles=["------"]
    return render_template("/pages/training_setup.html", liste_modeles=liste_modeles)



@app.route("/start_training", methods=['GET', 'POST'])
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

@app.route("/check_training_status")
def check_training_status():
    if os.environ["complete"]== "True":
        return redirect(url_for('result'))
    return render_template('/pages/loading.html')

    #Config du réentraînement
@app.route("/training_result")
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


#la route ci dessous récupère des images pour tester le modèle et fait passer le modèle dessus
@app.route("/test_modele", methods=['GET', 'POST'])
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

@app.route("/dispatch_corrections",methods=['GET', 'POST'])
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


@app.route("/evaluate_model",methods=['GET', 'POST'])
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
    parts = png_path.parts
    idx = parts.index("version_appli")
    png_path = Path(*parts[idx + 1:])
    # Generate the confusion matrix
    matrix_path=create_confusion_matrix(project_folder, yolo_model_folder)
    parts = matrix_path.parts
    idx = parts.index("version_appli")
    matrix_path = Path(*parts[idx + 1:])
    return render_template("/pages/model_evaluation.html", project_name=project_name, png_path=Path(png_path).as_posix(), matrix_path=Path(matrix_path).as_posix())

@app.route("/test_pre_trained_model",methods=['GET', 'POST'])
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

@app.route("/test_images_pretrained", methods=["GET","POST"])
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


