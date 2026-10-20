"""
Service pour la gestion des projets TiamaT : listing des projets existants,
creation de la structure de dossiers d'un nouveau projet sous projects/,
et suppression d'un projet existant.
"""

import shutil
from pathlib import Path

import pandas as pd


projects_folder = Path.cwd() / "projects"
projects_folder.mkdir(exist_ok=True)
 
def list_existing_projects() -> list:
    """Liste les noms (str) des projets existants sous projects_folder."""
    return [
        project.name for project in projects_folder.iterdir()
        if project.is_dir() and not project.name.startswith('.')
    ]
 
 
def project_exists(project_name: str) -> bool:
    """Verifie si un dossier de projet de ce nom existe deja."""
    return (projects_folder / project_name).is_dir()
 
 
def create_project_structure(project_name: str) -> Path:
    """
    Cree l'arborescence d'un nouveau projet (images + annotations, ground
    truth + eval) sous projects/{project_name}/. Ne fait rien si le projet
    existe deja (idempotent).
    """
    project_path = projects_folder / project_name
    if not project_path.is_dir():
        (project_path / "image_inputs" / "ground_truth_images").mkdir(parents=True, exist_ok=True)
        (project_path / "image_inputs" / "eval_images").mkdir(parents=True, exist_ok=True)
        (project_path / "annotations" / "ground_truth").mkdir(parents=True, exist_ok=True)
        (project_path / "annotations" / "prediction_corrections").mkdir(parents=True, exist_ok=True)
    return project_path

 
def _remove_folder_inside(folder: Path, parent: Path) -> None:
    """
    Supprime folder (s'il existe) uniquement s'il est un sous-dossier direct de parent.
    Garde-fou contre un nom detourne ("..", "a/../b"...) qui ferait supprimer
    un dossier ailleurs sur le disque.
    """
    if folder.resolve().parent != parent.resolve():
        print(f"Suppression ignorée, chemin hors de {parent} : {folder}")
        return
    shutil.rmtree(folder, ignore_errors=True)


def delete_project(project_name: str) -> None:
    """
    Supprime un projet et tout ce qui lui est associe :
    - projects/{project_name}/
    - data/{project_name}/
    - pour chaque modele liste dans projects/{project_name}/{project_name}.csv :
      output/train/{modele}/ et output/predict/{project_name}_{modele}/
 
    Les dossiers deja absents sont ignores, pour qu'une suppression ne
    s'arrete jamais a mi-chemin.
 
    :raises ValueError: si project_name n'est pas un projet existant. C'est le
        garde-fou essentiel : un nom vide designerait le dossier projects/
        lui-meme, et tous les projets seraient supprimes.
    """
    if project_name not in list_existing_projects():
        raise ValueError(f"Projet introuvable : '{project_name}'")
 
    root = projects_folder.parent
    project_path = projects_folder / project_name
    train_folder = root / "output" / "train"
    predict_folder = root / "output" / "predict"
 
    csv_file = project_path / f"{project_name}.csv"
    if csv_file.is_file():
        model_names = pd.read_csv(csv_file)["model_name"].dropna().astype(str).tolist()
    else:
        model_names = []
 
    for model_name in model_names:
        # Dossier d'entrainement du modele : output/train/{modele}
        _remove_folder_inside(train_folder / model_name, train_folder)
        # Dossier des predictions du modele sur ce projet : output/predict/{projet}_{modele}
        _remove_folder_inside(predict_folder / f"{project_name}_{model_name}", predict_folder)

    # Donnees preparees pour l'entrainement : data/{projet}
    _remove_folder_inside(root / "data" / project_name, root / "data")
    # Dossier du projet lui-meme : projects/{projet} (nom deja verifie en debut de fonction)
    shutil.rmtree(project_path)
