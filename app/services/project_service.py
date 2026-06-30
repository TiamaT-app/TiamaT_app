"""
Routes de gestion des projets TiamaT : choix/création d'un projet (page
d'accueil) et page d'import d'images pour ce projet.
"""

from pathlib import Path

projects_folder = Path.cwd() / "projects"
projects_folder.mkdir(exist_ok=True)
 
def list_existing_projects() -> list:
    """Liste les noms (str) des projets existants sous projects_folder."""
    return [
        x.name for x in projects_folder.iterdir()
        if x.is_dir() and not x.name.startswith('.')
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
