"""
Service centralisé pour la lecture/écriture du fichier config.json.
Remplace les 'with open("config.json", ...)' dispersés dans toutes les routes.
"""

import json
from pathlib import Path

CONFIG_PATH = Path("config.json")
DEFAULT_CONFIG = {"CURRENT_PROJECT_NAME": "", "LAST_MODEL_PATH": ""}
LS_URL = ""


def load_config() -> dict:
    """Lit config.json et renvoie le dict complet (valeurs par défaut si le fichier n'existe pas encore)."""
    if not CONFIG_PATH.exists():
        return dict(DEFAULT_CONFIG)
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def load_current_project() -> str:
    """Raccourci pour récuperer uniquement le nom du projet courant."""
    return load_config().get("CURRENT_PROJECT_NAME", "")


def load_last_model_path() -> str:
    """Raccourci pour recupérer uniquement le chemin du dernier modèle entrainé."""
    return load_config().get("LAST_MODEL_PATH", "")


def save_config(project_name: str = None, last_model_path=None) -> dict:
    """
    Met à jour config.json en ne modifiant QUE les clés fournies (merge avec
    le contenu existant). Corrige le comportement actuel où certaines routes
    écrivaient un config_dict partiel et risquaient d'écraser l'autre clé
    (ex: accueil_projet() qui ne renseignait que CURRENT_PROJECT_NAME).
    """
    config = load_config()
    if project_name is not None:
        config["CURRENT_PROJECT_NAME"] = project_name
    if last_model_path is not None:
        config["LAST_MODEL_PATH"] = str(last_model_path)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f)
    return config