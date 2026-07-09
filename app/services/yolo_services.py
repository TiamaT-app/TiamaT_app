"""
Service pour le lancement des opérations YOLO longues (entraînement, prédiction)
dans des threads séparés avec gestion d'erreur.

- _run_training()      : thread d'entraînement avec gestion d'erreur
- launch_training()    : démarre le thread, initialise les variables d'état
- _run_prediction()    : thread de prédiction avec gestion d'erreur  
- launch_prediction()  : démarre le thread, initialise les variables d'état
"""

import os
import traceback
from threading import Thread

from ..src.scripts.data_preparation_and_training import training


training_state = {
    "complete": False,
    "error": None,
}

def _run_training(project_name: str, nombre_epoch: int, dropout: float, model: str, model_name: str) -> None:
    try:
        training(project_name, nombre_epoch, dropout, model, model_name)
        training_state["complete"] = True
    except Exception as e:
        traceback.print_exc()
        training_state["error"] = str(e)
        training_state["complete"] = "Error"


def launch_training(project_name: str, nombre_epoch: int, dropout: float, model: str, model_name: str) -> Thread:
    """Lance l'entraînement dans un thread séparé."""
    thread = Thread(target=_run_training, args=(project_name, nombre_epoch, dropout, model, model_name))
    thread.start()
    return thread