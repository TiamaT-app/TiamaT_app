"""
Service pour la sélection d'un echantillon d'images à afficher en apercu
(pages d'accueil de projet et d'upload).
"""

import random
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_PREVIEW_IMAGES = 8


def get_preview_images(folder: Path, limit: int = MAX_PREVIEW_IMAGES):
    """
    Renvoie un tuple (liste_images, liste_noms, nbre_images) pour un dossier donne.

    - liste_images : chemins relatifs a la racine du repo (posix), ou le message
      "Pas encore d'images uploadees" si le dossier est vide ou absent.
    - liste_noms : noms de fichiers correspondants.
    - nbre_images : nombre total d'images trouvees (avant troncature a `limit`).
    """
    if not folder.is_dir():
        return "Pas encore d'images uploadées", [], 0

    images = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS and not f.name.startswith('.')
    ]
    nbre_images = len(images)

    if nbre_images == 0:
        return "Pas encore d'images uploadées", [], 0

    random.shuffle(images)
    selected = images[:limit]
    liste_noms = [img.name for img in selected]
    liste_images = [img.relative_to(Path.cwd()).as_posix() for img in selected]
    return liste_images, liste_noms, nbre_images