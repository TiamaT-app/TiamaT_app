"""
Routes qui servent des fichiers au navigateur.

- serve_image()  (/images/<chemin>)       : images affichees dans les pages (corpus, matrices...)
- serve_image2() (/serve_images/<chemin>) : idem, utilisee par la page d'evaluation
- download()     (/download/<chemin>)     : telechargement des poids d'un modele (best.pt)

Ces routes recoivent un chemin dans l'URL. Elles ne servent que les fichiers
attendus (images dans projects/, output/ et data/, poids .pt dans output/train/)
et renvoient une erreur 404 pour tout le reste : sans cette verification,
n'importe qui sur le reseau pourrait lire le .env, la configuration ou
n'importe quel fichier de l'ordinateur.
"""

import os
from pathlib import Path

from flask import Blueprint, abort, send_file

files_bp = Blueprint("files", __name__)

IMAGE_FOLDERS = ["projects", "output", "data"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def _allowed_file(requested: str, allowed_folders: list[str], allowed_extensions: set[str]) -> Path:
    """
    Renvoie le chemin du fichier demande s'il est autorise, sinon interrompt la requete (404).

    Un fichier est autorise s'il :
    - se trouve dans l'un des allowed_folders (relatifs au dossier de l'app),
    - ne passe par aucun fichier ou dossier cache (.env, .git...),
    - a une extension de allowed_extensions,
    - existe.
    """
    root = Path.cwd()
    candidate = Path(requested)
    if not candidate.is_absolute():
        candidate = root / candidate
    # abspath supprime les ".." sans suivre les liens symboliques : un dossier
    # d'images rattache par un lien symbolique dans projects/ reste accessible.
    candidate = Path(os.path.abspath(candidate))

    for folder in allowed_folders:
        base = Path(os.path.abspath(root / folder))
        if candidate.is_relative_to(base):
            break
    else:
        abort(404)

    if any(part.startswith(".") for part in candidate.relative_to(base).parts):
        abort(404)
    if candidate.suffix.lower() not in allowed_extensions or not candidate.is_file():
        abort(404)
    return candidate


@files_bp.route('/images/<path:filename>')
def serve_image(filename):
    return send_file(_allowed_file(filename, IMAGE_FOLDERS, IMAGE_EXTENSIONS))


@files_bp.route('/serve_images/<path:relpath>')
def serve_image2(relpath):
    return send_file(_allowed_file(relpath, IMAGE_FOLDERS, IMAGE_EXTENSIONS))


@files_bp.route('/download/<path:filepath>')
def download(filepath):
    # Les liens de telechargement contiennent un chemin absolu (/Users/...) :
    # Flask fusionne le double slash de /download//Users/..., on remet donc le / initial.
    if not Path(filepath).is_absolute():
        filepath = '/' + filepath
    return send_file(_allowed_file(filepath, ["output/train"], {".pt"}), as_attachment=True)