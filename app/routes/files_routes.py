import os
from pathlib import Path

from flask import send_file, send_from_directory, Blueprint

files_bp = Blueprint("files", __name__)

#la route suivante sert juste à afficher une image
@files_bp.route('/images/<path:filename>')
def serve_image(filename):
    print((Path.cwd()))
    print(Path(filename))
    
    return send_from_directory(os.getcwd(), filename)

@files_bp.route('/serve_images/<path:relpath>')
def serve_image2(relpath):
    filename= Path(relpath).name
    directory = Path.cwd() / relpath
    directory = directory.parents[0]
    return send_from_directory(directory, filename)

@files_bp.route('/download/<path:filepath>')
def download(filepath):
    if not Path.is_absolute(Path(filepath)):
        filepath='/'+filepath
    return send_file(filepath, as_attachment=True)

