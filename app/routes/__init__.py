
"""
Point d'entrée des blueprints de l'application.
Centralise l'enregistrement de toutes les routes pour ne garder qu'un seul
appel dans app.py, plutôt que d'y répéter un import + register_blueprint
par fichier de routes.
"""

from .files_routes import files_bp
from .labeling_routes import labeling_bp
from .project_route import project_bp
from .statistics_route import statistics_bp
from .testing_routes import testing_bp
from .training_routes import training_bp
from .upload_routes import upload_bp
 
def register_blueprints(app):
    app.register_blueprint(project_bp)
    app.register_blueprint(files_bp)
    app.register_blueprint(upload_bp)
    app.register_blueprint(labeling_bp)
    app.register_blueprint(statistics_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(testing_bp)
 
