from flask import Flask
from app.config import Config
import jinja2
import logging
from pathlib import Path
import os
from flask_bootstrap import Bootstrap5

app = Flask(__name__, template_folder="templates", static_folder='statics', static_url_path='/static')

env_path = Path('.env')
if not env_path.exists():
        with open('.env', 'w') as f:
            f.write(f'SECRET_KEY={os.urandom(24)}')
bootstrap = Bootstrap5(app)
app.config.from_object(Config)

# Configuration de Jinja pour gérer les erreurs de variables indéfinies
app.jinja_env.undefined = jinja2.StrictUndefined

from app.routes.project_route import project_bp
app.register_blueprint(project_bp)

from app.routes.files_routes import files_bp
app.register_blueprint(files_bp)

from app.routes.upload_routes import upload_bp
app.register_blueprint(upload_bp)

from app.routes.labeling_routes import labeling_bp
app.register_blueprint(labeling_bp)

from app.routes.statistics_route import statistics_bp
app.register_blueprint(statistics_bp)

from app.routes.training_routes import training_bp
app.register_blueprint(training_bp)

from app.routes.testing_routes import testing_bp
app.register_blueprint(testing_bp)