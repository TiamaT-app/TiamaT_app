from flask import Flask
from app.config import Config
import jinja2
import logging
from pathlib import Path
from flask_bootstrap import Bootstrap5

app = Flask(__name__, template_folder="templates", static_folder='statics', static_url_path='/static')

bootstrap = Bootstrap5(app)

app.config.from_object(Config)

# Configuration de Jinja pour gérer les erreurs de variables indéfinies
app.jinja_env.undefined = jinja2.StrictUndefined

from app.routes import register_blueprints
register_blueprints(app)