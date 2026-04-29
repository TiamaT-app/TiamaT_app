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

from .routes import generales