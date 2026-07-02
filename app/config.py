import os
import platform
from pathlib import Path
from dotenv import load_dotenv

env_path = Path('.env')
if not env_path.exists():
    env_path.write_text(f'SECRET_KEY={os.urandom(24).hex()}')

load_dotenv(env_path)

class Config():
    DEBUG = True
    PLATFORM = str(platform.system())
    WTF_CSRF_ENABLED = True
    SECRET_KEY = os.environ.get("SECRET_KEY")
    LABEL_STUDIO_PORT = int(os.environ.get("LABEL_STUDIO_PORT", 8080))