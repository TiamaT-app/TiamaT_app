import os
import subprocess
from pathlib import Path

from app.config import Config

LS_PORT = Config.LABEL_STUDIO_PORT

def launch_LS():
    '''Function that launches Label Studio in a non-blocking subprocess'''
    os.environ['LOCAL_FILES_DOCUMENT_ROOT'] = Path.cwd().as_posix()
    os.environ['LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT'] = Path.cwd().as_posix()
    os.environ['LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED'] = 'true'

    token = os.environ.get("LABEL_STUDIO_API_TOKEN", "")
    
    cmd: list[str] = [
        'label-studio', 'start',
        '--port', str(LS_PORT),
        '--user-token', token,
        '--enable-legacy-api-token',
    ]
    print(f"DEBUG cmd : {cmd}")
    subprocess.Popen(cmd, env=os.environ.copy())