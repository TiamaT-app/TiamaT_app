import os
from pathlib import Path

def launch_LS():
    '''Function that launches LS'''
    os.environ['LOCAL_FILES_DOCUMENT_ROOT'] = Path.cwd().as_posix()
    os.environ['LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT'] = Path.cwd().as_posix()
    os.environ['LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED'] = 'true'
    
    print("launching LS")
    print(f"Document root: {Path.cwd().as_posix()}")
    os.system('label-studio start --port 8081')