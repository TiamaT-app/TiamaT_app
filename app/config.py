from dotenv import load_dotenv
import os
import platform

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv('.env')

class Config():
    DEBUG = True
    PLATFORM =str(platform.system())
    WTF_CSRF_ENABLED = True
    SECRET_KEY = os.environ.get("SECRET_KEY")