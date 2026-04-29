from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, SelectMultipleField, TextAreaField, SubmitField, FloatField, FileField
from wtforms.validators import DataRequired, Length, Optional
from flask_wtf.file import MultipleFileField, FileAllowed, FileRequired
from flask_uploads import UploadSet, IMAGES

class NomDuProjet(FlaskForm):
    nom=StringField("Nom du projet", validators=[Optional(),Length(min=0)],)
    projet_existant = SelectField('Projet', choices=[], validators=[Optional()])    # submit = SubmitField('Submit')

images = UploadSet('images', IMAGES)
class ImportImages(FlaskForm):
    fichiers=MultipleFileField("Fichiers", validators =[FileRequired(), FileAllowed(images, "Images svp")])
    # submit = SubmitField('Submit')

class CheminDuModele(FlaskForm):
    
    modele_existant = SelectField('Modele', choices=[], validators=[Optional()])
    # submit = SubmitField('Submit')

images2 = UploadSet('images', IMAGES)
class ImportImages2(FlaskForm):
    fichiers=MultipleFileField("Fichiers", validators =[FileRequired(), FileAllowed(images, "Images svp")])
    modele = SelectField('Modele', choices=[], validators=[Optional()])
    # submit = SubmitField('Submit')