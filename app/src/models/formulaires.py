from flask_wtf import FlaskForm
from wtforms import StringField, SelectField
from wtforms.validators import InputRequired, Length, Optional, Regexp
from flask_wtf.file import MultipleFileField, FileAllowed, FileRequired, FileField
from flask_uploads import UploadSet, IMAGES

# Noms utilises comme noms de dossiers (projets, modeles) : lettres (accents compris),
# chiffres, tirets et underscores uniquement. Interdit notamment / . et les espaces,
# qui permettraient de creer ou de supprimer des dossiers en dehors de projects/.
SAFE_NAME = Regexp(r"^[\w-]+\Z",
                   message="Utilisez uniquement des lettres, des chiffres, des tirets (-) et des underscores (_).")


class NomDuProjet(FlaskForm):
    nom = StringField("Nom du projet", validators=[Optional(), Length(max=50), SAFE_NAME])
    projet_existant = SelectField('Projet', choices=[], validators=[Optional()])


class ConfirmationSuppression(FlaskForm):
    """Formulaire vide : il ne sert qu'a envoyer la suppression en POST, avec son jeton CSRF."""


images = UploadSet('images', IMAGES)
class ImportImages(FlaskForm):
    fichiers=MultipleFileField("Fichiers", validators =[FileRequired(), FileAllowed(images, "Images svp")])


class CheminDuModele(FlaskForm):
    modele_existant = SelectField('Modele', choices=[], validators=[Optional()])


images2 = UploadSet('images', IMAGES)
class ImportImages2(FlaskForm):
    fichiers=MultipleFileField("Fichiers", validators=[FileRequired(), FileAllowed(images, "Images svp")])
    modele = SelectField('Modele', choices=[], validators=[Optional()])


class ImportModel(FlaskForm):
    nom = StringField("Nom du modèle", validators=[InputRequired(), Length(max=50), SAFE_NAME])
    model = FileField('modele', validators=[
        FileRequired(),
        FileAllowed(['pt'], 'PyTorch Tensor (.pt) only!')
    ])


class LsToken(FlaskForm):
    # Le token est ecrit dans le fichier .env : on refuse tout caractere
    # (retour a la ligne, espace, =...) qui permettrait d'y ajouter d'autres lignes.
    token = StringField("token", validators=[
        InputRequired(),
        Regexp(r"^[\w.-]+\Z", message="Le token ne doit contenir ni espace ni retour à la ligne."),
    ])