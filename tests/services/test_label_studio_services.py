"""Tests des services Label Studio : lancement, attente du serveur, projets d'annotation et de correction.

Label Studio n'est jamais contacté : son client (LabelStudio, du SDK) est remplacé par un faux
client qui garde les projets et les tâches en mémoire.
"""

import json
import os
import types

import pytest

import app.services.label_studio_api_service as api
import app.services.label_studio_service as service
from app.services.project_service import create_project_structure

PREFIXE = "/data/local-files/?d=projects/manuscrits/image_inputs/ground_truth_images/"


# --- Un faux Label Studio, en mémoire ---------------------------------------------------------------

class FauxLabelStudio:
    def __init__(self):
        self.projets = {}       # id -> {"title", "label_config"}
        self.taches = {}        # id projet -> liste de tâches {"id", "data", "annotations"}
        self.synchronisations = []
        self.suppressions = 0
        self._id = 0

        faux = self
        self.projects = types.SimpleNamespace(
            list=lambda: [types.SimpleNamespace(id=i, title=p["title"]) for i, p in faux.projets.items()],
            create=faux._creer_projet,
            update=faux._mettre_a_jour_projet,
            get=lambda id: types.SimpleNamespace(id=id, label_config=faux.projets[id]["label_config"],
                                                 description=faux.projets[id].get("description")),
            import_tasks=faux._importer,
        )
        self.tasks = types.SimpleNamespace(
            list=lambda project: [types.SimpleNamespace(id=t["id"], data=t["data"]) for t in faux.taches[project]],
            delete=faux._supprimer_tache,
        )
        storage = types.SimpleNamespace(create=lambda **kw: types.SimpleNamespace(id=faux._nouvel_id()),
                                        sync=lambda id: faux.synchronisations.append(id),
                                        list=lambda project: [])
        self.import_storage = types.SimpleNamespace(local=storage)
        self.export_storage = types.SimpleNamespace(local=storage)

    def _nouvel_id(self):
        self._id += 1
        return self._id

    def _creer_projet(self, title, label_config=None, description=None):
        i = self._nouvel_id()
        self.projets[i] = {"title": title, "label_config": label_config, "description": description}
        self.taches[i] = []
        return types.SimpleNamespace(id=i)

    def _mettre_a_jour_projet(self, id, **champs):
        self.projets[id].update(champs)

    def _importer(self, id, request):
        for tache in request:
            self.taches[id].append({"id": self._nouvel_id(), "data": tache["data"],
                                    "annotations": tache.get("annotations", [])})

    def _supprimer_tache(self, id):
        self.suppressions += 1
        for taches in self.taches.values():
            taches[:] = [t for t in taches if t["id"] != id]

    def images(self, projet):
        return sorted(t["data"]["image"] for t in self.taches[projet])


@pytest.fixture
def ls(mocker):
    faux = FauxLabelStudio()
    mocker.patch.object(api, "_get_client", return_value=faux)
    mocker.patch.object(api, "wait_for_label_studio", return_value="http://localhost:8080")
    return faux


@pytest.fixture
def projet(espace_projets):
    projet = create_project_structure("manuscrits")
    return {
        "images": projet / "image_inputs" / "ground_truth_images",
        "gt": projet / "annotations" / "ground_truth",
        "labels": espace_projets / "labels.txt",
    }


def image(projet, nom):
    (projet["images"] / nom).write_bytes(b"")


def annotation_gt(projet, numero, chemin_image, classe="miniature"):
    (projet["gt"] / str(numero)).write_text(json.dumps({
        "id": numero, "project": 1,
        "result": [{"value": {"x": 10, "y": 10, "width": 5, "height": 5, "rectanglelabels": [classe]}}],
        "task": {"data": {"image": chemin_image}}}))


def setup(projet):
    return api.setup_label_studio_project("manuscrits", projet["images"], projet["gt"], projet["labels"])


# --- Normalisation des chemins ------------------------------------------------------------------------

def test_normalize_image_url():
    encode = "/data/local-files/?d=projects/noam%C3%A9/image_inputs/ground_truth_images/f%201.jpg"
    brut = "/data/local-files/?d=projects/noamé/image_inputs/ground_truth_images/f 1.jpg"
    decompose = "/data/local-files/?d=projects/noame\u0301/image_inputs/ground_truth_images/f 1.jpg"  # forme macOS
    assert api.normalize_image_url(encode) == api.normalize_image_url(brut) == api.normalize_image_url(decompose)


# --- Projet d'annotation de la vérité terrain ------------------------------------------------------------

def test_creation_du_projet(ls, projet):
    image(projet, "f1.jpg")
    projet["labels"].write_text("'0': 'miniature'\n'1': 'initiale'\n", encoding="utf-8")

    projet_id, url = setup(projet)

    assert ls.projets[projet_id]["title"] == "manuscrits"
    assert api._labels_in_config(ls.projets[projet_id]["label_config"]) == ["miniature", "initiale"]
    assert len(ls.synchronisations) == 1   # les tâches sont créées par la synchronisation du stockage
    assert url == "http://localhost:8080"


def test_une_image_deja_dans_label_studio_n_est_pas_importee_deux_fois(ls, projet):
    # Le doublement observé : Label Studio enregistre le chemin encodé, TiamaT le chemin brut.
    projet_id = ls._creer_projet("manuscrits").id
    ls._importer(projet_id, [{"data": {"image": PREFIXE.replace("manuscrits", "noam%C3%A9") + "f1.jpg"}}])
    images = create_project_structure("noamé") / "image_inputs" / "ground_truth_images"
    (images / "f1.jpg").write_bytes(b"")
    ls.projets[projet_id]["title"] = "noamé"

    api.setup_label_studio_project("noamé", images, images.parent.parent / "annotations" / "ground_truth")

    assert len(ls.taches[projet_id]) == 1


def test_image_corrigee_importee_avec_son_annotation(ls, projet):
    # Fin de cycle : l'image corrigée et sa correction rejoignent la vérité terrain.
    # Au cycle suivant, elle doit arriver dans Label Studio avec son annotation, même si
    # le chemin enregistré dans la correction est encodé (cause des corrections « disparues »).
    projet_id = ls._creer_projet("manuscrits").id
    image(projet, "folio é.jpg")
    annotation_gt(projet, 7, PREFIXE + "folio%20%C3%A9.jpg")

    setup(projet)

    tache = ls.taches[projet_id][0]
    assert tache["annotations"][0]["result"][0]["value"]["rectanglelabels"] == ["miniature"]


def test_seules_les_nouvelles_images_sont_importees(ls, projet):
    projet_id = ls._creer_projet("manuscrits").id
    ls._importer(projet_id, [{"data": {"image": PREFIXE + "f1.jpg"}}])
    for nom in ["f1.jpg", "f2.JPG", "f3.tif", ".DS_Store"]:
        image(projet, nom)

    setup(projet)

    assert ls.images(projet_id) == [PREFIXE + "f1.jpg", PREFIXE + "f2.JPG", PREFIXE + "f3.tif"]


def test_les_classes_creees_dans_label_studio_sont_conservees(ls, projet):
    # Une classe créée dans l'interface de Label Studio n'est pas encore dans labels.txt :
    # la mise à jour de la configuration ne doit pas la faire disparaître.
    projet_id = ls._creer_projet("manuscrits", label_config=api.build_ls_label_config(["miniature", "enluminure"])).id
    projet["labels"].write_text("'0': 'miniature'\n'1': 'initiale'\n", encoding="utf-8")

    setup(projet)

    assert api._labels_in_config(ls.projets[projet_id]["label_config"]) == ["miniature", "initiale", "enluminure"]


# --- Projet de correction des prédictions --------------------------------------------------------------------

@pytest.fixture
def fichiers_correction(tmp_path):
    def ecrire(*noms, modele="modele_1"):
        taches = [{"data": {"image": f"/data/local-files/?d=projects/manuscrits/image_inputs/eval_images/{n}"},
                   "predictions": [{"model_version": modele, "result": []}]} for n in noms]
        (tmp_path / "taches.json").write_text(json.dumps(taches))
        (tmp_path / "config.txt").write_text(api.build_ls_label_config(["miniature"]))
        return tmp_path / "taches.json", tmp_path / "config.txt"
    return ecrire


def correction(fichiers):
    taches, config = fichiers
    return api.setup_ls_correction_project("manuscrits", "corrections", "eval_images", taches, config)


def test_creation_du_projet_de_correction(ls, fichiers_correction):
    projet_id, _ = correction(fichiers_correction("t1.jpg", "t2.jpg"))
    assert ls.projets[projet_id]["title"] == "manuscrits_corrections"
    assert len(ls.taches[projet_id]) == 2


def test_recharger_la_page_de_correction_garde_les_taches(ls, fichiers_correction):
    # Avant, chaque rechargement supprimait et réimportait toutes les tâches,
    # ce qui faisait perdre les corrections en cours.
    fichiers = fichiers_correction("t1.jpg", "t2.jpg")
    projet_id, _ = correction(fichiers)
    ids_avant = [t["id"] for t in ls.taches[projet_id]]

    correction(fichiers)

    assert [t["id"] for t in ls.taches[projet_id]] == ids_avant
    assert ls.suppressions == 0


def test_autre_modele_sur_les_memes_images_remplace_les_taches(ls, fichiers_correction):
    # Mêmes images, mais prédictions d'un autre modèle : les tâches doivent être remplacées,
    # sinon on corrigerait les prédictions de l'ancien modèle.
    projet_id, _ = correction(fichiers_correction("t1.jpg", modele="modele_1"))

    correction(fichiers_correction("t1.jpg", modele="modele_2"))

    assert ls.suppressions == 1
    assert len(ls.taches[projet_id]) == 1


def test_nouvelles_images_de_test_remplacent_les_anciennes(ls, fichiers_correction):
    projet_id, _ = correction(fichiers_correction("t1.jpg", "t2.jpg"))

    correction(fichiers_correction("t3.jpg"))

    assert [t["data"]["image"].rsplit("/", 1)[-1] for t in ls.taches[projet_id]] == ["t3.jpg"]


# --- Connexion et lancement ------------------------------------------------------------------------------

def test_le_token_est_lu_au_moment_de_l_appel(mocker, monkeypatch):
    # Un token ajouté par le formulaire après le démarrage de l'app doit être utilisé tout de suite.
    monkeypatch.setenv("LABEL_STUDIO_API_TOKEN", "nouveau-token")
    client = mocker.patch.object(api, "LabelStudio")
    api._get_client()
    assert client.call_args.kwargs["api_key"] == "nouveau-token"


def test_attente_token_refuse(mocker, monkeypatch):
    # Label Studio répond mais refuse le token : erreur claire et immédiate, pas 3 minutes d'attente.
    class TokenRefuse(Exception):
        status_code = 401
    mocker.patch.object(api, "is_label_studio_running", return_value=True)
    mocker.patch.object(api, "LabelStudio").return_value.projects.list.side_effect = TokenRefuse()

    with pytest.raises(PermissionError, match="token"):
        api.wait_for_label_studio(timeout=5)


def test_attente_label_studio_injoignable(mocker):
    mocker.patch.object(api, "is_label_studio_running", return_value=False)
    assert api.wait_for_label_studio(timeout=0.2, interval=0.1) is None


def test_lancement_asynchrone_appelle_on_ready(mocker):
    # Même si Label Studio tournait déjà (launch_LS ne lance rien), on_ready doit être appelé.
    mocker.patch.object(service, "launch_LS", return_value=False)
    appels = []

    thread = service.launch_label_studio_async(on_ready=lambda: appels.append("pret"))
    thread.join(timeout=2)

    assert appels == ["pret"]


def test_meme_racine_que_launch_ls(espace_projets, monkeypatch):
    # La racine des fichiers locaux doit être le dossier de l'app, comme dans launch_LS :
    # les chemins des tâches commencent par ?d=projects/...
    for variable in ["LOCAL_FILES_DOCUMENT_ROOT", "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT",
                     "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"]:
        monkeypatch.delenv(variable, raising=False)  # remises dans leur état d'origine après le test
    service.configure_label_studio_root()
    assert os.environ["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] == espace_projets.as_posix()