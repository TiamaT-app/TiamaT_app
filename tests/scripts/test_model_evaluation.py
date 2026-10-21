"""Tests de l'évaluation d'un modèle : comparaison des prédictions et des corrections."""

import json

import pandas as pd
import pytest

import app.src.scripts.model_evaluation as ev


# --- IoU et appariement des boîtes ----------------------------------------------------------------

@pytest.mark.parametrize(
    "boite_b, iou_attendue",
    [
        ([0, 0.10, 0.10, 0.10, 0.10], 1.0),      # même boîte
        ([0, 0.15, 0.10, 0.10, 0.10], 1 / 3),    # décalée d'une demi-largeur : intersection 1/2, union 3/2
        ([0, 0.90, 0.90, 0.10, 0.10], 0.0),      # aucun recouvrement
        ([0, 0.10, 0.10, 0.05, 0.05], 0.25),     # boîte 4 fois plus petite, contenue dans l'autre
    ],
    ids=["identiques", "moitie", "disjointes", "contenue"],
)
def test_calculate_iou(boite_b, iou_attendue):
    # Avant correction, le « +1 » donnait 0,91 au lieu de 0,33 pour deux boîtes à moitié superposées.
    assert ev.calculate_iou([0, 0.10, 0.10, 0.10, 0.10], boite_b) == pytest.approx(iou_attendue)


def test_une_correction_n_est_appariee_qu_une_fois():
    # Deux prédictions sur le même objet : une seule peut être un vrai positif.
    predictions = ["0 0.5 0.5 0.2 0.2 0.9", "0 0.51 0.5 0.2 0.2 0.6"]
    corrections = ["0 0.5 0.5 0.2 0.2"]

    appariees, corrigees, faux_positifs, faux_negatifs = ev.match_boxes(predictions, corrections)

    assert appariees == ["0 0.5 0.5 0.2 0.2 0.9"]
    assert faux_positifs == ["0 0.51 0.5 0.2 0.2 0.6"]
    assert faux_negatifs == []


# Deux objets réels qui se chevauchent fortement (décalés de 0,04), de classes différentes
OBJET_A = "0 0.50 0.5 0.20 0.20"
OBJET_B = "1 0.54 0.5 0.20 0.20"


def test_objets_qui_se_chevauchent_independant_de_l_ordre():
    # Une prédiction imprécise, à cheval sur A et B, placée en premier dans le fichier,
    # ne doit pas « voler » la correction de A à la prédiction parfaite de A.
    imprecise = "1 0.52 0.5 0.20 0.20 0.6"
    parfaite_a = "0 0.50 0.5 0.20 0.20 0.9"
    parfaite_b = "1 0.54 0.5 0.20 0.20 0.9"

    for predictions in ([imprecise, parfaite_b, parfaite_a], [parfaite_a, parfaite_b, imprecise]):
        appariees, corrigees, faux_positifs, faux_negatifs = ev.match_boxes(predictions, [OBJET_A, OBJET_B])
        assert sorted(zip(appariees, corrigees)) == sorted([(parfaite_a, OBJET_A), (parfaite_b, OBJET_B)])
        assert faux_positifs == [imprecise]
        assert faux_negatifs == []


def test_objets_qui_se_chevauchent_priorite_a_la_meme_classe():
    # La prédiction de A recouvre un peu plus B que A : elle doit tout de même être
    # associée à A, de la même classe, plutôt que comptée comme erreur de classe sur B.
    prediction_a = "0 0.53 0.5 0.20 0.20 0.9"
    assert ev.calculate_iou([float(v) for v in prediction_a.split()], [float(v) for v in OBJET_B.split()]) > \
        ev.calculate_iou([float(v) for v in prediction_a.split()], [float(v) for v in OBJET_A.split()])

    appariees, corrigees, _, faux_negatifs = ev.match_boxes([prediction_a], [OBJET_A, OBJET_B])

    assert corrigees == [OBJET_A]
    assert faux_negatifs == [OBJET_B]


def test_doublons_aussi_bien_places_le_plus_confiant_est_retenu():
    # À IoU égale, la confiance départage. Si un doublon est mieux placé, c'est lui qui
    # est retenu, même moins confiant : l'IoU passe avant la confiance.
    doublons = ["0 0.5 0.5 0.2 0.2 0.4", "0 0.5 0.5 0.2 0.2 0.9"]
    appariees, _, faux_positifs, _ = ev.match_boxes(doublons, ["0 0.5 0.5 0.2 0.2"])
    assert appariees == ["0 0.5 0.5 0.2 0.2 0.9"]
    assert faux_positifs == ["0 0.5 0.5 0.2 0.2 0.4"]


def test_objets_de_meme_classe_qui_se_chevauchent():
    # Scène de banquet : 5 chaises en perspective, chacune recouvrant sa voisine,
    # et un doublon sur la 3e. Quel que soit l'ordre des prédictions, chaque chaise est
    # associée à une prédiction qui la recouvre, et le doublon donne un seul faux positif.
    import random
    chaises = [f"0 {0.30 + 0.06 * i:.2f} 0.50 {0.20 - 0.015 * i:.3f} 0.30" for i in range(5)]
    predictions = [f"0 {0.31 + 0.06 * i:.2f} 0.51 {0.20 - 0.015 * i:.3f} 0.30 0.9" for i in range(5)]
    predictions.append("0 0.43 0.50 0.17 0.30 0.4")
    x = lambda boite: float(boite.split()[1])

    for graine in range(20):
        random.Random(graine).shuffle(predictions)
        appariees, corrigees, faux_positifs, faux_negatifs = ev.match_boxes(predictions, chaises)
        assert len(appariees) == 5 and len(faux_positifs) == 1 and faux_negatifs == []
        assert all(abs(x(p) - x(c)) < 0.02 for p, c in zip(appariees, corrigees))


def test_une_prediction_trop_decalee_n_est_pas_appariee():
    # IoU de 1/3 < seuil de 0,5 : un faux positif et un faux négatif.
    _, _, faux_positifs, faux_negatifs = ev.match_boxes(["0 0.15 0.1 0.1 0.1 0.9"], ["0 0.1 0.1 0.1 0.1"])
    assert len(faux_positifs) == 1 and len(faux_negatifs) == 1


# --- Conversion des corrections de Label Studio ---------------------------------------------------

def ecrire_correction(cycle, numero, image, boites):
    """Correction exportée par Label Studio. boites = [(classe, x, y, w, h, avec_id), ...]"""
    resultat = []
    for classe, x, y, w, h, avec_id in boites:
        item = {"value": {"x": x, "y": y, "width": w, "height": h, "rectanglelabels": [classe]}, "score": 0.9}
        if avec_id:
            item["id"] = f"boite{len(resultat)}"
        resultat.append(item)
    (cycle["corrections"] / f"{numero}.json").write_text(json.dumps({
        "id": numero, "result": resultat,
        "task": {"data": {"image": f"/data/local-files/?d=projects/manuscrits/image_inputs/eval_images/{image}"}}}))


@pytest.fixture
def labels_resultats(cycle):
    cycle["resultats"].mkdir(parents=True)
    (cycle["resultats"] / "labels.txt").write_text("'0': 'miniature'\n'1': 'initiale'\n", encoding="utf-8")


def test_corrections_converties_au_format_yolo(cycle, labels_resultats):
    ecrire_correction(cycle, 1, "f%201.jpg", [("initiale", 10, 20, 30, 40, True),
                                              ("miniature", 0, 0, 5, 5, False)])   # boîte supprimée

    ev.get_corrected_label_files(cycle["projet"], cycle["modele"])

    fichier = cycle["resultats"] / "correctedLabels" / "f 1.txt"   # %20 décodé
    classe, *coordonnees = fichier.read_text().split()
    assert classe == "1"
    assert [float(v) for v in coordonnees] == pytest.approx([0.25, 0.4, 0.3, 0.4])
    # Le score de confiance est retiré du fichier de correction
    assert "score" not in json.loads((cycle["corrections"] / "1.json").read_text())["result"][0]


def test_correction_avec_une_classe_inconnue(cycle, labels_resultats):
    ecrire_correction(cycle, 1, "f1.jpg", [("enluminure", 10, 20, 30, 40, True)])
    with pytest.raises(ValueError, match="enluminure"):
        ev.get_corrected_label_files(cycle["projet"], cycle["modele"])


# --- Métriques et matrice de confusion ------------------------------------------------------------

@pytest.fixture
def predictions_et_corrections(cycle, labels_resultats):
    """
    f1 : bonne détection (miniature)          -> 1 TP
    f2 : bonne boîte, mauvaise classe         -> 1 FP (miniature) et 1 FN (initiale)
    f3 : objet manqué par le modèle           -> 1 FN (initiale)
    f4 : détection d'un objet inexistant      -> 1 FP (initiale)
    """
    predictions = cycle["resultats"] / "labels"
    corrigees = cycle["resultats"] / "correctedLabels"
    predictions.mkdir()
    corrigees.mkdir()
    (predictions / "f1.txt").write_text("0 0.5 0.5 0.2 0.2 0.9\n")
    (corrigees / "f1.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (predictions / "f2.txt").write_text("0 0.3 0.3 0.2 0.2 0.8\n")
    (corrigees / "f2.txt").write_text("1 0.3 0.3 0.2 0.2\n")
    (corrigees / "f3.txt").write_text("1 0.7 0.7 0.1 0.1\n")
    (predictions / "f4.txt").write_text("1 0.2 0.8 0.1 0.1 0.5\n")
    (corrigees / "f4.txt").write_text("")


def test_metriques_d_evaluation(cycle, predictions_et_corrections):
    ev.get_csv_results(cycle["projet"], cycle["modele"], all_results=False)
    ev.get_txt_results(cycle["projet"], cycle["modele"])

    df = pd.read_csv(cycle["resultats"] / "results" / "results_for_evaluation.csv", sep=";")
    assert sorted(df["TP/FP/FN"]) == ["FN", "FP", "FP_class", "TP"]

    resume = (cycle["resultats"] / "results" / "results_for_evaluation.txt").read_text()
    assert "Number of TP: 1\n" in resume
    assert "Number of FP : 2\n" in resume
    assert "Number of FN: 2\n" in resume
    assert (cycle["resultats"] / "results" / "results_for_evaluation.png").exists()


def test_matrice_de_confusion(cycle, predictions_et_corrections):
    ev.get_csv_results(cycle["projet"], cycle["modele"], all_results=False)
    chemin = ev.create_confusion_matrix(cycle["projet"], cycle["modele"])
    assert chemin.name == "confusion_matrix.png" and chemin.exists()


def test_modele_importe_sans_liste_d_entrainement(cycle, predictions_et_corrections):
    # Un modèle importé n'a pas de training_dataset.txt : l'évaluation doit fonctionner
    # quand même, sans exclure d'image (elle plantait avant avec une FileNotFoundError).
    assert ev.get_img_from_training(cycle["projet"], cycle["modele"]) == []
    ev.get_csv_results(cycle["projet"], cycle["modele"], all_results=False)


def lister_pour_l_entrainement(cycle, *noms):
    """Écrit le training_dataset.txt du modèle, comme le fait l'entraînement."""
    stats = cycle["modele"] / "dataset_statistics"
    stats.mkdir(exist_ok=True)
    (stats / "training_dataset.txt").write_text("".join(f"{cycle['data'] / 'images' / nom}\n" for nom in noms))


def lignes_du_csv(cycle):
    return pd.read_csv(cycle["resultats"] / "results" / "results_for_evaluation.csv", sep=";")["Filename"].tolist()


def omises(cycle):
    return {o["image"]: o["raison"] for o in ev.read_omitted_images(cycle["projet"], cycle["modele"])}


def test_une_image_d_entrainement_est_exclue_et_signalee(cycle, predictions_et_corrections):
    # f1.jpg a servi à l'entraînement, et l'image de test est la même (même contenu).
    (cycle["gt_images"] / "f1.jpg").write_bytes(b"image-f1")
    (cycle["eval"] / "f1.jpg").write_bytes(b"image-f1")
    lister_pour_l_entrainement(cycle, "f1.jpg")

    ev.get_csv_results(cycle["projet"], cycle["modele"], all_results=False)

    assert "f1.txt" not in lignes_du_csv(cycle)
    assert omises(cycle) == {"f1.jpg": ev.OMITTED_TRAINING}


def test_un_homonyme_d_une_image_d_entrainement_reste_evalue(cycle, predictions_et_corrections):
    # Même nom qu'une image d'entraînement, mais une autre image : elle doit être évaluée.
    (cycle["gt_images"] / "f1.jpg").write_bytes(b"image-d-entrainement")
    (cycle["eval"] / "f1.jpg").write_bytes(b"autre-image")
    lister_pour_l_entrainement(cycle, "f1.jpg")

    ev.get_csv_results(cycle["projet"], cycle["modele"], all_results=False)

    assert "f1.txt" in lignes_du_csv(cycle)
    assert omises(cycle) == {}


def test_image_d_entrainement_introuvable_exclue_par_prudence(cycle, predictions_et_corrections):
    # Impossible de comparer les contenus : l'image est exclue, et signalée.
    (cycle["eval"] / "f1.jpg").write_bytes(b"image-f1")
    lister_pour_l_entrainement(cycle, "f1.jpg")

    ev.get_csv_results(cycle["projet"], cycle["modele"], all_results=False)

    assert omises(cycle) == {"f1.jpg": ev.OMITTED_TRAINING}


def test_une_image_non_corrigee_est_omise_et_signalee(cycle, predictions_et_corrections):
    # f5 a des prédictions mais n'a pas été validée dans Label Studio : ses prédictions
    # ne doivent pas être comptées comme des faux positifs.
    (cycle["resultats"] / "labels" / "f5.txt").write_text("0 0.5 0.5 0.2 0.2 0.9\n")
    (cycle["eval"] / "f5.jpg").write_bytes(b"")
    (cycle["eval"] / "f6.png").write_bytes(b"")   # ni prédiction, ni correction

    ev.get_csv_results(cycle["projet"], cycle["modele"], all_results=False)
    ev.get_txt_results(cycle["projet"], cycle["modele"])

    assert "f5.txt" not in lignes_du_csv(cycle)
    assert omises(cycle) == {"f5.jpg": ev.OMITTED_NOT_CORRECTED, "f6.png": ev.OMITTED_NOT_CORRECTED}
    resume = (cycle["resultats"] / "results" / "results_for_evaluation.txt").read_text()
    assert "Number of FP : 2\n" in resume   # inchangé : f5 ne compte pas
    assert "Images omitted from the evaluation : 2" in resume
    assert " - f5.jpg : image non corrigée" in resume