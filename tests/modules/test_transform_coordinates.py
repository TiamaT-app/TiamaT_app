"""Tests des conversions de coordonnées entre Label Studio, YOLO et pixels."""

import pytest

from app.src.modules.transform_coordinates_functions import (
    from_ls_to_yolo,
    from_relative_coordinates_to_absolute,
)


# --- from_ls_to_yolo ---------------------------------------------------------

def test_ls_to_yolo_boite_image_entiere():
    # Une boîte qui couvre toute l'image : centre au milieu, taille 1.
    assert from_ls_to_yolo(0, 0, 100, 100) == ("0.5", "0.5", "1.0", "1.0")


def test_ls_to_yolo_cas_typique():
    # Coin haut-gauche à (10 %, 20 %), taille 30 % x 40 %.
    # Centre attendu : x = (10 + 30/2) / 100 = 0.25, y = (20 + 40/2) / 100 = 0.4
    result = [float(v) for v in from_ls_to_yolo(10, 20, 30, 40)]
    assert result == pytest.approx([0.25, 0.4, 0.3, 0.4])


def test_ls_to_yolo_renvoie_des_chaines():
    # Les valeurs sont écrites telles quelles dans les fichiers .txt de YOLO.
    assert all(isinstance(v, str) for v in from_ls_to_yolo(10, 20, 30, 40))


def test_ls_to_yolo_valeurs_entre_0_et_1():
    for v in from_ls_to_yolo(5.5, 12.3, 40.1, 60.7):
        assert 0 <= float(v) <= 1


# --- from_relative_coordinates_to_absolute -----------------------------------

def test_relatif_vers_absolu_boite_image_entiere():
    assert from_relative_coordinates_to_absolute(0.5, 0.5, 1, 1, 800, 600) == (0, 0, 800, 600)


def test_relatif_vers_absolu_cas_typique():
    # Image 1000 x 500, boîte centrée en (500, 250), taille 200 x 200 px.
    # Coin haut-gauche attendu : (500 - 100, 250 - 100) = (400, 150)
    assert from_relative_coordinates_to_absolute(0.5, 0.5, 0.2, 0.4, 1000, 500) == (400, 150, 200, 200)


def test_relatif_vers_absolu_renvoie_des_entiers():
    result = from_relative_coordinates_to_absolute(0.37, 0.61, 0.13, 0.27, 1234, 987)
    assert all(isinstance(v, int) for v in result)


# --- Aller-retour : Label Studio -> YOLO -> pixels ---------------------------

@pytest.mark.parametrize(
    "ls_box, img_size, pixels_attendus",
    [
        ((10, 20, 30, 40), (1000, 500), (100, 100, 300, 200)),
        ((0, 0, 100, 100), (640, 480), (0, 0, 640, 480)),
        ((25, 25, 50, 50), (2000, 3000), (500, 750, 1000, 1500)),
    ],
)
def test_aller_retour_ls_yolo_pixels(ls_box, img_size, pixels_attendus):
    # Une annotation Label Studio convertie en YOLO puis en pixels doit
    # retomber sur la même boîte que le calcul direct pourcentage x taille.
    yolo = [float(v) for v in from_ls_to_yolo(*ls_box)]
    assert from_relative_coordinates_to_absolute(*yolo, *img_size) == pixels_attendus


def test_aller_retour_sans_perte_de_pixel():
    # 46 % de 3000 px = 1380 px exactement. Les calculs en nombres flottants
    # donnent 1379.9999…, que int() tronque à 1379 : la boîte décale d'un pixel.
    yolo = [float(v) for v in from_ls_to_yolo(46, 4, 22, 31)]
    upper_left_x, _, _, _ = from_relative_coordinates_to_absolute(*yolo, 3000, 480)
    assert upper_left_x == 1380