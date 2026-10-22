"""Tests du calcul des coins d'une boîte et de la conversion inverse."""

import pytest

from app.src.modules.corners_functions import from_corners_to_relative, get_corners
from app.src.modules.transform_coordinates_functions import (
    from_ls_to_yolo,
    from_relative_coordinates_to_absolute,
)


# --- get_corners ---------------------------------------------------------------

def test_get_corners_cas_typique():
    # Image 1000 x 500, boîte centrée en (500, 250), taille 200 x 200 px.
    assert get_corners(0.5, 0.5, 0.2, 0.4, 1000, 500) == [
        [400, 150],  # haut-gauche
        [600, 150],  # haut-droit
        [600, 350],  # bas-droit
        [400, 350],  # bas-gauche
    ]


def test_get_corners_image_entiere():
    assert get_corners(0.5, 0.5, 1, 1, 800, 600) == [[0, 0], [800, 0], [800, 600], [0, 600]]


def test_get_corners_accepte_des_chaines():
    # Les valeurs lues dans les fichiers .txt de YOLO arrivent sous forme de texte.
    assert get_corners("0.5", "0.5", "0.2", "0.4", "1000", "500") == get_corners(0.5, 0.5, 0.2, 0.4, 1000, 500)


def test_get_corners_forme_un_rectangle():
    ul, ur, br, bl = get_corners(0.37, 0.61, 0.13, 0.27, 1234, 987)
    assert ul[1] == ur[1]  # haut horizontal
    assert bl[1] == br[1]  # bas horizontal
    assert ul[0] == bl[0]  # gauche vertical
    assert ur[0] == br[0]  # droite vertical


def test_get_corners_coherent_avec_transform_coordinates():
    # Les deux modules calculent la même boîte en pixels : ils doivent tomber
    # sur les mêmes valeurs. Cas où int() et round() divergent :
    # 46 % de 3000 px = 1380 px, et 31 % de 480 px = 148,8 px.
    yolo = [float(v) for v in from_ls_to_yolo(46, 4, 22, 31)]
    x, y, w, h = from_relative_coordinates_to_absolute(*yolo, 3000, 480)
    assert get_corners(*yolo, 3000, 480) == [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


# --- from_corners_to_relative --------------------------------------------------

def test_from_corners_to_relative_cas_typique():
    result = from_corners_to_relative([400, 150], [600, 350], 1000, 500)
    assert result == pytest.approx((0.5, 0.5, 0.2, 0.4))


def test_from_corners_to_relative_accepte_des_tuples():
    assert from_corners_to_relative((400, 150), (600, 350), 1000, 500) == \
        from_corners_to_relative([400, 150], [600, 350], 1000, 500)


# --- Aller-retour ----------------------------------------------------------------

@pytest.mark.parametrize(
    "box, img_size",
    [
        ((0.5, 0.5, 0.2, 0.4), (1000, 500)),
        ((0.37, 0.61, 0.13, 0.27), (1234, 987)),
        ((0.1, 0.9, 0.05, 0.1), (4000, 3000)),
    ],
)
def test_aller_retour_relatif_coins_relatif(box, img_size):
    # Relatif -> coins en pixels -> relatif : on retrouve la boîte de départ,
    # à un pixel près (l'arrondi aux pixels entiers fait perdre un peu de précision).
    img_w, img_h = img_size
    ul, _, br, _ = get_corners(*box, img_w, img_h)
    result = from_corners_to_relative(ul, br, img_w, img_h)
    tolerance = 1 / min(img_w, img_h)
    assert result == pytest.approx(box, abs=tolerance)