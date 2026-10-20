"""Tests de la détection du matériel de calcul (GPU NVIDIA, GPU Apple, TPU, CPU).

On ne peut pas changer le matériel de la machine qui lance les tests : on
remplace donc les fonctions de détection de PyTorch par des « mocks », de
fausses fonctions dont on choisit la réponse. Chaque cas est ainsi testé
sur n'importe quel ordinateur, y compris sur la machine de la CI.
"""

import sys
import types

import pytest

from app.src.modules.device_function import which_device


@pytest.fixture
def materiel(mocker):
    """Simule une machine sans GPU et sans TPU. Chaque test active ce qu'il veut."""
    cuda = mocker.patch("torch.cuda.is_available", return_value=False)
    mocker.patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090")
    mps = mocker.patch("torch.backends.mps.is_available", return_value=False)
    # Mettre None dans sys.modules fait échouer l'import : torch_xla est « absent ».
    mocker.patch.dict(sys.modules, {"torch_xla": None, "torch_xla.core": None,
                                    "torch_xla.core.xla_model": None})
    return types.SimpleNamespace(cuda=cuda, mps=mps)


def simuler_torch_xla(mocker, xla_device):
    """Installe un faux module torch_xla dont la fonction xla_device est fournie."""
    xla_model = types.ModuleType("torch_xla.core.xla_model")
    xla_model.xla_device = xla_device
    core = types.ModuleType("torch_xla.core")
    core.xla_model = xla_model
    torch_xla = types.ModuleType("torch_xla")
    torch_xla.core = core
    mocker.patch.dict(sys.modules, {"torch_xla": torch_xla, "torch_xla.core": core,
                                    "torch_xla.core.xla_model": xla_model})


def test_cpu_par_defaut(materiel):
    assert which_device() == "cpu"


def test_cuda(materiel, capsys):
    materiel.cuda.return_value = True
    assert which_device() == "cuda"
    # capsys capture ce que la fonction affiche avec print.
    assert "NVIDIA RTX 4090" in capsys.readouterr().out


def test_mps(materiel):
    materiel.mps.return_value = True
    assert which_device() == "mps"


def test_cuda_prioritaire_sur_mps(materiel):
    # Une machine avec les deux (cas théorique) doit choisir CUDA.
    materiel.cuda.return_value = True
    materiel.mps.return_value = True
    assert which_device() == "cuda"


def test_tpu(materiel, mocker):
    simuler_torch_xla(mocker, xla_device=lambda: "xla:0")
    assert which_device() == "xla"


def test_torch_xla_installe_mais_sans_tpu(materiel, mocker):
    # torch_xla peut être installé sur une machine sans TPU : xla_device()
    # lève alors une RuntimeError. La fonction doit se rabattre sur le CPU.
    def pas_de_tpu():
        raise RuntimeError("No TPU devices found")

    simuler_torch_xla(mocker, xla_device=pas_de_tpu)
    assert which_device() == "cpu"