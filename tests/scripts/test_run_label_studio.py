"""Tests du lancement de Label Studio. Aucun serveur n'est réellement démarré : subprocess.Popen est remplacé par un mock."""

import os
import socket

import pytest

import app.src.scripts.run_label_studio as run_ls

TOKEN = "abc123def456"


@pytest.fixture
def popen(mocker, monkeypatch, tmp_path):
    """Label Studio n'est pas encore lancé ; on intercepte la commande de lancement."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LABEL_STUDIO_API_TOKEN", raising=False)
    for variable in ["LOCAL_FILES_DOCUMENT_ROOT", "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT",
                     "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"]:
        monkeypatch.delenv(variable, raising=False)
    mocker.patch.object(run_ls, "is_label_studio_running", return_value=False)
    return mocker.patch.object(run_ls.subprocess, "Popen")


def commande(popen):
    return popen.call_args.args[0]


def test_lancement_avec_token(popen, monkeypatch):
    monkeypatch.setenv("LABEL_STUDIO_API_TOKEN", TOKEN)

    assert run_ls.launch_LS() is True

    cmd = commande(popen)
    assert cmd[:4] == ["label-studio", "start", "--port", str(run_ls.LS_PORT)]
    assert cmd[cmd.index("--user-token") + 1] == TOKEN


def test_lancement_sans_token(popen):
    run_ls.launch_LS()
    # Pas d'argument --user-token vide
    assert "--user-token" not in commande(popen)


def test_le_token_n_est_jamais_affiche(popen, monkeypatch, capsys):
    monkeypatch.setenv("LABEL_STUDIO_API_TOKEN", TOKEN)
    run_ls.launch_LS()
    assert TOKEN not in capsys.readouterr().out


def test_dossier_des_fichiers_locaux(popen, tmp_path):
    run_ls.launch_LS()
    env = popen.call_args.kwargs["env"]
    assert env["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] == tmp_path.as_posix()
    assert env["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] == "true"


def test_pas_de_second_lancement(popen, mocker):
    run_ls.is_label_studio_running.return_value = True
    assert run_ls.launch_LS() is False
    popen.assert_not_called()


def test_launch_ls2_utilise_le_meme_port(popen):
    # Le lancement sans token doit utiliser le port de la configuration, comme launch_LS
    run_ls.launch_LS2()
    cmd = commande(popen)
    assert cmd[cmd.index("--port") + 1] == str(run_ls.LS_PORT)


def test_detection_d_un_serveur_deja_lance():
    # Un vrai serveur local, sur un port libre choisi par le système
    with socket.socket() as serveur:
        serveur.bind(("127.0.0.1", 0))
        serveur.listen()
        port = serveur.getsockname()[1]
        assert run_ls.is_label_studio_running(port) is True
    assert run_ls.is_label_studio_running(port) is False