def test_app_se_charge(app):
    assert app is not None

def test_blueprints_enregistres(app):
    assert len(app.blueprints) > 0