import re
from pathlib import Path


def get_model_list(project_folder: str) -> tuple[str, ...]:
    """
    List the trained models available for a given project.

    Scans the shared "output/train" folder (common to all projects) and
    keeps only the subfolders whose name matches the pattern
    "{project_name}_{YYYYMMDD}_{HHMMSS}" -- i.e. models that were actually
    trained for this specific project (named that way in start_training()).

    Always includes "------" as the first entry, used as a placeholder
    meaning "no previously trained model / train from scratch".
    """
    
    project_name = str(Path(project_folder).name)
    models = ["------"]
    root = Path.cwd()
    trained_folder = Path(root / "output" / "train")
    list_models = [str(i.name) for i in trained_folder.iterdir() if not str(i.name).startswith('.')]
    print(list_models)  # debug: show every folder found under output/train

    for i in list_models:
        # keep only folders matching "{project_name}_{date}_{time}"
        match = re.match(re.escape(project_name) + r"_[0-9]{8}_[0-9]{6}$", i)
        if match:
            models.append(i)
        else:
            pass

    return tuple(models)