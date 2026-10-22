"""
Functions for listing the YOLO models trained for a TiamaT project.
"""

from pathlib import Path

import pandas as pd


def get_model_list(project_name: str) -> list[str]:
    """
    List the trained models available for a given project.

    The models are read from the "model_name" column of the project's CSV file,
    projects/<project_name>/<project_name>.csv. The path is relative to the
    current working directory, so the app must be launched from the repository root.

    :param project_name:
        - Type: str
        - Description: Name of the project (not its full path), e.g. "manuscrits".

    :return:
        - Type: list of str
        - Description: "------" first, used as a placeholder meaning "no previously trained
                       model / train from scratch", followed by the names of the trained models.
                       Returns ["------"] if the CSV file does not exist yet.
    """
    csv_file = Path("projects") / project_name / f"{project_name}.csv"

    if not csv_file.is_file():
        return ["------"]

    models = pd.read_csv(csv_file)["model_name"].tolist()
    return ["------"] + models