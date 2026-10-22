import codecs
import re
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator


from ..modules.folders_path import get_data_folder
from ..modules.generate_labels import get_labels
from .data_preparation_and_training import AUGMENTED_SUFFIX

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Prefixe ajoute par Label Studio aux fichiers importes : 8 caracteres hexadecimaux et un tiret
LS_PREFIX = re.compile(r"^[0-9a-f]{8}-")


def _stats_folders(project_folder: str | Path) -> tuple[Path, Path, Path, Path]:
    """Dossiers data/, data/images, data/labels et data/dataset_statistics (cree si besoin)."""
    data_folder = Path(get_data_folder(project_folder))
    data_stat_folder = data_folder / 'dataset_statistics'
    data_stat_folder.mkdir(parents=True, exist_ok=True)
    return data_folder, data_folder / 'images', data_folder / 'labels', data_stat_folder


def _dataset_images(img_folder: Path) -> list[Path]:
    """
    Images du dataset : sans les fichiers caches, ni les copies creees par l'augmentation
    de donnees (f1_PT.jpg). Si un entrainement a deja eu lieu dans le cycle, ces copies sont
    encore dans data/images, et les compter fausserait les statistiques du jeu annote.
    """
    return [f for f in img_folder.iterdir()
            if f.suffix.lower() in IMG_EXTENSIONS and not f.name.startswith('.')
            and not f.stem.endswith(AUGMENTED_SUFFIX)]


def _count_annotations(annotation_file: Path) -> int:
    """Nombre de boites d'un fichier YOLO : une par ligne non vide."""
    with open(annotation_file, 'r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())


def create_stats_folder(project_folder:str | Path) -> None:
    """
    Creates a 'dataset_statistics' subfolder in the training directory if it doesn't exist.

    Parameters
    ----------
    project_folder : str | Path
        Absolute path to the project folder.
    """
    _stats_folders(project_folder)


def clean_LS(project_folder:str | Path, annotated_with_LS:bool)-> None:
    """
    Renames files in the 'images' and 'labels' subdirectories of a dataset folder
    by removing the prefix added by Label Studio when files are uploaded to it
    (8 hexadecimal characters followed by a dash, e.g., 'a1b2c3d4-f1.jpg' -> 'f1.jpg').

    Only files that actually start with this prefix are renamed, and a file is never
    renamed if the new name is already taken.

    Parameters:
    ----------
    project_folder : str | Path
        The absolute path to the root project folder.

    annotated_with_LS : bool
        Indicates whether the files were uploaded to Label Studio.
        If True, the function proceeds with renaming.
    """
    if not annotated_with_LS:
        return

    _, img_folder, labels_folder, _ = _stats_folders(project_folder)

    for folder in (img_folder, labels_folder):
        for file in folder.iterdir():
            if not file.is_file() or not LS_PREFIX.match(file.name):
                continue
            new_filepath = folder / LS_PREFIX.sub("", file.name)
            if new_filepath.exists():
                print(f"⚠️ {new_filepath.name} existe déjà, {file.name} n'est pas renommé")
                continue
            file.rename(new_filepath)
            print(f"Renamed file : {file.name} -> {new_filepath.name}")


def get_annotation_files(img_folder:Path, labels_folder:Path) -> list[Path]:
    """
    Retrieves the list of '.txt' annotation files that correspond to image files
    in a given image folder. Only annotation files with a matching image file
    (based on filename without extension) are included.

    Parameters:
    ----------
    img_folder : Path
        Path to the folder containing image files (e.g., .jpg, .png, .jpeg, .tif, .tiff).

    labels_folder : Path
        Path to the folder containing annotation files (one .txt file per image).

    Returns:
    -------
    list of Path
        The .txt annotation files that have a corresponding image in the image folder.
    """
    annotation_files = []
    for image_file in _dataset_images(Path(img_folder)):
        annotation_file = Path(labels_folder) / f'{image_file.stem}.txt'
        if annotation_file.is_file():
            annotation_files.append(annotation_file)
    return annotation_files


def encoding(project_folder:str | Path) -> None:
    """
    Checks the encoding of annotation files and logs those not in UTF-8.

    Parameters
    ----------
    project_folder : str | Path
        Absolute path to the project folder containing the 'labels' subdirectory.

    Returns
    -------
    None
        Logs annotation files that are not UTF-8 encoded (e.g., ISO-8859-1).
    """
    _, img_folder, labels_folder, _ = _stats_folders(project_folder)

    for file_path in get_annotation_files(img_folder, labels_folder):
        rawdata = file_path.read_bytes()
        try:
            codecs.decode(rawdata, 'utf-8')
        except UnicodeDecodeError:
            try:
                codecs.decode(rawdata, 'iso-8859-1')
                print(f"{file_path.name} is encoded in ISO-8859-1")
            except UnicodeDecodeError:
                print(f"{file_path.name} encoding not recognized")


def img_without_annotations(img_folder:Path, labels_folder:Path) -> int:
    """
    This function identifies images in the specified image folder that do not have corresponding annotation files
    or have empty annotation files. It helps detect unannotated images, which may cause issues during model training.

    :param img_folder: path to the folder containing image files (.jpg, .jpeg, .png, .tif, .tiff).
    :param labels_folder: path to the folder containing .txt annotation files.
    :return: the number of unannotated images, including those without annotation files
             and those with empty annotation files.
    """
    count = 0
    for image_file in _dataset_images(Path(img_folder)):
        annotation_path = Path(labels_folder) / f"{image_file.stem}.txt"

        if not annotation_path.exists():
            count += 1
            print(f"Image {image_file} has no annotation file")
        elif _count_annotations(annotation_path) == 0:
            count += 1
            print(f"Image {image_file} has an empty annotation file")

    return count


def annotations_per_img(project_folder:str | Path) -> None:
    """
    Counts annotations per image and saves results to 'annotations_per_img.csv',
    sorted from the most to the least annotated image.

    Parameters
    ----------
    project_folder : str | Path
        Path to the project folder.
    """
    _, img_folder, labels_folder, data_stat_folder = _stats_folders(project_folder)

    # Each annotation file is paired with its image, so the image name is known directly
    counts = []
    for image_file in _dataset_images(img_folder):
        annotation_file = labels_folder / f"{image_file.stem}.txt"
        if annotation_file.is_file():
            counts.append((image_file.name, _count_annotations(annotation_file)))

    df = pd.DataFrame(counts, columns=['image_name', 'annotations_nb'])
    df = df.sort_values(['annotations_nb', 'image_name'], ascending=[False, True])

    csv_file_path = data_stat_folder / 'annotations_per_img.csv'
    df.to_csv(csv_file_path, index=False, sep=';')

    print(f'{csv_file_path} created')


def total_annotations(img_folder:Path, labels_folder:str | Path) -> int:
    """
    This function calculates the total number of annotations present in the specified training dataset
    by counting the number of non-empty lines in each annotation file. Each line in a `.txt` annotation file
    typically represents an individual bounding box or object annotation.

    :param img_folder: the folder where the images are stored.
    :param labels_folder: the folder where the annotation files are stored.
    :return: the total number of annotations across all images in the dataset.
    """
    total_lines = sum(_count_annotations(f) for f in get_annotation_files(img_folder, labels_folder))
    print(f"The total number of annotations is {total_lines}.")
    return total_lines


def classes_distribution(project_folder:str | Path)-> None:
    """
    Computes class distribution from label files.
    Saves results as a CSV and a horizontal bar chart.

    Every class of labels.txt appears, in the order of their ID, including those with
    no annotation: a class absent from the dataset is useful information before training.

    Parameters
    ----------
    project_folder : str | Path
        Absolute path to the project folder.

    Returns
    -------
    None
        Outputs 'class_distribution.csv' and 'class_distribution.png' to 'dataset_statistics'.
    """
    data_folder, img_folder, labels_folder, data_stat_folder = _stats_folders(project_folder)

    annotation_classes = get_labels(data_folder / 'labels.txt')
    occurrences = {code: 0 for code in annotation_classes}

    for annotation_file in get_annotation_files(img_folder, labels_folder):
        with open(annotation_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # ignore empty lines
                    code = line.split()[0]
                    occurrences[code] = occurrences.get(code, 0) + 1

    # Classes in the order of their ID; a code absent from labels.txt is shown rather than ignored
    codes = sorted(occurrences, key=lambda code: (not code.isdigit(), int(code) if code.isdigit() else 0, code))
    class_names = [annotation_classes.get(code, f"classe inconnue ({code})").strip() for code in codes]
    counts = [occurrences[code] for code in codes]

    df = pd.DataFrame({'class_name': class_names, 'nb_occurrences': counts})
    csv_file_path = data_stat_folder / 'class_distribution.csv'
    df.to_csv(csv_file_path, index=False, sep=';')
    print(f'{csv_file_path} created')

    # A Figure object rather than pyplot: pyplot keeps a global current figure, shared by
    # all the threads of the app, and two charts drawn at the same time could get mixed up.
    fig = Figure(figsize=(8, max(2.5, 0.4 * len(class_names) + 1)))
    ax = fig.subplots()
    if class_names:
        ax.barh(class_names, counts, color="#cb300e", edgecolor="#BA3D04")
        ax.invert_yaxis()  # first class at the top
    else:
        ax.text(0.5, 0.5, "Aucune classe annotée", ha="center", va="center", transform=ax.transAxes)
        ax.set_yticks([])

    ax.set_xlabel('Nombre d\'occurrences')
    ax.set_ylabel('Classes')
    ax.set_title('Distribution des classes')
    # Integer tick marks on the X axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(left=0)

    fig.savefig(data_stat_folder / 'class_distribution.png', bbox_inches='tight')


def get_global_results(project_folder:str | Path) -> None:
    """
    Generates a summary of key dataset metrics and saves it in 'global_data.csv'.

    Metrics:
    - Number of images without annotations
    - Total number of annotations

    Parameters
    ----------
    project_folder : str | Path
        Absolute path to the project folder.
    """
    _, img_folder, labels_folder, data_stat_folder = _stats_folders(project_folder)

    metrics = {
        'Number of files without annotations': img_without_annotations(img_folder, labels_folder),
        'Total number of annotations': total_annotations(img_folder, labels_folder)
    }

    df = pd.DataFrame(metrics.items(), columns=['metric', 'value'])
    csv_file_path = data_stat_folder / 'global_data.csv'
    df.to_csv(csv_file_path, index=False, sep=';')

    print(f'{csv_file_path} created')