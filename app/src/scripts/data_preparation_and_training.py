import shutil
import random
import traceback
from pathlib import Path
import os

import cv2
import yaml
import numpy as np
from ultralytics import YOLO


from ..modules.folders_path import get_data_folder
from ..modules.device_function import which_device
from ..modules.generate_labels import get_labels
from ..modules.corners_functions import get_corners, from_corners_to_relative

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Suffixe des images creees par l'augmentation de donnees (f1.jpg -> f1_PT.jpg)
AUGMENTED_SUFFIX = "_PT"


def _list_images(folder: Path) -> list[Path]:
    """Images du dossier, sans les fichiers caches (._f1.jpg crees par macOS sur les disques externes)."""
    return [img for img in Path(folder).iterdir()
            if img.suffix.lower() in IMG_EXTENSIONS and not img.name.startswith('.')]


def _original_stem(stem: str) -> str:
    """Nom de l'image d'origine d'une image augmentee : "f1_PT" -> "f1" ("f1" reste "f1")."""
    return stem.removesuffix(AUGMENTED_SUFFIX)


def _delete_augmented_copy(img_file: Path, labels_folder: Path) -> None:
    """Supprime une image augmentee (f1_PT.jpg) et son annotation (f1_PT.txt)."""
    img_file.unlink(missing_ok=True)
    (labels_folder / f"{img_file.stem}.txt").unlink(missing_ok=True)


def _dataset_folders(project_folder: str | Path) -> dict[str, Path]:
    """Dossiers utilises pendant l'entrainement (data/ du projet et datasets/ temporaire)."""
    data_folder = Path(get_data_folder(project_folder))
    project_name = Path(project_folder).name
    dataset_folder = data_folder.parent / 'datasets' / project_name
    return {
        "data": data_folder,
        "images": data_folder / 'images',
        "labels": data_folder / 'labels',
        "dataset": dataset_folder,
        "img_train": dataset_folder / 'images' / 'train',
        "img_val": dataset_folder / 'images' / 'val',
        "labels_train": dataset_folder / 'labels' / 'train',
        "labels_val": dataset_folder / 'labels' / 'val',
        "yaml": dataset_folder / f"{project_name}.yaml",
    }


def clean_comma(project_folder:str | Path) -> None:
    """
    This function removes any commas that may appear in the annotation `.txt` files within the specified training folder.
    This is particularly useful when annotation files are generated or modified from CSV files,
    as commas can accidentally be included and cause issues during model training.

    :param project_folder:
        - Type: str
        - Description: The absolute path to the folder named after the project.

    :return:
        - Type: None
        - Description: This function modifies `.txt` files in place, removing any commas that are found.
    """
    labels_folder = _dataset_folders(project_folder)["labels"]

    for label_file in labels_folder.iterdir():
        if label_file.is_file() and label_file.suffix.lower() == '.txt':
            content = label_file.read_text(encoding='utf-8')
            label_file.write_text(content.replace(',', ''), encoding='utf-8')


def perspective_transformation(img_file:str) -> np.ndarray:
    """
    Applies a random perspective transformation to the input image and saves the result.

    The function randomly adjusts the dimensions of the image to simulate a different viewing angle,
    which is useful for data augmentation in machine learning workflows. The transformed image is
    saved with the same name as the original, with '_PT' appended before the file extension.

    Parameters
    ----------
    img_file : str
        Absolute path to the image file to be transformed.

    Returns
    -------
    numpy.ndarray
        The 3x3 perspective transformation matrix (homography) used to warp the original image.

    Notes
    -----
    - The transformation randomly resizes the width and height to 30–80% of the original dimensions.
    - The output image is saved in the same directory as the input.
    """

    # Open image and get dimensions
    img = cv2.imread(img_file)
    rows, cols = img.shape[:2]

    # Define the points of origin for the perspective transformation.
    # These points form a quadrilateral covering the entire original image.
    pts1 = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])

    # Generate a new random width and height for the transformed image, between 30% and 80% of the original width.
    new_width = random.randint(int(cols*0.3), int(cols*0.8))
    new_height = random.randint(int(rows*0.3), int(rows*0.8))

    # Define the new points for the perspective transformation
    pts2 = np.float32([[0, 0], [new_width, 0], [new_width, new_height], [0, new_height]])

    # Calculate the perspective transformation matrix
    M = cv2.getPerspectiveTransform(pts1, pts2)

    # Apply the perspective transformation to the original image.
    dst = cv2.warpPerspective(img, M, (new_width, new_height))

    # Save the transformed image in the output folder
    img_path = Path(img_file)
    transformed_img_path = img_path.with_name(f"{img_path.stem}{AUGMENTED_SUFFIX}{img_path.suffix}")

    cv2.imwrite(str(transformed_img_path), dst)

    return M


def perspective_transformation_annotation(ann_file:str, img_file:str, M:np.ndarray) -> list:
    """
    This function applies a perspective transformation matrix to the bounding box annotations of an image
    and saves the new transformed annotations in a separate file. The transformed annotations correspond to
    the modified perspective and dimensions of the image after applying the perspective transformation.

    :param ann_file: absolute path to the YOLO annotation file (label, x_center, y_center, width, height).
    :param img_file: absolute path to the original image file.
    :param M: transformation matrix returned by perspective_transformation.
    :return: list of transformed annotations (label, x_center, y_center, width, height), in relative coordinates.
    """

    img_height, img_width = cv2.imread(img_file).shape[:2]
    img_path = Path(img_file)
    transformed_img_path = img_path.with_name(f"{img_path.stem}{AUGMENTED_SUFFIX}{img_path.suffix}")
    TP_img_height, TP_img_width = cv2.imread(str(transformed_img_path)).shape[:2]

    # Initialising a list to store the new bounding box coordinates
    bb_coordinates = []

    with open(ann_file, 'r', encoding='utf-8') as annotations:
        for line in annotations:
            if not line.strip():
                continue #skip empty lines

            # Extraire les coordonnées de l'annotation
            label, x_center, y_center, width, height = line.strip().split()

            # Convertir les coordonnées relatives en coordonnées absolues
            corners = get_corners(x_center, y_center, width, height, img_width, img_height)

            # Appliquer la transformation aux coins de la boîte d'annotation
            corners = np.array(corners, dtype=np.float32).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(corners, M).reshape(-1, 2)

            # Calculer les nouvelles coordonnées relatives
            new_upper_left = transformed_corners[0]
            new_bottom_right = transformed_corners[2]

            # Transformer les nouvelles coordonnés en relatives
            transformed_x_center, transformed_y_center, transformed_width, transformed_height = from_corners_to_relative(
                new_upper_left, new_bottom_right, TP_img_width, TP_img_height)

            bb_coordinates.append((label, transformed_x_center, transformed_y_center, transformed_width, transformed_height))

    annotations_path = Path(ann_file)
    new_annotation_path = annotations_path.with_name(f"{annotations_path.stem}{AUGMENTED_SUFFIX}{annotations_path.suffix}")

    with open(new_annotation_path, 'w', encoding='utf-8') as transformed_annotations:
        for bb in bb_coordinates:
            label, x, y, w, h = bb
            transformed_annotations.write(f"{int(label)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    return bb_coordinates


def generate_transformed_data(project_folder:str | Path) -> None:
    """
    Generates one transformed copy (suffix _PT) of each annotated image of data/images,
    with its adjusted annotation file in data/labels.

    Images that are themselves augmented copies (f1_PT.jpg) are skipped: without this,
    each training cycle would transform the copies of the previous cycle again
    (f1_PT_PT.jpg, f1_PT_PT_PT.jpg...) and the dataset would double at every cycle.
    The copies of the original images are regenerated (overwritten) at each training.
    """

    print('Image tranformation has started..')

    folders = _dataset_folders(project_folder)
    labels_folder, img_folder = folders["labels"], folders["images"]

    images = [img for img in _list_images(img_folder) if not img.stem.endswith(AUGMENTED_SUFFIX)]

    for img_file in images:
        ann_file = labels_folder / f"{img_file.stem}.txt"

        if ann_file.exists():
            try:
                M = perspective_transformation(str(img_file))
                perspective_transformation_annotation(str(ann_file), str(img_file), M)
            except Exception as e:
                print(f"Error processing {img_file.name}: {e}")

    print(f'New images stored in {img_folder}\nNew annotations stored in {labels_folder}')


def create_training_dataset(project_folder:str | Path, pretrained_model:str | None, preexisting_distribution:bool) -> None:
    """
    Prepares training and validation datasets from a directory of images and labels.
    Generates, in data/dataset_statistics/:
    1. traindata.txt – about 80% of the images
    2. valdata.txt – about 20% of the images
    3. training_dataset.txt – all images

    The split is made on the original images, and augmented copies (f1_PT.jpg) only go
    to the training set: the validation set contains original images only. Otherwise the
    model would be validated on near-copies of its training images (or on artificial
    images), and the validation metrics would be overestimated. The augmented copies of
    the validation images are deleted.

    If a pre-existing split is provided, it's reused. Otherwise, a new random split is created.
    Images and labels are moved into organized folders for training and validation.

    :param project_folder: the absolute path to the folder named after the project.
    :param pretrained_model: folder of a previous model, used only with preexisting_distribution.
    :param preexisting_distribution: if True, reuses the train/val split saved in pretrained_model.
    :raises ValueError: if there are less than 2 images (one is needed for training, one for validation).
    """

    folders = _dataset_folders(project_folder)
    data_folder, img_folder = folders["data"], folders["images"]

    data_stat_folder = data_folder / 'dataset_statistics'
    data_stat_folder.mkdir(parents=True, exist_ok=True)

    train_data = data_stat_folder / 'traindata.txt'
    val_data = data_stat_folder / 'valdata.txt'
    training_dataset = data_stat_folder / 'training_dataset.txt'

    if preexisting_distribution:
        if not pretrained_model:
            raise ValueError("A pre-trained model folder is needed to reuse a pre-existing distribution.")
        print(f'Use pre-existing files from {pretrained_model}.')
        previous_distribution = Path(pretrained_model) / 'dataset_statistics'
        if not previous_distribution.exists():
            raise FileNotFoundError(f"No dataset_statistics found in {pretrained_model}")
        shutil.copytree(previous_distribution, data_stat_folder, dirs_exist_ok=True)

    else:
        image_files = _list_images(img_folder)
        if not image_files:
            raise ValueError(f"No images found in {img_folder}. Please check your dataset.")

        # Group each original image with its augmented copy, then split the groups
        groups: dict[str, list[Path]] = {}
        for image_file in sorted(image_files):
            groups.setdefault(_original_stem(image_file.stem), []).append(image_file)

        originals = sorted(groups)
        if len(originals) < 2:
            raise ValueError("Au moins 2 images annotées sont nécessaires : une pour l'entraînement, une pour la validation.")
        random.shuffle(originals)

        # TODO: Add sklearn stratified split here for balanced class distribution

        # About 80% for training, but always at least one image in each set
        num_train = min(max(int(len(originals) * 0.8), 1), len(originals) - 1)
        train_files = [img for original in originals[:num_train] for img in groups[original]]
        val_files = []
        for original in originals[num_train:]:
            for img in groups[original]:
                if img.stem.endswith(AUGMENTED_SUFFIX):
                    _delete_augmented_copy(img, folders["labels"])
                else:
                    val_files.append(img)
        print(f"{len(train_files)} images assigned to training.")
        print(f"{len(val_files)} images assigned to validation.")

        all_files = train_files + val_files
        for list_file, files in [(train_data, train_files), (val_data, val_files), (training_dataset, all_files)]:
            list_file.write_text("".join(f"{image_file}\n" for image_file in files), encoding='utf-8')
            print(f"File created: {list_file}")

    # Split images and txt files into folders from a .txt file
    split_data_for_training(str(train_data), str(folders["labels"]),
                            str(folders["img_train"]), str(folders["labels_train"]))
    split_data_for_training(str(val_data), str(folders["labels"]),
                            str(folders["img_val"]), str(folders["labels_val"]))

    # Create the yaml file
    write_yaml_file(project_folder)


def split_data_for_training(img_list:str, labels_folder:str, output_img_folder:str, output_labels_folder:str) -> None:
    """
    Organizes images and annotation files into the appropriate YOLO train/val subdirectories.

    Parameters
    ----------
    img_list : str
        Path to a `.txt` file containing absolute paths to image files (one per line).
    labels_folder : str
        Path to the folder containing corresponding `.txt` YOLO annotation files.
    output_img_folder : str
        Destination folder where the images will be moved (e.g., images/train or images/val).
    output_labels_folder : str
        Destination folder where the annotation `.txt` files will be moved (e.g., labels/train or labels/val).
    """

    # Create the output folder if it does not already exist
    Path(output_img_folder).mkdir(parents=True, exist_ok=True)
    Path(output_labels_folder).mkdir(parents=True, exist_ok=True)

    with open(img_list, "r", encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            image_path = Path(line.strip())
            txt_file = Path(labels_folder) / f"{image_path.stem}.txt"

            try:
                shutil.move(str(image_path), str(Path(output_img_folder) / image_path.name))
            except FileNotFoundError:
                print(f"Image file {image_path} not found.")

            try:
                shutil.move(str(txt_file), str(Path(output_labels_folder) / txt_file.name))
            except FileNotFoundError:
                print(f'Text file {txt_file} does not exist')
    print(f'Image files move in {output_img_folder}')
    print(f'Text files move in {output_labels_folder}')


def write_yaml_file(project_folder:str | Path) -> None:
    """
    Creates a `.yaml` configuration file for YOLO training, specifying:
    - Dataset path
    - Train and validation folder structure
    - Class label mapping from `labels.txt`

    The file is written with the yaml library rather than by hand, so that a class
    name containing a quote or a colon ("lettrine d'or") still gives a valid file.
    """

    folders = _dataset_folders(project_folder)
    labels_file = folders["data"] / 'labels.txt'

    annotation_classes = {int(key): value for key, value in get_labels(labels_file).items()}

    yaml_data = {
        "path": str(folders["dataset"]),
        "train": "images/train",
        "val": "images/val",
        "names": annotation_classes,
    }

    folders["dataset"].mkdir(parents=True, exist_ok=True)
    with open(folders["yaml"], 'w', encoding='utf-8') as yaml_file:
        yaml.safe_dump(yaml_data, yaml_file, allow_unicode=True, sort_keys=False)

    print(f"File written to {folders['yaml']}")


def yolo_training(project_folder:str | Path, use_model:str, img_size:int,
                  epochs:int, batch:int, workers:int, dropout:float, model_name:str,
                  pretrained_model:str | None) -> str:
    """
    Trains a YOLO model using a specified dataset and configuration.

    :param project_folder: the absolute path to the folder named after the project.
    :param use_model: the YOLO model to start from (e.g., 'yolo26s.pt' or the best.pt of a previous model).
    :param img_size: the size of the input images.
    :param epochs: the number of epochs to train the model.
    :param batch: the batch size (-1 for AutoBatch).
    :param workers: the number of workers for data loading.
    :param dropout: the dropout factor, between 0 and 1.
    :param model_name: name of the output folder, in output/train/.
    :param pretrained_model: path to a pre-trained model, if any; overrides use_model.
    :return: the name of the model folder.
    """

    folders = _dataset_folders(project_folder)
    project_name = Path(project_folder).name
    yaml_file = folders["yaml"]

    if not yaml_file.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_file}")

    # Determine which model to use
    if pretrained_model is not None:
        use_model = pretrained_model
        model_name = f'{model_name}_{Path(use_model).stem}'

    # Check if the GPU is available - if not, use the CPU
    device = which_device()

    # Load a YOLO model
    model = YOLO(use_model).to(device)

    # Train the model
    model.train(
       data = yaml_file, # path to the datasets and classes
       imgsz = img_size, #image size
       epochs = epochs,
       batch = batch,
       dropout = dropout,
       workers = workers, # increases training speed, default setting is 8
       name = model_name, # output folder
       project = Path.cwd() / 'output' / 'train'
    )

    # Evaluate the model's performance on the validation set
    model.val(
        name = f"{model_name}/{str(project_name)}_val",
        project = Path.cwd() / 'output' / 'train')

    print(f"Training completed. Validation results saved to {model_name}/{str(project_name)}_val")
    return model_name


def resume_training(project_folder:str, interrupted_model_folder:str) -> None:
    """
    Resumes an interrupted YOLO model training session from the last saved checkpoint.

    :param project_folder: path to the root project folder.
    :param interrupted_model_folder: folder of the partially trained model, with a 'weights/last.pt' file.
    """
    last_weight = Path(interrupted_model_folder) / 'weights' / 'last.pt'

    if not last_weight.exists():
        raise FileNotFoundError(f"No checkpoint found at {last_weight}")

    model_name = Path(interrupted_model_folder).name
    project_name = Path(project_folder).name

    device = which_device()
    model = YOLO(last_weight).to(device)  # load a partially trained model
    model.train(resume=True)
    model.val(name = f"{model_name}/{str(project_name)}_val")


def restore_dataset(project_folder: str | Path) -> None:
    """
    Moves the images and annotations of the temporary datasets/ folder back to
    data/images and data/labels, then deletes the temporary folder.

    Called after the training, and also when the training fails: otherwise the
    images would stay in datasets/, data/images would be empty, and the next
    training would fail with "No images found".
    """
    folders = _dataset_folders(project_folder)

    for source, destination in [(folders["img_train"], folders["images"]), (folders["img_val"], folders["images"]),
                                (folders["labels_train"], folders["labels"]), (folders["labels_val"], folders["labels"])]:
        destination.mkdir(parents=True, exist_ok=True)
        if source.exists():
            for file in source.iterdir():
                shutil.move(str(file), str(destination / file.name))
            print(f"Files from {source} moved into {destination}")

    if folders["dataset"].exists():
        shutil.rmtree(folders["dataset"])
        print(f"The {folders['dataset']} folder has been deleted")


def dispatch_data(project_folder: str | Path, model_name: str) -> Path:
    """
    Stores the files used for the training in the model folder (output/train/{model_name}):
    the .yaml file, labels.txt and the dataset_statistics folder (train/val split).
    Then restores the dataset structure in data/ (restore_dataset).

    :return: the model folder.
    """
    folders = _dataset_folders(project_folder)
    project_name = Path(project_folder).name
    model_folder = Path.cwd() / 'output' / 'train' / model_name

    shutil.copy2(folders["yaml"], model_folder / f"{project_name}.yaml")
    print(f'The .yaml file has been copied into {model_folder}')

    shutil.copy2(folders["data"] / 'labels.txt', model_folder / 'labels.txt')
    print(f'The labels.txt file has been copied in {model_folder}')

    shutil.move(str(folders["data"] / 'dataset_statistics'), str(model_folder / 'dataset_statistics'))
    print(f'The statistics folder with the training data have been moved to {model_folder}.')

    restore_dataset(project_folder)

    print(f"✅ All data successfully dispatched")
    return model_folder


def training(project_folder:str | Path, nombre_epoch:int, dropout:float, model:str, model_name:str) -> None:
    """
    Complete training session, run in a background thread by the training route:
    data preparation, YOLO training, then storage of the training files in the model folder.

    The result is reported in os.environ["complete"]: "True" if the training succeeded,
    "Error" otherwise (the detail of the error is printed in the terminal). In both cases,
    the images are moved back to data/, ready for a new training.
    """
    try:
        clean_comma(project_folder)
        generate_transformed_data(project_folder)
        create_training_dataset(project_folder, model, preexisting_distribution=False)

        yolo_training(project_folder, use_model=model, img_size=640, epochs=nombre_epoch,
                      batch=-1, # -1 for AutoBatch
                      workers=8, # https://docs.ultralytics.com/modes/train/#train-settings
                      dropout=dropout, model_name=model_name, pretrained_model=None)

        # Done here rather than on the result page: the data is restored even if
        # nobody opens the result page, and reloading that page does not redo it.
        dispatch_data(project_folder, model_name)
        os.environ["complete"] = "True"

    except Exception as e:
        traceback.print_exc()
        print(f"Erreur pendant l'entraînement : {e}")
        try:
            restore_dataset(project_folder)
        except Exception:
            traceback.print_exc()
        os.environ["complete"] = "Error"