import shutil
from pathlib import Path

import pandas as pd
from PIL import Image
from slugify import slugify


from ..modules.transform_coordinates_functions import from_ls_to_yolo
from ..modules.class_names_functions import get_labels, get_class_code
from ..modules.folders_path import get_img_folder_training, get_ground_truth_folder_training, get_data_folder
from ..modules.manipulate_files import open_json_file


def clean_image_name(project_folder) -> None:
    ground_truth_folder = Path(get_img_folder_training(project_folder))

    renamed_files = []
    for img_file in ground_truth_folder.iterdir():
        if not img_file.is_file() or img_file.stem.startswith("."):
            continue
        
        # Extract name without extension and normalized
        filename = img_file.stem
        normalized_filename = slugify(filename, separator="_")

        # Create a new path if the name isn't already normalized 
        new_filepath = img_file.with_stem(normalized_filename)

        if new_filepath == img_file:
            continue
        
        if new_filepath.exists():
            print(f"⚠️ Fichier déjà existant, ignoré : {img_file.name}")
            continue

        # Rename the file
        img_file.rename(new_filepath)
        renamed_files.append((img_file.name, new_filepath.name))

    if renamed_files:
        print("✅ Fichiers renommés :")
        for old_name, new_name in renamed_files:
            print(f"{old_name} → {new_name}")
    else:
        print("ℹ️ Aucun fichier à renommer.")

def create_csv_file(project_folder:str) -> None:
    """
    Generates a CSV file with metadata for all images in the project.

    Parameters
    ----------
    project_folder : str
        Path to the project folder containing images.

    Returns
    -------
    None
        Saves a CSV file in the image folder with metadata for each image.

    The CSV includes:
        - Image name (without extension)
        - Folder name
        - Absolute path
        - Format (JPEG, PNG, etc.)
        - Width and height
        - Total pixel count (width × height)
    """

    project_name = Path(project_folder).name
    img_folder = Path(get_img_folder_training(project_folder))

    if not img_folder.exists():
        raise FileNotFoundError(f"Image folder not found: {img_folder}")
    
    img_exts = {".jpg", ".jpeg", ".png", ".tiff"}
    
    images = [img for img in img_folder.iterdir() if img.suffix.lower() in img_exts]

    # Retrieve the size for each image and save the relevant information in a dictionary
    data = []
    for img_path in images:
        try:
            with Image.open(img_path) as img:
                format = img.format
                width, height  = img.size
        except Exception as e:
            print(f"Failed to read image {img_path.name}: {e}")
            continue

        data.append({
              'Image_name' : img_path.stem,
              'Folder' : img_folder.name,
              'Absolute_path' : str(img_path.resolve()),
              'Format' : format,
              'Width' : width,
              'Height': height,
              'Image_size' : width*height
        })
        
    # Create a DataFrame from the image data list
    df = pd.DataFrame(data)
    
    # Save DataFrame to a CSV file
    csv_filename = img_folder / f"{project_name}_data.csv"
    df.to_csv(csv_filename, sep=';', index=False)
    
    print(f"Image data saved to {csv_filename}")

def create_labels_file(project_folder:str) -> None:
        """
        Creates a labels.txt file containing all unique class labels found in the annotation JSON files.
        
        :param project_folder: 
            - Type: str
            - Description: The absolute path to the folder named after the project. This folder should contain 
                        the annotation files, which are used to extract the class labels.

        :return: 
            - Type: None
            - Description: This function does not return a value. It creates a text file named 'labels.txt' 
                        in the project folder's image subdirectory.
        
        The resulting text file (`labels.txt`) is saved in the image folder of the project directory, 
        and can be used for further reference during model training or evaluation.
        """

        data_folder = Path(get_data_folder(project_folder))
        data_folder.mkdir(parents=True, exist_ok=True)
        
        annotation_folder = Path(get_ground_truth_folder_training(project_folder))
        labels_file = data_folder / 'labels.txt'
        
        annotation_files = [file for file in annotation_folder.iterdir() if not file.name.startswith('.')]
        
        unique_classes = set()
        
        for annotation_file in annotation_files:
            try:
                annotations = open_json_file(annotation_file)
                
                for i, result in enumerate(annotations['result']):
                    value = result['value']
                    label = value['rectanglelabels'][0]
                    
                    unique_classes.add(label)
            except ValueError as e:
                print(f"Il y a un problème avec le fichier {annotation_file}. Erreur : {e}")
                raise

        classes = list(unique_classes)
        print(classes)

        with open(labels_file, 'w', encoding='utf-8') as file:
            for index, classe in enumerate(classes):
                file.write(f"'{index}': '{classe}'\n")
        
        print(f"Labels file written in {labels_file} ")
    
def create_annotations_file(project_folder:str) -> None:
    """
    Converts JSON annotations into YOLO format and saves them as .txt files.

    Parameters
    ----------
    project_folder : str
        Absolute path to the project folder containing image and annotation data.

    Returns
    -------
    None
        Creates one YOLO-format .txt file per image in the 'labels' subdirectory.

    Notes
    -----
    - Requires a 'labels.txt' file to map class names to IDs.
    - Output files are named <image_name>.txt and stored in the 'labels' folder.
    - Each line in the output file represents a bounding box in YOLO format.
    """
    
    annotation_folder_ground_truth = Path(get_ground_truth_folder_training(project_folder))
    data_folder = Path(get_data_folder(project_folder))
    
    labels_folder = data_folder / 'labels'
    labels_folder.mkdir(parents=True, exist_ok=True)

    labels_file = data_folder / 'labels.txt'
    if not labels_file.exists():
        raise FileNotFoundError(f"'labels.txt' not found at {labels_file}")
    
    # Get the classes of the dataset from the labels file created with create_labels_file
    labels = get_labels(str(labels_file))
    
    # Get a list of the annotation files
    annotation_files = [file for file in annotation_folder_ground_truth.iterdir() if not file.name.startswith('.')]
        
    for annotation_file in annotation_files:
        annotations = open_json_file(annotation_file)

        # Get the name of the image
        img_path = annotations['task']['data']['image']
        img_path =img_path.replace("%5C","\\")
        img_name = Path(img_path).stem
        
        with open(labels_folder / f"{img_name}.txt", 'w') as yolo_annotation:
            print(f"création d'{img_name}.txt.")
            for result in annotations['result']:
                value = result['value']
                x, y, w, h = from_ls_to_yolo(value['x'], value['y'], value['width'], value['height'])
                classe_name = value['rectanglelabels'][0]
                classe_id = get_class_code(classe_name, labels)

                yolo_annotation.write(f"{classe_id} {x} {y} {w} {h}\n")
    
    print(f"Annotations successfully converted and saved")


def clean_classes_file(project_folder:str) -> None:
    """
    Cleans and converts a 'classes.txt' file into a YOLO-style 'labels.txt' format.

    Parameters
    ----------
    project_folder : str
        Path to the project folder containing the 'classes.txt' file.

    The resulting 'labels.txt' will contain each class with its corresponding index, e.g.:
    '0': 'car'
    '1': 'person'
    """
    project_folder = Path(project_folder)
    classes_txt_path = project_folder / 'classes.txt'
    labels_txt_path = project_folder / 'labels.txt'

    try:
        # Read class names from classes.txt
        with open(classes_txt_path, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]

        # Write cleaned content to labels.txt
        with open(labels_txt_path, 'w') as f:
            for i, class_name in enumerate(classes):
                f.write(f"'{i}': '{class_name}'\n")

        # Remove the original classes.txt
        classes_txt_path.unlink()
        print(f"Labels file written and renamed to : {labels_txt_path}")

    except Exception as e:
        print(f"There was a probleme loading the classes file :\n{e}")
    
def get_img_training_data(project_folder:str) -> None:
    """
    Copies ground truth images into the training folder under an 'images' subdirectory.

    Parameters
    ----------
    project_folder : str
        Absolute path to the project folder containing the ground truth images.

    Returns
    -------
    None
        Images are copied into the training folder under 'images/'.
    """
    data_folder = Path(get_data_folder(project_folder))
    data_folder.mkdir(parents=True, exist_ok=True)

    img_folder = data_folder / 'images'
    img_folder.mkdir(parents=True, exist_ok=True)
    
    img_folder_training = Path(get_img_folder_training(project_folder))
    if not img_folder_training.exists() or not img_folder_training.is_dir():
        raise NotADirectoryError(f"Can't find the ground truth image folder: {img_folder_training}")
    
    img_exts = {'.jpg', '.jpeg', '.png', '.tiff'}
    img_files = [img for img in img_folder_training.iterdir() if img.suffix.lower()in img_exts and not img.name.startswith('.')]
    
    for img_path in img_files:
        shutil.copy2(img_path, img_folder /img_path.name)

    print(f"Images copied in {img_folder}")

def create_dataset(project_folder, manually_downloaded):
    """
    Prepares a dataset for training by organizing files and generating required metadata.

    Parameters
    ----------
    project_folder : str
        Path to the project directory containing 'classes.txt' and an 'images' subfolder.

    manually_downloaded : bool
        If True, processes a manually downloaded dataset:
        - Cleans and formats 'classes.txt'.
        - Copies image files into the project structure.
        - Generates a CSV file from the image data.

        If False, assumes the project is structured and runs the full pipeline:
        - create_csv_file
        - create_labels_file
        - create_annotations_file
        - get_img_training_data

    Returns
    -------
    None
    """
    
    data_folder = Path(get_data_folder(project_folder))

    if manually_downloaded:
        project_name = Path(project_folder).name
        image_folder = data_folder / 'images'
        labels_folder = data_folder / 'labels'
        labels_file = data_folder / 'labels.txt'
    
        clean_classes_file(project_folder)

        # Validation: folders and files must exist
        if not data_folder.exists():
            print(f"[ERREUR] Data folder doesn't exist: {data_folder}")
            return

        if not image_folder.exists():
            print(f"[ERREUR] Image folder doesn't exist: {image_folder}")
            return
        
        if not labels_folder.exists():
            print(f"[ERREUR] Labels folder doesn't exist: {labels_folder}")
            return

        if not labels_file.exists():
            print(f"[ERREUR] Labels file doesn't exist: {labels_file}")
            return

        # Path to ground truth image folder
        ground_truth_folder_training = Path(get_img_folder_training(project_folder))
        ground_truth_folder_training.mkdir(parents=True, exist_ok=True)

        # Copy images from project_folder/images
        for file_path in image_folder.iterdir():
            if file_path.is_file():
                shutil.copy2(file_path, ground_truth_folder_training / file_path.name)

        create_csv_file(project_name)

    else:
        labels_file = data_folder / 'labels.txt'
        data_folder.mkdir(parents=True, exist_ok=True)
        print(f"Data will be stored in {data_folder}")

        
        create_csv_file(project_folder)
        # IMPORTANT pour éviter qu'un labels.txt avec le mauvais code soit recréé sur les images ajoutées en fin de cycle 
        if not labels_file.exists():
            create_labels_file(project_folder)
        create_annotations_file(project_folder)
        get_img_training_data(project_folder)