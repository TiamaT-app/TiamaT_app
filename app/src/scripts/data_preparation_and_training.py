import shutil
import random
from pathlib import Path
from datetime import datetime
import os

import cv2
import yaml
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt



from ..modules.folders_path import get_data_folder
from ..modules.device_function import which_device
from ..modules.generate_labels import get_labels
from ..modules.corners_functions import get_corners, from_corners_to_relative

def clean_comma(project_folder:str) -> None:
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

    This function ensures that annotation files are formatted correctly, preventing errors during the training process.
    """
    data_folder = Path(get_data_folder(project_folder))
    labels_folder = data_folder / 'labels'
    
    for file in labels_folder.iterdir():
        if file.is_file() and file.suffix.lower() == '.txt':       
            # Read the file content
            with open(file, 'r') as f:
                content = f.read()
            # Remove commas
            content_without_comma = content.replace(',', '')
            
            # Write the modified content in the file
            with open(file, 'w') as file:
                file.write(content_without_comma)

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
    - Useful for generating training data with varied viewpoints.
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
    new_filename = f"{img_path.stem}_PT{img_path.suffix}"
    transformed_img_path = img_path.with_name(new_filename)

    cv2.imwrite(str(transformed_img_path), dst)

    return M

def perspective_transformation_annotation(ann_file:str, img_file:str, M:np.ndarray) -> list:
    """ 
    This function applies a perspective transformation matrix to the bounding box annotations of an image 
    and saves the new transformed annotations in a separate file. The transformed annotations correspond to
    the modified perspective and dimensions of the image after applying the perspective transformation.

    :param ann_file: 
        - Type: str
        - Description: The absolute path to the annotation file (`.txt`) associated with the image. 
                       The file should contain bounding box annotations in YOLO format 
                       (label, x_center, y_center, width, height).

    :param img_file: 
        - Type: str
        - Description: The absolute path to the original image file. This is used to retrieve 
                       the original image dimensions and the dimensions of the transformed image.

    :param M: 
        - Type: numpy.ndarray
        - Description: The transformation matrix used for perspective transformation. This matrix 
                       is used to transform the bounding box coordinates to match the new image perspective.

    :return: 
        - Type: list of tuples
        - Description: Returns a list of transformed bounding box annotations. Each tuple contains 
                       the label and new relative coordinates (x_center, y_center, width, height) 
                       after applying the perspective transformation.

    The function ensures that the bounding box annotations remain consistent with the perspective changes applied to the image,
    which is essential for maintaining annotation accuracy after transformations.
    """

    img_height, img_width = cv2.imread(img_file).shape[:2]
    img_path = Path(img_file)
    new_filename = f"{img_path.stem}_PT{img_path.suffix}"
    transformed_img_path = img_path.with_name(new_filename)
    TP_img_height, TP_img_width = cv2.imread(str(transformed_img_path)).shape[:2]
    
    # print(f"Origal size: {img_height}, {img_width}\nNew size: {TP_img_height}, {TP_img_width}")

    # Initialising a list to store the new bounding box coordinates 
    bb_coordinates = []

    with open(ann_file, 'r') as annotations:
        for line in annotations:
            if not line.strip():
                continue #skip empty lines

            # Extraire les coordonnées de l'annotation
            label, x_center, y_center, width, height = line.strip().split()
            
            #print(type(label), type(x_center), type(y_center), type(width), type(height))

            # Convertir les coordonnées relatives en coordonnées absolues
            corners = get_corners(x_center, y_center, width, height, img_width, img_height)

            # Appliquer la transformation aux coins de la boîte d'annotation
            corners = np.array(corners, dtype=np.float32).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(corners, M).reshape(-1, 2)

            # Calculer les nouvelles coordonnées relatives
            new_upper_left = transformed_corners[0]
            new_bottom_right = transformed_corners[2]

            #print(f'new_upper_left = {new_upper_left}, new_bottom_right = {new_bottom_right}')

            # Transformer les nouvelles coordonnés en relatives
            transformed_x_center, transformed_y_center, transformed_width, transformed_height = from_corners_to_relative(
                new_upper_left, new_bottom_right, TP_img_width, TP_img_height)

            bb_coordinates.append((label, transformed_x_center, transformed_y_center, transformed_width, transformed_height))
    
    annotations_path = Path(ann_file)
    new_annotations_filename = f"{annotations_path.stem}_PT{annotations_path.suffix}"
    new_annotation_path = annotations_path.with_name(new_annotations_filename)

    with open(new_annotation_path, 'w') as transformed_annotations:
        for bb in bb_coordinates:
            label, x, y, w, h = bb
            transformed_annotations.write(f"{int(label)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


    return bb_coordinates

def generate_transformed_data(project_folder:str) -> None:
    """
    This function generates a set of transformed images and their corresponding annotations by applying 
    perspective transformations to each image and adjusting the bounding box annotations accordingly. 
    The new images and annotations are saved in the appropriate folders within the specified training folder.

    :param project_folder: 
        - Type: str
        - Description: The absolute path to the folder named after the project.

    :return: 
        - Type: None
        - Description: This function does not return a value. It generates transformed images and annotation files 
                       in place, saving them in the same subdirectories with modified filenames.

    This function automates the data augmentation process by generating new variations of the dataset, 
    which can be used to enhance model robustness during training.
    """
    
    print('Image tranformation has started..')
    
    data_folder = Path(get_data_folder(project_folder))
    labels_folder = data_folder / 'labels'
    img_folder = data_folder / 'images'

    img_exts = {".jpg", ".jpeg", ".png", ".tiff"}
    images = [img for img in img_folder.iterdir() if img.suffix.lower() in img_exts]

    for img_file in images:
        img_name = img_file.stem
        ann_file = labels_folder / f"{img_name}.txt"
        
        if ann_file.exists():
            try:
                M = perspective_transformation(str(img_file))
                perspective_transformation_annotation(str(ann_file), str(img_file), M)
            except Exception as e:
                print(f"Error processing {img_file.name}: {e}")

    print(f'New images stored in {img_folder}\nNew annotations stored in {labels_folder}')

def create_training_dataset(project_folder:str, pretrained_model:str | None, preexisting_distribution:bool) -> None:
    """
    Prepares training and validation datasets from a directory of images and labels.
    Generates:
    1. traindata.txt – 80% of images
    2. valdata.txt – 20% of images
    3. training_dataset.txt – all images

    If a pre-existing split is provided, it's reused. Otherwise, a new random split is created.
    Images and labels are copied into organized folders for training and validation.

    Parameters
    ----------

    :param project_folder: 
        - Type: str
        - Description: The absolute path to the folder named after the project.

    :param pretrained_model: 
        - Type: str
        - Description: The absolute path to the folder containing the pre-trained model data. 
                       This parameter is currently unused in the function but may be required for future enhancements.

    :param preexisting_distribution: 
        - Type: bool
        - Description: If `preexisting_distribution` is True, the function reuses a previous train/val split from 
                        the `pretrained_model` folder. Otherwise, it creates a new random split.


    :return: 
        - Type: None
        - Description: This function does not return a value. It creates and saves the text files 
                       `traindata.txt`, `valdata.txt`, and `training_dataset.txt` and organizes the images 
                       and labels into separate subdirectories for training and validation.

    This function ensures that the training and validation data are correctly organized and ready for model training.
    """
    
    data_folder = Path(get_data_folder(project_folder))
    project_name = Path(project_folder).name

    img_folder = data_folder / 'images'
    img_train_folder = data_folder.parent / 'datasets' / project_name / 'images' / 'train'
    img_val_folder = data_folder.parent / 'datasets' / project_name / 'images' / 'val'

    labels_folder = data_folder / 'labels'
    labels_train_folder = data_folder.parent / 'datasets' / project_name / 'labels' / 'train'
    labels_val_folder = data_folder.parent / 'datasets' / project_name / 'labels' / 'val'
    
    data_stat_folder = data_folder / 'dataset_statistics'
    data_stat_folder.mkdir(parents=True, exist_ok=True)

    train_data = data_stat_folder / 'traindata.txt'
    val_data = data_stat_folder / 'valdata.txt'
    training_dataset = data_stat_folder / 'training_dataset.txt'

    
    img_exts = {".jpg", ".jpeg", ".png", ".tiff"}

    if preexisting_distribution:
        if pretrained_model:
            print(f'Use pre-existing files from {pretrained_model}.')
            previous_distribution = Path(pretrained_model) / 'dataset_statistics'
            if not previous_distribution.exists():
                raise FileNotFoundError(f"No dataset_statistics found in {pretrained_model}")
            shutil.copytree(previous_distribution, data_stat_folder, dirs_exist_ok=True)
            # TODO: Copy yaml file

        else:
            pretrained_model = str(input("Please indicate the path to the pre-trained model folder: "))
            if not isinstance(pretrained_model, str):
                raise TypeError(f"The pre-trained model folder path must be a string, not {type(pretrained_model)}")

    else:
        # Get a list of the images
        image_files = [img for img in img_folder.iterdir() if img.suffix.lower() in img_exts]
        if not image_files:
            raise ValueError(f"No images found in {img_folder}. Please check your dataset.")

        # Shuffle file names randomly
        random.shuffle(image_files)


        # TODO: Add sklearn stratified split here for balanced class distribution

        # Calcul le nombre d'images pour chaque ensemble
        num_images = len(image_files)
        num_train = int(num_images * 0.8)
        num_val = int(num_images - num_train)

        # Divide file names into two sets : one for the training, one for the validation
        train_files = image_files[:num_train]
        print(f"{len(train_files)} images assigned to training.")
        val_files = image_files[num_train:num_train+num_val]
        print(f"{len(val_files)} images assigned to validation.")
        

        # Create a file with the list for the train data
        with open(train_data, 'w') as f:
            for image_file in train_files:
                f.write(str(image_file) + "\n")
        print(f"File created: {train_data}")

        # Create a file with the list for valdidation data
        with open(val_data, 'w') as f:
            for image_file in val_files:
                f.write(str(image_file) + "\n")
        print(f"File created: {val_data}")


        # Create a file with all the dataset
        with open(training_dataset, 'w') as f:
            for image_file in image_files:
                    f.write(str(image_file) + "\n")
            print(f"File created: {training_dataset}")
    

    # Split images and txt files into folders from a .txt file
    split_data_for_training(str(train_data), 
                            str(labels_folder),
                            str(img_train_folder),
                            str(labels_train_folder))
    
    split_data_for_training(str(val_data),
                            str(labels_folder),
                            str(img_val_folder),
                            str(labels_val_folder))
    
    # Create the yaml file
    write_yaml_file(project_folder)

def split_data_for_training(img_list:str, labels_folder:str, output_img_folder:str, output_labels_folder:str) -> None:
    """
    Organizes images and annotation files into the appropriate YOLOv8 train/val subdirectories.

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

    Returns
    -------
    None
        Files are moved into the specified YOLO directory structure.
    """


    # Create the output folder if it does not already exist
    Path(output_img_folder).mkdir(parents=True, exist_ok=True)
    Path(output_labels_folder).mkdir(parents=True, exist_ok=True)
    
    # Open the text file containing the image paths
    with open(img_list, "r") as f:
        # Browse through each line of the file
        for line in f:
            # Get the image path and text file name
            image_path = Path(line.strip())
            image_name = image_path.stem

            txt_file = Path(labels_folder) / f"{image_name}.txt"

            try:
                # Copy image to output folder
                shutil.move(str(image_path), str(Path(output_img_folder) / image_path.name))
            except FileNotFoundError:
                print(f"Image file {image_path} not found.")
        
            # Copy text file to output folder
            try:
                shutil.move(str(txt_file), str(Path(output_labels_folder) / txt_file.name))

            except FileNotFoundError:
                print(f'Text file {txt_file} does not exist')
    print(f'Image files move in {output_img_folder}')
    print(f'Text files move in {output_labels_folder}')

def write_yaml_file(project_folder:str) -> None:
    """
    Creates a `.yaml` configuration file for YOLOv8 training, specifying:
    - Dataset path
    - Train and validation folder structure
    - Class label mapping from `labels.txt`

    Parameters
    ----------
    project_folder : str
        Absolute path to the root project folder. Must contain a `labels.txt` file 
        and will be used to locate or generate the `datasets/<project_name>` directory.

    Returns
    -------
    None
        Writes a YAML file in the datasets folder for YOLO training.
    """

    data_folder = Path(get_data_folder(project_folder))
    project_name = Path(project_folder).name

    dataset_folder = data_folder.parent / 'datasets' / project_name
    labels_file = data_folder / 'labels.txt'
    yaml_path = dataset_folder / f"{project_name}.yaml"

    # Get the annotations classes
    annotation_classes = get_labels(labels_file)
    
    # Convertir les clés du dictionnaire annotation_classes en entiers
    annotation_classes_int = {int(key): value for key, value in annotation_classes.items()}

    # Formater la chaîne avec les éléments dans l'ordre souhaité
    yaml_data = [
        f"path: {dataset_folder}",
        f"train: 'images/train'",
        f"val: 'images/val'",
        "",
        f"#class names",
        f"names:"]
    
    for class_id, label in annotation_classes_int.items():
        yaml_data.append(f"  {class_id}: '{label}'")
        
        # TODO: Ajouter Albumentation

    with open(yaml_path, 'w') as yaml_file:
        yaml_file.write('\n'.join(yaml_data))

    print(f"File written to {yaml_path}")

def yolo_training(project_folder:str, use_model:str, img_size:int, 
                  epochs:int, batch:int, workers:int, dropout:float, model_name:str,
                  pretrained_model:str | None) -> str:
    """
    Trains a YOLO model using a specified dataset and configuration.

    :param project_folder: 
        - Type: str
        - Description: The absolute path to the folder named after the project.

    :param use_model: 
        - Type: str
        - Description: The YOLO model architecture to use for training (e.g., 'yolo11x.pt'). 
                       If a pre-trained model is provided, this parameter is overridden.

    :param img_size: 
        - Type: int
        - Description: The size of the input images. Larger image sizes can increase model accuracy 
                       but may also increase computational load.

    :param epochs: 
        - Type: int
        - Description: The number of epochs to train the model. More epochs allow the model to learn 
                       better but may result in overfitting if set too high.

    :param batch: 
        - Type: int
        - Description: The batch size to use during training. Larger batch sizes require more memory 
                       but can stabilize gradient updates.

    :param workers: 
        - Type: int
        - Description: The number of workers for data loading. Increasing this number can speed up data 
                       loading but may require more computational resources.

    :param dropout: 
        - Type: float
        - Description: The dropout factor applied to the labels to prevent overconfidence in predictions. 
                       Typically set between 0 and 1.

    :param pretrained_model: 
        - Type: str
        - Description: The path to a pre-trained model, if any. If provided, the function will use this model 
                       as the starting point for training. If not provided, it uses the `use_model` parameter 
                       to select the model architecture.

    :return: 
        - Type: None
        - Description: This function does not return a value. It trains the YOLO model using the provided 
                       parameters and saves the results to a specified output folder.

    This function automates the YOLO training process, providing flexibility in configuration and managing results storage.
    """
    
    # Derive additional paths and model name
    data_folder = Path(get_data_folder(project_folder))
    project_folder_path = Path(project_folder)
    project_name = project_folder_path.name

    dataset_folder = data_folder.parent / 'datasets' / project_name
    yaml_file = dataset_folder / f"{project_name}.yaml"


    # Check if yaml_file exists
    if not yaml_file.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_file}")

    # Determine which model to use
    if pretrained_model == None:
        model_name = model_name
        # model_name = f'{str(project_name)}_{date}_{use_model}_i{img_size}_e{epochs}_b{batch}_w{workers}'
    else:
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

    Parameters
    ----------
    project_folder : str
        Path to the root project folder (used to derive output path).
    
    interrupted_model_folder : str
        Path to the folder containing the partially trained model's data. 
        This folder must include a 'weights/last.pt' file.

    Returns
    -------
    None
        Resumes training and evaluates the model. Results are saved in the same folder.
    """
    last_weight = Path(interrupted_model_folder) / 'weights' / 'last.pt'
    
    if not last_weight.exists():
        raise FileNotFoundError(f"No checkpoint found at {last_weight}")
    
    model_name = Path(interrupted_model_folder).name

    project_name = Path(project_folder).name

    # Check if the GPU is available - if not, use the CPU
    device = which_device()
    
    # Load a model
    model = YOLO(last_weight).to(device)  # load a partially trained model

    # Resume training
    results = model.train(resume=True)

    # Evaluate the model's performance on the validation set
    val_results = model.val(
        name = f"{model_name}/{str(project_name)}_val")
    
def dispatch_data(project_folder:str | Path, use_model:str | Path, img_size:int, 
                  epochs:int, batch:int, workers:int, dropout:float, model_name:str | Path,
                  pretrained_model:str | Path | bool, interrupted_model_folder:str | bool) -> Path:
    """
    This function organizes and finalizes the data used for training by moving relevant files and directories 
    into the model folder. It also restores the original structure of the dataset folder by moving image and 
    annotation files back to their respective subdirectories and deletes the temporary training folder.

    :return: 
        - Type: None
        - Description: This function does not return a value. It organizes and moves files into appropriate folders, 
                       restores the original dataset structure, and deletes the temporary training folder.

    This function ensures that all data and configurations used for training are stored in a dedicated model folder, 
    making it easy to track and manage different training sessions.
    """

    data_folder = Path(get_data_folder(project_folder))
    project_name = Path(project_folder).name

    img_folder = data_folder / 'images'
    img_train_folder = data_folder.parent / 'datasets' / project_name / 'images' / 'train'
    img_val_folder = data_folder.parent / 'datasets' / project_name / 'images' / 'val'

    labels_folder = data_folder / 'labels'
    labels_train_folder = data_folder.parent / 'datasets' / project_name / 'labels' / 'train'
    labels_val_folder = data_folder.parent / 'datasets' / project_name / 'labels' / 'val'
    

    dataset_folder = data_folder.parent / 'datasets' / project_name
    yaml_file = dataset_folder / f"{project_name}.yaml"
    print(yaml_file)

    
    # Determine which model name to use
    if interrupted_model_folder:
        model_folder = Path(str(interrupted_model_folder))
        model_name = model_folder.name
    else:
        if not pretrained_model:
            model_name = model_name
        else:
            model_name = f'{Path(str(pretrained_model)).name}'
        
        model_folder = Path.cwd() / 'output' / 'train' / model_name
    
    print(model_folder)    
    # Move the data used for the training session into the model folder
    shutil.copy2(str(yaml_file), str(model_folder / f"{project_name}.yaml"))
    print(f'The .yaml file has been moved into {model_folder}')
    
    shutil.copy2(str(data_folder / 'labels.txt'), str(model_folder / 'labels.txt'))
    print(f'The labels.txt file has been copied in {model_folder}')
    
    shutil.move(str(data_folder / 'dataset_statistics'), str(model_folder))
    print(f'The statistics folder with the training data have been moved to {model_folder}.')
  
    img_folder.mkdir(parents=True, exist_ok=True)
    if img_train_folder.exists():
        for file in img_train_folder.iterdir():
            shutil.move(str(file), str(img_folder / file.name))
        print(f"Files from {img_train_folder} have been moved into {img_folder}")
    else:
        print(f"Warning: {img_train_folder} does not exist.")
    
    if img_val_folder.exists():
        for file in img_val_folder.iterdir():
            shutil.move(str(file), str(img_folder / file.name))
        print(f"Files from {img_val_folder} move into {img_folder}")
    else:
        print(f"Warning: {img_val_folder} does not exist.")

    labels_folder.mkdir(parents=True, exist_ok=True)
    if labels_train_folder.exists():
        for file in labels_train_folder.iterdir():
            shutil.move(str(file), str(labels_folder / file.name))
        print(f"Files from {labels_train_folder} move into {labels_folder}")
    else:
        print(f"Warning: {labels_train_folder} does not exist.")

    if labels_val_folder.exists():    
        for file in labels_val_folder.iterdir():
            shutil.move(str(file), str(labels_folder / file.name))
        print(f"Files from {labels_val_folder} move into {labels_folder}")
    else:
        print(f"Warning: {labels_val_folder} does not exist.")

    shutil.rmtree(str(data_folder.parent / 'datasets' / project_name))
    
    print(f"The {data_folder.parent / 'datasets' / project_name} has been deleted")

    print(f"✅ All data successfully dispatched")
    return model_folder

def entrainement(project_folder, nombre_epoch, dropout, model, model_name):
    clean_comma(project_folder)
    generate_transformed_data(project_folder)
    create_training_dataset(project_folder, model, preexisting_distribution=False)

#Config du réentraînement
    use_model = model # to be changed as needed, by default use 'yolov11x.pt'
    img_size = 640 # to be changed as needed, by default use 640
    epochs = nombre_epoch # to be changed as needed
    batch = -1 # to be changed as needed, by default use 8 or or -1 for AutoBatch
    workers = 8 # to be changed as needed, by default 24, or 8 (https://docs.ultralytics.com/modes/train/#train-settings)
    dropout = dropout # Elimine aléatoirement 10% connaissance à chaque époque

    yolo_training(project_folder, use_model, img_size, epochs, 
                  batch, workers, dropout, 
                  model_name, pretrained_model = None)
    os.environ["complete"] = "True"