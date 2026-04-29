import uuid
import json
import random
import unicodedata
from glob import glob
from pathlib import Path

import cv2
import torch
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import re

import sys
sys.path.append(str(Path.cwd().parent / 'modules'))

from ..modules.device_function import which_device
from ..modules.folders_path import get_results_folder
from ..modules.class_names_functions import get_labels, get_class_name, get_class_code
from ..modules.transform_coordinates_functions import from_relative_coordinates_to_absolute

def process_images_with_yolo(project_folder:str, yolo_model_folder:str) -> None:
    """
    Processes all image files in the 'eval_images' subdirectory of a project folder using a YOLO model.

    This function recursively scans the 'image_inputs/eval_images' directory, detects valid image files
    (e.g., .jpg, .jpeg, .png, .tiff), and runs object detection on each image using the provided YOLO model.
    Detection results are saved in a structured format for later analysis.

    Note:
    -----
    Hidden files and images located inside hidden directories (whose names start with '.') are automatically ignored.

    Related:
    --------
    See: https://github.com/ultralytics/ultralytics/issues/2143

    Parameters:
    -----------
    project_folder : str
        Path to the root project directory containing the 'image_inputs/eval_images' folder.

    yolo_model_folder : str
        Path to the folder containing the YOLO model weights.
        The function expects a file at: 'weights/best.pt' within this folder.

    Returns:
    --------
    None
        This function does not return anything.
        It performs detection on each image and saves results (typically in a 'labels' folder).
    """

    eval_folder = Path(project_folder) /'image_inputs' / 'eval_images'
    img_exts = {".jpg", ".jpeg", ".png", ".tiff"}

     # Recursively search for all image files
    for img_path in eval_folder.rglob("*"):
        # Skip non-image files or files in hidden folders
        if not img_path.is_file() or img_path.suffix.lower() not in img_exts:
            continue
        if any(part.startswith('.') for part in img_path.parts):
            continue
        
        # Run YOLO object detection on the image
        process_single_image_with_yolo(project_folder, yolo_model_folder, str(img_path))
        print(f"Processed: {img_path}")
                
def process_single_image_with_yolo(project_folder:str, yolo_model_folder:str, img_path:str) -> None:
    """
    Runs YOLO object detection on a single image and saves the results as a label file in YOLO format.

    Parameters
    ----------
    project_folder : str
        Path to the root of the project. Used to determine where to save results.

    yolo_model_folder : str
        Path to the folder containing the YOLO model weights. The model file should be located at:
        '<yolo_model_folder>/weights/best.pt'

    img_path : str
        Absolute path to the image to be processed.

    Returns
    -------
    None
        The function saves a .txt file in the 'labels' subfolder under the results directory, with the format:
        <class_id> <x_center> <y_center> <width> <height> <confidence>
    """

    # Check if the GPU is available - if not, use the CPU
    device = which_device()
    
    # Load YOLO model
    model_path = Path(yolo_model_folder) / 'weights' / 'best.pt'
    
    # Prepare output directory
    results_folder = get_results_folder(project_folder, yolo_model_folder)
    labels_folder = Path(results_folder) / 'labels'
    labels_folder.mkdir(parents=True, exist_ok=True)
    
    img_name = Path(img_path).stem
    image = cv2.imread(str(img_path))

    yolo_model = YOLO(model_path)

    # Process the image using YOLO
    results = yolo_model.predict(source=image,
                                 device=device,
                                 agnostic_nms=True,
                                 imgsz=640,
                                 save_txt=False,
                                 save_conf=False,
                                 verbose=False
                                )
    
    boxes = results[0].boxes
    
    if boxes is None or len(boxes) == 0:
            print(f"No detections found in {img_path}")
            return
    
    # Save prediction to a YOLO-format .txt file
    label_path = labels_folder / f"{img_name}.txt"
    
    with open(label_path, 'w') as label_file:
        for box in boxes:
            xywh = " ".join([f"{value:.4f}" for value in box.xywhn.cpu().squeeze().tolist()])
            class_id = int(box.cls.cpu().item())
            confidence = box.conf.cpu().item()
            label_line = f"{class_id} {xywh} {confidence:.4f}\n"
            label_file.write(label_line)
    
    print(f"✅ Saved predictions for {img_name} to {label_path}")

def get_image_data(project_folder:str) -> None:
    """
    Generates a CSV file containing metadata for each image in the 'eval_images' subfolder
    of the specified project. Metadata includes image name, format, dimensions, and paths.

    Parameters
    ----------
    project_folder : str
        Absolute path to the root project directory. The function will look for images in:
        project_folder/image_inputs/eval_images.

    Returns
    -------
    None
        The function creates a CSV file named '<folder>.csv' in the 'eval_images' folder
        with metadata for each image.
    """

    eval_folder = Path(project_folder) /'image_inputs' / 'eval_images'
    project_name = eval_folder.name
    img_exts = {".jpg", ".jpeg", ".png", ".tiff"}
    
    data = []

    images = [img for img in eval_folder.rglob('*') 
              if img.is_file() 
              and img.suffix.lower() in img_exts
              and not img.name.startswith('.')]
    
    for file in images:
        try:
            with Image.open(file) as img:
                absolute_path = str(file.resolve())
                format = img.format
                width, height  = img.size
        except Exception as e:
            print(f"Could not read {file.name}: {e}")
            continue

        img_data = {
              'Image_name' : file.stem,
              'Folder' : str(file.parent),
              'Absolute_path' : absolute_path,
              'Format' : format,
              'Width' : width,
              'Height': height,
              'Image_size': int(width)*int(height)
        }

        data.append(img_data)
        
    # Create and export the DataFrame
    df = pd.DataFrame(data)
    csv_filename = eval_folder /f"{project_name}.csv"
    df.to_csv(csv_filename, sep=';', index=False)
    print(f"Metadata CSV saved to: {csv_filename}")

def normalize_filename(filename:str) -> str:
    """
    Normalize the filename to remove special characters and ensure consistency.
    This function converts the filename to ASCII, removing accents and other special characters, 
    making it easier to match filenames across different platforms.
    
    :param filename: 
        - Type: str
        - Description: The filename to be normalized.

    :return: 
        - Type: str
        - Description: The normalized filename, with special characters removed.
    """
    return unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode('ASCII')

def yolo_to_csv(project_folder:str, yolo_model_folder:str) -> None:
    """
    Converts YOLO-format annotation files into a CSV file with full metadata and bounding box information.

    Parameters
    ----------
    project_folder : str
        Root path of the project.

    yolo_model_folder : str
        Path to the YOLO model folder (must contain weights and labels).

    Returns
    -------
    None
        Generates a CSV file in the 'results' folder with annotations and metadata.
    """

    eval_folder = Path(project_folder) /'image_inputs' / 'eval_images'
    project_name = Path(project_folder).name
    
    results_folder = get_results_folder(project_folder, yolo_model_folder)
    labels_file = Path(yolo_model_folder) / 'labels.txt'

    labels_folder = Path(results_folder) / 'labels'
    labels_folder.mkdir(parents=True, exist_ok=True)
    
    final_results_folder = Path(results_folder) / 'results'
    final_results_folder.mkdir(parents=True, exist_ok=True)
    
    csv_file = final_results_folder / f"{project_name}.csv"
    img_exts = {".jpg", ".jpeg", ".png", ".tiff"}

    annotation_files = [file for file in labels_folder.iterdir() if file.suffix.lower() == '.txt']

    all_rows = []

    # Recursively search for all image files
    for img_path in eval_folder.rglob("*"):
        if any(part.startswith('.') for part in img_path.parts):
            continue

        if not img_path.is_file() or img_path.suffix.lower() not in img_exts:
            continue

        try:
            with Image.open(img_path) as img:
                image_width, image_height = img.size
        except Exception as e:
            print(f"Could not read image {img_path}: {e}")
            continue
        
        if not annotation_files:
            print(f'No annotations found in {labels_folder}.')
            continue
        
        # Trouver les annotations correspondantes
        matching_annotations = [ann_file for ann_file in annotation_files if str(ann_file.stem) == str(img_path.stem)]
        
        # If no matching annotation, continue
        if not matching_annotations:
            print(f"No annotation found for image {img_path}.")
            all_rows.append({
                        'Image_Path': str(img_path),
                        'Image_Width': image_width,
                        'Image_Height': image_height,
                        'YOLO_Results_File': '',
                        'Class_Id': '',
                        'Class_Name': '',
                        'Detected_coordinates': '',
                        'Absolute_coordinates': '',
                        'Confidence': '',
                    })
            # continue

        # Process matching annotations
        for matching_annotation in matching_annotations:
            with open(matching_annotation, 'r') as f:
                for line in f.readlines():
                    class_id, x_center, y_center, width, height, confidence = map(float, line.strip().split())
                    # Convert relative YOLO coordinates to absolute
                    x, y, abs_width, abs_height = from_relative_coordinates_to_absolute(
                        x_center, y_center, width, height, image_width, image_height)

                    # Add row of data for the DataFrame
                    all_rows.append({
                        'Image_Path': str(img_path),
                        'Image_Width': image_width,
                        'Image_Height': image_height,
                        'YOLO_Results_File': str(matching_annotation),
                        'Class_Id': int(class_id),
                        'Class_Name': get_class_name(int(class_id), get_labels(str(labels_file))),
                        'Detected_coordinates': f'{x_center} {y_center} {width} {height}',
                        'Absolute_coordinates': f"{x} {y} {abs_width} {abs_height}",
                        'Confidence': confidence,
                    })
            print(f"Processed annotation for {img_path}")

    # Generate and save the CSV with results
    if all_rows:
        df = pd.DataFrame(all_rows)
        df_sorted = df.sort_values('Image_Path')
        df_sorted.to_csv(csv_file, sep=';', index=False)
        print(f'The file {csv_file} has been created.')
    else:
        print("No correspondence found between images and annotations.")

def convert_yolo_annotations_to_label_studio_format(yolo_annotations:str, img_path:str, yolo_model_folder:str) -> list:
    """
    This function converts YOLO annotation data into Label Studio's JSON format. The converted annotations can 
    then be imported into Label Studio for visualization, review, and further editing. The function uses the 
    YOLO annotation values (class ID, bounding box coordinates, and confidence score) to generate a compatible 
    JSON structure for Label Studio.

    Documentation: [Label Studio Converter](https://github.com/heartexlabs/label-studio-converter/blob/master/label_studio_converter/imports/yolo.py#L85)

    :param yolo_annotations: 
        - Type: list of str
        - Description: A list of annotation strings in YOLO format. Each string contains class ID, bounding box coordinates 
                       (x_center, y_center, width, height), and confidence score, separated by spaces.

    :param img_path: 
        - Type: str
        - Description: The absolute path to the image file corresponding to the annotations. The image dimensions are 
                       used to convert YOLO relative coordinates into absolute coordinates for Label Studio.

    :param yolo_model_folder: 
        - Type: str
        - Description: The path to the folder containing the YOLO model. This folder is used to retrieve class names 
                       from the `labels.txt` file and set the model version in the JSON output.

    :return: 
        - Type: list
        - Description: Returns a list containing the formatted JSON data compatible with Label Studio. The JSON includes 
                       image metadata, bounding box annotations, and additional properties required for visualization.

    This function helps streamline the process of converting YOLO annotations into Label Studio format, making it easier 
    to visualize and refine the predictions in an interactive environment.
    """
    labels_file = Path(yolo_model_folder) / 'labels.txt'

    results = []

        # Get the image dimensions
    with Image.open(img_path) as img:
        image_width, image_height = img.size
        # print(f'Largeur: {image_width}, Hauteur: {image_height}')
    
    # Get the bounding_boxes coordinates
    for line in yolo_annotations:
        class_id, x_center, y_center, width, height, confidence = map(float, line.split())
            # print(f'class id: {class_id}, x center: {x_center}, y center: {y_center}, width: {width}, height: {height}')

        result = {
                "id": f'{uuid.uuid1()}',
                "type": "rectanglelabels",
                "from_name": "label",
                "to_name": "image",
                "original_width": image_width,
                "original_height": image_height,
                "image_rotation": 0,
                "value":{
                    "rotation": 0,
                    "x": (x_center - width / 2) * 100,
                    "y": (y_center - height / 2) * 100,
                    "width": width * 100,
                    "height": height * 100,
                    "rectanglelabels": [f"{get_class_name(int(class_id), get_labels(str(labels_file)))}"]
                },
            "score": confidence
        }
        results.append(result)
        # print(results)

    label_studio_format = [{
        "data": {
            "image": img_path
        },
        "predictions":[{
            "model_version": str(Path(yolo_model_folder).name),
            "score": '',
            "result": results
        }]
        
    }]

    #print(label_studio_format)
    return label_studio_format

def convert_unannotated_to_label_studio_format(img_path: str, yolo_model_folder: str) -> list:
    """
    Build a Label Studio–compatible JSON entry for an image that has no YOLO annotations.

    :param img_path:
        - Type: str
        - Description: Path or URL to the image to import into Label Studio.
    :param yolo_model_folder:
        - Type: str
        - Description: Path to the YOLO model folder; its basename is used as the `model_version` field.
    :return:
        - Type: list of dict
        - Description: A one‐element list containing the Label Studio task JSON with an empty
                       `result` array so that the image appears unannotated in the UI.
    """
    # Read image size (not strictly required when result is empty, but retained for completeness)
    with Image.open(img_path) as img:
        width, height = img.size

    entry = {
        "data": {
            "image": img_path
        },
        "predictions": [
            {
                "model_version": str(Path(yolo_model_folder).name),
                "score": "",
                "result": []  # no annotations yet
            }
        ]
    }

    return [entry]


def get_ls_for_local_files(project_folder: str, yolo_model_folder: str) -> str:
    """
    Batch-convert all images in the project's `eval_images` folder into a single Label Studio JSON import file,
    handling both images with YOLO annotations (.txt files) and unannotated images.

    Parameters
    ----------
    project_folder : str
        Absolute path to the root project directory. The function will scan:
        project_folder/image_inputs/eval_images for image files (.jpg, .png, .tiff, etc.).

    yolo_model_folder : str
        Absolute path to the YOLO model folder, which contains the prediction outputs
        (in a `labels` subfolder) and a `labels.txt` file used to interpret class indices.
        Also sets the model_version field in the Label Studio payload.

    Returns
    -------
    str
        Path to the generated JSON file aggregating all Label Studio task entries.
        This file can be directly imported into Label Studio via:
        `label-studio import tasks --format json`.

    Notes
    -----
    For each image, the function checks if a corresponding YOLO `.txt` file exists in the model's
    `labels` directory:
    - If it exists, it reads and converts the annotations to Label Studio's rectangle-label format.
    - If not, it generates an empty task so the image appears unannotated.

    All entries are aggregated and paths rewritten for local-files import, then saved
    to a single JSON file inside the model’s `results` directory.
    """


    # Path construction
    eval_folder = Path(project_folder) /'image_inputs' / 'eval_images'
    project_name = Path(project_folder).name
    
    results_folder = get_results_folder(project_folder, yolo_model_folder)
    labels_folder = Path(results_folder) / 'labels'
    labels_folder.mkdir(parents=True, exist_ok=True)
    
    final_results_folder = Path(results_folder) / 'results'
    final_results_folder.mkdir(parents=True, exist_ok=True)
    
    json_file = final_results_folder / f"{project_name}_ls_local_files.json"
    img_exts = {".jpg", ".jpeg", ".png", ".tiff"}

    all_ls = []

    # Recursively search for all image files
    for img_path in eval_folder.rglob("*"):
        if any(part.startswith('.') for part in img_path.parts):
            continue
        if not img_path.is_file() or img_path.suffix.lower() not in img_exts:
            continue
        
        img_name = img_path.stem
        label_txt = labels_folder / f"{img_name}.txt"

        if label_txt.exists():
            # 1) Annotated image → Label Studio conversion
            with open(label_txt, 'r') as f:
                lines = f.read().splitlines()
            entries = convert_yolo_annotations_to_label_studio_format(
                lines, str(img_path), yolo_model_folder
            )
        else:
            # 2) Unannotated image → empty entry
            entries = convert_unannotated_to_label_studio_format(
                str(img_path), yolo_model_folder
            )
            
        all_ls.extend(entries)

    # Rewriting paths for Label Studio
    new_prefix = '/data/local-files/?d=' + (eval_folder.relative_to(Path(project_folder).parent)).as_posix()

    for ann in all_ls:        
        ann['data']['image'] = ann['data']['image'].replace(str(eval_folder), new_prefix).replace("\\",'/')

    # Writing the JSON file
    with open(json_file, 'w') as f:
        json.dump(all_ls, f, indent=2)
    print(f"Label Studio annotations written to {json_file}")
    
    return json_file

def generate_random_colours() -> str:
    """
    This function generates a random color in hexadecimal RGB format. The color is created by selecting 
    random values for the red, green, and blue channels, and then formatting these values into a hex string.

    :return: 
        - Type: str
        - Description: A string representing the random color in hexadecimal format (e.g., `#a1b2c3`).
    """
    r = random.randint(2, 255)
    g = random.randint(2, 255)
    b = random.randint(2, 255)

    hex_colour = '#{:02x}{:02x}{:02x}'.format(r, g, b)
    
    return hex_colour

def get_labeling_code(project_folder:str, yolo_model_folder:str) -> None:
    """
    Generates a Label Studio XML configuration template using class labels from a YOLO model.
    Each label is assigned a random background color for display in Label Studio.

    !!! Note:
        The generated file is a text file and must be manually copied into the configuration of a new
        Label Studio project. It is not a direct import.

    Parameters
    ----------
    project_folder : str
        Path to the root project folder. Used to determine the dataset and where to store the output.

    yolo_model_folder : str or Path
        Path to the YOLO model directory. Must contain a `labels.txt` file.

    Returns
    -------
    None
        A text file is saved in the model's `results` directory, containing the Label Studio config.
    """

    # Path construction
    project_name = Path(project_folder).name
    
    results_folder = Path(get_results_folder(project_folder, yolo_model_folder))
    results_folder.mkdir(parents=True, exist_ok=True)

    final_results_folder = results_folder / 'results'
    final_results_folder.mkdir(parents=True, exist_ok=True)

    labeling_file = final_results_folder / f"{str(project_name)}_labeling_code.txt" 
    
    labels_file = Path(yolo_model_folder) / "labels.txt"
    labels = get_labels(labels_file)
    label_names = labels.values()
    
    # Add the generated colour to your model for each label usiung the Label Studio template for bounding boxes
    labeling_template = """<View>
    <View style="display:flex;align-items:start;gap:8px;flex-direction:row">
        <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
        <RectangleLabels name="label" toName="image" showInline="false">        
    {label_backgrounds}    </RectangleLabels>
    </View>
    </View>
    """
    
    # Generate the part of the model for each label with a random colour
    label_backgrounds = ""
    for label in label_names:
        random_colour = generate_random_colours()
        label_backgrounds += f'        <Label value="{label}" background="{random_colour}"/>\n'
    
    # Intégrez la partie du modèle générée pour chaque étiquette
    labeling_template = labeling_template.format(label_backgrounds=label_backgrounds)
    
    with open(labeling_file, 'w') as file:
        file.write(labeling_template)
    
    # Utilisez le modèle avec les couleurs générées
    print(f"The labeling template is saved in {labeling_file}")
    return labeling_template

def get_model_list(project_folder):#DRACONES a mettre dans modules/folders_path 
    project_name = str(Path(project_folder).name)
    print(project_name)
    models=["------"]
    root = Path.cwd()
    trained_folder = Path( root /"output" / "train")
    list_models =[str(i.name) for i in trained_folder.iterdir()]
    print(list_models)
    for i in list_models:
        match = re.match(re.escape(project_name) + r"_[0-9]{8}_[0-9]{6}$", i)
        if match :
           models.append(i)
        else :
            pass
    return models