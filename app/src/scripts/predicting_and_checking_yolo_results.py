import uuid
import json
import traceback
import unicodedata
from pathlib import Path
import os

import pandas as pd
from PIL import Image
from ultralytics import YOLO

from ..modules.device_function import which_device
from ..modules.folders_path import get_results_folder
from ..modules.class_functions import get_class_name
from ..modules.transform_coordinates_functions import from_relative_coordinates_to_absolute
from ..modules.generate_labels import get_labels

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _eval_images(eval_folder: Path) -> list[Path]:
    """
    Images du dossier eval_images (sous-dossiers compris), sans les fichiers ou dossiers caches.

    Seule la partie du chemin situee sous eval_images est examinee : si le projet lui-meme
    se trouve dans un dossier cache (~/.local/..., par exemple), ses images restent prises en compte.
    """
    images = []
    for img_path in sorted(Path(eval_folder).rglob("*")):
        if not img_path.is_file() or img_path.suffix.lower() not in IMG_EXTENSIONS:
            continue
        if any(part.startswith('.') for part in img_path.relative_to(eval_folder).parts):
            continue
        images.append(img_path)
    return images

def process_images_with_yolo(project_folder:str | Path, yolo_model_folder:str | Path) -> None:
    """
    Processes all image files in the 'eval_images' subdirectory of a project folder using a YOLO model.

    This function recursively scans the 'image_inputs/eval_images' directory and runs object detection
    on each image. Detection results are saved in YOLO format, one .txt file per image, in the
    'labels' folder of the results directory.

    The model is loaded once for all the images. The result is reported in os.environ["test_complete"]:
    "True" when all the images have been processed, "Error" if the processing failed (the detail
    of the error is printed in the terminal).

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
    """
    try:
        eval_folder = Path(project_folder) / 'image_inputs' / 'eval_images'
        yolo_model = YOLO(Path(yolo_model_folder) / 'weights' / 'best.pt')
        device = which_device()

        for img_path in _eval_images(eval_folder):
            process_single_image_with_yolo(project_folder, yolo_model_folder, str(img_path),
                                           yolo_model=yolo_model, device=device)
            print(f"Processed: {img_path}")
        os.environ["test_complete"] = "True"

    except Exception as e:
        traceback.print_exc()
        print(f"Erreur pendant les prédictions : {e}")
        os.environ["test_complete"] = "Error"


def process_single_image_with_yolo(project_folder:str | Path, yolo_model_folder:str | Path, img_path:str | Path,
                                   yolo_model=None, device=None) -> None:
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

    yolo_model, device : optional
        An already loaded YOLO model and the device to use. If not given, they are loaded here
        (loading a model takes time: process_images_with_yolo loads it once for all the images).

    Returns
    -------
    None
        The function saves a .txt file in the 'labels' subfolder under the results directory, with the format:
        <class_id> <x_center> <y_center> <width> <height> <confidence>
    """
    if device is None:
        device = which_device()
    if yolo_model is None:
        yolo_model = YOLO(Path(yolo_model_folder) / 'weights' / 'best.pt')

    # Prepare output directory
    results_folder = get_results_folder(project_folder, yolo_model_folder)
    labels_folder = Path(results_folder) / 'labels'
    labels_folder.mkdir(parents=True, exist_ok=True)

    label_path = labels_folder / f"{Path(img_path).stem}.txt"

    # The path is given directly to YOLO rather than an image read with cv2.imread,
    # which cannot open a path containing accents under Windows.
    results = yolo_model.predict(source=str(img_path),
                                 device=device,
                                 agnostic_nms=True,
                                 imgsz=640,
                                 save_txt=False,
                                 save_conf=False,
                                 verbose=False
                                )

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        # Remove the predictions of a previous run on an image with the same name,
        # which would otherwise be taken for the predictions of this one
        label_path.unlink(missing_ok=True)
        print(f"No detections found in {img_path}")
        return

    with open(label_path, 'w', encoding='utf-8') as label_file:
        for box in boxes:
            xywh = " ".join([f"{value:.4f}" for value in box.xywhn.squeeze().tolist()])
            class_id = int(box.cls.item())
            confidence = box.conf.item()
            label_file.write(f"{class_id} {xywh} {confidence:.4f}\n")

    print(f"✅ Saved predictions for {Path(img_path).stem} to {label_path}")


def get_image_data(project_folder:str | Path) -> None:
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
        The function creates a CSV file named '<project_name>.csv' in the 'eval_images' folder
        with metadata for each image.
    """

    eval_folder = Path(project_folder) /'image_inputs' / 'eval_images'
    # Nom du projet (et non du dossier eval_images) : add_csv_data cherche ce fichier sous ce nom
    project_name = Path(project_folder).name

    data = []
    for file in _eval_images(eval_folder):
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

def yolo_to_csv(project_folder:str | Path, yolo_model_folder:str | Path) -> None:
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
    labels = get_labels(str(labels_file))

    annotation_files = [file for file in labels_folder.iterdir() if file.suffix.lower() == '.txt']

    all_rows = []

    for img_path in _eval_images(eval_folder):
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
            with open(matching_annotation, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    if not line.strip():
                        continue
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
                        'Class_Name': get_class_name(int(class_id), labels),
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

def convert_yolo_annotations_to_label_studio_format(yolo_annotations:list, img_path:str | Path, yolo_model_folder:str | Path) -> list:
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
    labels = get_labels(str(Path(yolo_model_folder) / 'labels.txt'))

    results = []

    # Get the image dimensions
    with Image.open(img_path) as img:
        image_width, image_height = img.size
        # print(f'Largeur: {image_width}, Hauteur: {image_height}')
    
    # Get the bounding_boxes coordinates
    for line in yolo_annotations:
        if not line.strip():
            continue
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
                    "rectanglelabels": [get_class_name(int(class_id), labels)]
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

def convert_unannotated_to_label_studio_format(img_path: str | Path, yolo_model_folder: str | Path) -> list:
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

def get_ls_for_local_files(project_folder: str | Path, yolo_model_folder: str | Path) -> Path:
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

    all_ls = []

    for img_path in _eval_images(eval_folder):
        img_name = img_path.stem
        label_txt = labels_folder / f"{img_name}.txt"

        if label_txt.exists():
            # 1) Annotated image → Label Studio conversion
            with open(label_txt, 'r', encoding='utf-8') as f:
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
    new_prefix = '/data/local-files/?d=projects/' + (eval_folder.relative_to(Path(project_folder).parent)).as_posix()

    for ann in all_ls:        
        ann['data']['image'] = ann['data']['image'].replace(str(eval_folder), new_prefix).replace("\\",'/')

    # Writing the JSON file
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_ls, f, indent=2)
    print(f"Label Studio annotations written to {json_file}")
    
    return json_file