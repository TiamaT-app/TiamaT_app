"""
Label generation functions and associated configurations for TiamaT projects.

This module centralises all aspects of creating and managing labels and classes 
within a project, whether this involves extracting them from annotation files, 
generating reference files (labels.txt) or building interface configurations for 
Label Studio.

Contents:

- get_labels()  : reads a labels.txt file and returns a dict {index: class_name}.
- create_labels_file()  : extracts unique classes from the JSON annotations
                        and generates the reference labels.txt file.
- add_new_labels()  : detects new classes that have appeared in the manual 
                    annotation files and updates labels.txt by assigning them new IDs
- generate_random_colours() : generates a random hexadecimal colour,used to visually 
                            distinguish between classes in the Label Studio interface.
- build_ls_label_config()   : constructs the Label Studio configuration XML from a 
                            list of classes (pure function, without reading or writing files).
- get_labeling_code()   : coordinates the complete generation of the Label Studio
                        configuration file (.txt) from a trained YOLO model
"""
from pathlib import Path
import shutil
import random

from modules.folders_path import *
from modules.manipulate_files import open_json_file


def get_labels(labels_file):
    """
    This functions checks if the file 'labels.txt' exists. 
    If not, it generated a .txt file with the generic names for each existing class "class1" to "classN". 
    The users can then change the names later.
    
    **Beware: if defined classes have not been used in the training dataset, they will not appear in this labels.txt file.**

    :param labels_file: 
        - Type: str
        - Description: The path to the 'labels.txt' file which contains the class IDs and corresponding class names.
    
    :return: 
        - Type: dict
        - Description: A dictionary where keys are class IDs (as strings) and values are class names.    
    """
    labels_dict = {}
    with open(labels_file, 'r') as labels:
        for line in labels:
            key, value = line.strip().split(': ')
            key = key.strip("'")
            value = value.strip("'\n")
            labels_dict[key] = value
    
    return labels_dict

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

def add_new_labels(project_folder:str | Path, yolo_model_folder:str | Path) -> None:
    """
    Updates the YOLO labels file with new classes found in manually corrected prediction files.

    If new classes are detected in the correction JSON files that are not already listed in
    the model's labels.txt, they are added with new IDs. The updated labels file is saved
    to the results folder. If no new classes are found, the original file is simply copied.

    Parameters
    ----------
    project_folder : str | Path
        Path to the main project directory.

    yolo_model_folder :str | Path
        Path to the folder containing the trained YOLO model and its labels.txt file.

    Returns
    -------
    None
        A new labels.txt file is saved in the results folder.
    """

    # Load existing labels (may be a dict {"0":"label"} or a list ["label"])
    labels_file = Path(yolo_model_folder) / 'labels.txt'
    labels = get_labels(str(labels_file))
    
    # Get results folder (destination for corrected labels) and ensure it exists
    results_folder = Path(get_results_folder(project_folder, yolo_model_folder))
    results_folder.mkdir(parents=True, exist_ok=True)
    label_dict_file = results_folder / 'labels.txt'

    # Normalize labels to a dictionary {id: name}
    if isinstance(labels, dict):
        train_labels = dict(labels)
    else:
        train_labels = {str(i): name for i, name in enumerate(labels)}
    existing_values = set(train_labels.values())

    # Folder containing manual correction files
    corrections_folder = Path(get_corrections_folder_inference(project_folder)) 
    correction_files = [f for f in corrections_folder.iterdir() if not f.name.startswith('.')]
    
    unique_classes = set()
    
    # Extract all unique corrected classes from correction file
    for correction_file in correction_files:
        corrections = open_json_file(str(correction_file))
        
        for i, result in enumerate(corrections['result']):
            value = result.get('value', {})
            labels_ls = value.get('rectanglelabels', [])
            if labels_ls:
                unique_classes.add(labels_ls[0])

    corrected_classes = list(unique_classes)
    
    # Identify labels that are not already in the training set
    new_labels = [c for c in corrected_classes if c not in existing_values]
    
    if new_labels:
        print(f"{len(new_labels)} new label(s) found in the correction files: {new_labels}")
        
        # Assign new incremental IDs starting after the last current ID
        max_id = max(map(int, train_labels.keys())) if train_labels else -1
        for i, cls, in enumerate(new_labels, start=max_id+1):
            train_labels[str(i)] = cls
        
        # Write the updated labels file
        with open(label_dict_file, "w", encoding='utf-8') as f:
            for k, v in train_labels.items():
                f.write(f"'{k}': '{v}'\n")
        print(f"Labels file written in {label_dict_file} ")
    
    else:
        # No new labels → copy the existing file
        shutil.copy2 (labels_file, label_dict_file)
        print(f"No new class found. Labels file copied to {label_dict_file}")

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

def build_ls_label_config(labels: list | dict) -> str:
    """
    Construit le XML de configuration Label Studio pour une liste de classes.
    Fonction pure : ne lit ni n'écrit aucun fichier.
    """

    if isinstance(labels, dict):
        labels_name = labels.values()
    else:
        labels_name = labels

    labeling_template = """<View>
    <View style="display:flex;align-items:start;gap:8px;flex-direction:row">
        <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
        <RectangleLabels name="label" toName="image" showInline="false">        
    {label_backgrounds}    </RectangleLabels>
    </View>
    </View>
    """
    label_backgrounds = ""
    for label in labels_name:
        random_colour = generate_random_colours()
        label_backgrounds += f'        <Label value="{label}" background="{random_colour}"/>\n'

    return labeling_template.format(label_backgrounds=label_backgrounds)

def get_labeling_code(project_folder:str | Path, yolo_model_folder:str | Path) -> str:
    """
    Generates a Label Studio XML configuration template using class labels from a YOLO model.
    Each label is assigned a random background color for display in Label Studio.

    Note:
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
    labeling_template :str
        The Label Studio XML configuration template.
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
    
    # Add the generated colour to your model for each label usiung the Label Studio template for bounding boxes
    labeling_template = build_ls_label_config(labels)
    
    with open(labeling_file, 'w') as file:
        file.write(labeling_template)
    
    print(f"The labeling template is saved in {labeling_file}")
    return labeling_template
