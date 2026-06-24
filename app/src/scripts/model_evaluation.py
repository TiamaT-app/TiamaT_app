import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt




from ..modules.folders_path import *
from ..modules.transform_coordinates_functions import from_ls_to_yolo
from ..modules.class_names_functions import get_labels, get_class_name, get_class_code
from ..modules.manipulate_files import open_json_file, save_json_file, get_files, exclude_training_images, load_data_from_files

def add_new_labels(project_folder:str, yolo_model_folder:str) -> None:
    """
    Updates the YOLO labels file with new classes found in manually corrected prediction files.

    If new classes are detected in the correction JSON files that are not already listed in
    the model's labels.txt, they are added with new IDs. The updated labels file is saved
    to the results folder. If no new classes are found, the original file is simply copied.

    Parameters
    ----------
    project_folder : str
        Path to the main project directory.

    yolo_model_folder : str
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


def get_img_from_training(project_folder:str | Path, yolo_model_folder:str | Path) -> list:
    """
    Returns a list of images from the dataset folder that were used during the YOLO model training.

    Parameters
    ----------
    project_folder : str

    yolo_model_folder : str
        Path to the YOLO model folder (should contain 'dataset_statistics/training_dataset.txt').

    Returns
    -------
    list
        A list of image filenames that were used for training.
    """

    yolo_model_folder = Path(yolo_model_folder)
    
    training_dataset = yolo_model_folder / 'dataset_statistics'/ 'training_dataset.txt'
    training_folder = Path(get_img_folder_training(project_folder))
    
    if not training_dataset.exists():
        raise FileNotFoundError(f"Training dataset file not found: {training_dataset}")
    
    with open(training_dataset, 'r') as train_data_file:
            train_image_names = [Path(line.strip()).name for line in train_data_file if line.strip()]
    
    img_exts = {'.jpg', '.jpeg', '.png', '.tiff'}  
    image_files = [file.name for file in training_folder.iterdir() if file.suffix.lower() in img_exts]
    
    matching_images = [image_name for image_name in train_image_names if Path(image_name).name in image_files]
   
    if matching_images:
        print("✅ The following images were used to train the model:")
        for img in matching_images:
            print(f" - {img}")
        
    else:
        print(f"⚠️ No matching images found in {training_folder} for the model {yolo_model_folder.name}.")
    
    return matching_images


def calculate_iou(box1:list, box2:list) -> float:
    """
    This function calculates the Intersection over Union (IoU) between two bounding boxes. IoU is a measure 
    of the overlap between two bounding boxes and is commonly used to evaluate the accuracy of object detection models.

    The function is adapted from the 'bb_intersection_over_union' function on PyImageSearch, which uses 
    bounding box coordinates in (x_min, y_min, x_max, y_max) format. The adaptation accounts for the 
    fact that YOLOv8 provides bounding box coordinates in relative format (x_center, y_center, width, height).

    :param box1: 
        - Type: list or tuple
        - Description: The first bounding box defined as a list or tuple of values [class_id, x_center, y_center, width, height]. 
                       The coordinates are relative to the image dimensions.
    :param box2: 
        - Type: list or tuple
        - Description: The second bounding box defined as a list or tuple of values [class_id, x_center, y_center, width, height]. 
                       The coordinates are relative to the image dimensions.
    
    :return: 
        - Type: float
        - Description: The IoU value, which ranges from 0 to 1. A value of 0 indicates no overlap, 
                       while a value of 1 indicates perfect overlap between the two bounding boxes.

    This function is useful for evaluating object detection models and determining how well the predicted bounding boxes 
    match the ground truth annotations.
    """
    
    # Convert coordinates (x, y, w, h) in (x_min, y_min, x_max, y_max)
    box1_x_min = box1[1] - box1[3] / 2
    box1_y_min = box1[2] - box1[4] / 2
    box1_x_max = box1[1] + box1[3] / 2
    box1_y_max = box1[2] + box1[4] / 2
    
    box2_x_min = box2[1] - box2[3] / 2
    box2_y_min = box2[2] - box2[4] / 2
    box2_x_max = box2[1] + box2[3] / 2
    box2_y_max = box2[2] + box2[4] / 2
    
    # Calculate coordinates (x,y) of the overlap
    x_min = max(box1_x_min, box2_x_min)
    y_min = max(box1_y_min, box2_y_min)
    x_max = min(box1_x_max, box2_x_max)
    y_max = min(box1_y_max, box2_y_max)
    #DRACONES sur le +1 
    # Calculate the area of the overlap
    intersection_area = max(0, x_max - x_min+1) * max(0, y_max - y_min+1)

    # Calculer the area of the two bounding boxes
    box1_area = (box1_x_max - box1_x_min+1) * (box1_y_max - box1_y_min+1)
    box2_area = (box2_x_max - box2_x_min+1) * (box2_y_max - box2_y_min+1)
    
    # Calculate the Intersection over Union (IoU)
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    
    return iou

def get_best_iou_matches(predictions:list, corrected_predictions:list) -> list:
    """
    This function finds the best matching corrected bounding box for each predicted bounding box based on 
    the Intersection over Union (IoU) value. For each prediction, it calculates the IoU with all corrected 
    bounding boxes and selects the one with the highest IoU as the best match.
    
    :param predictions: 
        - Type: list of str
        - Description: A list of predicted bounding boxes in YOLO format (class_id, x_center, y_center, width, height). 
                       Each bounding box is represented as a string of space-separated values.
    :param corrected_predictions: 
        - Type: list of str
        - Description: A list of corrected bounding boxes in YOLO format (class_id, x_center, y_center, width, height). 
                       Each bounding box is represented as a string of space-separated values.
    
    :return: 
        - Type: list of tuples
        - Description: A list of tuples, where each tuple contains:
            - The predicted bounding box (str)
            - The best matching corrected bounding box (str) based on the highest IoU
            - The IoU value (float) for the best match
    
    This function is useful for evaluating the performance of a model by comparing its predictions with manually corrected 
    ground truth annotations, identifying the best matches based on spatial overlap.
    """

    # Create an empty list for the best matches
    best_matches = []

    for prediction in predictions:
        prediction_box = prediction.split()
        prediction_box = [float(coord) for coord in prediction_box]
        best_iou = 0
        best_correction = None

        for correction in corrected_predictions:
            correction_box = correction.split()
            correction_box = [float(coord) for coord in correction_box]

            iou = calculate_iou(prediction_box, correction_box)
            print(f"{iou} EST LA VALEUR DE L IOU ")
            print(f"calculter l iou de {correction} correction, la longueur est{len(corrected_predictions)} ")
            if iou > best_iou:
                best_iou = iou
                best_correction = correction
                print("iou validé par la rue")
        
        best_matches.append((prediction, best_correction, best_iou))
    
    return best_matches

def save_results_to_csv(rows:list, output_file:str | Path) -> None:
    """
    This function saves a list of generated and corrected annotations into a CSV file. If no annotations are provided, 
    it logs a message indicating that no corrections were made and exits the function. Otherwise, it creates a 
    DataFrame from the provided data, sorts it by the 'Filename' column, and writes it to the specified CSV file.
    
    :param rows: 
        - Type: list of dict
        - Description: A list of dictionaries containing the generated and corrected annotations. Each dictionary should 
                       represent a single annotation entry with keys as column names.
    :param output_file: 
        - Type: str
        - Description: The path where the CSV file will be created. This file will store the sorted annotations for easy review.
    
    :return: 
        - Type: None
        - Description: This function does not return a value. It either creates the CSV file or prints a message if no 
                       annotations are provided.
    
    This function is useful for storing annotation results in a structured format, facilitating further analysis or 
    review of corrected and generated annotations.
    """

    if not rows:
        print('No correction made')
        return
    df = pd.DataFrame(rows)
    df_sorted = df.sort_values('Filename')
    df_sorted.to_csv(output_file, sep=';',index=False)
    print(f"The {output_file} file has been created.")
    

def get_csv_results(project_folder:str | Path, yolo_model_folder:str | Path, all_results:bool) -> None:
    """
    Generate a CSV file summarizing the evaluation of YOLO model predictions against manually corrected annotations.

    Each prediction is evaluated as:
        - TP (True Positive): correct class and IoU ≥ 0.5
        - FP (False Positive): incorrect or unmatched prediction
        - FP_class: correct box but wrong class (IoU ≥ 0.75)
        - FN (False Negative): missing prediction for a corrected annotation

    Parameters
    ----------
    project_folder : str
        Path to the project directory.
    
    yolo_model_folder : str
        Path to the folder containing the YOLO model and its associated output (e.g. labels.txt, predictions).

    all_results : bool
        If True, evaluates all predictions.
        If False, excludes predictions from images used during training (based on training_dataset.txt).

    Returns
    -------
    None
        The evaluation is saved as a CSV file in the results folder under 'results/results_for_evaluation.csv'.

    Notes
    -----
    - Uses best IoU matching between predictions and corrected labels.
    - Assumes YOLO annotations follow standard YOLO format (class x y w h confidence).
    - Corrected labels are expected in 'correctedLabels' folder.
    """

    results_folder = Path(get_results_folder(project_folder, yolo_model_folder))
    label_dict = get_labels(str(results_folder / 'labels.txt'))

    prediction_folder = results_folder / 'labels'
    predictions_files = get_files(str(prediction_folder), 'txt')

    correction_folder = results_folder / 'correctedLabels'
    corrected_files = get_files(str(correction_folder), 'txt')

    output_file = results_folder / 'results' / 'results_for_evaluation.csv'

    if not all_results:
        img_use_for_training = get_img_from_training(project_folder, yolo_model_folder)
        predictions_files = exclude_training_images(predictions_files, img_use_for_training)
        corrected_files = exclude_training_images(corrected_files, img_use_for_training)

    rows = []

    pred_map = {Path(path).name: Path(path) for path in predictions_files}
    corr_map = {Path(path).name: Path(path) for path in corrected_files}
    
    
    # Browse through all the predictions
    for basename, pred_path in pred_map.items():
        # Retrieve the correction file if it exists
        corr_path = corr_map.get(basename)

        # HIC SUNT DRACONES
        if corr_path:
            predictions = sorted(load_data_from_files([str(pred_path)]), key=lambda x: (float(x.split()[1]), float(x.split()[2])))
            corrections = sorted(load_data_from_files([str(corr_path)]), key=lambda x: (float(x.split()[1]), float(x.split()[2])))
            best_matches = get_best_iou_matches(predictions, corrections)
            compteur=0
            for prediction, best_correction, best_iou in best_matches:
                
                pred_box = list(map(float, prediction.split()))
                cls_pred = int(pred_box[0])
                compteur+=1
                print(best_correction)
                print("VOILA LE TYPE DE BEST CORRECTION" + str(type(best_correction))+f"compteur ={compteur} ")
                cls_corr = int(best_correction.split()[0])

                if best_iou >= 0.5 and cls_pred == cls_corr:
                    tp_fp_fn = 'TP'
                # 
                elif best_iou >= 0.75 and cls_pred != cls_corr:
                    tp_fp_fn = 'FP_class'
                else:
                    tp_fp_fn = 'FP'

                if tp_fp_fn == 'FP':
                    rows.append({
                    'Filename': basename,
                    'Predicted_coordinates': ', '.join(map(str, pred_box)),
                    'Predicted_class': get_class_name(str(cls_pred), label_dict),
                    'TP/FP/FN': tp_fp_fn,
                    'Corrected_class': '',
                    'Corrected_coordinates': '',
                    'IoU': 0.0,
                    'Confidence_score': pred_box[5] if len(pred_box) > 5 else 0.0
                })
                else: 
                    rows.append({
                        'Filename': basename,
                        'Predicted_coordinates': ', '.join(map(str, pred_box)),
                        'Predicted_class': get_class_name(str(cls_pred), label_dict),
                        'TP/FP/FN': tp_fp_fn,
                        'Corrected_class': get_class_name(str(cls_corr), label_dict),
                        'Corrected_coordinates': best_correction,
                        'IoU': best_iou,
                        'Confidence_score': pred_box[5] if len(pred_box) > 5 else 0.0
                    })
    
            matched_corrs = {c for _, c, _ in best_matches}
            for corr in corrections:
                if corr not in matched_corrs:
                    box_corr = list(map(float, corr.split()))
                    cls_corr = int(box_corr[0])
                    rows.append({
                        'Filename': basename,
                        'Predicted_coordinates': '',
                        'Predicted_class': '',
                        'TP/FP/FN': 'FN',
                        'Corrected_class': get_class_name(str(cls_corr), label_dict),
                        'Corrected_coordinates': ', '.join(map(str, box_corr)),
                        'IoU': 0.0,
                        'Confidence_score': 0.0
                    })

        else:
            # No correction file at all → all predictions can be considered FP
            predictions = load_data_from_files([pred_path])
            for pred in predictions:
                box = list(map(float, pred.split()))
                cls = int(box[0])
                rows.append({
                    'Filename': basename,
                    'Predicted_coordinates': ', '.join(map(str, box)),
                    'Predicted_class': get_class_name(str(cls), label_dict),
                    'TP/FP/FN': 'FP',
                    'Corrected_class': '',
                    'Corrected_coordinates': '',
                    'IoU': 0.0,
                    'Confidence_score': box[5] if len(box) > 5 else 0.0
                })

    # HIC SUNT DRACONES

    # Process *orphan* corrections (without associated predictions)
    for basename, corr_path in corr_map.items():
        if basename not in pred_map:
            corrections = load_data_from_files([corr_path])
            for corr in corrections:
                box = list(map(float, corr.split()))
                cls = int(box[0])
                rows.append({
                    'Filename': basename,
                    'Predicted_coordinates': '',
                    'Predicted_class': '',
                    'TP/FP/FN': 'FN',
                    'Corrected_class': get_class_name(str(cls), label_dict),
                    'Corrected_coordinates': ', '.join(map(str, box)),
                    'IoU': 0.0,
                    'Confidence_score': 0.0
                })
    
    save_results_to_csv(rows, output_file)

def get_txt_results(project_folder:str, yolo_model_folder:str) -> Path:
    
    """
    Generate a text summary and visual table (PNG) of evaluation metrics from YOLO predictions.

    Metrics include TP, FP, FN, recall, precision, and F1-score, both globally and per class.
    Data is sourced from a CSV file generated by the evaluation pipeline.

    
    :param project_folder: 
        - Type: str
        - Description: Path to the project folder.
    
    :return: 
        - Type: None
        - Description: This function does not return a value. It generates a `.txt` file with a summary of evaluation 
                       results and a `.png` file with a table displaying the calculated metrics.
    """
    
    results_folder = Path(get_results_folder(project_folder, yolo_model_folder))
    csv_with_results = results_folder / 'results'/ 'results_for_evaluation.csv'
    if not csv_with_results.exists():
        raise FileNotFoundError(f"No CSV found at {csv_with_results}")

 
    df = pd.read_csv(csv_with_results, sep=';')
    output_file_txt = csv_with_results.with_suffix('.txt')
    output_png = csv_with_results.with_suffix('.png')
    
    # Collect all unique classes present in the DataFrame
    all_classes = np.unique(np.concatenate([df['Predicted_class'].dropna().unique(), df['Corrected_class'].dropna().unique()]))
    print(f'Classes : {all_classes}')
 
    table_data = []
 
    # Initialize TP, FP and FN counters for all classes
    class_TP = {classe: 0 for classe in all_classes}
    class_FP = {classe: 0 for classe in all_classes}
    class_FN = {classe: 0 for classe in all_classes}
 
    # Browse DataFrame rows
    for _, row in df.iterrows():
        pred_class = row['Predicted_class']
        corr_class = row['Corrected_class']
        # Check TP/FP/FN status for line
        if row['TP/FP/FN'] == 'TP':
            class_TP[pred_class] += 1
        elif row['TP/FP/FN'] == 'FP':
            class_FP[pred_class] += 1
        elif row['TP/FP/FN'] == 'FN':
            class_FN[corr_class] += 1
        elif row['TP/FP/FN'] == 'FP_class':
            class_FN[corr_class] += 1
            class_FP[pred_class] += 1
            
 
    # Calculate global totals
    total_TP = sum(class_TP.values())
    total_FP = sum(class_FP.values())
    total_FN = sum(class_FN.values())
    total_support = total_TP + total_FN
 
    # Recall computation
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) != 0 else 0
 
    # Precision computation
    precision = total_TP / (total_TP + total_FP) if(total_TP + total_FP) != 0 else 0
 
    # Calculation of the overall F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
 
    # Open the file in write mode
    with open(output_file_txt, 'w') as file:
        # Écrire les résultats globaux
        file.write("Overall results :\n")
        file.write("Number of TP: {}\n".format(total_TP))
        file.write("Number of FP : {}\n".format(total_FP))
        file.write("Number of FN: {}\n".format(total_FN))
        file.write("Recall (Recall) : {}\n".format(recall))
        file.write("Precision : {}\n".format(precision))
        file.write("Score F1 global : {}\n".format(f1_score))
        file.write(f"Support : {total_support}\n")
        file.write("\n")
 
        # Write results by class
        file.write("Results per class :\n")
        for classe in all_classes:
            tp = class_TP[classe]
            fp = class_FP[classe]
            fn = class_FN[classe]
            support = tp + fn
 
            recall_class = tp / (tp + fn) if (tp + fn) != 0 else 0
            precision_class = tp / (tp + fp) if (tp + fp) != 0 else 0
            f1_score_class = 2 * (precision_class * recall_class) / (precision_class + recall_class) if (precision_class + recall_class) != 0 else 0
            
            recall_formated, precision_formated, f1_score_formated = reformated_decimal(tp, fn, recall_class, precision_class, f1_score_class)
            table_data.append([classe, tp, fp, fn, precision_formated, recall_formated, f1_score_formated, support])
            
            
            file.write("Class {}\n".format(classe))
            file.write("Number of TP: {}\n".format(tp))
            file.write("Number of FP : {}\n".format(fp))
            file.write("Number of FN: {}\n".format(fn))
            file.write("Recall (Recall): {}\n".format(recall_class))
            file.write("Precision : {}\n".format(precision_class))
            file.write("Score F1 : {}\n".format(f1_score_class))
            file.write(f"Support : {support}\n")
            file.write("\n")
 
    print(f"The {output_file_txt} file has been created.")

    recall_formated, precision_formated, f1_score_formated = reformated_decimal(total_TP, total_FN, recall, precision, f1_score)
    table_data.append(['Overall', total_TP, total_FP, total_FN, precision_formated, recall_formated, f1_score_formated, total_support])
    
    # Generate a PNG file with a table
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.axis('tight')
    
    table = ax.table(cellText=table_data, colLabels=['Classes', 'Nb TP', 'Nb FP', 'Nb FN', 'Precision', 'Rappel', 'Score F1', 'Support'],
                     loc='center', cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width([0, 1, 2, 3, 4, 5, 6,7])
    
    plt.savefig(output_png, bbox_inches='tight')
    # plt.show()
    
    print(f"The {output_png} file has been created.")
    return output_png


def get_corrected_label_files(project_folder:str, yolo_model_folder:str) -> None:
    """
    Converts corrected annotation files (Label Studio format) into YOLOv8-compatible .txt files.

    This function processes all JSON files containing manually corrected annotations. It removes
    confidence scores, skips deleted annotation boxes, and writes new YOLO-format label files for
    each image. The output files are saved in the `correctedLabels` folder under the results directory.

    Parameters
    ----------
    project_folder : str
        Path to the project directory containing the corrected JSON files.

    yolo_model_folder : str
        Path to the YOLO model folder containing the 'labels.txt' file for class mapping.

    Returns
    -------
    None
        YOLO-format annotation files are saved in:
        `<runs/predict/<project>_<model>/correctedLabels>`.
    """
    
    corrections_folder = Path(get_corrections_folder_inference(project_folder))
    results_folder = Path(get_results_folder(project_folder, yolo_model_folder))
    
    label_dict_file = results_folder / 'labels.txt'
    labels = get_labels(label_dict_file)
    
    label_dict_folder = Path(get_correctedLabels_folder(project_folder, yolo_model_folder))
    label_dict_folder.mkdir(parents=True, exist_ok=True)
    

    # Retrieve corrected JSON files as a list and open them
    corrected_files = [f for f in corrections_folder.iterdir() if not f.name.startswith('.')]
    
    for corrected_file in corrected_files:
        corrections = open_json_file(corrected_file)

        for result_item in corrections['result']:
            result_item.pop('score', None)
        save_json_file(corrected_file, corrections)
        
        # Retrieve image name from corrected annotations file
        name = corrections['task']['data']['image']
        img_name = Path(name).stem
        result = corrections['result']
        

        # Create a .txt file with annotation data
        with open(label_dict_folder / f"{img_name}.txt", 'w') as yolo_correction:
            for item in result:
                if "id" not in item:
                    #Skipped (deleted box)
                    print("Prediction box erased.")
                    continue

                
                # Retrieve annotation box coordinates
                value = item['value']
                x, y, w, h = from_ls_to_yolo(value['x'], value['y'], value['width'], value['height'])

                # Retrieve the annotation label and associate it with its number in the "labels.txt" file
                class_name = value['rectanglelabels'][0]
                class_id = get_class_code(class_name, labels)
                
                yolo_correction.write(f"{class_id} {x} {y} {w} {h}\n")

    print(f"✅ All corrected annotations have been converted to YOLO format in: {label_dict_folder}")

def reformated_decimal(tp:int, fn:int, recall_class:float, precision_class:float, f1_score_class:float) -> tuple:
    """
    This function formats the values of recall, precision, and F1 score to a consistent number of decimal places 
    based on the length of the input values for True Positives (TP), False Positives (FP), and False Negatives (FN).
    
    :param tp: 
        - Type: int
        - Description: The number of True Positives (TP) for a given class.
    :param fn: 
        - Type: int
        - Description: The number of False Negatives (FN) for a given class.
    :param recall_class: 
        - Type: float
        - Description: The recall value for the class, calculated as `TP / (TP + FN)`.
    :param precision_class: 
        - Type: float
        - Description: The precision value for the class, calculated as `TP / (TP + FP)`.
    :param f1_score_class: 
        - Type: float
        - Description: The F1 score value for the class, calculated as `2 * (precision * recall) / (precision + recall)`.
    
    :return: 
        - Type: tuple of str
        - Description: Returns a tuple containing the formatted values for recall, precision, and F1 score, 
                       with a consistent number of decimal places based on the length of the input values.
    """

    # Establish the number of decimal places to use for formatting base on the support's numbers
    support = tp + fn

    if support < 10:
        max_decimal = 1
    elif support < 100:
        max_decimal = 2
    elif support < 1000:
        max_decimal = 3
    else:
        max_decimal = 4


    # Ensure that the results are displayed consistently
    recall_formated = "{:.{}f}".format(recall_class, max_decimal)
    precision_formated = "{:.{}f}".format(precision_class, max_decimal)
    f1_score_formated = "{:.{}f}".format(f1_score_class, max_decimal)
    
    return recall_formated, precision_formated, f1_score_formated



def create_confusion_matrix(project_folder:str, yolo_model_folder:str) -> Path:
    """
    Generate and save a confusion matrix from YOLO prediction results.

    The confusion matrix compares predicted classes to corrected (ground truth) classes
    and includes a 'Background' class to handle false positives and false negatives.

    Parameters
    ----------
    project_folder : str
        Path to the project folder.

    yolo_model_folder : str
        Path to the folder containing the YOLO model and its output data.

    Returns
    -------
    confusion_matrix_path : Path
        The path to the PNG file containing the confusion matrix.

    Notes
    -----
    - The CSV file must exist at 'results/results_for_evaluation.csv'.
    - NaN values in predictions or corrections are mapped to 'Background'.
    - The matrix is saved as 'confusionMatrice.png'.
    """

    results_folder = Path(get_results_folder(project_folder, yolo_model_folder))
    csv_with_results = results_folder / 'results'/ 'results_for_evaluation.csv'
    if not csv_with_results.exists():
        raise FileNotFoundError(f"No CSV found at {csv_with_results}")

    # Load labels from the labels file
    dict_labels = get_labels(str(results_folder / 'labels.txt'))
    display_labels=list(dict_labels.values())
    
    # Add the 'Background' class used for file the NaN results
    display_labels.append('Background')

    confusion_matrix_path = results_folder / 'results' / 'confusion_matrix.png'
    
    # Open the csv with results
    results = pd.read_csv(csv_with_results, sep=';')

    # Replace the NaN results with 'Background', the class will be used to show the FP and FN
    predictions = results['Predicted_class'].fillna('Background')
    corrections = results['Corrected_class'].fillna('Background')

    # Create the confusion matrix
    confusion_matix = metrics.confusion_matrix(y_pred=predictions, y_true=corrections, labels=display_labels)
    # print(confusion_matix)

    # To create a more interpretable visual display we need to convert the table into a confusion matrix display
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matix, display_labels=display_labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Costumizing and visualizing the display with rotation of x-axis labels
    cm_display.plot(ax=ax, xticks_rotation=90, cmap='Blues', values_format='d')

    plt.title('Confusion matrice')

    plt.tight_layout()
    plt.savefig(confusion_matrix_path)
    # plt.show()
    plt.clf()
    
    return confusion_matrix_path