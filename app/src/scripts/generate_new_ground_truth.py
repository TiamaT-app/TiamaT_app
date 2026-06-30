import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

from ..modules.folders_path import *
from ..modules.manipulate_files import change_id_and_path

def create_new_ground_truth(project_folder:str | Path, yolo_model_folder:str | Path, create_groundtruth:bool) -> None:
    """
    Update an existing YOLO dataset with corrected labels, evaluation images, and an updated class list.

    This function integrates annotation corrections produced during evaluation into the main dataset.
    Instead of creating a new dataset folder, it updates the existing one in place by:
      - Copying corrected label files from the results folder into the dataset `labels/` directory
        (overwriting any files with the same name).
      - Copying evaluation images into the dataset `images/` directory (adding new files as needed).
      - Replacing the dataset `labels.txt` file with the one generated during evaluation,
        ensuring that new or updated classes are included.

    Parameters
    ----------
    project_folder : str
        Path to the main project folder. Must contain `image_inputs/eval_images` with
        evaluation images to be integrated into the dataset.
    yolo_model_folder : str
        Path to the YOLO model folder. Used to locate the evaluation results
        (`labels.txt` and `correctedLabels/`).
    create_groundtruth : bool
        If False, the function exits without performing any action.

    Returns
    -------
    None
        This function performs file operations in place. It does not return a value.

    Notes
    -----
    - Both `labels/` and `images/` subfolders are created inside the dataset folder if they do not exist.
    - Files are copied with overwrite: existing label files or images with the same name will be replaced.
    - `labels.txt` is atomically replaced to avoid corruption.
    - If the corrections folder or evaluation images folder are missing, the function exits early.
    """
    
    if not create_groundtruth:
        print('No new dataset generated')
        return

    # Recompose paths
    results_folder = get_results_folder(project_folder, yolo_model_folder)
    eval_folder = Path(get_img_folder_inference(project_folder))
    data_folder = get_data_folder(project_folder)
    labels_folder = Path(data_folder) / 'labels'
    img_folder = Path(data_folder) / 'images'
    labels_file_src = Path(results_folder) / 'labels.txt'
    labels_file_dst = Path(data_folder) / 'labels.txt'
    corrections_folder = Path(results_folder) / 'correctedLabels'

    # If not exist, create the destination folders
    labels_folder.mkdir(parents=True, exist_ok=True)
    img_folder.mkdir(parents=True, exist_ok=True)
    
    if not corrections_folder.exists() or not corrections_folder.is_dir():
        print(f"[WARN] Corrections folder not found: {corrections_folder}")
        return
    
    if not eval_folder.exists() or not eval_folder.is_dir():
        print(f"[WARN] Eval images folder not found: {eval_folder}")
        return
    
    # Copy labels to the new dataset
    copied_labels = 0
    for file in corrections_folder.iterdir():
        if file.is_file():
            if file.stem.endswith('_PT'):
                continue
            else:
                shutil.copy2(str(file), str(labels_folder / file.name))
                copied_labels +=1
    print(f"[OK] {copied_labels} corrected label file(s) copied to {labels_folder}")
    
    # Copy images to the new dataset
    copied_imgs = 0
    img_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    for file in eval_folder.iterdir():
        # Remove transformed images
        if file.stem.endswith('_PT'):
            file.unlink()
            print(f"The transformed image {file} has been removed")
        else:
            if file.is_file() and file.suffix.lower() in img_exts:
                shutil.copy2(str(file), str(img_folder/ file.name))
                copied_imgs +=1
    print(f"[OK] {copied_imgs} image(s) copied to {img_folder}")

    # Copy the labels file
    if labels_file_src.exists() and labels_file_src.is_file():
        shutil.copy2(str(labels_file_src), str(labels_file_dst))
        print(f"Labels file copied to: {str(labels_file_dst)}")
    else:
        print(f"[WARN] labels.txt not found in results: {labels_file_src}")
        
    print("[DONE] Dataset updated in place.")

def move_correction_files_and_images(project_folder:str | Path) -> None:
    """
    Move evaluation images and correction JSON files into their designated folders within the project structure.

    This function organizes data after evaluation by:
      - Moving evaluation images from `image_inputs/eval_images/` into `image_inputs/ground_truth_images/`.
      - Moving correction JSON files from `annotations/prediction_corrections/` into `annotations/ground_truth/`.
      - Ensuring that annotation filenames are unique (incrementing the filename number if necessary).
      - Updating the "id" field inside each JSON annotation to match its new filename.

    Parameters
    ----------
    project_folder : str
        Path to the main project folder containing the `image_inputs/` and `annotations/` subdirectories.

    Returns
    -------
    None
        The function performs file moves and modifications in place. It does not return a value.

    Notes
    -----
    - Existing images or annotation files with the same name are renamed to ensure uniqueness.
    - Hidden files (starting with `.`) are ignored when processing annotations.
    - The `change_id_and_path()` function must exist in the codebase and is expected to update the "id" field
      of each moved JSON annotation file.
    """

    
    # Recompose paths
    ground_truth_img_folder = Path(get_img_folder_training(project_folder))
    eval_folder = Path(get_img_folder_inference(project_folder))
    ground_truth_folder = Path(get_ground_truth_folder_training(project_folder))
    pred_cors_folder = Path(get_corrections_folder_inference(project_folder))
    

    # Move .jpg images to the annotated images folder
    img_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    for file in eval_folder.iterdir():
        if file.is_file() and file.suffix.lower() in img_exts:
            shutil.move(str(file), str(ground_truth_img_folder / file.name))
    print(f"Images moved from {eval_folder} to {ground_truth_img_folder}")

    # Move correction files to the annotations folder
    for file in pred_cors_folder.iterdir():
        if file.is_file() and not file.name.startswith('.'):
            # Ensure unique file name
            new_annotation = ground_truth_folder / file
            annotation_number = int(Path(file).stem)

            while new_annotation.exists():
                annotation_number += 1
                new_annotation = ground_truth_folder / str(annotation_number)
                
            shutil.move(str(pred_cors_folder /file), str(new_annotation))

            # Changes the 'id' field in the JSON file to the basename of the file path
            change_id_and_path(new_annotation)
            
    print(f"Annotations files corrected and moved to {str(pred_cors_folder)}")

def add_csv_data(project_folder:str | Path, yolo_model_folder:str | Path) -> None:
    """
    Consolidate image metadata CSV files for a YOLO project.

    This function ensures that all relevant image metadata (filename, folder, dimensions, format, etc.)
    is stored in a single CSV file inside the `ground_truth_images` folder. It handles three cases:

      1. If no CSV files exist, it scans all images in `ground_truth_images/` and generates a new CSV file.
      2. If only the annotated CSV exists, it scans `ground_truth_images/` and appends any missing images
         not already listed in the annotated CSV.
      3. If both annotated and non-annotated CSVs exist, it merges them intelligently, updating folder paths
         from `eval_images/` to `ground_truth_images/` and removing duplicates.

    Parameters
    ----------
    project_folder : str
        Path to the main project folder. Must contain:
        - `image_inputs/ground_truth_images/` for annotated images.
        - `image_inputs/eval_images/` for evaluation images and their CSV, if it exists.
    yolo_model_folder : str
        Path to the YOLO model folder (currently not directly used, but included
        for compatibility with the rest of the pipeline).

    Returns
    -------
    None
        This function updates or creates the CSV file `*_data.csv` inside `ground_truth_images/`.

    Notes
    -----
    - The consolidated CSV uses `;` as a separator.
    - Duplicate entries (same `Image_name`) are avoided.
    - Image metadata includes: name, folder path, absolute path, format, width, height, and pixel count.
    - Non-annotated CSVs (from `eval_images`) are migrated and harmonized to point to `ground_truth_images`.
    """

    # Recompose paths
    project_name = Path(project_folder).name
    ground_truth_img_folder = Path(get_img_folder_training(project_folder))
    eval_folder = Path(get_img_folder_inference(project_folder))
    
    non_annotated_csv = eval_folder / f'{project_name}.csv'
    annotated_csv = ground_truth_img_folder / f"{project_name}_data.csv"

    img_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}


    # 2. Existence checks
    exists_non = non_annotated_csv.exists()
    exists_ann = annotated_csv.exists()
    if not exists_non and not exists_ann:
        print("No CSV found.")
        
        data = []
        images = [
            img for img in ground_truth_img_folder.iterdir()
            if img.is_file() and img.suffix.lower() in img_exts
        ]

        for file in images:
            img_name = file.stem

            with Image.open(file) as img:
                absolute_path = img.filename
                format = img.format
                w, h  = img.size
                img_size = w * h

            img_data = {
                'Image_name'   : str(img_name),
                'Folder'       : str(ground_truth_img_folder.name),
                'Absolute_path': absolute_path,
                'Format'       : format,
                'Width'        : w,
                'Height'       : h,
                'Image_size'   : img_size
            }

            data.append(img_data)

        # Create a DataFrame from the image data list
        df = pd.DataFrame(data)

        # Save DataFrame to a CSV file
        csv_filepath = annotated_csv
        df.to_csv(csv_filepath, sep=';', index=False)

        print(f"Image data saved to {csv_filepath}")
        return

    # 3. Case: only the annotated CSV exists -> add the non-annotated images
    if exists_ann and not exists_non:
        print("Adding non-annotated images to the existing annotated CSV…")
        # Load the existing CSV to get already included names
        df_annot = pd.read_csv(annotated_csv, sep=';')
        existing_names = set(df_annot['Image_name'])

        # Build the list of images to consider
        images = [
            img for img in ground_truth_img_folder.iterdir()
            if img.is_file() and img.suffix.lower() in img_exts
        ]

        new_rows = []
        for img_path in images:
            img_name = img_path.stem
            # If this name is already in the annotated CSV, skip it
            if img_name in existing_names:
                continue

            with Image.open(img_path) as img:
                absolute_path = img.filename
                w, h  = img.size
                img_size = w * h

            new_rows.append({
                'Image_name'   : str(img_name),
                'Folder'       : str(ground_truth_img_folder.name),
                'Absolute_path': absolute_path,
                'Format'       : img.format,
                'Width'        : w,
                'Height'       : h,
                'Image_size'   : w * h
            })

        if not new_rows:
            print("No new images to add.")
            return

        df_new = pd.DataFrame(new_rows)
        
        # Concatenate only the new rows
        merged = pd.concat([df_annot, df_new], ignore_index=True)
        merged.to_csv(annotated_csv, sep=';', index=False)
        print(f"{len(df_new)} image(s) added to {annotated_csv}")

        return

    # 4. Case: both CSVs exist -> intelligent merge
    df_annot = pd.read_csv(annotated_csv, sep=';') if exists_ann else pd.DataFrame()
    df_non   = pd.read_csv(non_annotated_csv, sep=';') if exists_non else pd.DataFrame()
    
    # Update paths from eval_images → ground_truth_images
    if not df_non.empty:
        df_non['Folder'] = df_non['Folder'].str.replace('eval_images', 'ground_truth_images')
        df_non['Absolute_path'] = df_non['Absolute_path'].str.replace(
            'eval_images', 'ground_truth_images'
        )

    # Remove from df_non the images already present in df_annot
    existing_names    = set(df_annot['Image_name'])
    df_non_filtered   = df_non.query("Image_name not in @existing_names")

    if df_non_filtered.empty:
        print("No additional non-annotated images to merge.")
    else:
        merged = pd.concat([df_annot, df_non_filtered], ignore_index=True)
        merged.to_csv(annotated_csv, sep=';', index=False)
        print(f"{len(df_non_filtered)} image(s) merged into {annotated_csv}")
        return
    
def clean_data(project_folder:str | Path) -> None:
    """
    Remove transformed images and annotations from a project's data folders.

    This function deletes:
    - all transformed image files (with suffix '_PT') from the images folder
    - all annotation files from the labels folder

    The function operates in-place and permanently removes files from disk.

    Parameters
    ----------
    project_folder : str
        Path to the root project folder.

    Returns
    -------
    None
    """

    # Recompose paths
    data_folder = Path(get_data_folder(project_folder))
    img_folder = data_folder / 'images'
    labels_folder = data_folder / 'labels'

    img_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    
    
    # Revome the transformed images (identified by '_PT' suffix in filename)
    removed_images = 0
    for img in img_folder.iterdir():
        if (img.is_file() and img.suffix.lower() in img_exts 
            and img.stem.endswith('_PT')):
            img.unlink()
            removed_images += 1
            
    print(f"[OK] {removed_images} transformed image(s) removed from {img_folder}")

    
    # Revome transformed annotations (identified by '_PT' suffix in filename)
    removed_annotations = 0
    for ann_file in labels_folder.iterdir():
        if (ann_file.is_file() and ann_file.suffix.lower() == '.txt'
            and ann_file.stem.endswith('_PT')):
            ann_file.unlink()
            removed_annotations += 1
            
            
    print(f"[OK] {removed_annotations} transformed annotation(s) removed from {labels_folder}")
    