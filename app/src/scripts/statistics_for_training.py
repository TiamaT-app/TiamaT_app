import codecs
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



from ..modules.folders_path import get_data_folder
from ..modules.generate_labels import get_labels
from ..modules.manipulate_files import find_image_path

def create_stats_folder(project_folder:str | Path) -> None:
    """
    Creates a 'dataset_statistics' subfolder in the training directory if it doesn't exist.

    Parameters
    ----------
    project_folder : str | Path
        Absolute path to the project folder.
    """
    stats_folder = Path(get_data_folder(project_folder)) / 'dataset_statistics'
    stats_folder.mkdir(parents=True, exist_ok=True)

def clean_LS(project_folder:str | Path, annotated_with_LS:bool)-> None:
    """
    Renames files in the 'images' and 'labels' subdirectories of a dataset folder
    by removing the prefix added by Label Studio (typically an 8-character alphanumeric
    string followed by a dash, e.g., 'abcd1234-').

    This function is intended to clean up file names after annotation with Label Studio,
    ensuring compatibility with downstream processing pipelines that expect original file names.

    Parameters:
    ----------
    project_folder : str | Path
        The absolute path to the root project folder.

    annotated_with_LS : bool
        Indicates whether the files were annotated using Label Studio.
        If True, the function proceeds with renaming.

    Returns:
    -------
    None
        This function modifies filenames in place and does not return a value.
    """

    
    if not annotated_with_LS:
        return
    
    else:
        data_folder = Path(get_data_folder(project_folder))
        img_folder = data_folder / 'images'
        labels_folder = data_folder / 'labels'


        # Browse the files in the 'images' directory
        for img_file in img_folder.iterdir():
            if img_file.is_file() and len(img_file.name) > 9:
                new_img_filename = img_file.name[9:]
                new_img_filepath = img_folder / new_img_filename
                img_file.rename(new_img_filepath)
            print(f"Renamed image file : {img_file} -> {new_img_filename}")

        # Browse the files in the 'labels' directory
        for label_file in labels_folder.iterdir():
            if label_file.is_file() and len(label_file.name) > 9:
                new_label_filename = label_file.name[9:]
                new_label_filepath = labels_folder / new_img_filename
                label_file.rename(new_label_filepath)
            print(f"Renamed label file: {label_file} -> {new_label_filename}")
        

def get_annotation_files(img_folder:Path, labels_folder:Path) -> list:
    """
    Retrieves the list of '.txt' annotation files that correspond to image files
    in a given image folder. Only annotation files with a matching image file
    (based on filename without extension) are included.

    Parameters:
    ----------
    img_folder : Path
        Path to the folder containing image files (e.g., .jpg, .png, .jpeg, .tiff).

    labels_folder : Path
        Path to the folder containing annotation files (one .txt file per image).

    Returns:
    -------
    list of Path
        A list of Path objects representing .txt annotation files that have a
        corresponding image in the image folder.

    Notes:
    ------
    - This ensures consistency between images and annotations, which is critical
      for tasks like object detection or image classification training.
    """

    img_exts = {".jpg", ".jpeg", ".png", ".tiff"}
    image_files = [f for f in img_folder.iterdir() if f.suffix.lower() in img_exts]

    annotation_files = []
    
    for image_file in image_files:
        image_name = image_file.stem
        annotation_file = labels_folder / f'{image_name}.txt'
        
        if annotation_file.exists() and annotation_file.is_file():
            annotation_files.append(str(annotation_file))
            
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
    data_folder = Path(get_data_folder(project_folder))
    img_folder = data_folder / 'images'
    labels_folder = data_folder / 'labels'
    
    annotation_files = get_annotation_files(img_folder, labels_folder)

    for file_path in annotation_files:
        with open(file_path, 'rb') as f:
            rawdata = f.read()
        try:
            result = codecs.decode(rawdata, 'utf-8')
        except UnicodeDecodeError:
            try:
                result = codecs.decode(rawdata, 'iso-8859-1')
                print(f"{file_path.name} is encoded in ISO-8859-1")
            except UnicodeDecodeError:
                print(f"{file_path.name} encoding not recognized")


def img_without_annotations(img_folder:Path, labels_folder:Path) -> int:
    """
    This function identifies images in the specified image folder that do not have corresponding annotation files 
    or have empty annotation files. It helps detect unannotated images, which may cause issues during model training.
    This function helps ensure that the dataset is clean and consistent before starting a training session, 
    preventing potential errors or suboptimal model performance caused by unannotated or empty images.

    :param img_folder: 
        - Type: Path
        - Description: Path to the folder containing image files (.jpg, .jpeg, .png, .tiff).

    :param labels_folder: 
        - Type: Path
        - Description: Path to the folder containing .txt annotation files.

    :return: 
        - Type: int
        - Description: The number of unannotated images found, including those without annotation files 
                       and those with empty annotation files.
    """

    annotation_files = get_annotation_files(img_folder, labels_folder)
    
    img_exts = {".jpg", ".jpeg", ".png", ".tiff"}
    image_files = [f for f in img_folder.iterdir() if f.suffix.lower() in img_exts]
    
    count = 0
    unannotated_image = []
    
    # Images without corresponding annotation files or with an empty annotation file
    for image_file in image_files:
        annotation_path = labels_folder / f"{image_file.stem}.txt"
        
        if not annotation_path.exists():
            count += 1
            unannotated_image.append(image_file)
            print(f"Image {image_file} has no annotation file")
        
        elif annotation_path.stat().st_size == 0:
            count += 1
            unannotated_image.append(image_file)
            print(f"Image {image_file} has an empty annotation file")

    
    # Annotation files that are empty
    # if unannotated_image:
    #     delete = input(f'You have {len(unannotated_image)} unannotated images in your dataset. Do you want to delete them? (yes/no) : ').strip().lower()
    #     if delete == 'yes':
    #         for image in unannotated_image:
    #             image.unlink()
    #             print(f"Deleted image: {image.name}")
    #     else:
    #         print('Warning! You will start a training session with unannotated images')
    
    return count

def annotations_per_img(project_folder:str | Path) -> None:
    """
    Counts annotations per image and saves results to 'annotations_per_img.csv'.

    Parameters
    ----------
    project_folder : str | Path
        Path to the folder containing 'images', 'labels', and 'dataset_statistics'.

    Returns
    -------
    None
    """

    # Retrieve Annotation Files
    data_folder = Path(get_data_folder(project_folder))
    data_stat_folder = data_folder / 'dataset_statistics'
    data_stat_folder.mkdir(parents=True, exist_ok=True)

    img_folder = data_folder / 'images'
    labels_folder = data_folder / 'labels'
    
    annotation_files = get_annotation_files(img_folder, labels_folder)

    # Count Annotations per Image
    lines_per_file = {}
   
    for annotation_file in annotation_files:
        with open(annotation_file, 'r') as ann_file:
                nb_lines = sum(1 for _ in ann_file)

        image_name = Path(annotation_file).stem  # Get the image name without extension
        image_path = find_image_path(img_folder, image_name)
        lines_per_file[image_path] = nb_lines
    
    sorted_counts = dict(sorted(lines_per_file.items(), key=lambda x: x[1], reverse=True))

    # Create a DataFrame from the results
    df = pd.DataFrame(sorted_counts.items(), columns=['image_name', 'annotations_nb'])

    # Write the DataFrame to a CSV file with ';' as the separator
    csv_file_path = data_stat_folder / 'annotations_per_img.csv'
    df.to_csv(csv_file_path, index=False, sep=';')

    print(f'{csv_file_path} created')    

def total_annotations(img_folder:Path, labels_folder:str | Path) -> int:
    """
    This function calculates the total number of annotations present in the specified training dataset 
    by counting the number of non-empty lines in each annotation file. Each line in a `.txt` annotation file 
    typically represents an individual bounding box or object annotation.

    :param img_folder: 
        - Type: Path
        - Description: The absolute path to the folder where the images are stored. 
                       The function will look for image files to identify corresponding annotation files.

    :param labels_folder: 
        - Type: str | Path
        - Description: The absolute path to the folder where the annotation files are stored. 
                       The function will look for `.txt` files containing annotation data.

    :return: 
        - Type: int
        - Description: The total number of annotations across all images in the dataset. This count includes 
                       all valid lines from the `.txt` annotation files, excluding empty lines.

    This function helps provide an overview of the dataset's annotation density,
    which can be useful for dataset analysis and model training considerations.
    """

    # Retrieve Annotation Files
    annotation_files = get_annotation_files(img_folder, labels_folder)

    # Count Annotations
    total_lines = 0

    for annotation_file in annotation_files:
        # with open(labels_folder / annotation_file, 'r') as f:
        # labels_folder est deja dans annotation file
        with open(annotation_file, 'r') as f:
            nb_lines = 0
            for line in f:
                if line.strip():  # ignore empty lignes
                    nb_lines += 1
            total_lines += nb_lines
    print(f"The total number of annotations is {total_lines}.")
    return total_lines

def classes_distribution(project_folder:str | Path)-> None:
    """
    Computes class distribution from label files.
    Saves results as a CSV and a horizontal bar chart.

    Parameters
    ----------
    project_folder : str | Path
        Absolute path to the project folder containing 'labels' and 'dataset_statistics'.

    Returns
    -------
    None
        Outputs 'class_distribution.csv' and 'class_distribution.png' to 'dataset_statistics'.
    """

    data_folder = Path(get_data_folder(project_folder))
    img_folder = data_folder / 'images'
    labels_folder = data_folder / 'labels'
    labels_file = data_folder / 'labels.txt'
    data_stat_folder = data_folder / 'dataset_statistics'
    data_stat_folder.mkdir(parents=True, exist_ok=True)

    # Get the labels from the labels.txt file
    annotation_classes = get_labels(labels_file)
    annotation_files = get_annotation_files(img_folder, labels_folder)

    # Count Annotations per Class
    occurrences = {}
    for annotation_file in annotation_files:
        with open(annotation_file, 'r', encoding='ascii') as f:
            for line in f:
                annotation_code = line.split()[0]
                if annotation_code not in occurrences:
                    occurrences[annotation_code] = 1
                else:
                    occurrences[annotation_code] += 1

    # Map annotation codes to class names
    class_names = [annotation_classes[code].strip() for code in occurrences.keys()]
    
    # Create a DataFrame from the results
    df = pd.DataFrame({'class_name': class_names, 'nb_occurrences': occurrences.values()})

    # Write the DataFrame to a CSV file with ';' as the separator
    csv_file_path = data_stat_folder / 'class_distribution.csv'
    df.to_csv(csv_file_path, index=False, sep=';')

    print(f'{csv_file_path} created')
    
    # Creating a stacked bar chart
    plt.barh(class_names, occurrences.values())

    # Setting axis and title labels
    plt.xlabel('Nombre d\'occurrences')
    plt.ylabel('Classes')
    plt.title('Distribution des classes')
    
    figure_path = data_stat_folder / 'class_distribution.png'
    # Display and save the graph
    plt.savefig(figure_path, bbox_inches='tight')
    plt.clf()

def get_global_results(project_folder:str | Path) -> None:
    """
    Generates a summary of key dataset metrics and saves it in 'global_data.csv'.

    Metrics:
    - Number of images without annotations
    - Total number of annotations

    Parameters
    ----------
    project_folder : str | Path
        Absolute path to the project folder containing 'images', 'labels', and 'dataset_statistics' subfolders.

    Returns
    -------
    None
        Creates 'global_data.csv' in the 'dataset_statistics' folder.
    """
    data_folder = Path(get_data_folder(project_folder))
    img_folder = data_folder / 'images'
    labels_folder = data_folder / 'labels'
    data_stat_folder = data_folder / 'dataset_statistics'
    data_stat_folder.mkdir(parents=True, exist_ok=True)

    # Calculate the metrics
    metrics = {
        'Number of files without annotations': img_without_annotations(img_folder, labels_folder),
        'Total number of annotations': total_annotations(img_folder, labels_folder)
    }

    # Create a DataFrame from the results
    df = pd.DataFrame(metrics.items(), columns=['metric', 'value'])

    # Write the DataFrame to a CSV file with ';' as the separator

    csv_file_path = data_stat_folder / 'global_data.csv'
    df.to_csv(csv_file_path, index=False, sep=';')

    print(f'{csv_file_path} created')