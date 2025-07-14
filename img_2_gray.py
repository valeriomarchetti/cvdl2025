import argparse
import os
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# Function to check if a file is an image based on its extension
def is_image_file(filename):
    """Check if the file is an image based on its extension."""
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
    return filename.lower().endswith(tuple(img_extensions))

# Function to convert an image to grayscale and copy it to the destination
def convert_and_copy(src_file, dst_file):
    """Convert an image to grayscale and copy it to the destination."""
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    if is_image_file(src_file.name):
        # Converte direttamente in scala di grigi senza controllare la modalità per ottimizzare
        with Image.open(src_file) as img:
            img = img.convert('L')
            img.save(dst_file)
    else:
        # Copio normalmente
        shutil.copy2(src_file, dst_file)

# Main function to process the dataset
def main():
    """Main function to convert images to grayscale and reconstruct the dataset."""
    parser = argparse.ArgumentParser(description="Script per convertire immagini in scala di grigi e ricostruire il dataset.")
    parser.add_argument('--path', type=str, required=True, help='Percorso della cartella contenente il dataset di input')
    args = parser.parse_args()

    input_dir = Path(args.path)
    if not input_dir.is_dir():
        tqdm.write(f"Il percorso {input_dir} non esiste o non è una directory.")
        sys.exit(1)

    # Crete a new directory for the grayscale images
    output_dir = input_dir.parent / (input_dir.name + "_Grey")
    output_dir.mkdir(exist_ok=True)

    # This will include all files in the directory and its subdirectories
    all_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            src_file = Path(root) / f
            rel_path = src_file.relative_to(input_dir)
            dst_file = output_dir / rel_path
            all_files.append((src_file, dst_file))

    # Using tqdm to show progress
    # Using ThreadPoolExecutor to process files in parallel for multiple cores
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(convert_and_copy, src, dst): (src, dst) for src, dst in all_files}

        for f in tqdm(as_completed(futures), desc="Elaborazione file", total=len(all_files), unit="file"):
            pass  

    # Final check: ensure all images have corresponding labels
    # Take all images in the output directory
    images_list = []
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if is_image_file(f):
                images_list.append(Path(root) / f)

    missing_labels = []
    # Check for corresponding labels with a progress bar
    # Assuming labels name is the same as image name but with .txt extension
    for img_path in tqdm(images_list, desc="Verifica labels", unit="file"):
        label_path = img_path.with_suffix('.txt')
        if not label_path.exists():
            missing_labels.append(img_path.relative_to(output_dir))

    if missing_labels:
        tqdm.write("Attenzione! Le seguenti immagini non hanno una label corrispondente:")
        for img in missing_labels:
            tqdm.write(str(img))
    else:
        tqdm.write("Tutte le immagini convertite hanno la loro label corrispondente.")

    tqdm.write("Operazione completata.")

if __name__ == "__main__":
    main()
