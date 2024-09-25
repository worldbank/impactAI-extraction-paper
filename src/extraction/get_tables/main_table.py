import fitz
from utils import (
    initialize_models,
    detect_tables_in_document,
    create_output_folder,
    save_pdf_pages_as_images,
    extract_table,
)

from paddleocr import PaddleOCR
import os
import torch
import argparse
import openai
from dotenv import load_dotenv  # Import dotenv to load .env file

# Load environment variables from .env file
load_dotenv()

# Fetch the OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Ensure the API key is available
if not api_key:
    raise ValueError("OpenAI API key is missing. Please set it in the .env file.")

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="en")


# Argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process PDFs to detect and extract tables."
    )

    parser.add_argument(
        "--pdf_path", type=str, required=True, help="Path to the input PDF file."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data/TableExtraction",
        help="Path to the output folder.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=False,
        help="API key if needed for specific operations.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Device to run the models on, either "cuda" or "cpu".',
    )
    parser.add_argument(
        "--crop_padding",
        type=int,
        default=10,
        help="Padding for cropping detected tables.",
    )

    return parser.parse_args()


# Main function
def main():
    args = parse_arguments()

    # Load PDF
    pdf_name = os.path.basename(args.pdf_path)
    doc = fitz.open(args.pdf_path)

    # Initialize models
    model, structure_model = initialize_models()

    # Detect tables in the document
    pages_tables_contain = detect_tables_in_document(doc)
    intersection = pages_tables_contain

    # Create output folders
    output_folder_image_path = create_output_folder(args.output_folder)

    page_numbers_to_convert = intersection

    base_name = os.path.splitext(pdf_name)[0]

    output_folder_image_path = output_folder_image_path + "/" + base_name + "/Images"

    output_folder_image_path = create_output_folder(output_folder_image_path)

    # Save PDF pages as images
    save_pdf_pages_as_images(
        args.pdf_path, page_numbers_to_convert, output_folder_image_path
    )

    temp_output_folder = os.path.join(args.output_folder, base_name)
    temp_output_folder = temp_output_folder + "/"
    print(temp_output_folder)
    os.makedirs(temp_output_folder, exist_ok=True)

    # Set thresholds for detection classes
    detection_class_thresholds = {"table": 0.75, "table rotated": 0.5, "no object": 10}

    evaluate_dict = {}
    main_evaluate_dict = {}

    # Extract tables from detected pages
    extract_table(
        Pages_Contains_Tables=intersection,
        ocr=ocr,
        evaluate_dict=evaluate_dict,
        main_evaluate_dict=main_evaluate_dict,
        temp_output_folder=temp_output_folder,
        output_folder_image_path=output_folder_image_path,
        doc=doc,
        model=model,
        structure_model=structure_model,
        device=args.device,
        detection_class_thresholds=detection_class_thresholds,
        crop_padding=args.crop_padding,
        data_cropped_path=args.output_folder,
    )


if __name__ == "__main__":
    main()
