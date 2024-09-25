# Requirements

- Python 3.9
- Poetry
- Git

# Configuration
1. **Clone the repository**

    Use the following command to clone the repository:
    ```bash
    git clone <repository-url>
    ```

2. **Install poetry**

    Poetry is a tool for dependency management in Python. Install it with the following command:

    ```bash
    curl -sSL https://install.python-poetry.org | python -
    ```

3. **Install dependencies**

    Use Poetry to install the project dependencies:

    ```bash
    poetry install
    ```

4. **Configure pre-commit**

    Pre-commit is a tool that runs checks on your code before you commit it. It is configured in the `.pre-commit-config.yaml` file. To install it, run the following command:

    ```bash
    poetry run pre-commit install
    ```

5. **Set-up environment variables**

    Create a `.env` file at the root of the project and add the following environment variables
    - OPENAI_API_KEY
	- GOOGLE_API_KEY

    And activate the variables with:

    ```bash
    source .env
    ```

# Data structure
Most scripts in this repository are written to perform extraction and subsequent transforms on a batch of article pdfs.
As a suggestion, prior to running the extraction pipeline, pdf files for scientific articles could be put as a batch in a new subfolder of the ```data/raw_pdfs``` folder, e.g. ```data/raw_pdfs/batch_01/```. The ```data``` folder should also contain an extraction subfolder, ```data/extraction``` where the output of extraction scripts will be stored as a batch subfolder, e.g. ```data/extraction/tables/```.
The last suggestion is to also include annotations in a subfolder such as ```data/annotations/batch_01/```.


# Extraction pipeline

## Table Extraction

   Use the following command to run the `main_table.py` script:

   ```bash
   python src/extraction/get_tables/main_table.py --pdf_path <path_to_your_pdf> --output_folder <output_folder_path>
   ```

   Replace <path_to_your_pdf> with the path to your input PDF file.
   Replace <output_folder_path> with the desired output folder where images and CSV files will be saved.

    Example Command:

    ```bash
    python src/extraction/get_tables/main_table.py --pdf_path data/pdf/A1.pdf --output_folder data/TableExtraction
    ```

    Script Workflow
    1. The script will process the specified PDF file and extract Tables.
    2. Extracted data will be saved as images (metadata) and in a .pkl files in the specified output folder.
    3. Ensure the output folder exists or will be created automatically in the script.


## Extraction of interventions and outcomes from tables

This script aims to get information about interventions and outcomes (their names and descriptions) from extracted tables.

To run the parsing with Google Gemini, run:

```bash
python src/extraction/get_io/get_io_tables.py --tables_folder <path_to_tables> --out_folder <path_to_extraction> --batch <name_of_batch>
```

With:
- `tables_folder`: the path to the input tables for the systematic reviews considered.
- `out_folder`: the path to output folder where extractions will be saved as a csv file.
- `batch`: the batch of pdfs being processed.

For example:

```bash
python src/extraction/get_io/extract_from_tables.py --tables_folder data/extraction/tables/batch_01/ --out_folder data/extraction/io_tables/ --batch batch_01
```

The output folder will contain csv files corresponding to input pdf files, each containing the interventions, outcomes and descriptions for a given systematic review.


# Evaluation

These scripts are aimed at evaluating files previously extracted from source pdfs against annotation files.

To run the evaluation, you should first run the merging described in the section above.


## Intervention-outcome evaluation

This script evaluates the extracted interventions and outcomes in both mentions from the text and tables prior to merging. It should be ran as follows :

```bash
python src/evaluation/evaluate_io.py --batch first_batch --separator \t --eval_type sim
```

The output folder will by default be `data/evaluation/scores/` and will contain csv files containing the evaluation scores, as well as pairwise similarities if the `save_similarities` argument was set to `True`.
