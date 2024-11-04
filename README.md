# 📚 Table of Contents
- [🛠️ Requirements](#-requirements)
- [⚙️ Configuration](#-configuration)
- [📁 Data Structure](#-data-structure)
- [🔄 Extraction Pipeline](#extraction-pipeline)
  - [📊 Table Extraction](#table-extraction)
  - [🎯 Extraction of Interventions and Outcomes](#extraction-of-interventions-and-outcomes-from-tables)
  - [📝 Metadata Extraction](#metadata-extraction)
- [📊 Evaluation](#evaluation)

# 🛠️ Requirements

- Python 3.11
- Poetry
- Git

# ⚙️ Configuration
1. **Clone the repository**

    Use the following command to clone the repository:
    ```bash
    git clone git@github.com:worldbank/impactAI-extraction-paper.git
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

# 📁 Data Structure
Most scripts in this repository are written to perform extraction and subsequent transforms on a batch of article pdfs.
As a suggestion, prior to running the extraction pipeline, pdf files for scientific articles could be put as a batch in a new subfolder of the ```data/raw_pdfs``` folder, e.g. ```data/raw_pdfs/batch_01/```. The ```data``` folder should also contain an extraction subfolder, ```data/extraction``` where the output of extraction scripts will be stored as a batch subfolder, e.g. ```data/extraction/tables/```.
The last suggestion is to also include annotations in a subfolder such as ```data/annotations/batch_01/```.


# 🔄 Extraction Pipeline

## 📊 Table Extraction

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


## 🎯 Extraction of Interventions and Outcomes from Tables

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


## 📝 Metadata Extraction

This script extracts key metadata from academic papers in PDF format, including:
- Title
- Year of Publication
- Authors
- Abstract
- Keywords

### Usage

Run the metadata extraction script using:

```bash
python src/extract_metadata/extract_metadata.py
```

The script will:
1. Process all PDF files in the `data/raw` directory in parallel
2. Extract metadata using GPT-4
3. Save results to `processed/metadata.json`

### Configuration

You can customize the extraction by modifying `src/extract_metadata/settings.py`:
```python
@dataclass
class Settings:
    path_folder: Path = Path("data/raw")          # Input PDF folder
    path_prompt: Path = Path("config/prompts/metadata-extraction.prompt")  # Prompt template
    path_output: Path = Path("processed/metadata.json")  # Output file
    temperature: float = 0.0                      # Model temperature
    model: str = "gpt-4"                         # OpenAI model
    max_tokens: int = 4096                       # Max response tokens
    batch_size: int = 10                         # Parallel processing batch size
```

### Output Format

The script generates a JSON file with the following structure:
```json
{
  "path/to/paper.pdf": {
    "filename": "paper.pdf",
    "metadata": {
      "title": "Paper Title",
      "year": "2023",
      "authors": "Author 1, Author 2",
      "abstract": "Paper abstract...",
      "keywords": "keyword1, keyword2, keyword3"
    }
  }
}
```

### Error Handling

If a PDF cannot be processed, the output will include an error message:
```json
{
  "path/to/paper.pdf": {
    "filename": "paper.pdf",
    "error": "Error message details"
  }
}
```

### Requirements

Make sure you have:
1. Set up your OpenAI API key in `.env`
2. Installed all dependencies using Poetry
3. PDF files in the input directory


## 🎯 Intervention-outcome evaluation

This script evaluates the extracted interventions and outcomes in both mentions from the text and tables prior to merging. It should be ran as follows :

```bash
python src/evaluation/evaluate_io.py --batch first_batch --separator \t --eval_type sim
```

The output folder will by default be `data/evaluation/scores/` and will contain csv files containing the evaluation scores, as well as pairwise similarities if the `save_similarities` argument was set to `True`.
