# 📚 Table of Contents
- [🛠️ Requirements](#-requirements)
- [⚙️ Configuration](#-configuration)
- [📁 Data Structure](#-data-structure)
- [🔄 Extraction Pipeline](#extraction-pipeline)
  - [🔄 PDF to Markdown](#pdf-to-markdown)
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

## 📄 PDF to Markdown

The `parse_pdf.py` script implements a robust PDF processing pipeline that converts academic papers into clean markdown format. This pipeline has been extensively tested against various approaches (Zerox, OCR on PDF images, PyMuPDF with and without LLM) and has proven to be the most effective solution.

### Usage

```bash
# Basic usage
python src/parse/parse_pdf.py

# Process with verbose logging
python src/parse/parse_pdf.py --verbose

# Process specific number of samples
python src/parse/parse_pdf.py --n_samples 5
```

### Configuration

Configure the pipeline in `src/parse/settings.py`:

```python
model_text = "gpt-4o-mini"  # For text processing
model_tables = "gpt-4o"     # For table processing
```

### Pipeline Steps

1. **Document Loading**
   - Scans input directory for PDF files
   - Supports batch processing with configurable concurrency limits
   - Handles nested directory structures

2. **Docling Ingestion**
   - Converts PDFs using Docling's document processing pipeline
   - Extracts text, tables, and structural elements
   - Generates high-quality page images for table processing

3. **Content Separation**
   - Divides content into two streams:
     - Main text content
     - Tables and their associated captions
   - Preserves document structure and relationships

4. **LLM Post-processing**
   - Text Processing:
     - Uses GPT-4-mini for optimal performance/cost ratio
     - Cleans formatting, removes noise (headers, footers, page numbers)
     - Preserves academic structure and references
   - Table Processing:
     - Uses GPT-4 with vision capabilities
     - Verifies and corrects table structure
     - Ensures accuracy of numerical data
     - Preserves notes and statistical indicators

5. **Output Generation**
   - Generates clean markdown files
   - Creates separate files for main text and tables
   - Saves processing metrics for analysis

### Best Practices

Our testing revealed that the optimal configuration uses:
- `gpt-4o-mini` for text post-processing
- `gpt-4o` for table post-processing

This combination provides the best balance of accuracy and cost-effectiveness.

### Comparison with Other Approaches

We evaluated several alternative approaches:
- Zerox PDF processing [here](https://github.com/getomni-ai/zerox/)
- OCR on PDF images (from scratch)
- PyMuPDF (with and without LLM)
- Adding Yolov10 to extract tables from PDF images

Docling + LLM post-processing consistently outperformed these methods in terms of:
- Text extraction accuracy
- Table structure preservation
- Document formatting retention
- Processing speed

### Known Limitations and Future Improvements

Current limitations that will be addressed in future iterations:

1. **Document Completeness**
   - Occasional missing pages in Docling output
   - Solution: Implement page verification and recovery system

2. **Table Accuracy**
   - Some instances of missing rows in complex tables
   - Improvement planned: Enhanced table structure validation

3. **Numerical Sequences**
   - Reduced accuracy with very long number sequences
   - Future enhancement: Work step by step on the rows with prompt

4. **Processing Speed**
   - Current sequential processing of tables
   - Planned: Improved parallelization for table processing

Despite these limitations, the current pipeline produces high-quality results suitable for most academic papers.

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

## 🤖 RCT Classification with Zero-Shot Learning

This script uses GPT-4o to classify research papers as Randomized Controlled Trials (RCTs) or not using zero-shot learning.

### Usage

After extracting metadata, run the classification script:

```bash
python src/rct_clf/zsl_classify.py
```

The script will:
1. Load metadata from `data/processed/metadata.json`
2. Process each paper using GPT-4o for RCT classification
3. Save results to `data/processed/metadata_rct_classified.json`

### Configuration

Customize the classification by modifying `src/rct_clf/settings.py`:
```python
@dataclass
class ZSLSettings:
    path_prompt: Path = Path("config/prompts/RCT_ZSL.prompt")
    path_input: Path = Path("data/processed/metadata.json")
    path_output: Path = Path("data/processed/metadata_rct_classified.json")
    system_content: str = "You are an expert in economic research."
    temperature: float = 1.0
    model: str = "gpt-4o"
    max_tokens: int = 1024
    batch_size: int = 10
```

### Output Format

The script generates a JSON file that includes the original metadata plus RCT classification:
```json
{
  "path/to/paper.pdf": {
    "filename": "paper.pdf",
    "metadata": {
      "title": "Paper Title",
      "abstract": "Paper abstract...",
      "keywords": "keyword1, keyword2"
    },
    "rct": "True"  // or "False"
  }
}
```

### Error Handling

If classification fails, the output will include an error message:
```json
{
  "path/to/paper.pdf": {
    "filename": "paper.pdf",
    "metadata": {...},
    "error": "Error message details"
  }
}
```

## 🤖 MetaData extraction with RCT Classification with Zero-Shot Learning

This script uses GPT-4o to extract metadata and classify research papers as Randomized Controlled Trials (RCTs) or not using zero-shot learning.

### Usage

Run the metadata extraction and RCT classification script using:

```bash
python src/rct_clf/zsl_from_pdf.py
```

The script will:
1. Process all PDF files in the `data/raw` directory in parallel
2. Extract metadata using GPT-4o
3. Save results to `metadata_pdf_rct_classified.json`

### Configuration

You can customize the extraction by modifying `src/rct_clf/settings.py`:
```python
@dataclass
class PDFZSLSettings:
    path_folder: Path = Path("data/raw") # Input PDF folder
    path_prompt: Path = Path("config/prompts/RCT_metadata-extraction_ZSL.prompt")  # Prompt template
    path_output: Path = Path("data/processed/metadata_pdf_rct_classified.json") # Output file
    system_content: str = "You are an expert that extracts metadata and classify whether the study is Randomized Controlled Trial (RCT) or not from academic papers." # system message
    temperature: float = 0.0 # Model temperature
    model: str = "gpt-4o" # OpenAI model
    max_tokens: int = 1024  # Max response tokens
    batch_size: int = 10   # Parallel processing batch size
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
    },
    "rct": "True", // or "False",
    "explanation": "text"
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

### Evaluation

To evaluate the RCT classification, run the following command:

```bash
python src/rct_clf/evaluate.py
```

The script will:
1. Load predictions from `data/processed/metadata_rct_classified.json`
2. Load ground truth from `data/raw/RCT_GT.csv`
3. Compute metrics
4. Save results to `data/processed/ZSL_two_steps_metrics.json`

You can customize the evaluation by modifying `src/rct_clf/settings.py`:
```python
@dataclass
class EvaluationParams:
    path_preds: Path = Path("data/processed/metadata_rct_classified.json")
    path_true: Path = Path("data/raw/RCT_GT.csv")
    path_output: Path = Path("data/processed/ZSL_two_steps_metrics.json")
```

### Output Format

The script generates a JSON file with the following structure:
```json
{
    "accuracy": 92.72727272727272,
    "precision": 100.0,
    "recall": 89.74358974358975,
    "f1": 94.5945945945946
}
```






## 🎯 Intervention-outcome evaluation

This script evaluates the extracted interventions and outcomes in both mentions from the text and tables prior to merging. It should be ran as follows :

```bash
python src/evaluation/evaluate_io.py --batch first_batch --separator \t --eval_type sim
```

The output folder will by default be `data/evaluation/scores/` and will contain csv files containing the evaluation scores, as well as pairwise similarities if the `save_similarities` argument was set to `True`.
