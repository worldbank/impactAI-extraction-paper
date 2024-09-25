import google.generativeai as genai
import re
import os
import pickle as pkl
import pandas as pd
import json
import argparse
from dotenv import load_dotenv

load_dotenv()


def load_tables(tables_path):
    """Function that loads extracted tables at a given location."""
    tables = {}
    for pdf_name in os.listdir(tables_path):
        if not pdf_name.startswith("."):
            pdf_tables = []
            tables_subfolder = tables_path + pdf_name + "/"
            for filename in os.listdir(tables_subfolder):
                if filename.endswith(".pkl"):
                    file_path = tables_subfolder + filename
                    pdf_tables.append(pkl.load(open(file_path, "rb")))
            tables[pdf_name] = pdf_tables
    return tables


def get_io_from_latex_with_google(latex_content, model):
    """Function that gets intervention and outcome information from tables."""
    # Preprocess the LaTeX content to get only the table part
    table_pattern = re.compile(r"\\begin{tabular}.*?\\end{tabular}", re.DOTALL)
    table_match = table_pattern.search(latex_content)

    # If no table found, raise an error
    if not table_match:
        raise ValueError("No table found in the LaTeX content")

    table_content = latex_content
    # Escape curly braces in the LaTeX content
    escaped_table_content = table_content.replace("{", "{{").replace("}", "}}")

    # Prepare the prompt for Google Gemini
    prompt = f"""
    You are an assistant that extracts information about interventions, outcomes along with their description, and countries in latex tables from
    economics RCT/impact evaluation studies. You are given a LaTeX table as an input, which contains information for a given impact evaluation study.
    Look for specific keywords and phrases that indicate the table is about interventions and/or outcomes. Extract the intervention, outcome and
    their description from the following latex table along with the study name. The output should be a dictionary with five keys : 'intervention', 'outcome', 'intervention_description', 'outcome_description', and 'country' respectively for the name of the study (author
    and eventually year), the intervention, the name of the outcome, the description of the intervention, the description of the outcome, and the
    country where the study was implemented. If any of these four fields is not found, the associated value should be xx. Assert the well-formedness
    of the output JSON.

    JSON Format:
    {{"intervention": "xx", "outcome": "xx", "intervention_description": "xx", "outcome_description": "xx", "country": "xx"}}

    Latex: \n {escaped_table_content}"


    Please provide the results in JSON format.

    Json:
    """

    response = model.generate_content(prompt)
    response.resolve()
    return response.text


def extract_io(tables, out_subdir, model, save=True):
    all_extractions = {}
    for pdf_name, pdf_tables in tables.items():
        extracted_jsons = []
        print(f"Performing extraction for {pdf_name}...")
        for t in pdf_tables:
            latex_str = t.reset_index().to_latex()
            try:
                io_response = get_io_from_latex_with_google(latex_str, model)
                io_response = io_response.replace("\n", " ")
                json_as_list_candidates = re.findall("\[.*\]", io_response)
                json_as_dict_candidates = re.findall("\{.*\}", io_response)

                if json_as_dict_candidates:
                    as_list = False
                    if json_as_list_candidates:
                        if len(json_as_list_candidates[0]) > len(
                            json_as_dict_candidates[0]
                        ):
                            as_list = True
                            io_as_json = json.loads(json_as_list_candidates[0])
                            for io_dict in io_as_json:
                                extracted_jsons.append(io_dict)
                    if as_list is False:
                        io_as_dict = json.loads(json_as_dict_candidates[0])
                        extracted_jsons.append(io_as_dict)
            except Exception:
                print(f"Encountered an error while processing {pdf_name}.")
        if save:
            extraction_df = pd.DataFrame(extracted_jsons)
            extraction_df.to_csv(out_subdir + pdf_name + ".csv", index=False)
        all_extractions[pdf_name] = extraction_df
    return all_extractions


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Get information from extracted tables using google's Gemini model."
    )

    parser.add_argument("--tables_folder", type=str, help="Path to the input tables.")
    parser.add_argument("--out_folder", type=str, help="Path to output.")
    parser.add_argument("--batch", type=str, help="Batch of pdfs being processed.")

    args = parser.parse_args()

    tables = load_tables(args.tables_folder)

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    out_folder = args.out_folder

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    out_subdir = out_folder + args.batch + "/"

    if not os.path.exists(out_subdir):
        os.mkdir(out_subdir)

    all_extractions = extract_io(tables, out_subdir, model, save=True)

    stats = {"PDF name": [], "# Interventions": [], "# Outcomes": [], "# Countries": []}
    for pdf_name in tables:
        df = all_extractions[pdf_name]
        stats["PDF name"].append(pdf_name)
        stats["# Interventions"].append(
            len(set([i for i in df.intervention if i != "xx"]))
        )
        stats["# Outcomes"].append(len(set([i for i in df.outcome if i != "xx"])))
        stats["# Countries"].append(len(set([i for i in df.country if i != "xx"])))

    print("Displaying extraction statistics...")
    print(pd.DataFrame(stats))
