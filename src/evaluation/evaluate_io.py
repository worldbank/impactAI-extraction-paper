import os
import argparse
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL = genai.GenerativeModel("gemini-1.5-flash")


ANNOT_FIELDS = [
    "descriptive name of intervention",
    "Outcome name used in the paper",
    "details of the intervention",
]
EXTRACTION_FIELDS = ["intervention", "outcome", "intervention_description"]


def load_annotations(annotations_folder, sep="\t"):
    annotations = {}
    for f in os.listdir(annotations_folder):
        try:
            if f.endswith(".tsv") or f.endswith(".csv"):
                file_path = annotations_folder + f
                annotations[f.split(".")[0].split(" - ")[1]] = pd.read_csv(
                    file_path, sep=sep
                )
        except Exception:
            pass
    return annotations


def get_embeddings(extraction, annot_list, model_type):
    if model_type == "google":
        emb_extr = np.array(
            genai.embed_content("models/text-embedding-004", extraction)["embedding"]
        ).reshape(1, -1)
        emb_annot_list = [
            np.array(
                genai.embed_content("models/text-embedding-004", a)["embedding"]
            ).reshape(1, -1)
            for a in annot_list
        ]
    elif model_type == "stella":
        model = SentenceTransformer(
            "dunzhang/stella_en_1.5B_v5", token=os.getenv("HUGGINGFACE_TOKEN")
        )
        emb_extr = np.array(model.encode(extraction)).reshape(1, -1)
        emb_annot_list = [np.array(model.encode(a)).reshape(1, -1) for a in annot_list]
    return emb_extr, emb_annot_list


def get_annot_interv(annotations_df):
    ground_truth_col = [
        c for c in annotations_df.columns.values if c.startswith("Ground_Truth")
    ][0]
    annots = {
        field: annotations_df[annotations_df["Field Name"].str.startswith(field)][
            ground_truth_col
        ].tolist()
        for field in ANNOT_FIELDS
    }
    for field, values in annots.items():
        annots[field] = list(set(values))
    return annots


def load_io_dfs_tables(io_tables_subdir):
    io_dfs = {}
    for filename in os.listdir(io_tables_subdir):
        if filename.endswith(".csv"):
            io_df = pd.read_csv(io_tables_subdir + filename)
            io_dfs[filename[:-4]] = io_df
    return io_dfs


def look_for_io_annotation(annot, extraction_list, eval_type="em", model_type="google"):
    if eval_type == "em":
        if annot in extraction_list:
            return 1
        else:
            return 0
    elif eval_type == "sim":
        threshold = 0.6
        emb_annot, emb_extr_list = get_embeddings(annot, extraction_list, model_type)
        sims = [cosine_similarity(emb_annot, emb_extr) for emb_extr in emb_extr_list]
        if sims:
            if max(sims) > threshold:
                return 1
            else:
                return 0
        else:
            return 0


def evaluate_io_for_sr(
    sr_annotations, sr_io_mentions, eval_type="em", model_type="google"
):
    """Evaluates extraction for one systematic review"""
    sr_eval = [0] * len(ANNOT_FIELDS)
    total = [0] * len(ANNOT_FIELDS)
    annots = get_annot_interv(sr_annotations)
    for i, field in enumerate(ANNOT_FIELDS):
        for annot in annots[field]:
            extractions = [
                e
                for e in sr_io_mentions[EXTRACTION_FIELDS[i]].dropna().unique().tolist()
                if e != "xx"
            ]
            total[i] += 1
            sr_eval[i] += look_for_io_annotation(
                annot, extractions, eval_type, model_type=model_type
            )
    return sr_eval, total


def evaluate_io(
    io_dfs, annotations, print_b=False, eval_type="em", model_type="google"
):
    """Evaluates extraction for all systematic reviews"""
    all_eval = {
        k: []
        for k in ["pdf"] + ["{}_{}".format(eval_type, f) for f in EXTRACTION_FIELDS]
    }
    for pdf, pdf_io_mentions in io_dfs.items():
        try:
            print("Evaluating extraction for {}".format(pdf))
            pdf_eval, total = evaluate_io_for_sr(
                annotations[pdf], pdf_io_mentions, eval_type, model_type
            )
            all_eval["pdf"].append(pdf)
            for k, field in enumerate(EXTRACTION_FIELDS):
                score = pdf_eval[k] / max(1, total[k])
                all_eval["{}_{}".format(eval_type, field)].append(score)
                if print_b:
                    print(
                        "Proportion of {} recall for field {} : {}".format(
                            eval_type, field, round(score, 2)
                        )
                    )
        except Exception:
            print("Found issue with {}".format(pdf))
    return all_eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate extracted interventions and outcomes along with their descriptions against annotations for each source pdf file."
    )

    parser.add_argument("--batch", type=str, help="Name of the batch.")
    parser.add_argument(
        "--separator", type=str, help="Separator used to parse annotation files."
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        help="Type of evaluation to compute recall of annotations among extraction.",
        default="sim",
    )

    args = parser.parse_args()

    if not os.path.exists("scores/"):
        os.mkdir("scores/")

    batch = args.batch
    eval_type = args.eval_type

    # Load and process annotations
    sep = args.separator

    annotations_folder = f"data/annotations/{batch}/"
    annotations = load_annotations(annotations_folder, sep=sep)

    ## Evaluate io from tables
    io_tables_folder = f"data/extraction/io_tables/{batch}/"

    io_dfs_tables = load_io_dfs_tables(io_tables_folder)
    all_scores = evaluate_io(
        io_dfs_tables,
        annotations,
        print_b=False,
        eval_type=eval_type,
        model_type="google",
    )

    pd.DataFrame(all_scores).to_csv(
        f"src/evaluation/scores/io_eval_{eval_type}_{batch}.csv"
    )
