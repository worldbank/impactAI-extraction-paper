import asyncio
import base64
from io import BytesIO
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import time
from tqdm import tqdm
from jinja2 import Template
from openai import AsyncOpenAI
from PIL import Image
from google.generativeai import GenerativeModel
from dataclasses import dataclass
from pdf2image import convert_from_path
import tempfile

from docling_core.types.doc import DoclingDocument
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import DocItemLabel
from docling_core.types.doc import TableItem, TextItem


from src.parse.settings import PDF2MarkdownSettings
from src.utils.file_management import load_prompt_async, get_secret
from src.utils.openai_response import (
    process_text_chunk,
    process_text_chunk_gemini,
    generate_completion_with_image_docling,
    generate_completion_with_image_gemini,
)


def setup_clients(
    settings: PDF2MarkdownSettings,
) -> Tuple[
    Union[AsyncOpenAI, GenerativeModel, None], Union[AsyncOpenAI, GenerativeModel, None]
]:
    """
    Setup OpenAI and Gemini clients.
    """
    if not settings.model_text:
        client_texts = None
    elif settings.model_text in ["gpt-4o-mini", "gpt-4o"]:
        client_texts = AsyncOpenAI(api_key=get_secret("openai-api-key"))
    elif "gemini" in settings.model_text:
        client_texts = GenerativeModel(settings.model_text)
    else:
        client_texts = None

    if not settings.model_tables:
        client_tables = None
    elif settings.model_tables in ["gpt-4o-mini", "gpt-4o"]:
        client_tables = AsyncOpenAI(api_key=get_secret("openai-api-key"))
    elif "gemini" in settings.model_tables:
        client_tables = GenerativeModel(settings.model_tables)
    else:
        client_tables = None

    return client_texts, client_tables


def setup_logger(verbose: bool) -> logging.Logger:
    """
    Set up logger with appropriate verbosity.

    Args:
        verbose: If True, set level to INFO, else ERROR

    Returns:
        Logger instance
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def log_summary_metrics(
    metrics: Dict, total_files: int, logger: logging.Logger
) -> None:
    """
    Calculate and log summary metrics.

    Args:
        metrics: Dictionary of metrics for all files
        total_files: Total number of files processed
        logger: Logger instance
    """
    logger.info("Conversion complete:")
    logger.info(
        f"Successfully converted: {metrics['successful_conversions']}/{metrics['total_documents']} files"
    )
    logger.info(f"Total pages processed: {metrics['total_pages']}")
    logger.info(
        f"Average time per document: {metrics['average_time_per_document']:.2f} seconds"
    )
    logger.info(
        f"Average time per page: {metrics['average_time_per_page']:.2f} seconds"
    )


async def save_metrics(
    metrics: Dict, metrics_path: Path, logger: logging.Logger
) -> None:
    """
    Save metrics to JSON file.

    Args:
        metrics: Dictionary of metrics to save
        metrics_path: Path to save metrics
        logger: Logger instance
    """
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.get_event_loop().run_in_executor(
        None, lambda: json.dump(metrics, open(metrics_path, "w"), indent=4)
    )
    logger.info(f"Metrics saved to {metrics_path}")


async def main_process_pdfs(
    pdf_files: List[Path],
    settings: PDF2MarkdownSettings,
    logger: logging.Logger,
    client_texts: Optional[Union[AsyncOpenAI, GenerativeModel, None]] = None,
    client_tables: Optional[Union[AsyncOpenAI, GenerativeModel, None]] = None,
) -> Dict:
    """
    Process PDFs with specified converter function.

    Args:
        pdf_files: List of PDF files to process
        settings: Settings for the converter
        logger: Logger instance
        client: Optional AsyncOpenAI client

    Returns:
        Dictionary containing metrics for all processed files
    """
    semaphore = asyncio.Semaphore(settings.batch_size)
    total_pages = 0
    total_time = 0
    successful = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    documents = await ingest_pdfs_with_docling(pdf_files, logger, settings)

    with tqdm(total=len(pdf_files), desc="Converting PDFs") as pbar:
        for pdf_file, document in zip(pdf_files, documents):
            relative_path = pdf_file.relative_to(settings.path_input)
            output_path = settings.path_output / relative_path.with_suffix(".md")

            success, metrics = await process_pdf_with_docling(
                document=document,
                pdf_name=pdf_file,
                output_path=output_path,
                client_texts=client_texts,
                client_tables=client_tables,
                settings=settings,
                logger=logger,
                semaphore=semaphore,
            )

            if success:
                total_pages += metrics.get("pages", 0)
                total_time += metrics.get("processing_time", 0)
                successful += 1

                # Safely extract token metrics
                token_metrics = metrics.get("token_metrics", {})
                total_prompt_tokens += token_metrics.get("total_prompt_tokens", 0)
                total_completion_tokens += token_metrics.get(
                    "total_completion_tokens", 0
                )
                total_tokens += token_metrics.get("total_tokens", 0)

            pbar.update(1)

    return {
        "total_documents": len(pdf_files),
        "successful_conversions": successful,
        "total_pages": total_pages,
        "average_time_per_document": total_time / len(pdf_files) if pdf_files else 0,
        "average_time_per_page": total_time / total_pages if total_pages else 0,
        "token_metrics": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "avg_prompt_tokens_per_page": total_prompt_tokens / total_pages
            if total_pages > 0
            else 0,
            "avg_completion_tokens_per_page": total_completion_tokens / total_pages
            if total_pages > 0
            else 0,
            "avg_total_tokens_per_page": total_tokens / total_pages
            if total_pages > 0
            else 0,
        },
    }


async def ingest_pdfs_with_docling(
    input_folder_path: Path, logger: logging.Logger, settings: PDF2MarkdownSettings
) -> DoclingDocument:
    """
    Ingest PDF file using Docling.

    Args:
        input_folder_path: Path to input folder
        logger: Logger instance

    Returns:
        DoclingDocument instance
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.do_ocr = True
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=4, device=AcceleratorDevice.AUTO
    )
    # if settings.model_tables:
    #    logger.info("Using table model, generating images")
    #    pipeline_options.images_scale = 2.0
    #    pipeline_options.generate_page_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    try:
        conv_results = doc_converter.convert_all(input_folder_path)
        return conv_results
    except Exception as e:
        logger.error(f"Error ingesting PDF {input_folder_path}: {e}")
        raise


async def split_document_content(
    document: DoclingDocument, logger: logging.Logger
) -> Tuple[DoclingDocument, DoclingDocument]:
    """
    Split document into text and tables/captions.

    Args:
        document: DoclingDocument instance
        logger: Logger instance

    Returns:
        Tuple containing:
        - Text document
        - Tables document
    """
    try:
        text_doc = DoclingDocument(name="text")
        table_doc = DoclingDocument(name="table")

        previous_label = None
        for item in document.iterate_items():
            label = item[0].label.value
            if label == DocItemLabel.TABLE:
                table_doc.add_table(data=item[0].data, prov=item[0].prov)
            elif (
                previous_label in [DocItemLabel.TABLE, DocItemLabel.CAPTION]
                and label == DocItemLabel.TEXT
            ):
                table_doc.add_text(
                    text=item[0].text, label=DocItemLabel.TEXT, prov=item[0].prov
                )
            elif label == DocItemLabel.CAPTION:
                table_doc.add_text(
                    text=item[0].text, label=DocItemLabel.TEXT, prov=item[0].prov
                )
            elif label != DocItemLabel.PICTURE:
                text_doc.add_text(
                    text=item[0].text, label=item[0].label, prov=item[0].prov
                )
            previous_label = label

        return text_doc, table_doc
    except Exception as e:
        logger.error(f"Error splitting document content: {e}")
        raise


async def process_document_text(
    text_doc: DoclingDocument,
    client_texts: Union[AsyncOpenAI, GenerativeModel, None],
    settings: PDF2MarkdownSettings,
    prompt_template: Template,
    logger: logging.Logger,
    semaphore: asyncio.Semaphore,
) -> Tuple[str, Dict[str, int]]:
    """
    Process entire document text with parallelization.

    Args:
        text_doc: Text document
        client_texts: API client for text processing
        settings: Settings instance
        prompt_template: Prompt template
        logger: Logger instance
        semaphore: Semaphore for concurrency control

    Returns:
        Tuple containing:
        - Combined processed text
        - Aggregated metrics
    """
    try:
        async with semaphore:
            if not text_doc:
                logger.error("Received None document")
                return empty_metrics()

            markdown_text = text_doc.export_to_markdown()
            if not markdown_text:
                logger.error("Empty markdown text from document")
                return empty_metrics()

            parts = markdown_text.split("\n\n##")
            if not parts:
                logger.error("No parts found after splitting markdown")
                return empty_metrics()

            logger.info(f"Processing {len(parts)} text chunks")
            pbar = create_progress_bar(len(parts), table_processing=False)

            async def process_and_update(
                text: str, index: int
            ) -> Tuple[int, str, Dict[str, int]]:
                if not text:
                    logger.warning(f"Empty text chunk at index {index}")
                    pbar.update(1)
                    return (
                        index,
                        "",
                        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    )

                try:

                    if settings.model_text in ["gpt-4o-mini", "gpt-4o"]:
                        result = await process_text_chunk(
                            text="\n\n##" + text if text else "",
                            client=client_texts,
                            settings=settings,
                            prompt_template=prompt_template,
                            logger=logger,
                        )
                    elif "gemini" in settings.model_text:
                        result = await process_text_chunk_gemini(
                            text="\n\n##" + text if text else "",
                            settings=settings,
                            prompt_template=prompt_template,
                            logger=logger,
                            client=client_texts,
                        )
                    else:
                        result = text, empty_metrics()[1]
                    pbar.update(1)
                    return index, *result
                except Exception as e:
                    logger.error(f"Error processing chunk {index}: {str(e)}")
                    pbar.update(1)
                    return (
                        index,
                        "",
                        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    )

            # Create tasks with their indices
            tasks = [
                asyncio.create_task(process_and_update(part, i))
                for i, part in enumerate(parts)
                if part  # Only process non-empty parts
            ]

            if not tasks:
                logger.error("No valid tasks created")
                return empty_metrics()

            # Gather results
            results = await asyncio.gather(*tasks, return_exceptions=True)
            pbar.close()

            sorted_results = sorted(results, key=lambda x: x[0])

            final_text = ""
            total_metrics = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "total_blocks": len(parts),
                "failed_blocks": len(parts) - len(results),
            }

            # Process sorted results
            for _, text, metrics in sorted_results:
                if text:  # Only add non-empty text
                    final_text += text
                for k, v in metrics.items():
                    total_metrics[k] += v

            return final_text, total_metrics

    except Exception as e:
        logger.error(f"Error processing document text: {str(e)}")
        return empty_metrics()


def empty_metrics() -> Tuple[str, Dict[str, int]]:
    """Return empty text and zeroed metrics."""
    return "", {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "total_blocks": 0,
        "failed_blocks": 0,
    }


def create_progress_bar(total: int, table_processing: bool) -> tqdm:
    """Create a progress bar for processing text chunks."""
    if table_processing:
        return tqdm(
            total=total,
            desc="Processing table chunks",
            position=1,
            leave=False,
            ncols=100,
        )
    else:
        return tqdm(
            total=total,
            desc="Processing text chunks",
            position=1,
            leave=False,
            ncols=100,
        )


async def process_pdf_with_docling(
    document: DoclingDocument,
    pdf_name: str,
    output_path: Path,
    client_texts: Union[AsyncOpenAI, GenerativeModel, None],
    client_tables: Union[AsyncOpenAI, GenerativeModel, None],
    settings: PDF2MarkdownSettings,
    logger: logging.Logger,
    semaphore: asyncio.Semaphore,
) -> Tuple[bool, Dict[str, int]]:
    """
    Main processing function for Docling.

    Args:
        document: DoclingDocument instance
        pdf_name: Name of the PDF file
        output_path: Path to output file
        client_texts: API client for text processing
        client_tables: API client for table processing
        settings: Settings instance
        logger: Logger instance
        semaphore: Semaphore for concurrency control

    Returns:
        Tuple containing:
        - Boolean indicating success
        - Dictionary containing metrics
    """
    try:
        start_time = time.time()
        text_doc, table_doc = await split_document_content(document.document, logger)

        if client_tables:
            table_groups = associate_tables_and_texts(table_doc.tables, table_doc.texts)
            table_text, table_metrics = await process_document_tables(
                table_groups=table_groups,
                conv_result=document,
                client_tables=client_tables,
                settings=settings,
                logger=logger,
                semaphore=semaphore,
                pdf_name=pdf_name,
            )
        else:
            logger.info("No table model selected, using document tables")
            table_text = table_doc.export_to_markdown()
            table_metrics = empty_metrics()[1]

        if client_texts:
            prompt_template = await load_prompt_async(settings.path_prompt, logger)
            final_text, text_metrics = await process_document_text(
                text_doc=text_doc,
                client_texts=client_texts,
                settings=settings,
                prompt_template=prompt_template,
                logger=logger,
                semaphore=semaphore,
            )
        else:
            logger.info("No text model selected, using document text")
            final_text = text_doc.export_to_markdown()
            text_metrics = empty_metrics()[1]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(final_text)
        tables_path = output_path.with_stem(output_path.stem + "-tables")
        tables_path.write_text(table_text)

        with open(Path(output_path.parent, output_path.stem + ".json"), "w") as f:
            json_metrics = {
                "text_metrics": text_metrics,
                "table_metrics": table_metrics,
            }
            json.dump(json_metrics, f, indent=4, ensure_ascii=False)

        processing_time = time.time() - start_time

        return True, {
            "pages": len(document.pages),
            "processing_time": processing_time,
            "successful_blocks": text_metrics.get("total_blocks", 0)
            - text_metrics.get("failed_blocks", 0),
            "token_metrics": {
                "total_prompt_tokens": text_metrics.get("prompt_tokens", 0)
                + table_metrics.get("prompt_tokens", 0),
                "total_completion_tokens": text_metrics.get("completion_tokens", 0)
                + table_metrics.get("completion_tokens", 0),
                "total_tokens": text_metrics.get("total_tokens", 0)
                + table_metrics.get("total_tokens", 0),
            },
        }
    except Exception as e:
        logger.error(f"Error converting {output_path.stem}: {str(e)}")
        return False, {"error": str(e)}


@dataclass
class TableTextGroup:
    table: "TableItem"
    title: Optional["TextItem"] = None
    notes: Optional["TextItem"] = None
    page_no: int = None
    bbox: Optional[Tuple[float, float, float, float]] = None

    def export_to_markdown(self):
        markdown = ""
        if self.title:
            markdown += f"## {self.title.text}\n"
        markdown += self.table.export_to_markdown()
        if self.notes:
            markdown += f"\n{self.notes.text}\n"
        return markdown


def get_extended_bbox(
    main_bbox: Tuple[float, float, float, float], *other_bboxes
) -> Tuple[float, float, float, float]:
    """
    Extend the main bounding box to include other bounding boxes.

    Args:
        main_bbox: Original (l, t, r, b) bbox
        *other_bboxes: Additional bboxes to include

    Returns:
        Extended (l, t, r, b) bbox
    """
    all_bboxes = [main_bbox] + [bbox for bbox in other_bboxes if bbox is not None]

    # Get extremes of all bboxes
    left = min(bbox[0] for bbox in all_bboxes)
    top = max(bbox[1] for bbox in all_bboxes)
    right = max(bbox[2] for bbox in all_bboxes)
    bottom = min(bbox[3] for bbox in all_bboxes)

    return {"l": left, "t": top, "r": right, "b": bottom}


def associate_tables_and_texts(
    tables: List["TableItem"],
    texts: List["TextItem"],
    max_title_distance: float = 50,
    max_notes_distance: float = 50,
) -> List[TableTextGroup]:
    """
    Associate tables with their titles and notes based on spatial proximity.
    Extends bounding box to include associated text elements.

    Args:
        tables: List of TableItem objects
        texts: List of TextItem objects
        max_title_distance: Maximum vertical distance to consider for title association
        max_notes_distance: Maximum vertical distance to consider for notes association

    Returns:
        List of TableTextGroup objects containing associated elements
    """
    table_groups = []

    for table in tables:
        group = TableTextGroup(table=table)
        table_bbox = table.prov[0][0].bbox
        page_no = table.prov[0][0].page_no
        group.page_no = page_no

        current_bbox = (table_bbox.l, table_bbox.t, table_bbox.r, table_bbox.b)

        # Find potential title (text above table)
        potential_titles = [
            text
            for text in texts
            if text.prov[0][0].page_no == page_no
            and abs(text.prov[0][0].bbox.b - table_bbox.t)
            < max_title_distance  # Within distance
            and "Table" in text.text  # Contains "Table" keyword
        ]

        title_bbox = None
        if potential_titles:
            # Take the closest title
            group.title = min(
                potential_titles, key=lambda x: abs(x.prov[0][0].bbox.b - table_bbox.t)
            )
            # Get title bbox
            title_bbox = (
                group.title.prov[0][0].bbox.l,
                group.title.prov[0][0].bbox.t,
                group.title.prov[0][0].bbox.r,
                group.title.prov[0][0].bbox.b,
            )

        # Find potential notes (text below table)
        potential_notes = [
            text
            for text in texts
            if text.prov[0][0].page_no == page_no
            and text.prov[0][0].bbox.t < table_bbox.b  # Text is below table
            and abs(text.prov[0][0].bbox.t - table_bbox.b)
            < max_notes_distance  # Within distance
        ]

        notes_bbox = None
        if potential_notes:
            # Take the closest notes
            group.notes = min(
                potential_notes, key=lambda x: abs(x.prov[0][0].bbox.t - table_bbox.b)
            )
            # Get notes bbox
            notes_bbox = (
                group.notes.prov[0][0].bbox.l,
                group.notes.prov[0][0].bbox.t,
                group.notes.prov[0][0].bbox.r,
                group.notes.prov[0][0].bbox.b,
            )

        # Extend bbox to include title and notes
        group.bbox = get_extended_bbox(current_bbox, title_bbox, notes_bbox)

        table_groups.append(group)

    return table_groups


def crop_image_from_bbox(
    pil_image: Image.Image, bbox_dict, margin: int = 50, scale_factor: float = 2.0
) -> Image.Image:
    """
    Crop PIL image using bbox dictionary with BOTTOMLEFT coordinate origin and scaling.

    Args:
        pil_image: PIL Image object
        bbox_dict: Dictionary with 'l', 't', 'r', 'b' keys and BOTTOMLEFT coordinate origin
        margin: Optional margin around the crop in pixels
        scale_factor: Factor to scale the bbox coordinates

    Returns:
        Cropped PIL Image
    """
    # Get image dimensions
    img_width, img_height = pil_image.size

    # Extract and scale coordinates
    left = bbox_dict["l"] * scale_factor
    top = bbox_dict["t"] * scale_factor
    right = bbox_dict["r"] * scale_factor
    bottom = bbox_dict["b"] * scale_factor

    # Convert from bottom-left to top-left coordinate system
    crop_top = img_height - top  # Convert top coordinate
    crop_bottom = img_height - bottom  # Convert bottom coordinate

    # Create crop box with margin
    crop_box = (
        max(0, int(left) - margin),
        max(0, int(crop_top) - margin),
        min(img_width, int(right) + margin),
        min(img_height, int(crop_bottom) + margin),
    )

    return pil_image.crop(crop_box)


def encode_image_to_base64(pil_image):
    """Convert PIL Image to base64 string."""
    # Convert to RGB if image is in RGBA mode
    if pil_image.mode == "RGBA":
        pil_image = pil_image.convert("RGB")

    # Save image to bytes
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # Using JPEG instead of PNG
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


async def process_document_tables(
    table_groups: List[TableTextGroup],
    conv_result: DoclingDocument,
    client_tables: Union[AsyncOpenAI, GenerativeModel],
    settings: PDF2MarkdownSettings,
    logger: logging.Logger,
    semaphore: asyncio.Semaphore,
    pdf_name: str,
):
    """
    Process all tables in a document.

    Args:
        table_groups: List of TableTextGroup objects
        conv_result: Document containing page images
        client_tables: client for table processing
        settings: Settings instance
        logger: Logger instance
        semaphore: Semaphore for concurrency control

    Returns:
        Tuple containing:
        - List of processed table results
        - Aggregated metrics
    """
    system_prompt = await load_prompt_async(settings.path_prompt_tables, logger)

    results, metrics = await process_table_groups(
        table_groups=table_groups,
        pdf_name=pdf_name,
        conv_result=conv_result,
        client_tables=client_tables,
        settings=settings,
        system_prompt=system_prompt,
        logger=logger,
        semaphore=semaphore,
        image_scale=2.0,
    )

    logger.info("\nProcessing Metrics:")
    for k, v in metrics.items():
        logger.info(f"{k}: {v}")

    return results, metrics


async def process_table_groups(
    table_groups: List[TableTextGroup],
    pdf_name: str,
    conv_result: DoclingDocument,
    client_tables: Union[AsyncOpenAI, GenerativeModel],
    settings: PDF2MarkdownSettings,
    system_prompt: str,
    logger: logging.Logger,
    semaphore: asyncio.Semaphore,
    image_scale: float = 1.0,
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Process multiple table groups in parallel using GPT-4V.

    Args:
        table_groups: List of TableTextGroup objects
        conv_result: Document containing page images
        client: OpenAI API client
        settings: Settings instance
        system_prompt: System prompt for GPT-4V
        logger: Logger instance
        semaphore: Semaphore for concurrency control
        image_scale: Scale factor for image coordinates

    Returns:
        Tuple containing:
        - List of processed results (including table content and metadata)
        - Aggregated metrics
    """

    try:
        async with semaphore:
            if not table_groups:
                logger.error("No table groups to process")
                return [], empty_metrics()

            logger.info(f"Processing {len(table_groups)} tables")
            pbar = create_progress_bar(len(table_groups), table_processing=True)

            async def process_single_table(
                group: TableTextGroup, index: int
            ) -> Tuple[int, Dict, Dict[str, int]]:
                """Process a single table group and return results with index."""
                try:
                    # Get page image and crop to table
                    page_no = group.table.prov[0][0].page_no
                    # page_image = conv_result.pages[page_no].image
                    with tempfile.TemporaryDirectory() as temp_dir:
                        page_image = convert_from_path(
                            pdf_name,
                            dpi=300,
                            output_folder=temp_dir,
                            first_page=page_no,
                            last_page=page_no,
                            fmt="png",
                            thread_count=1,
                        )[0]
                    cropped_img = crop_image_from_bbox(
                        page_image,
                        group.bbox,
                        margin=30,
                        scale_factor=image_scale
                        * 2.0833333333333335,  # this is the scale factor for the table compared to the docling image with DPI of 300 and image scale of 2.0 for docling
                    )

                    # Encode image
                    image_base64 = encode_image_to_base64(cropped_img)

                    # Prepare messages
                    if settings.model_tables in ["gpt-4o-mini", "gpt-4o"]:
                        (
                            response,
                            metrics,
                        ) = await generate_completion_with_image_docling(
                            client=client_tables,
                            system_prompt=system_prompt,
                            image_base64=image_base64,
                            max_tokens=settings.max_tokens,
                            temperature=0.0,
                            user_markdown=group.export_to_markdown(),
                            model=settings.model_tables,
                        )
                    elif "gemini" in settings.model_tables:
                        response, metrics = await generate_completion_with_image_gemini(
                            system_prompt=system_prompt,
                            image_base64=image_base64,
                            max_tokens=settings.max_tokens,
                            client=client_tables,
                            temperature=0.0,
                            user_markdown=group.export_to_markdown(),
                        )
                    else:
                        response = group.export_to_markdown()
                        metrics = empty_metrics()[1]

                    pbar.update(1)
                    return index, response, metrics

                except Exception as e:
                    logger.error(
                        f"Error processing table {index} on page {group.page_no}: {str(e)}"
                    )
                    pbar.update(1)
                    return (
                        index,
                        None,
                        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    )

            # Create tasks for all tables
            tasks = [
                asyncio.create_task(process_single_table(group, i))
                for i, group in enumerate(table_groups)
            ]

            if not tasks:
                logger.error("No valid tasks created")
                return [], empty_metrics()

            # Gather results
            results = await asyncio.gather(*tasks, return_exceptions=True)
            pbar.close()

            # Sort results by original index
            sorted_results = sorted([r for r in results if r], key=lambda x: x[0])

            # Aggregate metrics
            total_metrics = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "total_tables": len(table_groups),
                "failed_tables": len(table_groups) - len(sorted_results),
            }

            # Process results and combine metrics
            processed_tables = ""
            for _, result, metrics in sorted_results:
                if result:  # Only add non-empty results
                    processed_tables += result
                for k, v in metrics.items():
                    total_metrics[k] += v

            return processed_tables, total_metrics

    except Exception as e:
        logger.error(f"Error in table group processing: {str(e)}")
        return [], empty_metrics()
