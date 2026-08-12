"""
extractor.py
------------
Responsible for ONE job: turning uploaded resume files (PDF, DOCX, TXT)
into plain text strings, regardless of format.

This isolates all "messy file handling" logic away from the AI logic,
so if we ever need to support a new file type (e.g., .rtf), we only
touch this file.
"""

import io
import pdfplumber
import docx


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extracts all text from a PDF file, page by page.

    Args:
        file_bytes: Raw bytes of the uploaded PDF file.

    Returns:
        A single string containing all extracted text.
    """
    text_chunks = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  # Some pages (e.g., scanned images) may return None
                text_chunks.append(page_text)
    return "\n".join(text_chunks)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """
    Extracts all text from a DOCX file, paragraph by paragraph.

    Args:
        file_bytes: Raw bytes of the uploaded DOCX file.

    Returns:
        A single string containing all extracted text.
    """
    document = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def extract_text_from_txt(file_bytes: bytes) -> str:
    """
    Decodes a plain text file into a string.

    Args:
        file_bytes: Raw bytes of the uploaded TXT file.

    Returns:
        Decoded string. Falls back to ignoring undecodable characters
        rather than crashing on weird encodings.
    """
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("utf-8", errors="ignore")


def extract_text(filename: str, file_bytes: bytes) -> str:
    """
    Dispatcher function: looks at the file extension and routes to the
    correct extraction function. This is the ONLY function other modules
    need to call — they never need to know the internals.

    Args:
        filename: Original filename (used to detect extension).
        file_bytes: Raw bytes of the uploaded file.

    Returns:
        Extracted plain text.

    Raises:
        ValueError: If the file type is not supported.
    """
    extension = filename.lower().split(".")[-1]

    if extension == "pdf":
        return extract_text_from_pdf(file_bytes)
    elif extension == "docx":
        return extract_text_from_docx(file_bytes)
    elif extension == "txt":
        return extract_text_from_txt(file_bytes)
    else:
        raise ValueError(
            f"Unsupported file type: '.{extension}'. "
            "Please upload a PDF, DOCX, or TXT file."
        )
