from __future__ import annotations

import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from .extractor import extract_document

app = FastAPI(
    title="Document Extraction API",
    version="1.0.0",
)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

MAX_WORKERS = 4
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
MAX_BATCH_FILES = 10

_executor = ThreadPoolExecutor(
    max_workers=MAX_WORKERS,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def generate_document_id() -> str:
    """Generate a short unique document ID."""
    return uuid.uuid4().hex[:8]


async def read_upload_file(file: UploadFile) -> bytes:
    """
    Read an uploaded file with basic validation.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file '{file.filename}' is empty.",
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File '{file.filename}' is too large. "
                f"Maximum size is {MAX_FILE_SIZE // (1024 * 1024)} MB."
            ),
        )
    print(f"Read file '{file.filename}' ({len(content)} bytes).")
    return content


def run_extraction(
    content: bytes,
    filename: str,
    document_id: str,
) -> dict[str, Any]:
    """
    Synchronous extraction wrapper.

    This function runs inside ThreadPoolExecutor so that
    OCR/LLM processing does not block FastAPI's event loop.
    """

    t0 = time.perf_counter()

    result = extract_document(
        content,
        filename,
        document_id,
    )

    wall_ms = (time.perf_counter() - t0) * 1000

    payload = result.model_dump()

    payload["wall_clock_ms"] = round(wall_ms, 1)

    return payload


# ----------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------


@app.on_event("shutdown")
def shutdown_event():
    """
    Gracefully shut down the thread pool when the application stops.
    """
    _executor.shutdown(
        wait=True,
        cancel_futures=True,
    )


# ----------------------------------------------------------------------
# Health Check
# ----------------------------------------------------------------------


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "document-extraction-api",
    }


# ----------------------------------------------------------------------
# Single Document Extraction
# ----------------------------------------------------------------------


@app.post("/extract")
async def extract_single(
    file: UploadFile = File(...),
):
    """
    Extract structured information from a single document.

    OCR + LLM processing runs in the thread pool so the
    FastAPI event loop remains responsive.
    """

    content = await read_upload_file(file)

    filename = file.filename or "unknown"

    doc_id = generate_document_id()

    loop = asyncio.get_running_loop()

    try:
        result = await loop.run_in_executor(
            _executor,
            run_extraction,
            content,
            filename,
            doc_id,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document extraction failed: {str(exc)}",
        ) from exc

    return JSONResponse(content=result)


# ----------------------------------------------------------------------
# Batch Document Extraction
# ----------------------------------------------------------------------


@app.post("/extract/batch")
async def extract_batch(
    files: list[UploadFile] = File(...),
):
    """
    Extract multiple documents concurrently.

    The ThreadPoolExecutor controls the maximum number of
    documents processed simultaneously.
    """

    # --------------------------------------------------------------
    # Validate batch
    # --------------------------------------------------------------

    if not files:
        raise HTTPException(
            status_code=400,
            detail="At least one file is required.",
        )

    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Maximum {MAX_BATCH_FILES} files are allowed " "per batch request."
            ),
        )

    # --------------------------------------------------------------
    # Read files
    # --------------------------------------------------------------

    contents: list[tuple[bytes, str]] = []

    for file in files:
        content = await read_upload_file(file)

        filename = file.filename or "unknown"

        contents.append((content, filename))

    # --------------------------------------------------------------
    # Safe extraction wrapper
    # --------------------------------------------------------------

    def _safe_run_extraction(
        content: bytes,
        filename: str,
        doc_id: str,
    ) -> dict[str, Any]:
        """
        Run document extraction safely.

        Always returns a dictionary, even if extraction fails.
        This prevents asyncio.gather() from returning a
        dict | Exception union type.
        """

        try:
            result = run_extraction(
                content,
                filename,
                doc_id,
            )

            result["success"] = True
            result["filename"] = filename

            return result

        except Exception as exc:
            return {
                "success": False,
                "filename": filename,
                "document_id": doc_id,
                "error": str(exc),
            }

    # --------------------------------------------------------------
    # Parallel extraction
    # --------------------------------------------------------------

    t0 = time.perf_counter()

    loop = asyncio.get_running_loop()

    tasks: list[asyncio.Future[dict[str, Any]]] = []

    for content, filename in contents:
        doc_id = generate_document_id()

        task = loop.run_in_executor(
            _executor,
            _safe_run_extraction,
            content,
            filename,
            doc_id,
        )

        tasks.append(task)

    # Because _safe_run_extraction catches exceptions and always
    # returns dict[str, Any], gather() also returns dictionaries.
    results: list[dict[str, Any]] = await asyncio.gather(*tasks)

    wall_ms = (time.perf_counter() - t0) * 1000

    # --------------------------------------------------------------
    # Metrics
    # --------------------------------------------------------------

    successful_results = [result for result in results if result.get("success") is True]

    total_cost = sum(
        float(result.get("cost_usd") or 0) for result in successful_results
    )

    latencies = [float(result.get("latency_ms") or 0) for result in successful_results]

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    success_count = len(successful_results)

    failed_count = len(results) - success_count

    # --------------------------------------------------------------
    # Response
    # --------------------------------------------------------------

    return JSONResponse(
        content={
            "success": failed_count == 0,
            "n_documents": len(results),
            "n_success": success_count,
            "n_failed": failed_count,
            "max_workers": MAX_WORKERS,
            "wall_clock_ms_total": round(
                wall_ms,
                1,
            ),
            "avg_latency_ms_per_doc": round(
                avg_latency,
                1,
            ),
            "total_cost_usd": round(
                total_cost,
                6,
            ),
            "results": results,
        }
    )
