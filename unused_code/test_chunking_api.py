#!/usr/bin/env python3
"""
FastAPI endpoint for testing Chonkie OSS chunking.
Upload PDFs via web interface and see chunked output with timing.
"""

import time
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from chonkie_oss_enhanced import chunk_document_with_metadata


app = FastAPI(
    title="Chonkie OSS Enhanced Test API",
    description="Test PDF chunking with FULL METADATA - Pages, Chapters, Headings",
    version="2.0.0"
)


class ChunkingResult(BaseModel):
    """Result of chunking operation with full metadata."""
    source_file: str
    processing_time_ms: float
    chunk_size: int
    overlap: int
    chunker_type: str
    total_chunks: int
    total_tokens: int
    avg_tokens_per_chunk: float
    chunks: List[Dict[str, Any]]
    # Enhanced metadata
    metadata: Dict[str, Any] = {}
    stats: Dict[str, Any] = {}


@app.get("/", response_class=HTMLResponse)
async def home():
    """Home page with upload form."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chonkie OSS PDF Chunking Test</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 900px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                padding: 40px;
            }
            h1 {
                color: #667eea;
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            .subtitle {
                color: #666;
                margin-bottom: 30px;
                font-size: 1.1em;
            }
            .form-group {
                margin-bottom: 25px;
            }
            label {
                display: block;
                margin-bottom: 8px;
                color: #333;
                font-weight: 600;
                font-size: 0.95em;
            }
            input[type="file"], input[type="number"], select {
                width: 100%;
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 1em;
                transition: border-color 0.3s;
            }
            input[type="file"]:focus, input[type="number"]:focus, select:focus {
                outline: none;
                border-color: #667eea;
            }
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px 40px;
                border: none;
                border-radius: 8px;
                font-size: 1.1em;
                font-weight: 600;
                cursor: pointer;
                width: 100%;
                transition: transform 0.2s;
            }
            button:hover {
                transform: translateY(-2px);
            }
            button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }
            #results {
                margin-top: 30px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
                display: none;
            }
            .loading {
                text-align: center;
                color: #667eea;
                font-size: 1.2em;
                margin: 20px 0;
            }
            .chunk {
                background: white;
                padding: 20px;
                margin: 15px 0;
                border-radius: 10px;
                border-left: 4px solid #667eea;
            }
            .chunk-header {
                display: flex;
                justify-content: space-between;
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 1px solid #e0e0e0;
            }
            .chunk-title {
                font-weight: 600;
                color: #667eea;
                font-size: 1.1em;
            }
            .chunk-meta {
                color: #666;
                font-size: 0.9em;
            }
            .chunk-text {
                color: #333;
                line-height: 1.6;
                white-space: pre-wrap;
                font-family: 'Courier New', monospace;
                background: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                max-height: 300px;
                overflow-y: auto;
            }
            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            .stat-card {
                background: white;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }
            .stat-value {
                font-size: 2em;
                font-weight: 700;
                color: #667eea;
            }
            .stat-label {
                color: #666;
                font-size: 0.9em;
                margin-top: 5px;
            }
            .info-box {
                background: #e3f2fd;
                border-left: 4px solid #2196f3;
                padding: 15px;
                margin: 20px 0;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Chonkie OSS Test</h1>
            <p class="subtitle">Upload a PDF and see the chunked output with timing</p>

            <div class="info-box">
                <strong>ℹ️ Info:</strong> This uses Chonkie OSS (free, local chunking).
                Upload any PDF to see how it chunks the text for RAG.
            </div>

            <form id="uploadForm" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="file">📄 PDF File</label>
                    <input type="file" id="file" name="file" accept=".pdf" required>
                </div>

                <div class="form-group">
                    <label for="chunk_size">📏 Chunk Size (tokens)</label>
                    <input type="number" id="chunk_size" name="chunk_size" value="512" min="100" max="2000">
                </div>

                <div class="form-group">
                    <label for="overlap">🔄 Overlap (tokens)</label>
                    <input type="number" id="overlap" name="overlap" value="80" min="0" max="500">
                </div>

                <div class="form-group">
                    <label for="chunker_type">⚙️ Chunker Type</label>
                    <select id="chunker_type" name="chunker_type">
                        <option value="recursive">Recursive (Best for general text)</option>
                        <option value="token">Token-based (Fixed-size chunks)</option>
                    </select>
                </div>

                <button type="submit" id="submitBtn">🚀 Chunk Document</button>
            </form>

            <div id="results"></div>
        </div>

        <script>
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();

                const submitBtn = document.getElementById('submitBtn');
                const resultsDiv = document.getElementById('results');

                submitBtn.disabled = true;
                submitBtn.textContent = '⏳ Processing...';
                resultsDiv.style.display = 'block';
                resultsDiv.innerHTML = '<div class="loading">⏳ Chunking your document...</div>';

                const formData = new FormData();
                formData.append('file', document.getElementById('file').files[0]);
                formData.append('chunk_size', document.getElementById('chunk_size').value);
                formData.append('overlap', document.getElementById('overlap').value);
                formData.append('chunker_type', document.getElementById('chunker_type').value);

                try {
                    const response = await fetch('/chunk', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();

                    if (response.ok) {
                        displayResults(data);
                    } else {
                        resultsDiv.innerHTML = `<div style="color: red;">❌ Error: ${data.detail}</div>`;
                    }
                } catch (error) {
                    resultsDiv.innerHTML = `<div style="color: red;">❌ Error: ${error.message}</div>`;
                } finally {
                    submitBtn.disabled = false;
                    submitBtn.textContent = '🚀 Chunk Document';
                }
            });

            function displayResults(data) {
                const resultsDiv = document.getElementById('results');

                let html = '<h2 style="color: #667eea; margin-bottom: 20px;">📊 Results</h2>';

                // Stats
                html += '<div class="stats">';
                html += `
                    <div class="stat-card">
                        <div class="stat-value">${data.processing_time_ms.toFixed(2)}</div>
                        <div class="stat-label">⏱️ Processing Time (ms)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${data.total_chunks}</div>
                        <div class="stat-label">📄 Total Chunks</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${data.total_tokens.toLocaleString()}</div>
                        <div class="stat-label">🔢 Total Tokens</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${data.avg_tokens_per_chunk.toFixed(1)}</div>
                        <div class="stat-label">📊 Avg Tokens/Chunk</div>
                    </div>
                `;
                html += '</div>';

                // Chunks
                html += '<h3 style="margin: 30px 0 15px 0; color: #333;">📝 Chunks</h3>';

                data.chunks.forEach((chunk, index) => {
                    const text = chunk.text.substring(0, 500) + (chunk.text.length > 500 ? '...' : '');
                    html += `
                        <div class="chunk">
                            <div class="chunk-header">
                                <div class="chunk-title">Chunk #${index + 1}</div>
                                <div class="chunk-meta">
                                    ${chunk.token_count} tokens |
                                    Chars ${chunk.start_index}-${chunk.end_index}
                                </div>
                            </div>
                            <div class="chunk-text">${escapeHtml(text)}</div>
                        </div>
                    `;
                });

                resultsDiv.innerHTML = html;
            }

            function escapeHtml(text) {
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }
        </script>
    </body>
    </html>
    """


@app.post("/chunk", response_model=ChunkingResult)
async def chunk_pdf(
    file: UploadFile = File(...),
    chunk_size: int = Form(512),
    overlap: int = Form(80),
    chunker_type: str = Form("recursive")
):
    """
    Chunk a PDF file and return the results with timing.

    Args:
        file: PDF file to chunk
        chunk_size: Target chunk size in tokens
        overlap: Overlap between chunks in tokens
        chunker_type: Type of chunker (recursive, token)
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Validate parameters
    if chunk_size < 100 or chunk_size > 2000:
        raise HTTPException(status_code=400, detail="Chunk size must be between 100 and 2000")

    if overlap < 0 or overlap >= chunk_size:
        raise HTTPException(status_code=400, detail="Overlap must be between 0 and chunk_size")

    if chunker_type not in ['recursive', 'token']:
        raise HTTPException(status_code=400, detail="Chunker type must be 'recursive' or 'token'")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name

    try:
        # Chunk with FULL METADATA extraction
        enhanced_result = chunk_document_with_metadata(
            file_path=tmp_path,
            chunk_size=chunk_size,
            overlap_tokens=overlap,
            tokenizer="cl100k_base",  # GPT-4 tokenizer
            chunker_type=chunker_type,
            # FORMAT OPTIONS ⭐
            recipe="markdown",  # Store in markdown format for Pinecone
            preserve_formatting=True,
            # ENABLE ALL METADATA EXTRACTION ⭐
            extract_metadata=True,
            detect_chapters=True,
            detect_fonts=True,
            detect_structure=True,
            pinecone_format=False  # Keep as regular format for testing
        )

        # Build response
        result = ChunkingResult(
            source_file=file.filename,
            processing_time_ms=enhanced_result['stats']['processing_time_ms'],
            chunk_size=chunk_size,
            overlap=overlap,
            chunker_type=chunker_type,
            total_chunks=enhanced_result['stats']['total_chunks'],
            total_tokens=enhanced_result['stats']['total_tokens'],
            avg_tokens_per_chunk=enhanced_result['stats']['avg_tokens_per_chunk'],
            chunks=enhanced_result['chunks'],
            metadata=enhanced_result.get('metadata', {}),
            stats=enhanced_result.get('stats', {})
        )

        # Save JSON to outputs directory
        import json
        output_dir = Path("chunking_outputs")
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / f"{Path(file.filename).stem}_chunks.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result.dict(), f, indent=2, ensure_ascii=False)

        print(f"\n💾 Saved JSON to: {output_file.absolute()}")

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chunking failed: {str(e)}")

    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Chonkie OSS Test API"}


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 Starting Chonkie OSS Test API")
    print("="*80)
    print("\n📍 Open in browser: http://localhost:8000")
    print("📚 API docs: http://localhost:8000/docs")
    print("\n💡 Upload a PDF to test Chonkie OSS chunking!\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
