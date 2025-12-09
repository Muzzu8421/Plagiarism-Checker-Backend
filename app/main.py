from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import tempfile
from typing import Optional

from app.plagiarism_checker import PlagiarismChecker
from app.models import PlagiarismResult
from app.utils.file_processor import FileProcessor

app = FastAPI(
    title="Plagiarism Checker API",
    description="AI-powered plagiarism detection API",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize plagiarism checker
plagiarism_checker = PlagiarismChecker()
file_processor = FileProcessor()

@app.get("/")
async def root():
    return {
        "message": "Plagiarism Checker API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/check", response_model=PlagiarismResult)
async def check_plagiarism(file: UploadFile = File(...)):
    """
    Check uploaded file for plagiarism
    Accepts: PDF, DOCX, TXT files
    """
    
    # Validate file type
    allowed_extensions = {'.pdf', '.docx', '.doc', '.txt'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Validate file size (10MB max)
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(
            status_code=400,
            detail="File size exceeds 10MB limit"
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Extract text from file
        text = file_processor.extract_text(tmp_file_path, file_ext)
        
        if not text or len(text.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="Document text is too short or couldn't be extracted"
            )
        
        # Check for plagiarism
        result = plagiarism_checker.check(text)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return result
        
    except Exception as e:
        # Clean up on error
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )

@app.post("/check-text", response_model=PlagiarismResult)
async def check_text_plagiarism(text: str):
    """
    Check plain text for plagiarism
    """
    if not text or len(text.strip()) < 50:
        raise HTTPException(
            status_code=400,
            detail="Text is too short (minimum 50 characters)"
        )
    
    try:
        result = plagiarism_checker.check(text)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error checking text: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)