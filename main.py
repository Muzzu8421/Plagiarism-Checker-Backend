import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
from docx import Document
import io
import requests
import re
from typing import List, Tuple

app = FastAPI(
    title="Plagiarism Checker API",
    description="AI-powered plagiarism detection with Wikipedia + Web",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 10 * 1024 * 1024

print("=" * 50)
print("Loading AI model...")
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model loaded!")
except Exception as e:
    print(f"❌ Error: {e}")
print("=" * 50)

def clean_text(text):
    """Clean text"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

def extract_text_from_txt(content):
    """Extract from TXT"""
    try:
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                return clean_text(content.decode(encoding))
            except:
                continue
        return clean_text(content.decode('utf-8', errors='ignore'))
    except Exception as e:
        raise HTTPException(400, f"TXT error: {str(e)}")

def extract_text_from_pdf(content):
    """Extract from PDF"""
    try:
        pdf_file = io.BytesIO(content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if not text.strip():
            raise HTTPException(400, "PDF is empty")
        return clean_text(text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"PDF error: {str(e)}")

def extract_text_from_docx(content):
    """Extract from DOCX"""
    try:
        docx_file = io.BytesIO(content)
        doc = Document(docx_file)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n".join(paragraphs)
        if not text.strip():
            raise HTTPException(400, "DOCX is empty")
        return clean_text(text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"DOCX error: {str(e)}")

def extract_text_from_file(file: UploadFile):
    """Extract text from file"""
    filename = file.filename.lower()
    print(f"\n{'='*50}")
    print(f"📁 File: {file.filename}")
    
    content = file.file.read()
    file_size = len(content)
    print(f"📏 Size: {file_size:,} bytes")
    
    if file_size > MAX_FILE_SIZE or file_size == 0:
        raise HTTPException(400, "Invalid file size")
    
    if filename.endswith('.txt'):
        return extract_text_from_txt(content)
    elif filename.endswith('.pdf'):
        return extract_text_from_pdf(content)
    elif filename.endswith(('.docx', '.doc')):
        return extract_text_from_docx(content)
    else:
        raise HTTPException(400, "Unsupported format")

def search_wikipedia_proper(query: str) -> List[Tuple[str, str]]:
    """
    Proper Wikipedia search using MediaWiki API
    This is the CORRECT way to search Wikipedia!
    """
    results = []
    
    try:
        # Step 1: Search for relevant articles
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            'action': 'query',
            'list': 'search',
            'srsearch': query,
            'srlimit': 3,
            'format': 'json'
        }
        
        response = requests.get(search_url, params=search_params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            search_results = data.get('query', {}).get('search', [])
            
            # Step 2: Get full content for each article
            for result in search_results[:2]:  # Top 2 results
                page_title = result.get('title', '')
                
                if page_title:
                    # Get article extract
                    extract_params = {
                        'action': 'query',
                        'titles': page_title,
                        'prop': 'extracts',
                        'exintro': True,
                        'explaintext': True,
                        'format': 'json'
                    }
                    
                    extract_response = requests.get(search_url, params=extract_params, timeout=5)
                    
                    if extract_response.status_code == 200:
                        extract_data = extract_response.json()
                        pages = extract_data.get('query', {}).get('pages', {})
                        
                        for page_id, page_info in pages.items():
                            extract = page_info.get('extract', '')
                            if extract and len(extract) > 100:
                                page_url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
                                results.append((extract, page_url))
                                print(f"  ✅ Wikipedia: {page_title}")
    
    except Exception as e:
        print(f"  ⚠️  Wikipedia error: {e}")
    
    return results

def search_duckduckgo_backup(query: str) -> List[Tuple[str, str]]:
    """
    DuckDuckGo as backup (for non-Wikipedia content)
    """
    results = []
    
    try:
        url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': 1,
            'skip_disambig': 1
        }
        
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            # Abstract (often from Wikipedia)
            abstract = data.get('Abstract', '')
            abstract_url = data.get('AbstractURL', '')
            if abstract and len(abstract) > 100:
                results.append((abstract, abstract_url))
                print(f"  ✅ DDG: {abstract_url[:50]}")
            
            # Related topics
            for topic in data.get('RelatedTopics', [])[:2]:
                if isinstance(topic, dict):
                    text = topic.get('Text', '')
                    url = topic.get('FirstURL', '')
                    if text and len(text) > 50:
                        results.append((text, url))
    
    except Exception as e:
        print(f"  ⚠️  DDG error: {e}")
    
    return results

def search_combined(sentence: str) -> List[Tuple[str, str]]:
    """
    HYBRID APPROACH:
    1. Try Wikipedia first (best for academic content)
    2. Fall back to DuckDuckGo (for web content)
    """
    # Extract meaningful keywords
    words = [w for w in sentence.split() if len(w) > 3][:8]
    if not words:
        return []
    
    query = ' '.join(words)
    print(f"  🔍 Searching: '{query[:60]}'")
    
    all_results = []
    
    # Priority 1: Wikipedia (academic sources)
    wiki_results = search_wikipedia_proper(query)
    all_results.extend(wiki_results)
    
    # Priority 2: DuckDuckGo (if Wikipedia found nothing or for additional sources)
    if len(all_results) < 2:
        ddg_results = search_duckduckgo_backup(query)
        all_results.extend(ddg_results)
    
    # Try shorter query if still no results
    if not all_results and len(words) > 4:
        shorter_query = ' '.join(words[:4])
        print(f"  🔍 Trying: '{shorter_query}'")
        wiki_results = search_wikipedia_proper(shorter_query)
        all_results.extend(wiki_results)
    
    return all_results[:3]  # Max 3 sources per sentence

@app.get("/")
def home():
    """Health check"""
    return {
        "status": "✅ Plagiarism Checker API",
        "version": "2.1.0",
        "search": "Wikipedia + DuckDuckGo",
        "model": "all-MiniLM-L6-v2"
    }

@app.post("/check")
async def check_plagiarism(file: UploadFile = File(...)):
    """Check plagiarism with Wikipedia priority + web backup"""
    try:
        if not file.filename:
            raise HTTPException(400, "No file")
        
        # Extract text
        text = extract_text_from_file(file)
        
        if len(text.strip()) < 50:
            raise HTTPException(400, "Text too short")
        
        print(f"✅ Text: {len(text)} chars, {len(text.split())} words")
        
        # Get sentences (filter out junk)
        sentences = []
        for sent in re.split(r'[.!?]+', text):
            sent = sent.strip()
            # Only real content: 40+ chars, 5+ words, not headers
            if (len(sent) > 40 and 
                len(sent.split()) > 5 and
                not re.match(r'^(page|section|chapter|department|year|\d+|signature|date|name)', sent.lower())):
                sentences.append(sent)
        
        sentences = sentences[:12]  # Analyze 12 sentences
        
        if not sentences:
            raise HTTPException(400, "No valid sentences")
        
        print(f"📊 Analyzing {len(sentences)} sentences")
        
        # Encode sentences
        print("🤖 AI encoding...")
        sentence_embeddings = model.encode(sentences, show_progress_bar=False)
        
        # Check plagiarism
        matches = []
        checked = 0
        
        print("\n" + "="*50)
        print("🔍 PLAGIARISM DETECTION")
        print("="*50)
        
        for i, sentence in enumerate(sentences):
            print(f"\n📝 [{i+1}/{len(sentences)}] {sentence[:60]}...")
            
            # Search with hybrid approach
            sources = search_combined(sentence)
            
            if sources:
                checked += 1
                
                for source_text, source_url in sources:
                    if len(source_text) < 50:
                        continue
                    
                    # Calculate similarity
                    source_embedding = model.encode([source_text], show_progress_bar=False)
                    similarity = cosine_similarity([sentence_embeddings[i]], source_embedding)[0][0]
                    similarity_percent = round(float(similarity * 100), 1)
                    
                    print(f"  📊 {similarity_percent}% similar")
                    
                    # Match threshold: 65%
                    if similarity > 0.65:
                        matches.append({
                            "matchedText": sentence[:300],
                            "similarity": similarity_percent,
                            "source": source_url or "Web source"
                        })
                        print(f"  ✅ PLAGIARISM DETECTED!")
                        break  # One match per sentence
        
        print("\n" + "="*50)
        print(f"✅ Checked: {checked}/{len(sentences)} sentences")
        print(f"✅ Matches found: {len(matches)}")
        
        # Calculate final scores
        if matches:
            match_ratio = len(matches) / len(sentences)
            avg_similarity = sum(m['similarity'] for m in matches) / len(matches)
            plagiarism_score = round(match_ratio * avg_similarity, 1)
        else:
            plagiarism_score = 0.0
        
        originality_score = round(100 - plagiarism_score, 1)
        
        print(f"\n🎯 RESULTS:")
        print(f"   Plagiarism: {plagiarism_score}%")
        print(f"   Originality: {originality_score}%")
        print("="*50 + "\n")
        
        return {
            "plagiarismScore": plagiarism_score,
            "originalityScore": originality_score,
            "text": text[:10000],
            "matches": matches
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🚀 Plagiarism Checker v2.1")
    print("📚 Wikipedia Priority + Web Backup")
    print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=7860)