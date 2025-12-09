from sentence_transformers import SentenceTransformer, util
import numpy as np
import requests
import re
from typing import List
import time

from app.models import PlagiarismResult, Match

class PlagiarismChecker:
    def __init__(self):
        print("Loading AI model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully!")
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PlagiarismChecker/1.0 (Educational Project)'
        })
    
    def check(self, text: str) -> PlagiarismResult:
        """
        Main plagiarism checking function
        """
        sentences = self._split_into_sentences(text)
        
        matches = []
        total_similarity = 0
        checked_sentences = 0
        
        print(f"\n🔍 Analyzing {len(sentences)} sentences...")
        
        for idx, sentence in enumerate(sentences):
            if len(sentence.strip()) < 20:
                continue
            
            checked_sentences += 1
            print(f"\nChecking sentence {idx + 1}: {sentence[:60]}...")
            
            # Extract key phrases (first 5-8 words are usually most important)
            key_phrase = ' '.join(sentence.split()[:8])
            
            # Search Wikipedia
            wiki_result = self._search_wikipedia(key_phrase, sentence)
            
            if wiki_result:
                print(f"✅ MATCH FOUND: {wiki_result['similarity']}% - {wiki_result['source']}")
                match = Match(
                    source=wiki_result['url'],
                    similarity=wiki_result['similarity'],
                    matchedText=sentence,
                    startIndex=text.find(sentence),
                    endIndex=text.find(sentence) + len(sentence)
                )
                matches.append(match)
                total_similarity += wiki_result['similarity']
            else:
                print(f"❌ No match found")
            
            # Small delay to avoid rate limiting
            time.sleep(0.3)
        
        # Calculate scores
        if checked_sentences > 0:
            avg_similarity = (len(matches) / checked_sentences) * 100
            plagiarism_score = round(avg_similarity, 2)
        else:
            plagiarism_score = 0.0
        
        originality_score = round(100 - plagiarism_score, 2)
        
        print(f"\n📊 Final Score: {plagiarism_score}% plagiarized, {len(matches)} matches found")
        
        # Word and character counts
        word_count = len(text.split())
        char_count = len(text)
        
        return PlagiarismResult(
            plagiarismScore=plagiarism_score,
            originalityScore=originality_score,
            matches=matches,
            text=text,
            wordCount=word_count,
            characterCount=char_count,
            sources=len(matches)
        )
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _search_wikipedia(self, query: str, full_sentence: str) -> dict:
        """
        Search Wikipedia API with better matching
        """
        try:
            # Use Wikipedia REST API
            search_url = "https://en.wikipedia.org/w/rest.php/v1/search/page"
            
            params = {
                "q": query,
                "limit": 3  # Get top 3 results
            }
            
            response = self.session.get(search_url, params=params, timeout=10)
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            
            # Check if we got results
            if 'pages' in data and len(data['pages']) > 0:
                best_match = None
                best_similarity = 0
                
                # Check top 3 results
                for page in data['pages'][:3]:
                    title = page.get('title', '')
                    description = page.get('description', '')
                    
                    print(f"  → Found: '{title}' - {description}")
                    
                    # Get page content
                    page_key = page.get('key', '')
                    content = self._get_wikipedia_content_by_title(page_key)
                    
                    if content:
                        # Calculate similarity
                        similarity = self._calculate_similarity(full_sentence, content)
                        
                        print(f"     Similarity: {similarity * 100:.1f}%")
                        
                        # Lower threshold for famous quotes
                        threshold = 0.45  # 45% threshold (was 70%)
                        
                        if similarity > threshold and similarity > best_similarity:
                            best_similarity = similarity
                            best_match = {
                                'url': f"https://en.wikipedia.org/wiki/{page_key}",
                                'similarity': round(similarity * 100, 2),
                                'source': title
                            }
                
                return best_match
            
            return None
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            return None
    
    def _get_wikipedia_content_by_title(self, page_key: str) -> str:
        """
        Get Wikipedia page content by page key
        """
        try:
            # Use Wikipedia REST API to get page summary
            content_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_key}"
            
            response = self.session.get(content_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                extract = data.get('extract', '')
                
                # Also get the full text if available
                full_text = data.get('extract_html', '')
                
                # Combine for better matching
                combined = extract + " " + full_text
                return combined[:1000]  # First 1000 characters
            
            return None
            
        except Exception as e:
            return None
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity using BERT with better preprocessing
        """
        try:
            if not text1 or not text2:
                return 0.0
            
            # Clean and normalize text
            text1_clean = text1.lower().strip()
            text2_clean = text2.lower().strip()
            
            # Encode both texts
            embeddings = self.model.encode([text1_clean, text2_clean])
            
            # Calculate cosine similarity
            similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
            
            return float(similarity)
            
        except Exception as e:
            print(f"  ❌ Similarity error: {str(e)}")
            return 0.0
