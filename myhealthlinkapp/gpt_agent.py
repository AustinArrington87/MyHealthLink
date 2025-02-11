import pytesseract
from pdf2image import convert_from_path
import re
from PIL import Image, ImageDraw
import io
import spacy
import numpy as np
import logging
from typing import List, Dict, Optional, Set, Union
from dataclasses import dataclass
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor
import time
from openai import OpenAI
import deepl
from tenacity import retry, stop_after_attempt, wait_exponential

# additional installations needed, NLP and OCR 
# brew install tesseract
# brew install tesseract-lang
# python3.11 -m spacy download en_core_web_sm

# Initialize OpenAI client
client = OpenAI(
    api_key="EnterKey",
    organization="EnterKey"
)
translation_api_key="EnterDeepLKey"
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('HealthRecordAnalyzer')

class PIIScrubber:
    def __init__(self, config=None):
        # Load SpaCy model for named entity recognition
        self.nlp = spacy.load("en_core_web_sm")
        logger.info("Successfully loaded SpaCy model")
        
        # Enhanced PII patterns to catch more variations
        self.pii_patterns = {
            'mrn': r'(?:MRN:?\s*|Medical Record Number:?\s*|Record Number:?\s*|#:?\s*)\d{5,}',
            'dob': r'(?:DOB:?\s*|Date of Birth:?\s*|Birth Date:?\s*|Born:?\s*)\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
            'name_pattern': r'(?:Name:?\s*|Patient:?\s*|Dr\.?:?\s*|Doctor:?\s*|PCP:?\s*|Dear\s+|Sincerely,?\s+)[A-Z][a-z]+(?:[-\s][A-Z][a-z]+)*(?:\s*,\s*[A-Z][A-Za-z.]*)?',
            'simple_name': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Catches standalone full names
            'provider_name': r'(?:Provider:?\s*|MD:?\s*|DO:?\s*|NP:?\s*|PA:?\s*)[A-Z][a-z]+(?:[-\s][A-Z][a-z]+)*(?:\s*,\s*[A-Z][A-Za-z.]*)?',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(?:\+\d{1,2}\s)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            'ssn': r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',
            'address': r'\d{1,5}\s+[A-Za-z0-9\s.,-]+(?:street|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|parkway|pkwy|circle|cir|boulevard|blvd)\b',
            'provider_id': r'(?:NPI:?\s*|Provider ID:?\s*|License:?\s*)\d{10}',
            'legal_name': r'Legal Name:?\s*[A-Z][a-z]+(?:[-\s][A-Z][a-z]+)*',
            # Add these new patterns
            'capitalized_name': r'\b[A-Z][a-z]+(?:[-\s][A-Z][a-z]+)+\b',  # Catches names like "Austin Arrington"
            'heme_profile': r'(?:Profile for|Results for|Patient:|Name:)\s*[A-Z][a-z]+(?:[-\s][A-Z][a-z]+)*',
            'doctor_full': r'(?:Dr\.|Doctor)\s+[A-Z][a-z]+(?:[-\s][A-Z][a-z]+)*'
        }
        self.config = config

    def _text_to_image_coords(self, start_char: int, end_char: int, text: str, image_shape: tuple) -> tuple:
        """Convert text position to approximate image coordinates"""
        height, width = image_shape[:2]
        
        # Calculate relative position in document
        total_chars = len(text)
        relative_start = start_char / total_chars
        relative_end = end_char / total_chars
        
        # Calculate vertical position (assume text takes up middle 80% of image)
        vertical_margin = height * 0.1  # 10% margin top and bottom
        usable_height = height * 0.8
        y_position = vertical_margin + (relative_start * usable_height)
        
        # Calculate horizontal position (assume text takes up middle 90% of image)
        horizontal_margin = width * 0.05  # 5% margin left and right
        usable_width = width * 0.9
        x_start = horizontal_margin + (relative_start * usable_width)
        x_end = horizontal_margin + (relative_end * usable_width)
        
        # Calculate box dimensions
        box_height = height * 0.05  # 5% of image height
        y1 = max(0, y_position - (box_height / 2))
        y2 = min(height, y_position + (box_height / 2))
        
        # Ensure x2 > x1 and coordinates are within bounds
        x1 = max(0, min(x_start, width - 1))
        x2 = max(x1 + 1, min(x_end, width))
        
        # Convert to integers
        return (int(x1), int(y1), int(x2), int(y2))

    def detect_pii_regions(self, text: str, image_shape: tuple) -> List[tuple]:
        """Detect regions containing PII in text"""
        pii_regions = []
        entities_found = set()
        
        # Use SpaCy for named entity recognition
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC']:
                region = self._text_to_image_coords(ent.start_char, ent.end_char, text, image_shape)
                pii_regions.append(region)
                entities_found.add(ent.text)
        
        # Use regex patterns to find additional PII
        for pattern_name, pattern in self.pii_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                matched_text = match.group()
                if matched_text not in entities_found:
                    region = self._text_to_image_coords(match.start(), match.end(), text, image_shape)
                    pii_regions.append(region)
                    entities_found.add(matched_text)
        
        return self._merge_overlapping_regions(pii_regions)

    def _merge_overlapping_regions(self, regions: List[tuple]) -> List[tuple]:
        """Merge overlapping redaction regions"""
        if not regions:
            return regions
            
        regions.sort(key=lambda x: (x[0], x[1]))
        merged = [regions[0]]
        
        for current in regions[1:]:
            previous = merged[-1]
            if current[0] <= previous[2]:  # Regions overlap
                merged[-1] = (
                    min(previous[0], current[0]),
                    min(previous[1], current[1]),
                    max(previous[2], current[2]),
                    max(previous[3], current[3])
                )
            else:
                merged.append(current)
        
        return merged

    def redact_pii(self, file_path: str) -> Optional[io.BytesIO]:
        """Redact PII from the given file"""
        try:
            # Handle both PDF and image files
            if file_path.lower().endswith('.pdf'):
                pages = convert_from_path(file_path)
                image = pages[0]  # Process first page for now
                logger.info(f"Successfully converted PDF to image: {file_path}")
            else:
                image = Image.open(file_path)
                logger.info(f"Successfully opened image file: {file_path}")

            # Convert to numpy array for processing
            img_array = np.array(image)
            
            # Extract text using OCR
            text = pytesseract.image_to_string(image)
            confidence = float(pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)['conf'].mean())
            
            if confidence < 75.0:  # Configurable threshold
                logger.warning(f"Low OCR confidence: {confidence:.2f}%")
            
            # Detect PII regions
            pii_regions = self.detect_pii_regions(text, img_array.shape)
            
            # Create a copy of the image for redaction
            redacted_image = image.copy()
            draw = ImageDraw.Draw(redacted_image)
            
            # Redact detected regions
            for region in pii_regions:
                draw.rectangle(region, fill='black')
            
            # Save to bytes buffer with proper filename
            buffer = io.BytesIO()
            redacted_image.save(buffer, format='PNG')
            buffer.seek(0)
            
            # Store the original filename for later use
            filename = os.path.splitext(os.path.basename(file_path))[0] + '.png'
            buffer.name = filename
            
            logger.info(f"Successfully redacted {len(pii_regions)} PII regions in {file_path}")
            return buffer

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            raise

class MedicalTranslator:
    def __init__(self, api_key):
        self.translator = deepl.Translator(api_key)
        
    def translate(self, text: str, target_lang: str) -> str:
        try:
            # Map our language names to DeepL codes
            lang_map = {
                'spanish': 'ES',
                'french': 'FR',
                'german': 'DE',
                'italian': 'IT',
                'japanese': 'JA',
                'chinese': 'ZH'
            }
            
            target_lang_code = lang_map.get(target_lang.lower())
            if not target_lang_code:
                raise ValueError(f"Unsupported language: {target_lang}")
                
            result = self.translator.translate_text(text, target_lang=target_lang_code)
            return result.text
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text  # Return original text if translation fails

def extract_section(text: str, section_name: str) -> str:
    """Extract specific sections from the analysis text"""
    try:
        # Section markers
        markers = {
            "Synopsis": [
                "### Synopsis", "## Synopsis", "Synopsis:", "Summary:", 
                "### Summary:", "## Summary", "### Overview:", "## Overview",
                "Overview:", "Analysis:", "### Analysis:", "## Analysis"
            ],
            "Insights and Anomalies": [
                "### Insights and Anomalies", "## Insights and Anomalies",
                "Insights and Anomalies:", "### Insights:", "## Insights",
                "Insights or Anomalies:", "**Insights and Anomalies:**",
                "## Insights and Anomalies", "**Insights or Anomalies:**"
            ],
            "Citations": [
                "### Citations", "## Citations", "Citations:", 
                "References:", "### References:", "## References",
                "Research Citations:", "### Research Citations:",
                "**Citations:**"
            ]
        }
        
        # Look for section content
        current_markers = markers.get(section_name, [])
        section_text = ""
        
        for marker in current_markers:
            if marker in text:
                start = text.find(marker) + len(marker)
                # Find next section marker
                next_markers = []
                for m_list in markers.values():
                    next_markers.extend(m_list)
                ends = []
                for next_marker in next_markers:
                    pos = text.find(next_marker, start)
                    if pos != -1:
                        ends.append(pos)
                
                if ends:
                    end = min(ends)
                else:
                    end = len(text)
                
                section_text = text[start:end].strip()
                break
        
        # Clean markdown and other formatting
        if section_text:
            # Remove markdown formatting
            section_text = re.sub(r'\*\*(.+?)\*\*', r'\1', section_text)  # Bold
            section_text = re.sub(r'\*(.+?)\*', r'\1', section_text)      # Italic
            section_text = re.sub(r'_(.+?)_', r'\1', section_text)        # Underscore
            section_text = re.sub(r'`(.+?)`', r'\1', section_text)        # Code
            # Remove multiple newlines
            section_text = re.sub(r'\n\s*\n', '\n', section_text)
            # Remove leading/trailing whitespace
            section_text = section_text.strip()
        
        if not section_text:
            return f"No {section_name.lower()} available."
        
        return section_text
        
    except Exception as e:
        logger.error(f"Error extracting {section_name}: {e}")
        return f"Error processing {section_name.lower()} section"

# Add this helper function for retrying file uploads
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def upload_file_with_retry(client, file_buffer):
    """Upload a file to OpenAI with retry logic"""
    try:
        response = client.files.create(
            file=file_buffer,
            purpose="assistants"
        )
        return response
    except Exception as e:
        logger.error(f"File upload attempt failed: {str(e)}")
        raise

def analyze_health_records(file_paths: List[str], target_language: str = None, config=None) -> Dict:
    """Process and analyze health records with PII scrubbing and optional translation"""
    logger.info(f"Starting analysis of {len(file_paths)} files")
    
    # First, preprocess files to remove PII
    processed_files = []
    scrubber = PIIScrubber(config)
    
    # Process files with progress bar
    for file_path in tqdm(file_paths, desc="Processing files", unit="file"):
        try:
            processed_file = scrubber.redact_pii(file_path)
            if processed_file:
                processed_files.append((file_path, processed_file))
            else:
                logger.error(f"Failed to process {file_path}")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
    
    if not processed_files:
        return {
            'success': False,
            'error': 'No files were successfully processed'
        }

    # Upload processed files to OpenAI with retry logic
    image_files = []
    
    for original_path, processed_file in processed_files:
        try:
            logger.info(f"Uploading processed file: {processed_file.name}")
            
            with io.BytesIO(processed_file.getvalue()) as temp_buffer:
                temp_buffer.name = processed_file.name
                # Use retry logic for file upload
                upload_response = upload_file_with_retry(client, temp_buffer)
            
            image_files.append(upload_response.id)
            logger.info(f"Successfully uploaded file with ID: {upload_response.id}")
            
            # Add a small delay between file uploads
            time.sleep(2)
                
        except Exception as e:
            logger.error(f"Failed to upload processed version of {original_path}: {e}")
            continue

    if not image_files:
        return {
            'success': False,
            'error': 'No files were successfully uploaded to OpenAI'
        }

    # Create and process the analysis request
    try:
        # Create thread
        thread = client.beta.threads.create()
        
        # Prepare prompt with multi-file context
        prompt = (
            "Analyze these health records and provide: \n"
            "1. A detailed synopsis\n"
            "2. Insights and anomalies with clinical significance\n"
            "3. Relevant research citations that support the key insights\n\n"
        )
        
        if len(file_paths) > 1:
            prompt += (
                "These documents are related and should be analyzed together. "
                "Please provide a comprehensive analysis that considers all documents "
                "and highlights any relationships or patterns between them. "
            )

        # Create message content with text and file references
        message_content = [
            {
                "type": "text",
                "text": prompt
            }
        ]
        
        # Add files as image_file content with delay between each
        for file_id in image_files:
            message_content.append({
                "type": "image_file",
                "image_file": {
                    "file_id": file_id
                }
            })
            time.sleep(1)  # Small delay between adding files

        # Send message
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=message_content
        )

        # Run analysis with increased timeout
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id="asst_1pBzntEcrVPWsbztmDtG9Hap"
        )

        # Wait for completion with longer timeout
        start_time = time.time()
        timeout = 600  # 10 minutes timeout for multiple files
        
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError("Analysis timed out after 10 minutes")
                
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            
            if run_status.status == "completed":
                break
            elif run_status.status == "failed":
                raise Exception(f"Analysis failed: {run_status.last_error}")
                
            logger.info("Waiting for response...")
            time.sleep(5)

        # Get results
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        
        if not messages.data:
            return {
                'success': False,
                'error': 'No response received from OpenAI'
            }

        # Extract content
        latest_message = messages.data[0].content
        if isinstance(latest_message, list):
            analysis_text = ' '.join(
                block.text.value if hasattr(block, 'text') and hasattr(block.text, 'value')
                else str(block)
                for block in latest_message
            )
        else:
            analysis_text = str(latest_message)

        # Extract sections
        synopsis = extract_section(analysis_text, "Synopsis")
        insights_anomalies = extract_section(analysis_text, "Insights and Anomalies")
        citations = extract_section(analysis_text, "Citations")

        # Get the analysis results first
        analysis_result = {
            'success': True,
            'result': {
                'synopsis': synopsis,
                'insights_anomalies': insights_anomalies,
                'citations': citations,
                'raw_text': analysis_text
            }
        }

        # Only attempt translation if we have successful analysis AND a target language
        if analysis_result['success'] and target_language and target_language != "":
            try:
                translator = MedicalTranslator(api_key=translation_api_key)
                
                # Translate each section
                analysis_result['result']['synopsis'] = translator.translate(
                    analysis_result['result']['synopsis'], 
                    target_language
                )
                analysis_result['result']['insights_anomalies'] = translator.translate(
                    analysis_result['result']['insights_anomalies'], 
                    target_language
                )
                analysis_result['result']['citations'] = translator.translate(
                    analysis_result['result']['citations'], 
                    target_language
                )
                
                logger.info(f"Successfully translated analysis to {target_language}")
                
            except Exception as e:
                logger.error(f"Translation failed: {str(e)}")
                # Continue with original text if translation fails
                pass

        return analysis_result

    except Exception as e:
        error_msg = f"Error in analysis: {str(e)}"
        logger.error(error_msg)
        if "rate_limit" in str(e).lower():
            error_msg = "Rate limit exceeded. Please try again in a few minutes."
        elif "authentication" in str(e).lower():
            error_msg = "Authentication error. Please check your API key."
        elif "permission" in str(e).lower():
            error_msg = "Permission denied. Please check your API access settings."
        return {
            'success': False,
            'error': error_msg
        }
