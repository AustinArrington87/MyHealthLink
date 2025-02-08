import openai
import time

# OpenAI API Credentials
openai.api_key = ""
openai.organization = ""

def extract_section(text, section_name):
    """
    Flexible helper function to extract sections from any type of analysis text.
    Handles various document types (medical records, blood tests, etc.)
    """
    try:
        # Common section markers that might appear in different types of analyses
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
        
        current_markers = markers.get(section_name, [])
        section_text = ""
        
        # Try to find the section and its content
        for marker in current_markers:
            if marker in text:
                start = text.find(marker) + len(marker)
                # Look for the next section marker
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
                    # Look for common ending phrases
                    ending_phrases = ["These citations", "This summary", "In conclusion"]
                    end_positions = [text.find(phrase, start) for phrase in ending_phrases]
                    end_positions = [pos for pos in end_positions if pos != -1]
                    end = min(end_positions) if end_positions else len(text)
                
                section_text = text[start:end].strip()
                break
        
        if not section_text and section_name == "Insights and Anomalies":
            # Try to find insights section by looking for numbered points
            lines = text.split('\n')
            in_insights = False
            insights_lines = []
            for line in lines:
                if any(marker in line for marker in current_markers):
                    in_insights = True
                    continue
                if in_insights and line.strip() and not line.startswith('###') and not line.startswith('##'):
                    insights_lines.append(line.strip())
            if insights_lines:
                section_text = '\n'.join(insights_lines)

        def clean_text(text):
            if not text:
                return ""
            
            # Remove markdown formatting
            text = text.replace('**', '')
            text = text.replace('###', '')
            text = text.replace('##', '')
            
            # Split into lines and process each line
            lines = text.split('\n')
            cleaned_lines = []
            for line in lines:
                # Handle numbered points
                if line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                    line = line.split('.', 1)[1] if '.' in line else line
                if line.strip():  # Only add non-empty lines
                    cleaned_lines.append(line.strip())
            
            # Join lines back together, preserving paragraph structure
            text = '\n\n'.join(cleaned_lines)
            return text.strip()
        
        cleaned_text = clean_text(section_text)
        if not cleaned_text:
            if section_name == "Synopsis":
                return "No synopsis available."
            elif section_name == "Insights and Anomalies":
                return "No significant findings to report."
            elif section_name == "Citations":
                return "No citations provided."
            else:
                return "No content available."
        
        return cleaned_text
        
    except Exception as e:
        print(f"Error extracting {section_name}: {e}")
        return f"Error processing {section_name} section"

def analyze_health_records(file_paths):
    """
    Sends image/PDF files to OpenAI API for analysis.
    """
    # Determine if single or multi-file analysis
    prompt = (
        "Analyze these health records and provide: \n"
        "1. A detailed synopsis\n"
        "2. Insights and anomalies with clinical significance\n"
        "3. Relevant research citations that support the key insights\n\n"
        "Format the response with clear sections for Synopsis, Insights and Anomalies, and Citations. "
        "For citations, include recent peer-reviewed research that supports the clinical insights provided. "
        "If analyzing multiple documents, include cross-document insights."
        if len(file_paths) > 1
        else "Analyze this health record and provide: \n"
        "1. A detailed synopsis\n"
        "2. Insights and anomalies with clinical significance\n"
        "3. Relevant research citations that support the key insights\n\n"
        "Format the response with clear sections for Synopsis, Insights and Anomalies, and Citations. "
        "For citations, include recent peer-reviewed research that supports the clinical insights provided."
    )

    # Upload files to OpenAI
    uploaded_files = []
    image_files = []
    for file_path in file_paths:
        with open(file_path, "rb") as f:
            is_image = file_path.endswith(('.png', '.jpg', '.jpeg'))
            upload_response = openai.files.create(
                purpose="vision" if is_image else "assistants",
                file=f
            )
            if is_image:
                image_files.append(upload_response.id)
            else:
                uploaded_files.append({"id": upload_response.id, "type": "file_search"})
    
    print(f"Uploaded files: {uploaded_files}, Image Files: {image_files}")

    # Create a new thread
    thread = openai.beta.threads.create()

    # Attach PDFs with correct tools (file_search)
    attachments = [{"file_id": file["id"], "tools": [{"type": "file_search"}]} for file in uploaded_files]

    # Create a message (text + images directly in content)
    message_content = [
        {"type": "text", "text": prompt}
    ]

    # Add images to message content
    for img_id in image_files:
        message_content.append({"type": "image_file", "image_file": {"file_id": img_id}})

    # Add user message
    openai.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=message_content,
        attachments=attachments
    )

    # Run the assistant
    assistant_id = "asst_1pBzntEcrVPWsbztmDtG9Hap"
    run = openai.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id
    )

    # Poll for the assistant's response
    while True:
        run_status = openai.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run_status.status == "completed":
            break
        print("Waiting for response...")
        time.sleep(5)

    # Retrieve the final analysis result
    messages = openai.beta.threads.messages.list(thread_id=thread.id)
    
    if messages.data:
        latest_message = messages.data[0].content
        print("\nAnalysis Result:\n", latest_message)
        
        # Extract text content from TextContentBlock
        if isinstance(latest_message, list):
            try:
                analysis_text = ' '.join(
                    block.text.value if hasattr(block, 'text') and hasattr(block.text, 'value')
                    else str(block)
                    for block in latest_message
                )
            except AttributeError:
                analysis_text = str(latest_message)
        else:
            analysis_text = str(latest_message)
            
        print("Extracted analysis text:", analysis_text)

        # Parse the analysis into sections
        synopsis = extract_section(analysis_text, "Synopsis")
        insights_anomalies = extract_section(analysis_text, "Insights and Anomalies")
        citations = extract_section(analysis_text, "Citations")

        # Create the response structure
        analysis_result = {
            'success': True,
            'result': {
                'synopsis': synopsis,
                'insights_anomalies': insights_anomalies,
                'citations': citations,
                'raw_text': analysis_text  # Include raw text as fallback
            }
        }
        return analysis_result
    else:
        return {
            'success': False,
            'error': 'No response received.'
        }

if __name__ == "__main__":
    file_paths = [
        "Medications.png",
        "Heme_Profile.pdf"
    ]  
    result = analyze_health_records(file_paths)
    print("\nFinal Output:\n", result)
