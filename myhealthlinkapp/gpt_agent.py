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
                "Based on", "Synopsis:", "Summary:", "### Summary:",
                "### Overview:", "Overview:", "Analysis:", "### Analysis:",
                "Results:", "### Results:", "Findings:", "### Findings:"
            ],
            "Anomalies": [
                "### Insights/Anomalies:", "Anomalies:", "### Anomalies:", 
                "Insights/Anomalies:", "### Findings of Interest:", 
                "### Areas of Concern:", "### Abnormal Results:",
                "### Notable Results:", "### Observations:", "### Key Points:",
                "### Important Findings:"
            ]
        }
        
        current_markers = markers.get(section_name, [])
        section_text = ""
        
        # First try to find specific section markers
        for marker in current_markers:
            if marker in text:
                start = text.find(marker) + len(marker)
                # Look for any of the next section markers
                all_possible_markers = sum(markers.values(), [])  # Combine all markers
                ends = [text.find(m, start) for m in all_possible_markers if text.find(m, start) != -1]
                # Also look for common endpoint markers
                common_endings = ["This summary", "In conclusion", "Please note", "Note:"]
                ends.extend([text.find(m, start) for m in common_endings if text.find(m, start) != -1])
                end = min(ends) if ends else len(text)
                section_text = text[start:end].strip()
                break
        
        # If no section was found with markers, try to intelligently split the content
        if not section_text:
            # For synopsis, take the first major section if it exists
            if section_name == "Synopsis":
                # Look for the first major section break
                for marker in markers["Anomalies"]:
                    index = text.find(marker)
                    if index != -1:
                        section_text = text[:index].strip()
                        break
                if not section_text:
                    # If no section break found, take the first part of the text
                    section_text = text.split('\n\n')[0].strip()
            
            # For anomalies/insights, take the latter part if it contains relevant keywords
            elif section_name == "Anomalies":
                relevant_keywords = ["note", "concern", "abnormal", "elevated", "decreased", 
                                  "high", "low", "irregular", "unusual", "attention",
                                  "significant", "important", "notable", "finding"]
                
                # Split text into paragraphs
                paragraphs = text.split('\n\n')
                relevant_paragraphs = []
                
                # Collect paragraphs that contain relevant keywords
                for para in paragraphs[1:]:  # Skip the first paragraph (likely synopsis)
                    if any(keyword in para.lower() for keyword in relevant_keywords):
                        relevant_paragraphs.append(para)
                
                if relevant_paragraphs:
                    section_text = '\n\n'.join(relevant_paragraphs)
        
        # Clean up the extracted text
        def clean_text(text):
            # Remove markdown formatting
            text = text.replace('**', '')
            text = text.replace('###', '')
            # Remove excessive whitespace
            text = ' '.join(text.split())
            # Remove list markers at the start of lines
            text = text.replace('\n- ', '\n')
            text = text.replace('\n* ', '\n')
            text = text.replace('\n1. ', '\n')
            return text.strip()
        
        return clean_text(section_text) if section_text else f"No {section_name.lower()} available."
        
    except Exception as e:
        print(f"Error extracting {section_name}: {e}")
        return "Analysis available. Please refer to the full text for details."

def analyze_health_records(file_paths):
    """
    Sends image/PDF files to OpenAI API for analysis.
    """
    # Determine if single or multi-file analysis
    prompt = (
        "Analyze these health records and provide a synopsis, any insights or anomalies "
        "(explore if insights occur through cross-analysis of the documents)."
        if len(file_paths) > 1
        else "Analyze this health record and provide a synopsis, any insights or anomalies."
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

    # Add images to message content (not as attachments)
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
        anomalies = extract_section(analysis_text, "Anomalies")

        # Create a more resilient response structure
        analysis_result = {
            'success': True,
            'result': {
                'synopsis': synopsis,
                'anomalies': anomalies,
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
