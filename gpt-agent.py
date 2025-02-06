import openai
import time

# OpenAI API Credentials
openai.api_key = ""
openai.organization = ""

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
                image_files.append(upload_response.id)  # Separate storage for images
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
    assistant_id = "asst_1pBzntEcrVPWsbztmDtG9Hap"  # Use your actual OpenAI Assistant ID
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
        time.sleep(5)  # Wait for processing to complete

    # Retrieve the final analysis result
    messages = openai.beta.threads.messages.list(thread_id=thread.id)
    
    if messages.data:
        latest_message = messages.data[0].content
        print("\nAnalysis Result:\n", latest_message)
        return latest_message
    else:
        print("\n❌ No valid response received. Check if the PDF contains selectable text.")
        return "No response received."

if __name__ == "__main__":
    file_paths = [
        "Medications.png",
        "Heme_Profile.pdf"
    ]  
    result = analyze_health_records(file_paths)
    print("\nFinal Output:\n", result)
