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
    for file_path in file_paths:
        with open(file_path, "rb") as f:
            upload_response = openai.files.create(purpose="assistants", file=f)
            uploaded_files.append(upload_response.id)
    
    print(f"Uploaded files: {uploaded_files}")

    # Create a new thread
    thread = openai.beta.threads.create()

    # Attach files when creating the user message
    openai.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt,
        attachments=[{"file_id": file_id, "tools": [{"type": "file_search"}]} for file_id in uploaded_files]  # ✅ Correct format
    )

    # Run the assistant (without file_ids)
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
    latest_message = messages.data[0].content if messages.data else "No response received."

    return latest_message

if __name__ == "__main__":
    file_paths = ["Medications.png"]  # Replace with actual file paths
    result = analyze_health_records(file_paths)
    print("\nAnalysis Result:\n", result)
