import requests

oauth_token = "EnterToken"

def get_patient_allergies(patient_id, clinical_status):
    # Base URL for the Epic FHIR API
    base_url = "https://fhir.epic.com/interconnect-fhir-oauth/api/FHIR/R4"
    
    # Query parameters
    params = {
        'clinical-status': clinical_status,
        'patient': patient_id
    }
    
    # Your Bearer token (you'll need to obtain this through OAuth 2.0 flow)
    # In production, this should be obtained dynamically through proper authentication
    bearer_token = oauth_token
    
    # Headers including the Bearer token
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Accept": "application/json"
    }
    
    # Construct the full URL with endpoint
    endpoint = f"{base_url}/AllergyIntolerance"
    
    try:
        # Make the GET request with query parameters
        response = requests.get(
            endpoint,
            params=params,
            headers=headers
        )
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Return the JSON response
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None

def show_allergy_results(response):
    """
    Parse and print FHIR AllergyIntolerance response in a human-readable format.
    """
    if not response.get('entry'):
        print("No allergy records found.")
        return
        
    print("=== Allergy Information Summary ===")
    print(f"Total Records: {response['total']}\n")
    
    for entry in response['entry']:
        resource = entry['resource']
        
        # Get patient info
        patient = resource.get('patient', {})
        patient_name = patient.get('display', 'Unknown Patient')
        print(f"Patient: {patient_name}")
        
        # Get clinical status
        clinical_status = resource.get('clinicalStatus', {}).get('coding', [{}])[0].get('display', 'Unknown Status')
        print(f"Clinical Status: {clinical_status}")
        
        # Get verification status
        verification_status = resource.get('verificationStatus', {}).get('coding', [{}])[0].get('display', 'Unknown Verification')
        print(f"Verification Status: {verification_status}")
        
        # Get allergy/intolerance information
        allergy_code = resource.get('code', {})
        allergy_text = allergy_code.get('text', 'No description available')
        allergy_display = allergy_code.get('coding', [{}])[0].get('display', 'No specific allergy coded')
        
        print(f"Allergy Status: {allergy_text}")
        print(f"Additional Details: {allergy_display}")
        print("-" * 50)

################# CALL FUNCTION ##################################

# Example usage -- GET Patient Allergies
patient_id = "edO5RwPdFu.l85iSJJ8z5-w3"
clinical_status = "active"
allergy_result = get_patient_allergies(patient_id, clinical_status)

if allergy_result:
    print("Successfully retrieved patient allergies:")
    print(allergy_result)
    print(show_allergy_results(allergy_result))
