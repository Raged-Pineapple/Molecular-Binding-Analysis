import requests
import json

BASE_URL = "http://127.0.0.1:8001"

def test_improve_endpoint():
    payload = {
        "protein_id": "0c326d7b-6744-4aad-96ae-42b66e508c9b",
        "ligand_smiles": "CCO",
        "target_score": 80,
        "mode": "heuristic"
    }
    
    print(f"Sending POST to {BASE_URL}/improve with payload: {payload}")
    try:
        resp = requests.post(f"{BASE_URL}/improve", json=payload)
        print(f"Status Code: {resp.status_code}")
        try:
            data = resp.json()
            print("Response JSON:")
            print(json.dumps(data, indent=2))
        except:
            print("Response Text (not JSON):")
            print(resp.text)
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_improve_endpoint()
