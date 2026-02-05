import requests
import json

BASE_URL = "http://127.0.0.1:8001"

def test_improve():
    # 1. First predict score to know base
    payload_pred = {
        "protein_id": "0c326d7b-6744-4aad-96ae-42b66e508c9b",
        "ligand": "c1cc2ccccc2[nH]1", # Indole
        "mode": "heuristic"
    }
    resp = requests.post(f"{BASE_URL}/predict", json=payload_pred)
    if resp.status_code != 200:
        print("Predict failed:", resp.text)
        return
    
    score = resp.json()["score"]
    print(f"Base Score from /predict: {score}")
    
    # 2. Call improve with target < score
    target = int(score) - 10
    print(f"Targeting: {target}")
    
    payload_imp = {
        "protein_id": "0c326d7b-6744-4aad-96ae-42b66e508c9b",
        "ligand_smiles": "c1cc2ccccc2[nH]1",
        "target_score": target,
        "mode": "heuristic"
    }
    
    try:
        resp = requests.post(f"{BASE_URL}/improve", json=payload_imp)
        data = resp.json()
        print("\nImprove Response:")
        print(f"Base Score: {data.get('base_score')}")
        print(f"Num Improvements: {len(data.get('improvements', []))}")
        print(f"Trace Length: {len(data.get('trace', []))}")
        
        if len(data.get('trace', [])) == 1:
            print("SUCCESS: Early exit triggered (Trace length 1).")
        else:
            print("FAILURE: Full optimization ran (Trace > 1).")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_improve()
