import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

try:
    from services.ml import scoring
    print("Imported scoring module successfully.")
except Exception as e:
    print(f"Failed to import scoring: {e}")
    sys.exit(1)

def test_score():
    smiles = "CCO"
    try:
        print(f"Testing scoring.predict for {smiles}...")
        res = scoring.predict(smiles, protein_id="0c326d7b-6744-4aad-96ae-42b66e508c9b")
        print(f"Result: {res}")
        score = res.get("score")
        print(f"Score: {score}")
    except Exception as e:
        print(f"Scoring failed: {e}")

if __name__ == "__main__":
    test_score()
