from __future__ import annotations
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    PredictRequest,
    PredictResponse,
    Interaction,
    Suggestion,
    ViewerPayload,
    PredictIn,
    PredictOut,
    ImproveIn,
    ImproveOut,
    ImprovedMolecule,
)
from .utils import smiles_from_input
from ..ml import scoring, explain

app = FastAPI(title="protein-ligand-mvp", version="0.1.0")

# CORS: allow all for MVP
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictOut, summary="Minimal mock predictor", tags=["predict"])
def predict(req: PredictIn) -> PredictOut:
    try:
        res = scoring.predict(req.ligand, protein_id=req.protein_id)
        return PredictOut(
            score=float(res["score"]),
            calibration_info=str(res["calibration_info"]),
            explanations=[str(x) for x in res.get("explanations", [])],
            viewer_payload=res.get("viewer_payload"),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/improve", response_model=ImproveOut, summary="Stub ligand improver", tags=["improve"])
def improve_stub(req: ImproveIn) -> ImproveOut:
    """Return mocked improved ligands without performing RDKit edits."""
    try:
        # First get a score for the original ligand
        base = scoring.predict(req.ligand_smiles)
        if base["score"] >= req.target_score:
            improvements = []
        else:
            improvements = scoring.stub_improve(req.ligand_smiles, req.target_score)
        return ImproveOut(
            improvements=[ImprovedMolecule(smiles=i["smiles"], score=float(i["score"])) for i in improvements]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
