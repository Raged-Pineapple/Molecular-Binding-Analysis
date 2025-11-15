from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
from pydantic import field_validator


class LigandInput(BaseModel):
    type: Literal["smiles"] = Field(description="Ligand representation type")
    value: str = Field(description="Ligand value string")


class Interaction(BaseModel):
    type: str
    residue: Optional[str] = None
    description: Optional[str] = None
    score: Optional[float] = None


class Suggestion(BaseModel):
    issue: str
    detail: str
    severity: Literal["low", "medium", "high"]


class ViewerPayload(BaseModel):
    protein_id: str
    ligand_sdf: str


class PredictRequest(BaseModel):
    protein_id: str = Field(description="Protein PDB ID")
    ligand: LigandInput
    solvent: str = Field(default="water")


class PredictResponse(BaseModel):
    job_id: str
    raw_score: float
    calibrated_score: int
    interactions: List[Interaction]
    lacking_factors: List[Suggestion]
    viewer_payload: ViewerPayload
    session: Dict[str, Any]


class ImproveRequest(BaseModel):
    protein_id: str
    ligand_smiles: str
    target_score: int = Field(default=90, ge=0, le=100)
    max_iters: int = Field(default=12, ge=1, le=100)

    @field_validator("ligand_smiles")
    @classmethod
    def non_empty_smiles(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("ligand_smiles must be a non-empty SMILES string")
        return v


class IterationRecord(BaseModel):
    iter: int
    smiles: str
    raw_score: float
    calibrated_score: int


class ImproveResponse(BaseModel):
    achieved: bool
    best_smiles: str
    best_score: int
    history: List[IterationRecord]


# Simple models for minimal /predict endpoint
class PredictIn(BaseModel):
    protein_id: str
    ligand: str


class PredictOut(BaseModel):
    score: float
    calibration_info: str
    explanations: List[str]
    viewer_payload: Optional[ViewerPayload] = None


class ImproveIn(BaseModel):
    protein_id: str
    ligand_smiles: str
    target_score: int


class ImprovedMolecule(BaseModel):
    smiles: str
    score: float


class ImproveOut(BaseModel):
    improvements: List[ImprovedMolecule]
