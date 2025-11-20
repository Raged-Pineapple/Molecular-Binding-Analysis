from __future__ import annotations
import uuid
from fastapi import FastAPI, HTTPException, UploadFile, File
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
    ProteinUploadResponse,
    ProteinPreviewInfo,
    BatchPredictIn,
    BatchPredictItem,
    BatchPredictOut,
    DockPredictIn,
    DockPredictOut,
)
from .utils import smiles_from_input
from pathlib import Path
import uuid as _uuid
from .utils import ensure_dir, parse_pdb_preview, write_bytes
from ..ml import scoring, explain
from ..ml import docking
from ..ml import model as ml_model
from ..ml import improve

import os, shutil
print("DEBUG VINA_BIN:", os.getenv("VINA_BIN"))
print("DEBUG which(vina):", shutil.which("vina"))

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
        mode = (req.mode or "heuristic").lower()
        if mode == "ml":
            ml_res = ml_model.score_smiles(req.ligand, req.protein_id)
            feats = ml_res.get("features", {})
            explanations = [f"{k}={v:.2f}" if isinstance(v, (int,float)) else f"{k}={v}" for k,v in feats.items()]
            return PredictOut(
                score=float(ml_res.get("score", 0.0)),
                calibration_info="ml-v1",
                explanations=explanations,
                viewer_payload=None,
            )
        elif mode == "docking":
            prot_path = Path("data") / "proteins" / f"{req.protein_id}.pdb"
            dock_res = docking.dock(req.ligand, str(prot_path))
            affinity = dock_res.get("affinity")
            score = 0.0 if affinity is None else max(1.0, min(100.0, (-float(affinity)) * 10.0))
            explanations = [] if affinity is None else [f"Affinity={affinity}"]
            # we don't map pose into viewer_payload; SDF may be large, frontends can request via dedicated route later
            return PredictOut(
                score=float(round(score,1)),
                calibration_info="docking-v1",
                explanations=explanations,
                viewer_payload=None,
            )
        else:
            # default heuristic
            res = scoring.predict(req.ligand, protein_id=req.protein_id)
            return PredictOut(
                score=float(res["score"]),
                calibration_info=str(res["calibration_info"]),
                explanations=[str(x) for x in res.get("explanations", [])],
                viewer_payload=res.get("viewer_payload"),
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/dock_predict", response_model=DockPredictOut, summary="Dock ligand with Vina", tags=["docking"])
def dock_predict(req: DockPredictIn) -> DockPredictOut:
    try:
        prot_path = Path("data") / "proteins" / f"{req.protein_id}.pdb"
        if not prot_path.exists():
            raise HTTPException(status_code=404, detail="Protein file not found. Upload via /upload_protein first.")

        res = docking.dock(req.ligand, str(prot_path))
        affinity = res.get("affinity")
        if affinity is None:
            score = 0.0
        else:
            score = max(1.0, min(100.0, (-float(affinity)) * 10.0))
        pose_sdf = res.get("pose_sdf")
        return DockPredictOut(score=float(round(score, 1)), affinity=float(affinity) if affinity is not None else None, pose=pose_sdf)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/batch_predict", response_model=BatchPredictOut, summary="Batch ligand scoring", tags=["predict"])
def batch_predict(req: BatchPredictIn) -> BatchPredictOut:
    try:
        results: list[BatchPredictItem] = []
        for smi in req.ligands or []:
            try:
                res = scoring.predict(smi, protein_id=req.protein_id)
                score = float(res.get("score", 0.0))
                exps = res.get("explanations", []) or []
                expl = "; ".join(str(x) for x in exps) if exps else ""
                results.append(BatchPredictItem(ligand=smi, score=score, explanation=expl))
            except Exception as inner:
                # On individual failure, include item with score 0 and error message
                results.append(BatchPredictItem(ligand=smi, score=0.0, explanation=f"error: {inner}"))
        return BatchPredictOut(results=results)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/upload_protein", response_model=ProteinUploadResponse, summary="Upload a PDB file", tags=["protein"])
async def upload_protein(file: UploadFile = File(...)) -> ProteinUploadResponse:
    """Accepts a PDB file, saves it under data/proteins/<uuid>.pdb, and returns a small preview."""
    try:
        filename = (file.filename or "").lower()
        if not filename.endswith(".pdb"):
            raise HTTPException(status_code=400, detail="Only .pdb files are accepted")

        pid = str(_uuid.uuid4())
        out_dir = Path("data") / "proteins"
        ensure_dir(out_dir)
        out_path = out_dir / f"{pid}.pdb"

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        write_bytes(out_path, content)

        # Parse minimal preview info
        text = content.decode("utf-8", errors="ignore")
        num_atoms, num_residues = parse_pdb_preview(text.splitlines())

        return ProteinUploadResponse(
            protein_id=pid,
            message="uploaded successfully",
            preview_info=ProteinPreviewInfo(num_atoms=num_atoms, num_residues=num_residues),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/improve", response_model=ImproveOut, summary="Stub ligand improver", tags=["improve"])
def improve_stub(req: ImproveIn) -> ImproveOut:
    """Deterministic improvement using mutation operators and GA-style loop.
    If optional advanced parameters are provided, runs advanced optimizer instead.
    """
    try:
        # Base score for original ligand
        base = scoring.predict(req.ligand_smiles, protein_id=req.protein_id)
        base_score = float(base.get("score", 0.0))

        # If target provided and already reached, return original with minimal trace
        if req.target_score is not None and base_score >= req.target_score:
            return ImproveOut(
                base_score=base_score,
                improvements=[ImprovedMolecule(smiles=req.ligand_smiles, score=base_score, step=0)],
                trace=[{"step": 0, "smiles": req.ligand_smiles, "score": base_score}],
            )

        # Choose optimizer: advanced if optional params present, else legacy GA
        use_advanced = any([
            req.mode is not None,
            req.n_iters is not None,
            req.pop_size is not None,
            req.mutate_rate is not None,
            bool(req.persist_debug),
        ])

        if use_advanced:
            # Resolve protein path if docking selected
            protein_path = None
            if (req.mode or "").lower() == "docking":
                p = Path("data") / "proteins" / f"{req.protein_id}.pdb"
                protein_path = str(p) if p.exists() else None

            dbg_dir = None
            if req.persist_debug:
                dbg_dir = str(Path("data") / "improve_debug")

            # Quick mode heuristic: if caller provided a custom size box, enable quick
            quick = False
            if (req.mode or "").lower() == "docking" and req.size:
                try:
                    # any indication of smaller-than-default box enables quick path
                    sx, sy, sz = [float(x) for x in req.size]
                    quick = any(v < 30.0 for v in (sx, sy, sz))
                except Exception:
                    quick = True

            ga = improve.advanced_genetic_optimize(
                seed_smiles=req.ligand_smiles,
                protein_path=protein_path,
                mode=(req.mode or "heuristic"),
                n_iters=int(req.n_iters) if req.n_iters is not None else 8,
                pop_size=int(req.pop_size) if req.pop_size is not None else 8,
                mutate_rate=float(req.mutate_rate) if req.mutate_rate is not None else 0.5,
                target_score=float(req.target_score) if req.target_score is not None else None,
                persist_debug_dir=dbg_dir,
                quick=quick,
            )
            final = ga.get("final", [])
            trace = ga.get("trace", [])
            improvements = []
            for it in final:
                meta = {
                    "mode": (req.mode or "heuristic"),
                    "protein_id": req.protein_id,
                    "quick": bool(quick),
                    "op_name": it.get("op_name"),
                    "parent_smiles": it.get("parent_smiles"),
                }
                expl = []
                if (req.mode or "").lower() == "docking":
                    expl.append(f"docking_affinity={it['score']}")
                improvements.append(
                    ImprovedMolecule(
                        smiles=it["smiles"],
                        score=float(it["score"]),
                        step=int(it.get("step", 0)),
                        op_name=str(it.get("op_name")) if it.get("op_name") is not None else None,
                        parent_smiles=str(it.get("parent_smiles")) if it.get("parent_smiles") is not None else None,
                        explanations=expl or None,
                        metadata=meta,
                    )
                )
            return ImproveOut(
                base_score=ga.get("base_score", base_score),
                improvements=improvements,
                trace=[{"step": int(t.get("step", 0)), "smiles": t["smiles"], "score": float(t["score"]) } for t in trace],
                run_metadata={
                    "mode": (req.mode or "heuristic"),
                    "protein_id": req.protein_id,
                    "n_iters": int(req.n_iters) if req.n_iters is not None else 8,
                    "pop_size": int(req.pop_size) if req.pop_size is not None else 8,
                    "mutate_rate": float(req.mutate_rate) if req.mutate_rate is not None else 0.5,
                    "quick": bool(quick),
                },
            )
        else:
            # Legacy GA-style optimization
            ga = improve.genetic_optimize(
                req.ligand_smiles,
                protein_path=None,
                n_iters=5,
                pop_size=6,
                target_score=float(req.target_score) if req.target_score is not None else None,
            )
            final = ga.get("final", [])
            trace = ga.get("trace", [])
            improvements = [
                ImprovedMolecule(
                    smiles=it["smiles"],
                    score=float(it["score"]),
                    step=int(it.get("step", 0)),
                    explanations=[f"heuristic_score={float(it['score']):.1f}"],
                    metadata={"mode": "heuristic", "protein_id": req.protein_id},
                ) for it in final
            ]
            return ImproveOut(
                base_score=base_score,
                improvements=improvements,
                trace=[{"step": int(t.get("step", 0)), "smiles": t["smiles"], "score": float(t["score"]) } for t in trace],
                run_metadata={"mode": "heuristic", "protein_id": req.protein_id},
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
