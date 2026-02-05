from __future__ import annotations
from typing import Tuple, Dict, Optional
import os
import subprocess
import shutil
import tempfile
from pathlib import Path
import json
import hashlib

from rdkit import Chem
from rdkit.Chem import AllChem

import logging
log = logging.getLogger("dock")

def _resolve_obabel_binary() -> Optional[str]:
    """Resolve Open Babel binary path robustly."""
    env_bin = os.getenv("OBABEL_BIN")
    if env_bin and Path(env_bin).exists():
        return str(env_bin)
    common_paths = [
        r"C:\Program Files\OpenBabel-3.1.1\bin\obabel.exe",
        r"C:\Program Files (x86)\OpenBabel-3.1.1\bin\obabel.exe",
        r"C:\Program Files\OpenBabel-3.1.0\bin\obabel.exe",
        r"C:\Program Files (x86)\OpenBabel-3.1.0\bin\obabel.exe",
    ]
    for p in common_paths:
        if Path(p).exists():
            return p
    for name in ("obabel", "obabel.exe", "babel", "babel.exe"):
        w = shutil.which(name)
        if w: return w
    return None

def _resolve_vina_binary() -> Optional[str]:
    return r"C:/Program Files (x86)/The Scripps Research Institute/Vina/vina.exe"

CACHE_DIR = Path("data") / "cache" / "docking"
RECEPTOR_PDBQT_DIR = Path("data") / "cache" / "receptors"

def prepare_receptor(input_pdb_path: str) -> str:
    """Prepare receptor: remove water, add Hs, convert to PDBQT."""
    in_path = Path(input_pdb_path)
    pid = in_path.stem
    RECEPTOR_PDBQT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RECEPTOR_PDBQT_DIR / f"{pid}.pdbqt"
    
    if out_path.exists():
        return str(out_path)
    
    tool = _resolve_obabel_binary()
    if not tool:
        raise RuntimeError("Open Babel required for protein preparation")
            
    cmd = [tool, str(in_path), "-O", str(out_path), "-xr", "-h"]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Protein preparation failed: {res.stderr or res.stdout}")
    return str(out_path)

def _rdkit_smiles_to_pdb(smiles: str, out_pdb: Path) -> None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 13
    if AllChem.EmbedMolecule(mol, params) != 0:
        if AllChem.EmbedMolecule(mol) != 0:
            raise RuntimeError("Failed to embed 3D coordinates for ligand")
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass
    pdb_block = Chem.MolToPDBBlock(mol)
    out_pdb.write_text(pdb_block)

def _convert_to_pdbqt(in_path: Path, out_path: Path) -> None:
    tool = _resolve_obabel_binary()
    if not tool:
        raise RuntimeError("Open Babel not found.")
    cmd = [tool, str(in_path), "-O", str(out_path)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Open Babel conversion failed: {res.stderr or res.stdout}")

def autodetect_pocket(pdb_path: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Simple pocket estimation using protein centroid and fixed box size."""
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            rec = line[0:6].strip().upper()
            if rec in ("ATOM", "HETATM"):
                name = line[12:16].strip()
                if name != "CA": continue
                try:
                    xs.append(float(line[30:38]))
                    ys.append(float(line[38:46]))
                    zs.append(float(line[46:54]))
                except Exception: continue
    if not xs:
        with open(pdb_path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                rec = line[0:6].strip().upper()
                if rec in ("ATOM", "HETATM"):
                    try:
                        xs.append(float(line[30:38]))
                        ys.append(float(line[38:46]))
                        zs.append(float(line[46:54]))
                    except Exception: continue
    if not xs:
        return (0.0, 0.0, 0.0), (28.0, 28.0, 28.0)
    cx, cy, cz = sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs)
    return (cx, cy, cz), (28.0, 28.0, 28.0)

def run_vina(protein_pdbqt: str, ligand_pdbqt: str,
             center: Tuple[float, float, float], size: Tuple[float, float, float]) -> Dict:
    vina_bin = _resolve_vina_binary()
    if not vina_bin:
        raise RuntimeError("AutoDock Vina not found.")

    with tempfile.TemporaryDirectory() as td:
        out_pdbqt = Path(td) / "out.pdbqt"
        log_path = Path(td) / "vina.log"
        cx, cy, cz = center
        sx, sy, sz = size
        cmd = [vina_bin, "--receptor", protein_pdbqt, "--ligand", ligand_pdbqt,
               "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
               "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
               "--out", str(out_pdbqt), "--log", str(log_path)]
        
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Vina failed: {res.stderr or res.stdout}")

        affinity: Optional[float] = None
        text = "\n".join([res.stdout, log_path.read_text(errors="ignore")])
        for line in text.splitlines():
            if "REMARK VINA RESULT:" in line:
                try:
                    parts = line.split()
                    for tok in parts:
                        if tok.replace('.', '', 1).replace('-', '', 1).isdigit():
                            affinity = float(tok)
                            break
                    if affinity is not None: break
                except: continue
        
        if affinity is None:
            for line in text.splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[0].isdigit():
                    try:
                        affinity = float(parts[1])
                        break
                    except: pass

        pose_txt = out_pdbqt.read_text(errors="ignore") if out_pdbqt.exists() else None
        return {"affinity": affinity, "pose_pdbqt_text": pose_txt}

def _pdbqt_to_sdf(pdbqt_txt: str) -> Optional[str]:
    tool = _resolve_obabel_binary()
    if not tool: return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdbqt") as pf:
        pf.write(pdbqt_txt.encode())
        pf_name = pf.name
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".sdf") as sf:
            sf_name = sf.name
        cmd = [tool, "-ipdbqt", pf_name, "-osdf", "-O", sf_name]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            return Path(sf_name).read_text()
    finally:
        for f in (pf_name, sf_name):
            try: os.unlink(f)
            except: pass
    return None

def dock(smiles: str, protein_path: str, quick: bool = False, force: bool = False) -> Dict:
    protein_path = Path(protein_path)
    protein_id = protein_path.stem
    
    cache_str = f"{protein_id}_{smiles}_{'quick' if quick else 'full'}"
    cache_id = hashlib.sha256(cache_str.encode()).hexdigest()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_id}.json"
    
    if not force and cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except: pass

    rec_pdbqt = prepare_receptor(str(protein_path))
    
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        lig_pdb = td / "ligand.pdb"
        lig_pdbqt = td / "ligand.pdbqt"
        _rdkit_smiles_to_pdb(smiles, lig_pdb)
        _convert_to_pdbqt(lig_pdb, lig_pdbqt)
        
        center, size = autodetect_pocket(str(protein_path))
        if quick: size = (18.0, 18.0, 18.0)
        
        vina_res = run_vina(rec_pdbqt, str(lig_pdbqt), center, size)

    pose_sdf = _pdbqt_to_sdf(vina_res["pose_pdbqt_text"]) if vina_res.get("pose_pdbqt_text") else None
    result = {
        "protein_id": protein_id,
        "ligand_smiles": smiles,
        "affinity": vina_res.get("affinity"),
        "pose_sdf": pose_sdf,
    }
    
    try:
        cache_file.write_text(json.dumps(result))
    except: pass
    
    return result
