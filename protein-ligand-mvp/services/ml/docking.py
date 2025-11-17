from __future__ import annotations
from typing import Tuple, Dict, Optional
import os
import subprocess
import shutil
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

import logging
log = logging.getLogger("dock")


def _check_tool(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _resolve_vina_binary() -> Optional[str]:
    return r"C:/Program Files (x86)/The Scripps Research Institute/Vina/vina.exe"



def prepare_pdbqt(input_pdb_path: str, output_path: str) -> None:
    """Prepare a PDBQT file using Open Babel if available.
    Falls back to simple copy for .pdbqt input.
    """
    in_path = Path(input_pdb_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if in_path.suffix.lower() == ".pdbqt":
        shutil.copyfile(in_path, out_path)
        return

    if not _check_tool("obabel") and not _check_tool("babel"):
        raise RuntimeError("Open Babel (obabel/babel) is required to prepare PDBQT")

    tool = shutil.which("obabel") or shutil.which("babel")
    # -xr to remove hydrogens for receptor is typical; keep minimal flags
    cmd = [tool, str(in_path), "-O", str(out_path), "-xr"]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Failed to prepare PDBQT: {res.stderr or res.stdout}")


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
    if not _check_tool("obabel") and not _check_tool("babel"):
        raise RuntimeError("Open Babel (obabel/babel) is required to convert to PDBQT")
    tool = shutil.which("obabel") or shutil.which("babel")
    cmd = [tool, str(in_path), "-O", str(out_path)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Open Babel conversion failed: {res.stderr or res.stdout}")


def autodetect_pocket(pdb_path: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Very simple pocket estimation: center of coordinates from CA atoms, fixed size.
    This is a naive heuristic to get Vina running without external tools.
    """
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            rec = line[0:6].strip().upper()
            if rec in ("ATOM", "HETATM"):
                # Prefer CA for proteins when present
                name = line[12:16].strip()
                if name != "CA":
                    continue
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    xs.append(x); ys.append(y); zs.append(z)
                except Exception:
                    continue
    if not xs:
        # Fallback to any atoms if no CA found
        with open(pdb_path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                rec = line[0:6].strip().upper()
                if rec in ("ATOM", "HETATM"):
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        xs.append(x); ys.append(y); zs.append(z)
                    except Exception:
                        continue
    if not xs:
        # Last resort center/size
        return (0.0, 0.0, 0.0), (20.0, 20.0, 20.0)
    cx = sum(xs)/len(xs); cy = sum(ys)/len(ys); cz = sum(zs)/len(zs)
    size = (30.0, 30.0, 30.0)  # enlarged default box for robustness
    return (cx, cy, cz), size


def run_vina(protein_pdbqt: str, ligand_pdbqt: str,
             center: Tuple[float, float, float], size: Tuple[float, float, float]) -> Dict:
    """Run AutoDock Vina via subprocess and return log and affinity.
    Requires 'vina' binary to be installed and available on PATH.
    """
    vina_bin = _resolve_vina_binary()
    if not vina_bin:
        raise RuntimeError("AutoDock Vina not found. Set VINA_BIN to full path or add 'vina' to PATH.")

    with tempfile.TemporaryDirectory() as td:
        out_pdbqt = Path(td) / "out.pdbqt"
        log_path = Path(td) / "vina.log"
        cx, cy, cz = center
        sx, sy, sz = size
        cmd = [
            vina_bin,
            "--receptor", protein_pdbqt,
            "--ligand", ligand_pdbqt,
            "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
            "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
            "--out", str(out_pdbqt), "--log", str(log_path)
        ]
        # Debug: print exact command
        try:
            log.info(f"Running Vina: {' '.join(cmd)}")

        except Exception:
            pass

        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout = res.stdout
        stderr = res.stderr
        # Debug: always print stdout/stderr
        try:
            print("[DOCK] Vina stdout:\n", stdout)
            print("[DOCK] Vina stderr:\n", stderr)
        except Exception:
            pass
        if res.returncode != 0:
            raise RuntimeError(f"Vina failed: {stderr or stdout}")

        # Parse best affinity from stdout or log
        affinity: Optional[float] = None
        text = "\n".join([stdout, log_path.read_text(errors="ignore")])
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("1 ") or line.startswith("-----+-----------+"):
                # skip table header lines
                pass
            if line and "REMARK VINA RESULT:" in line:
                try:
                    # e.g., REMARK VINA RESULT: -7.8  0.000  0.000
                    parts = line.split()
                    for i, tok in enumerate(parts):
                        if tok.replace('.', '', 1).replace('-', '', 1).isdigit():
                            affinity = float(tok)
                            break
                except Exception:
                    continue
        # Fallback simple parse: look for a float like -7.8 in lines with RESULT
        if affinity is None:
            for line in text.splitlines():
                # Pattern in Vina 1.1.x:
                # 1       -7.2      0.0      0.0
                parts = line.split()
                if len(parts) >= 2 and parts[0].isdigit():
                    try:
                        affinity = float(parts[1])
                        break
                    except:
                        pass

        return {"affinity": affinity, "log": text, "pose_pdbqt": str(out_pdbqt)}


def _pdbqt_to_sdf(pdbqt_path: Path) -> Optional[str]:
    if not _check_tool("obabel") and not _check_tool("babel"):
        try:
            log.warning("[DOCK] Open Babel not found on PATH for PDBQT->SDF conversion")
        except Exception:
            pass
        return None
    tool = shutil.which("obabel") or shutil.which("babel")
    with tempfile.TemporaryDirectory() as td:
        out_sdf = Path(td) / "pose.sdf"
        # Prefer explicit -i/-o flags for robustness on Windows
        cmd = [tool, "-ipdbqt", str(pdbqt_path), "-osdf", "-O", str(out_sdf)]
        try:
            log.info(f"[DOCK] Converting pose with Open Babel: {' '.join(cmd)}")
        except Exception:
            pass
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            log.info("[DOCK] obabel stdout:\n" + (res.stdout or ""))
            log.info("[DOCK] obabel stderr:\n" + (res.stderr or ""))
        except Exception:
            pass
        if res.returncode != 0:
            return None
        try:
            return out_sdf.read_text()
        except Exception:
            return None


def dock(smiles: str, protein_path: str) -> Dict:
    """High-level docking: build ligand PDBQT, autodetect pocket, run vina, return affinity and pose SDF if convertible.
    Requires Open Babel and Vina available on PATH.
    """
    protein_path = str(protein_path)
    if not Path(protein_path).exists():
        raise FileNotFoundError("Protein file not found")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        lig_pdb = td / "ligand.pdb"
        lig_pdbqt = td / "ligand.pdbqt"
        rec_pdbqt = td / "receptor.pdbqt"

        _rdkit_smiles_to_pdb(smiles, lig_pdb)
        _convert_to_pdbqt(lig_pdb, lig_pdbqt)
        prepare_pdbqt(protein_path, str(rec_pdbqt))

        center, size = autodetect_pocket(protein_path)
        vina_res = run_vina(str(rec_pdbqt), str(lig_pdbqt), center, size)

        pose_sdf = _pdbqt_to_sdf(Path(vina_res["pose_pdbqt"]))
        return {
            "affinity": float(vina_res.get("affinity")) if vina_res.get("affinity") is not None else None,
            "log": vina_res.get("log", ""),
            "pose_sdf": pose_sdf,
        }
