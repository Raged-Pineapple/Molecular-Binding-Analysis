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


def _resolve_obabel_binary() -> Optional[str]:
    """Resolve Open Babel binary path robustly.
    Priority: OBABEL_BIN env -> common Windows install dirs -> which(obabel/babel/exe).
    """
    # Env override
    env_bin = os.getenv("OBABEL_BIN")
    if env_bin and Path(env_bin).exists():
        return str(env_bin)
    # Common Windows paths
    common_paths = [
        r"C:\\Program Files\\OpenBabel-3.1.1\\bin\\obabel.exe",
        r"C:\\Program Files (x86)\\OpenBabel-3.1.1\\bin\\obabel.exe",
        r"C:\\Program Files\\OpenBabel-3.1.0\\bin\\obabel.exe",
        r"C:\\Program Files (x86)\\OpenBabel-3.1.0\\bin\\obabel.exe",
    ]
    for p in common_paths:
        if Path(p).exists():
            return p
    # which fallback
    for name in ("obabel", "obabel.exe", "babel", "babel.exe"):
        w = shutil.which(name)
        if w:
            return w
    return None


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
    tool = _resolve_obabel_binary()
    if not tool:
        raise RuntimeError("Open Babel (obabel/babel) not found. Set OBABEL_BIN or add to PATH.")
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

        # Read pose now before temp dir is removed
        pose_txt: Optional[str] = None
        try:
            pose_txt = out_pdbqt.read_text(errors="ignore") if out_pdbqt.exists() else None
        except Exception:
            pose_txt = None
        return {"affinity": affinity, "log": text, "pose_pdbqt": str(out_pdbqt), "pose_pdbqt_text": pose_txt}


def _pdbqt_to_sdf(pdbqt_path: Path) -> Optional[str]:
    tool = _resolve_obabel_binary()
    if not tool:
        try:
            log.warning("[DOCK] Open Babel not found for PDBQT->SDF conversion (set OBABEL_BIN or PATH)")
        except Exception:
            pass
        return None
    with tempfile.TemporaryDirectory() as td:
        out_sdf = Path(td) / "pose.sdf"
        # Prefer explicit -i/-o flags for robustness on Windows
        cmd = [tool, "-ipdbqt", str(pdbqt_path), "-osdf", "-O", str(out_sdf)]
        try:
            log.info(f"[DOCK] Converting pose with Open Babel: {tool}")
            log.info(f"[DOCK] CMD: {' '.join(cmd)}")
        except Exception:
            pass
        # Always echo to stdout for visibility
        try:
            print("[DOCK] Converting pose with Open Babel:", tool)
            print("[DOCK] CMD:", " ".join(cmd))
        except Exception:
            pass
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            log.info("[DOCK] obabel stdout:\n" + (res.stdout or ""))
            log.info("[DOCK] obabel stderr:\n" + (res.stderr or ""))
        except Exception:
            pass
        try:
            print("[DOCK] obabel stdout:\n" + (res.stdout or ""))
            print("[DOCK] obabel stderr:\n" + (res.stderr or ""))
        except Exception:
            pass
        # Consider empty output as failure as well
        if res.returncode != 0 or (out_sdf.exists() and out_sdf.stat().st_size == 0):
            # Retry with minimal invocation style as a fallback
            alt_cmd = [tool, str(pdbqt_path), "-O", str(out_sdf)]
            try:
                log.info(f"[DOCK] Retry conversion with: {' '.join(alt_cmd)}")
            except Exception:
                pass
            try:
                print("[DOCK] Retry conversion with:", " ".join(alt_cmd))
            except Exception:
                pass
            res2 = subprocess.run(alt_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                log.info("[DOCK] obabel retry stdout:\n" + (res2.stdout or ""))
                log.info("[DOCK] obabel retry stderr:\n" + (res2.stderr or ""))
            except Exception:
                pass
            try:
                print("[DOCK] obabel retry stdout:\n" + (res2.stdout or ""))
                print("[DOCK] obabel retry stderr:\n" + (res2.stderr or ""))
            except Exception:
                pass
            if res2.returncode != 0 or (out_sdf.exists() and out_sdf.stat().st_size == 0):
                # Fallback 2: try forcing coordinate generation
                gen_cmd = [tool, "-ipdbqt", str(pdbqt_path), "-osdf", "--gen3D", "-O", str(out_sdf)]
                try:
                    log.info(f"[DOCK] Fallback with --gen3D: {' '.join(gen_cmd)}")
                except Exception:
                    pass
                try:
                    print("[DOCK] Fallback with --gen3D:", " ".join(gen_cmd))
                except Exception:
                    pass
                res3 = subprocess.run(gen_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                try:
                    log.info("[DOCK] obabel gen3D stdout:\n" + (res3.stdout or ""))
                    log.info("[DOCK] obabel gen3D stderr:\n" + (res3.stderr or ""))
                except Exception:
                    pass
                try:
                    print("[DOCK] obabel gen3D stdout:\n" + (res3.stdout or ""))
                    print("[DOCK] obabel gen3D stderr:\n" + (res3.stderr or ""))
                except Exception:
                    pass
                if res3.returncode != 0 or (out_sdf.exists() and out_sdf.stat().st_size == 0):
                    # Fallback 3: convert to PDB, then RDKit -> SDF
                    pdb_tmp = Path(td) / "pose.pdb"
                    pdb_cmd = [tool, "-ipdbqt", str(pdbqt_path), "-opdb", "-O", str(pdb_tmp)]
                    try:
                        log.info(f"[DOCK] Fallback via PDB: {' '.join(pdb_cmd)}")
                    except Exception:
                        pass
                    try:
                        print("[DOCK] Fallback via PDB:", " ".join(pdb_cmd))
                    except Exception:
                        pass
                    res4 = subprocess.run(pdb_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    try:
                        log.info("[DOCK] obabel pdb stdout:\n" + (res4.stdout or ""))
                        log.info("[DOCK] obabel pdb stderr:\n" + (res4.stderr or ""))
                    except Exception:
                        pass
                    try:
                        print("[DOCK] obabel pdb stdout:\n" + (res4.stdout or ""))
                        print("[DOCK] obabel pdb stderr:\n" + (res4.stderr or ""))
                    except Exception:
                        pass
                    if res4.returncode != 0 or (not pdb_tmp.exists()) or pdb_tmp.stat().st_size == 0:
                        return None
                    # RDKit load PDB and write SDF block
                    try:
                        pdb_text = pdb_tmp.read_text(errors="ignore")
                        mol = Chem.MolFromPDBBlock(pdb_text, sanitize=False, removeHs=False)
                        if mol is None:
                            return None
                        try:
                            Chem.SanitizeMol(mol)
                        except Exception:
                            pass
                        sdf_block = Chem.MolToMolBlock(mol)
                        return sdf_block
                    except Exception:
                        return None
        try:
            return out_sdf.read_text()
        except Exception:
            return None


def dock(smiles: str, protein_path: str, quick: bool = False) -> Dict:
    """High-level docking: build ligand PDBQT, autodetect pocket, run vina, return affinity and pose SDF if convertible.
    Requires Open Babel and Vina available on PATH.
    quick: if True, use a smaller search box for faster docking.
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
        if quick:
            try:
                # shrink box moderately but keep reasonable coverage
                sx, sy, sz = size
                size = (
                    max(12.0, sx * 0.6),
                    max(12.0, sy * 0.6),
                    max(12.0, sz * 0.6),
                )
                log.info(f"[DOCK] quick mode on -> box size {sx,sy,sz} -> {size}")
            except Exception:
                # if any issue, fallback to a fixed smaller box
                size = (18.0, 18.0, 18.0)
        vina_res = run_vina(str(rec_pdbqt), str(lig_pdbqt), center, size)

        # Optional: persist debug artifacts if requested
        if os.getenv("DOCK_KEEP_TMP"):
            try:
                dbg = Path("data") / "dock_debug"
                dbg.mkdir(parents=True, exist_ok=True)
                import time
                ts = int(time.time())
                # Copy inputs and outputs
                (dbg / f"{ts}_ligand.pdb").write_text(lig_pdb.read_text(errors="ignore"))
                (dbg / f"{ts}_ligand.pdbqt").write_text(lig_pdbqt.read_text(errors="ignore"))
                (dbg / f"{ts}_receptor.pdbqt").write_text(rec_pdbqt.read_text(errors="ignore"))
                out_pdbqt_path = Path(vina_res.get("pose_pdbqt", "")) if vina_res.get("pose_pdbqt") else None
                if out_pdbqt_path and out_pdbqt_path.exists():
                    (dbg / f"{ts}_out.pdbqt").write_text(out_pdbqt_path.read_text(errors="ignore"))
                try:
                    log.info(f"[DOCK] Saved debug docking artifacts under {dbg} (ts={ts})")
                except Exception:
                    pass
            except Exception:
                pass

        # Ensure a persistent PDBQT path: run_vina returns a path in its own temp dir which is already deleted.
        # If we have the text, write it to a fresh temp file for conversion.
        pose_pdbqt_path = None
        pose_txt = vina_res.get("pose_pdbqt_text")
        if pose_txt:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdbqt") as fh:
                fh.write(pose_txt.encode("utf-8", errors="ignore"))
                pose_pdbqt_path = Path(fh.name)
        else:
            # Fallback: try the returned path (may be invalid already)
            p = vina_res.get("pose_pdbqt")
            pose_pdbqt_path = Path(p) if p else None

        pose_sdf = None
        if pose_pdbqt_path and pose_pdbqt_path.exists():
            pose_sdf = _pdbqt_to_sdf(pose_pdbqt_path)
        return {
            "affinity": float(vina_res.get("affinity")) if vina_res.get("affinity") is not None else None,
            "log": vina_res.get("log", ""),
            "pose_sdf": pose_sdf,
        }
