"""
compton_response.py
===================
Build a Compton imaging response (COSIpy-compatible HDF5) directly from
MEGAlib/revan .tra files.

API
----------

ComptonResponseCreator(
    Ei_edges, 
    Em_edges,
    tra_filename,
    nside_nulambda, 
    nside_psichi,
    phi_bins,
    save_path,
    pol_bins   = None,
    max_sq     = 7,
    overwrite  = False,
    n_workers  = 1,
    dtype      = np.float32,
)
    -> stats_dict


    The HDF5 file at save_path is fully compatible with COSIpy's
    FullDetectorResponse.open(save_path).
"""

from __future__ import annotations

import concurrent.futures
import gzip
import logging
import multiprocessing as mp
import platform
from pathlib import Path
from typing import Union

import h5py as h5
import hdf5plugin
import healpy as hp
import numpy as np
from scipy.special import erf
from histpy import Axes, Axis, HealpixAxis
from scoords import SpacecraftFrame
from tqdm.auto import tqdm

logger = logging.getLogger('cosipy')

_ME_KEV      = 510.998950   # electron rest mass [keV]
_RSP_VERSION = 2            # COSIpy FullDetectorResponse format version

def _inv_perm(p: np.ndarray) -> np.ndarray:
    """Invert a permutation array."""
    r = np.empty_like(p)
    for i, v in enumerate(p): r[v] = i
    return r


_AXIS_DESCRIPTION = {
    "Ei"       : "Initial simulated energy",
    "NuLambda" : "Location of the simulated source in the spacecraft coordinates",
    "Pol"      : "Polarization angle",
    "Em"       : "Measured energy",
    "PsiChi"   : "Location in the Compton Data Space",
    "Phi"      : "Compton angle",
}

# ── Input validation ──────────────────────────────────────────────────────────

def _validate_tra_path(path: Path) -> None:
    """Raise if path does not exist or is not a .tra or .tra.gz file."""
    if not path.exists():
        raise FileNotFoundError(f"TRA file not found: {path}")
    # Handle multi-part suffixes like RSL7_iso.inc1.id1.tra.gz
    if ".tra" not in path.suffixes:
        raise ValueError(
            f"Expected a .tra or .tra.gz file, got: {path.name}\n"
            "File must have a .tra extension (optionally followed by .gz)."
        )


# ── File helpers ──────────────────────────────────────────────────────────────

def _open_tra(path: Union[str, Path]):
    """Open a .tra or .tra.gz file for text reading."""
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding="utf-8", errors="replace")
    return open(p, "r", encoding="utf-8", errors="replace")


def _resolve_files(tra_path: Path, _seen: set = None) -> list[Path]:
    """
    Recursively resolve all leaf event files from a (nested) concat file.
    Each file is opened only once. _seen prevents circular reference loops.
    """
    if _seen is None:
        _seen = set()
    tra_path = tra_path.resolve()
    if tra_path in _seen:
        logger.warning(f"Circular reference, skipping: {tra_path}")
        return []
    _seen.add(tra_path)

    in_lines = []
    is_concat = False

    with _open_tra(tra_path) as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] in ("ET", "SE", "TB", "StartStream"):
                return [tra_path]           # leaf file with event data
            if t[0] == "IN":
                is_concat = True
                in_lines.append(" ".join(t[1:]).strip())

    if not is_concat:
        return [tra_path]                   # header-only or empty file

    leaf_files = []
    for rel in in_lines:
        child = (tra_path.parent / rel).resolve()
        leaf_files.extend(_resolve_files(child, _seen=_seen))
    return leaf_files


# ── TRA header parser ─────────────────────────────────────────────────────────

def _parse_tra_header(tra_path: Path) -> dict:
    """Read simulation metadata from the top-level TRA file header."""
    hdr = {"nevents_sim": 0, "area_sim": 0.0,
           "norm": None, "norm_params": (), "headers": {}}
    with _open_tra(tra_path) as f:
        for raw in f:
            t = raw.split()
            if not t:
                continue
            k = t[0]
            if k in ("ET", "SE"):
                break
            if k in ("SA", "SimulationStartAreaFarField"):
                try:
                    hdr["area_sim"] = float(t[1])
                    hdr["headers"]["SA"] = t[1]
                except (IndexError, ValueError):
                    pass
            elif k == "TS":
                try:
                    hdr["nevents_sim"] = int(t[1])
                    hdr["headers"]["TS"] = t[1]
                except (IndexError, ValueError):
                    pass
            elif k in ("SP", "SpectralType"):
                if len(t) > 1:
                    try:
                        params = _validate_norm_params(t[1], t[2:])
                        hdr["norm"] = t[1]
                        hdr["norm_params"] = params
                        hdr["headers"]["SP"] = " ".join(t[1:])
                    except ValueError as e:
                        logger.warning(f"SpectralType parse error: {e}")
            elif k in ("BE", "BeamType"):
                hdr["headers"]["BE"] = " ".join(t[1:])
    return hdr


def _validate_norm_params(norm: str, params) -> tuple:
    match norm:
        case "Mono":
            return () if len(params) == 0 else (int(params[0]),)
        case "Linear":
            if len(params) != 2: raise ValueError("Linear needs 2 params")
            return (int(params[0]), int(params[1]))
        case "Gaussian":
            if len(params) != 3: raise ValueError("Gaussian needs 3 params")
            return (float(params[0]), float(params[1]), float(params[2]))
        case "powerlaw":
            if len(params) != 3: raise ValueError("powerlaw needs 3 params")
            return (int(params[0]), int(params[1]), float(params[2]))
        case _:
            raise ValueError(f"Unknown norm '{norm}'")


# ── Effective area ────────────────────────────────────────────────────────────

def _compute_eff_area(Ei_edges: np.ndarray,
                      n_nulambda_bins: int,
                      hdr: dict) -> np.ndarray:
    """
    Per-Ei-bin effective area correction [cm2/count].
    Mirrors RspConverter._get_eff_area_correction exactly.
    """
    norm    = hdr["norm"];    params  = hdr["norm_params"]
    nevents = hdr["nevents_sim"]; area = hdr["area_sim"]
    e_lo = Ei_edges[:-1].copy(); e_hi = Ei_edges[1:].copy()

    if norm is None:
        norm = "Linear"; params = (int(Ei_edges[0]), int(Ei_edges[-1]))
        hdr["norm"] = norm; hdr["norm_params"] = params
        hdr["headers"]["SP"] = f"Linear {params[0]} {params[1]}"
        logger.warning("No SpectralType found; using Linear fallback.")

    match norm:
        case "Linear":
            emin, emax = params
            elo = np.clip(e_lo, emin, emax); ehi = np.clip(e_hi, emin, emax)
            npc = (ehi - elo) / (emax - emin)
        case "Mono":
            if params == ():
                npc = np.array([1.0])
            else:
                emono = params[0]; npc = np.zeros(len(e_lo))
                npc[(e_lo <= emono) & (e_hi >= emono)] = 1.0
        case "powerlaw":
            emin, emax, alpha = params
            elo = np.clip(e_lo, emin, emax); ehi = np.clip(e_hi, emin, emax)
            if alpha == 1.0:
                npc = np.log(ehi / np.where(elo>0, elo, 1e-30)) / np.log(emax/emin)
            else:
                a = 1 - alpha
                npc = (ehi**a - elo**a) / (emax**a - emin**a)
        case "Gaussian":
            mean, sdev, cutoff = params
            emin = mean - cutoff*sdev; emax = mean + cutoff*sdev
            elo = np.clip(e_lo, emin, emax); ehi = np.clip(e_hi, emin, emax)
            def cdf(x): return 0.5*(1 + erf((x-mean)/(sdev*np.sqrt(2))))
            npc = (cdf(ehi)-cdf(elo)) / (cdf(emax)-cdf(emin))
        case _:
            raise ValueError(f"Unknown norm '{norm}'")

    npc_total = npc * nevents / n_nulambda_bins
    zero = npc_total == 0
    if np.any(zero):
        logger.warning("Zero photons in some Ei bins; EFF_AREA=0 there.")
    ea = np.zeros(len(npc_total), dtype=np.float64)
    ea[~zero] = area / npc_total[~zero]
    return ea


# ── Single-file parser ────────────────────────────────────────────────────────

def _parse_single_file(tra_path: Path, max_sq: int) -> dict:
    """Parse one .tra/.tra.gz into raw numpy event arrays."""
    Ei_l=[]; nx_l=[]; ny_l=[]; nz_l=[]
    Eg_l=[]; Ee_l=[]
    x1_l=[]; y1_l=[]; z1_l=[]
    x2_l=[]; y2_l=[]; z2_l=[]
    n_total=0; n_sq_drop=0; ts_footer=0
    in_co=False; cur={}
    NEED = {"Ei","nx","ny","nz","Eg","Ee","x1","y1","z1","x2","y2","z2"}

    def _flush():
        if cur.get("sq_rejected"): return
        if not NEED.issubset(cur): return
        Ei_l.append(cur["Ei"])
        nx_l.append(cur["nx"]); ny_l.append(cur["ny"]); nz_l.append(cur["nz"])
        Eg_l.append(cur["Eg"]); Ee_l.append(cur["Ee"])
        x1_l.append(cur["x1"]); y1_l.append(cur["y1"]); z1_l.append(cur["z1"])
        x2_l.append(cur["x2"]); y2_l.append(cur["y2"]); z2_l.append(cur["z2"])

    with _open_tra(tra_path) as f:
        for raw in f:
            t = raw.split()
            if not t: continue
            k = t[0]
            if k == "SE" or k == "EN":
                if in_co: _flush(); cur={}
                in_co = False; continue
            if k == "ET":
                in_co = len(t)>1 and t[1]=="CO"
                if in_co: n_total += 1
                cur={}; continue
            if k == "TS" and not in_co:
                try:
                    ts_footer = int(t[1])
                except (IndexError, ValueError):
                    pass
                continue
            if not in_co: continue
            if k == "SQ":
                try:
                    if int(t[1]) > max_sq: cur["sq_rejected"]=True; n_sq_drop+=1
                except (IndexError, ValueError): pass
            elif k == "OI":
                try:
                    # tokens 4-6: photon travel direction → negate for source direction
                    cur["nx"]=-float(t[4]); cur["ny"]=-float(t[5]); cur["nz"]=-float(t[6])
                    cur["Ei"]=float(t[10])
                except (IndexError, ValueError): pass
            elif k == "CE":
                try: cur["Eg"]=float(t[1]); cur["Ee"]=float(t[3])
                except (IndexError, ValueError): pass
            elif k == "CD":
                try:
                    cur["x1"]=float(t[1]); cur["y1"]=float(t[2]); cur["z1"]=float(t[3])
                    cur["x2"]=float(t[7]); cur["y2"]=float(t[8]); cur["z2"]=float(t[9])
                except (IndexError, ValueError): pass
    if in_co and cur: _flush()

    def _a(l): return np.array(l, dtype=np.float64)
    return dict(Ei=_a(Ei_l), nx=_a(nx_l), ny=_a(ny_l), nz=_a(nz_l),
                Eg=_a(Eg_l), Ee=_a(Ee_l),
                x1=_a(x1_l), y1=_a(y1_l), z1=_a(z1_l),
                x2=_a(x2_l), y2=_a(y2_l), z2=_a(z2_l),
                n_events_total=n_total, n_sq_drop=n_sq_drop, ts_footer=ts_footer)


def _idx(v: np.ndarray, e: np.ndarray) -> np.ndarray:
    """
    Vectorised bin index for edges array e.
    v in [e[0], e[-1]] → bin index in [0, len(e)-2].
    v outside range    → -1.
    Values exactly on the upper edge are included in the last bin.
    """
    i = np.searchsorted(e, v, side="right") - 1
    in_range = (v >= e[0]) & (v <= e[-1])
    return np.where(in_range, np.clip(i, 0, len(e) - 2), -1).astype(np.int64)


# ── Vectorised bin: one file → sparse result ─────────────────────────────────

def _bin_file(tra_path: Path, max_sq: int,
              Ei_edges, Em_edges, phi_edges,
              nside_nl, nside_pc,
              use_pol, strides, total_bins) -> dict:
    """
    Parse + vectorised-bin one file.
    Returns SPARSE (nonzero_indices, counts) – memory ∝ n_events, not matrix size.
    """
    r = _parse_single_file(tra_path, max_sq=max_sq)
    n_total=r["n_events_total"]; n_sq_drop=r["n_sq_drop"]; ts_footer=r["ts_footer"]

    if len(r["Ei"]) == 0:
        return dict(idx=np.empty(0,np.int64), cts=np.empty(0,np.uint32),
                    n_total=n_total, n_sq_drop=n_sq_drop,
                    n_bad=0, n_oob=0, ts_footer=ts_footer)

    Eg=r["Eg"]; Ee=r["Ee"]; Em=Eg+Ee
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_phi = 1.0 - _ME_KEV/Eg + _ME_KEV/Em

    bad=~np.isfinite(cos_phi)|(cos_phi<-1.0)|(cos_phi>1.0)
    n_bad=int(bad.sum()); g=~bad
    phi_d = np.degrees(np.arccos(cos_phi[g]))

    i_ei = _idx(r["Ei"][g], Ei_edges)
    i_em = _idx(Em[g],       Em_edges)
    i_ph = _idx(phi_d,       phi_edges)
    i_nl = hp.vec2pix(nside_nl, r["nx"][g], r["ny"][g], r["nz"][g],
                      nest=False).astype(np.int64)

    dx=r["x1"][g]-r["x2"][g]; dy=r["y1"][g]-r["y2"][g]; dz=r["z1"][g]-r["z2"][g]
    nd=np.sqrt(dx*dx+dy*dy+dz*dz); safe=nd>0
    
    theta_pc, phi_pc = hp.vec2ang(np.column_stack([
    np.where(safe,dx/nd,0.), np.where(safe,dy/nd,0.), np.where(safe,dz/nd,1.)]))
    i_pc = hp.ang2pix(nside_pc, theta_pc, phi_pc, nest=False).astype(np.int64)

    valid = (i_nl>=0)&(i_ei>=0)&(i_em>=0)&(i_ph>=0)&(i_pc>=0)
    if use_pol:
        s1,s2,_,s4,s5 = strides   # s3 (Pol stride) vanishes since pol_idx=0
        flat = i_nl*s1 + i_ei*s2 + i_em*s4 + i_ph*s5 + i_pc
    else:
        s1,s2,s3,s4 = strides
        flat = i_nl*s1 + i_ei*s2 + i_em*s3 + i_ph*s4 + i_pc

    n_oob = int((~valid).sum())
    # np.unique sorts internally → no pre-sort needed; allocates O(n_events) not O(total_bins)
    uniq, counts = np.unique(flat[valid], return_counts=True)
    return dict(idx=uniq.astype(np.int64), cts=counts.astype(np.uint32),
                n_total=n_total, n_sq_drop=n_sq_drop,
                n_bad=n_bad, n_oob=n_oob, ts_footer=ts_footer)


def _bin_file_worker(args: tuple) -> dict:
    """Picklable wrapper for multiprocessing. Includes path in error messages."""
    path, *rest = args
    try:
        return _bin_file(Path(path), *rest)
    except Exception as exc:
        raise RuntimeError(f"Failed to process '{path}': {type(exc).__name__}: {exc}") from exc


# ── Public: parse_tra_file ────────────────────────────────────────────────────

def parse_tra_file(tra_filename: Union[str, Path], max_sq: int = 7) -> dict:
    """
    Parse a .tra/.tra.gz file (single or nested concat) into event arrays.

    Parameters
    ----------
    tra_filename : path to TRA or concat TRA file
    max_sq       : maximum sequence length accepted (default 7)

    Returns
    -------
    dict with keys: Ei, nu_x/y/z, Em, phi_rad, psichi_x/y/z,
                    n_events_total, n_events_bad_angle, n_events_sq_drop, header
    """
    tra_path = Path(tra_filename)
    _validate_tra_path(tra_path)

    leaf_files = _resolve_files(tra_path)
    tra_header = _parse_tra_header(tra_path)
    logger.info(f"Parsing {len(leaf_files)} file(s) [max_sq={max_sq}]")

    results=[]; ts_sum=0
    for p in leaf_files:
        r = _parse_single_file(p, max_sq=max_sq)
        results.append(r); ts_sum += r["ts_footer"]

    if tra_header["nevents_sim"] == 0 and ts_sum > 0:
        tra_header["nevents_sim"] = ts_sum
        tra_header["headers"]["TS"] = str(ts_sum)

    def _cat(k): return np.concatenate([r[k] for r in results])
    Ei=_cat("Ei"); nx=_cat("nx"); ny=_cat("ny"); nz=_cat("nz")
    Eg=_cat("Eg"); Ee=_cat("Ee")
    x1=_cat("x1"); y1=_cat("y1"); z1=_cat("z1")
    x2=_cat("x2"); y2=_cat("y2"); z2=_cat("z2")

    Em=Eg+Ee
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_phi = 1.0 - _ME_KEV/Eg + _ME_KEV/Em
    bad=~np.isfinite(cos_phi)|(cos_phi<-1)|(cos_phi>1)
    g=~bad; phi_rad=np.arccos(cos_phi[g])
    dx=x1-x2; dy=y1-y2; dz=z1-z2
    nd=np.sqrt(dx**2+dy**2+dz**2); safe=nd>0
    pcx=np.where(safe,dx/nd,0.); pcy=np.where(safe,dy/nd,0.); pcz=np.where(safe,dz/nd,1.)

    return dict(
        Ei=Ei[g], nu_x=nx[g], nu_y=ny[g], nu_z=nz[g],
        Em=Em[g], phi_rad=phi_rad,
        psichi_x=pcx[g], psichi_y=pcy[g], psichi_z=pcz[g],
        n_events_total=sum(r["n_events_total"] for r in results),
        n_events_bad_angle=int(bad.sum()),
        n_events_sq_drop=sum(r["n_sq_drop"] for r in results),
        header=tra_header,
    )


# ── Public: ComptonResponseCreator ────────────────────────────────────────────

def ComptonResponseCreator(
    tra_filename:   Union[str, Path],
    Ei_edges:       np.ndarray,
    Em_edges:       np.ndarray,
    nside_nulambda: int,
    nside_psichi:   int,
    phi_bins:       int,
    save_path:      Union[str, Path],
    pol_bins:       int   = None,
    max_sq:         int   = 7,
    overwrite:      bool  = False,
    compress:       bool  = True,
    n_workers:      int   = 1,
    dtype                 = np.float32,
) -> dict:
    """
    Parameters
    ----------
    tra_filename   : .tra/.tra.gz or nested concat file
    Ei_edges       : bin edges for initial energy [keV]
    Em_edges       : bin edges for measured energy [keV]
    nside_nulambda : HEALPix NSIDE for source direction
    nside_psichi   : HEALPix NSIDE for scatter direction
    phi_bins       : number of Compton-angle bins (0–180°)
    save_path      : output .h5 path
    pol_bins       : polarisation bins; None (default) = omit Pol axis
    max_sq         : max sequence length accepted (default 7)
    overwrite      : overwrite existing output file (default False)
    compress       : Bitshuffle compression in HDF5 (default True)
    n_workers      : 1 = sequential (default, safe in Jupyter)
                     N = N parallel worker processes
    dtype          : numpy dtype for EFF_AREA in HDF5 (default float32).

    Returns
    -------
    stats : dict with keys
        n_events_total, n_events_good, n_events_bad_angle,
        n_events_sq_drop, n_events_out_of_range, sparsity, save_path
    """
    # ── Validate inputs ────────────────────────────────────────────────────
    tra_path  = Path(tra_filename)
    save_path = Path(save_path)
    _validate_tra_path(tra_path)

    if save_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output file already exists: {save_path}\n"
                f"Pass overwrite=True to overwrite it."
            )
        logger.warning(f"Overwriting existing file: {save_path}")

    Ei_edges = np.asarray(Ei_edges, dtype=np.float64)
    Em_edges = np.asarray(Em_edges, dtype=np.float64)
    use_pol  = pol_bins is not None

    # ── Resolve leaf files + header ────────────────────────────────────────
    leaf_files = _resolve_files(tra_path)
    tra_header = _parse_tra_header(tra_path)
    n_files    = len(leaf_files)
    n_workers_eff = min(n_workers, n_files)
    if n_workers_eff < n_workers:
        logger.info(f"Capping n_workers to {n_workers_eff} (number of files)")
    logger.info(f"Found {n_files} tra-file(s)  [workers={n_workers_eff}]")

    # Validate all resolved paths before starting workers – catches path
    # resolution errors and missing files before any worker is spawned.
    missing = [p for p in leaf_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} resolved TRA file(s) do not exist:\n"
            + "\n".join(f"  {p}" for p in missing[:10])
            + ("\n  ..." if len(missing) > 10 else "")
        )

    # ── Build axes ─────────────────────────────────────────────────────────
    phi_edges = np.linspace(0.0, 180.0, phi_bins + 1)
    ax_nl  = HealpixAxis(nside=nside_nulambda, scheme="ring",
                         coordsys=SpacecraftFrame(), label="NuLambda")
    ax_Ei  = Axis(edges=Ei_edges, unit="keV", scale="log",    label="Ei")
    ax_Em  = Axis(edges=Em_edges, unit="keV", scale="log",    label="Em")
    ax_phi = Axis(edges=phi_edges, unit="deg", scale="linear", label="Phi")
    ax_pc  = HealpixAxis(nside=nside_psichi, scheme="ring",
                         coordsys=SpacecraftFrame(), label="PsiChi")

    if use_pol:
        pol_edges = np.linspace(0.0, 180.0, pol_bins + 1)
        ax_pol = Axis(edges=pol_edges, unit="deg", scale="linear", label="Pol")
        axes   = Axes([ax_nl, ax_Ei, ax_pol, ax_Em, ax_phi, ax_pc], copy_axes=False)
    else:
        axes   = Axes([ax_nl, ax_Ei, ax_Em, ax_phi, ax_pc], copy_axes=False)

    n_nl=ax_nl.nbins; n_ei=ax_Ei.nbins; n_po=pol_bins if use_pol else 1
    n_em=ax_Em.nbins; n_ph=ax_phi.nbins; n_pc=ax_pc.nbins
    total_bins = n_nl*n_ei*n_po*n_em*n_ph*n_pc
    shape = (n_nl,n_ei,n_po,n_em,n_ph,n_pc) if use_pol else (n_nl,n_ei,n_em,n_ph,n_pc)

    if use_pol:
        s5=n_pc; s4=n_ph*s5; s3=n_em*s4; s2=n_po*s3; s1=n_ei*s2
        strides=(s1,s2,s3,s4,s5)
    else:
        s4=n_pc; s3=n_ph*s4; s2=n_em*s3; s1=n_ei*s2
        strides=(s1,s2,s3,s4)

    logger.info(
        f"Shape: {'×'.join(str(s) for s in shape)} = {total_bins:,} bins "
        f"({total_bins * np.dtype(dtype).itemsize / 1e9:.2f} GB dense matrix with dtype={np.dtype(dtype)})"
    )

    # ── Parse + bin all files ──────────────────────────────────────────────
    fixed  = (max_sq, Ei_edges, Em_edges, phi_edges,
               nside_nulambda, nside_psichi, use_pol, strides, total_bins)
    parts  = []
    n_total=0; n_bad=0; n_sq_drop=0; n_oob=0; ts_sum=0

    def _collect(r):
        nonlocal n_total, n_bad, n_sq_drop, n_oob, ts_sum
        parts.append(r)
        n_total+=r["n_total"]; n_bad+=r["n_bad"]
        n_sq_drop+=r["n_sq_drop"]; n_oob+=r["n_oob"]; ts_sum+=r["ts_footer"]

    if n_workers_eff == 1:
        for path in tqdm(leaf_files, desc="Parsing TRA files", unit="file"):
            _collect(_bin_file(path, *fixed))
    else:
        ctx_name = "forkserver" if platform.system() != "Windows" else "spawn"
        logger.info(f"Parallel: {ctx_name}, {n_workers_eff} workers")
        worker_args = [(str(p),) + fixed for p in leaf_files]
        with tqdm(total=n_files, desc="Parsing TRA files", unit="file") as pbar:
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=n_workers_eff,
                    mp_context=mp.get_context(ctx_name)) as pool:
                futs = [pool.submit(_bin_file_worker, a) for a in worker_args]
                for fut in concurrent.futures.as_completed(futs):
                    try:
                        _collect(fut.result())
                    except Exception as exc:
                        logger.error(f"Worker failed: {exc}")
                        raise
                    pbar.update(1)

    # ── TS fallback ────────────────────────────────────────────────────────
    if tra_header["nevents_sim"] == 0:
        if ts_sum > 0:
            tra_header["nevents_sim"] = ts_sum
            tra_header["headers"]["TS"] = str(ts_sum)
            logger.info(f"TS from footer(s): {ts_sum:,}")
        else:
            logger.error("TS not found – EFF_AREA will be 0!")
    logger.info(f"Simulated Area: {tra_header['area_sim']} cm2")

    # ── Merge sparse results ───────────────────────────────────────────────
    logger.info("Merging sparse results ...")
    parts = [p for p in parts if len(p["idx"]) > 0]
    if parts:
        all_idx = np.concatenate([p["idx"] for p in parts])
        all_cts = np.concatenate([p["cts"] for p in parts]).astype(np.int64)
        del parts
        order    = np.argsort(all_idx, kind="stable")
        all_idx  = all_idx[order]; all_cts = all_cts[order]; del order
        uniq, starts = np.unique(all_idx, return_index=True); del all_idx
        merged_cts   = np.add.reduceat(all_cts, starts).astype(np.uint32)
        del all_cts
        merged_idx   = uniq; del uniq, starts
    else:
        del parts
        merged_idx = np.empty(0, dtype=np.int64)
        merged_cts = np.empty(0, dtype=np.uint32)

    n_good   = int(merged_cts.sum())
    sparsity = 1.0 - n_good / max(total_bins, 1)
    logger.info(
        f"Events: total={n_total:,}  sq_drop={n_sq_drop:,}  "
        f"bad_angle_drop={n_bad:,}  energy_drop={n_oob:,}  "
        f"good={n_good:,}  sparsity={100*sparsity:.1f}%"
    )

    # ── Effective area ─────────────────────────────────────────────────────
    eff_area = _compute_eff_area(Ei_edges, n_nl, tra_header)
    logger.info(f"EFF_AREA: [min/max] = [{eff_area.min():.6g}, {eff_area.max():.6g}] cm2/count")

    _write_hdf5_response(axes, merged_idx, merged_cts, eff_area, tra_header,
                         save_path, compress, dtype, n_ei, s1, s2,
                         n_po, n_em, n_ph, n_pc, use_pol)

    return dict(
        n_events_total        = n_total,
        n_events_good         = n_good,
        n_events_bad_angle    = n_bad,
        n_events_sq_drop      = n_sq_drop,
        n_events_out_of_range = n_oob,
        sparsity              = sparsity,
        save_path             = str(save_path),
    )


# ── Shared HDF5 writer ────────────────────────────────────────────────────────

def _write_hdf5_response(axes, merged_idx, merged_cts, eff_area, tra_header,
                         save_path, compress, dtype,
                         n_ei, s1, s2, n_po, n_em, n_ph, n_pc, use_pol):
    """
    Write a COSIpy-compatible HDF5 response file chunk by chunk.
    Peak RAM = one chunk buffer. Used by both ComptonResponseCreator
    and PhotoResponseCreator.
    """
    logger.info(f"Writing HDF5 → {save_path} ...")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_inner = n_po * n_em * n_ph * n_pc
    is_idx      = [ax.label in ("NuLambda", "Ei", "Pol") for ax in axes]
    inner_shape = tuple(ax.nbins for ax, ix in zip(axes, is_idx) if not ix)
    chunk_shp   = tuple(1 if ix else ax.nbins for ax, ix in zip(axes, is_idx))
    compr       = hdf5plugin.Bitshuffle() if compress else None
    vmax        = int(merged_cts.max()) if len(merged_cts) else 0
    c_dtype     = np.min_scalar_type(vmax) if vmax > 0 else np.dtype(np.uint8)

    with h5.File(save_path, mode="w") as hf:
        drm = hf.create_group("DRM")
        drm.attrs["VERSION"] = _RSP_VERSION
        drm.attrs["UNIT"]    = "cm2"
        drm.attrs["SPARSE"]  = False

        hdr_grp = drm.create_group("HEADERS")
        headers = tra_header.get("headers", {})
        for k, v in headers.items():
            hdr_grp.attrs[k] = str(v)
        drm.attrs["HEADER_ORDER"] = (
            _inv_perm(np.array([sorted(headers).index(k) for k in headers]))
            if headers else np.array([], dtype=np.int64)
        )

        axes.write(drm.create_group("AXES"))
        desc = drm.create_group("AXIS_DESCRIPTIONS")
        for lbl in axes.labels:
            desc.attrs[lbl] = _AXIS_DESCRIPTION.get(lbl, lbl)

        drm.create_dataset("EFF_AREA",
                           data=np.broadcast_to(eff_area, n_ei).astype(dtype),
                           track_times=False)

        ds = hf.create_dataset("DRM/COUNTS", shape=axes.shape, dtype=c_dtype,
                               chunks=chunk_shp, compression=compr,
                               track_times=False, fillvalue=0)

        if use_pol:
            idx_ranges = [(inl, iei, ipo)
                          for inl in range(axes["NuLambda"].nbins)
                          for iei in range(n_ei)
                          for ipo in range(n_po)]
        else:
            idx_ranges = [(inl, iei)
                          for inl in range(axes["NuLambda"].nbins)
                          for iei in range(n_ei)]

        s3_pol = (n_em * n_ph * n_pc) if use_pol else 0
        buf    = np.zeros(chunk_inner, dtype=np.uint32)

        for chunk_idx in tqdm(idx_ranges, desc="Writing HDF5 chunks",
                              unit="chunk", mininterval=5.0):
            chunk_offset = (chunk_idx[0]*s1 + chunk_idx[1]*s2
                            + (chunk_idx[2]*s3_pol if use_pol else 0))
            chunk_end = chunk_offset + chunk_inner
            lo = int(np.searchsorted(merged_idx, chunk_offset, side="left"))
            hi = int(np.searchsorted(merged_idx, chunk_end,    side="left"))
            if lo < hi:
                buf.fill(0)
                buf[merged_idx[lo:hi] - chunk_offset] = merged_cts[lo:hi]
                ds[chunk_idx] = buf.reshape(inner_shape).astype(c_dtype)

    logger.info(f"Saved → {save_path}")