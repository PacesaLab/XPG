"""Processing pipeline: the run / resume / checkpoint loop, the decision log and the light stages (DETECTOR,
BACKGROUND, SPOTS, MASK). The other stages are mixins of Pipeline:

    stage_index.py      INDEX: indexing with retries, lattice choice
    stage_integrate.py  INTEGRATE: per-block geometry, shoebox integration, forward model, second-pass decision
    stage_scale.py      SCALE: observation selection, symmetry, scaling, merging, outputs and report
    stage_filters.py    frame filters within SCALE
    stage_lattices.py   extra lattices: triage, joint merge, verdict
"""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import gemmi
import numpy as np

from .config import Params
from .geometry import Geometry
from .ice import detect_ice_rings
from .indexing import IndexingResult
from .init import analyse_background, static_mask
from .integrate import IntegrationModel
from .io.images import open_images
from .lattice import LatticeCandidate
from .models import Experiment
from .spotfind import SpotList, find_spots
from .stage_filters import FilterStage
from .stage_index import IndexStage
from .stage_integrate import IntegrateStage
from .stage_lattices import LatticeStage
from .stage_scale import ScaleStage

class Logger:
    def __init__(self, path=None, echo=True):
        self.f = open(path, "a") if path else None
        self.echo = echo

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        if self.echo:
            print(s, flush=True)
        if self.f:
            self.f.write(s + "\n")
            self.f.flush()


@dataclass
class State:
    params: Params
    geom: Geometry | None = None
    mask: np.ndarray | None = None
    gain: float = 1.0
    background: np.ndarray | None = None
    spots: SpotList | None = None
    indexing: IndexingResult | None = None
    exp: Experiment | None = None          # current experiment (reduced or conventional cell)
    lattice: LatticeCandidate | None = None
    space_group: gemmi.SpaceGroup | None = None
    model: IntegrationModel | None = None
    refl: dict | None = None
    results: dict = field(default_factory=dict)
# extra lattices re-integrated in pass 2 only when their pass-1 merge with the main lattice would be kept
# (_triage_extra_lattices) extra lattices are integrated again in pass 2 only if their pass-1 merge does not clearly
# cost resolution. The pass-1 tolerance is 5%, not the final merge's 1%: pass 2 recovers small losses what to try when
# a decision is a close call (log, xpg_summary.md, xpg serve)
CLOSE_CALL_HINTS = {
    "point group": "the alternatives' merged data are written as xpg_<SG>_merged_iso.mtz (xpg mr searches them); to "
                   "impose one, set [crystal] space_group (and unit_cell) and rerun from the integration: "
                   "xpg xpg.toml -o RUN --jobs INTEGRATE SCALE",
    "space group": "the candidates differ only in systematic absences (screw axes, enantiomorphs): search all of them in "
                   "MR",
    "frame filter": "compare with [processing] prune_image_groups = false, or exclude a range with [data] "
                    "exclude_frames; SCALE only from the final integration: xpg xpg.toml -o RUN --jobs SCALE --single-pass",
    "lattice merge": "both are written (all lattices: xpg_all_lattices_*.mtz; main lattice: xpg_merged_iso.mtz, or "
                     "xpg_lattice1_*.mtz when the merge is the primary output); refine against both",
}

class Pipeline(IndexStage, IntegrateStage, FilterStage, LatticeStage, ScaleStage):
    def __init__(self, params: Params, workdir=None, log=None, second_pass="auto", anisotropy=True,
                 chiral=True):
        self.p = params
        self.wd = Path(workdir or params.workdir)
        self.wd.mkdir(parents=True, exist_ok=True)
        base_log = log or Logger(self.wd / "xpg.log")
        # every log line is mirrored to events.jsonl (stages, flags) for the live dashboard (xpg serve)
        from .events import EventLog
        self.events = EventLog(self.wd)

        def _log(*a):
            base_log(*a)
            self.events.line(" ".join(str(x) for x in a))
        self.log = _log
        self.second_pass = second_pass
        self.anisotropy = anisotropy
        self.chiral = chiral
        self.st = State(params)
        self.images = open_images(params.name_template, params.workdir, params.data_range[0], params.data_range[1],
                                  min_valid=params.min_valid_pixel, overload=params.overload,
                                  offset=params.offset, dark_current_image=params.dark_current_image)
        if params.nx == 0:
            params.nx, params.ny = self.images.nx, self.images.ny
        for key, lo, n, n_exp in getattr(self.images, "recovered_blocks", []):
            self.log(f"xpg: {key} unreadable by HDF5; frames {lo}-{lo + n - 1} ({n} of {n_exp}) recovered from "
                     f"its raw compressed chunks")
        # frames without readable data (missing / corrupt files) are excluded like EXCLUDE_DATA_RANGE
        missing = [f for f in params.frames() if not self.images.available(f)]
        if missing:
            runs, start = [], missing[0]
            for a_, b_ in zip(missing, missing[1:] + [None]):
                if b_ != a_ + 1:
                    runs.append((start, a_))
                    start = b_
            params.exclude_data_range = list(params.exclude_data_range) + runs
            self.log(f"XPG WARNING: {len(missing)} frames have no readable data (missing or corrupt files"
                     f"{': ' + ', '.join(getattr(self.images, 'bad_blocks', [])) if getattr(self.images, 'bad_blocks', []) else ''}); "
                     f"excluded: {', '.join(f'{a_}-{b_}' for a_, b_ in runs)}")
        self.st.geom = Geometry.from_params(params)
        self.timing = {}

    # ------------------------------------------------------------------
    ORDER = ["DETECTOR", "BACKGROUND", "SPOTS", "INDEX", "MASK", "INTEGRATE", "SCALE"]
    # former stage names (legacy JOB= values, old --jobs arguments and checkpoints)
    ALIASES = {"XYCORR": "DETECTOR", "INIT": "BACKGROUND", "COLSPOT": "SPOTS", "IDXREF": "INDEX", "DEFPIX": "MASK",
               "CORRECT": "SCALE", "ALL": None}
    OLD_NAME = {v: k for k, v in ALIASES.items() if v}

    def decide(self, name, choice, margin=None, unit="", low=None, evidence=None, alternatives=None):
        """Record an automatic decision: choice, evidence, alternatives and its margin (posterior probability, log-
        likelihood gap, or distance from the threshold in noise units). Low-margin decisions are flagged in the log,
        the dashboard and the summary report. The latest record per name is kept (pass 2 replaces pass 1)."""
        rec = dict(name=name, choice=str(choice), margin=None if margin is None else float(margin), unit=unit,
                   low_margin=bool(low) if low is not None else False, evidence=evidence or {},
                   alternatives=[str(a) for a in (alternatives or [])])
        if rec["low_margin"] and name in CLOSE_CALL_HINTS:
            rec["hint"] = CLOSE_CALL_HINTS[name]
        dec = [d for d in self.st.results.get("decisions", []) if d["name"] != name]
        self.st.results["decisions"] = dec + [rec]
        self.events.emit("decision", **rec)
        m = "" if margin is None else f"; margin {margin:.3g}{(' ' + unit) if unit else ''}"
        alt = f"; alternatives {', '.join(rec['alternatives'])}" if rec["alternatives"] else ""
        self.log(f"  DECISION {name}: {choice}{m}{alt}{'  [LOW MARGIN: check]' if rec['low_margin'] else ''}"
                 + (f"\n    hint: {rec['hint']}" if rec.get("hint") else ""))
        return rec

    def _state_file(self, job):
        """Checkpoint of a stage: state_<STAGE>.pkl, or the former name's file from older runs."""
        f = self.wd / f"state_{job}.pkl"
        if not f.exists() and job in self.OLD_NAME and (self.wd / f"state_{self.OLD_NAME[job]}.pkl").exists():
            return self.wd / f"state_{self.OLD_NAME[job]}.pkl"
        return f

    def resume(self, first_job):
        """Load the checkpoint of the latest step preceding ``first_job`` ."""
        i = self.ORDER.index(first_job)
        for prev in reversed(self.ORDER[:i]):
            f = self._state_file(prev)
            if f.exists():
                with open(f, "rb") as fh:
                    d = pickle.load(fh)
                for k, v in d.items():
                    setattr(self.st, k, v)
                self.st.params = self.p
                self.log(f"xpg: resumed from checkpoint after {prev}")
                return prev
        return None

    def run(self, jobs=None):
        jobs = [j.upper() for j in (jobs or self.p.jobs)]
        if "ALL" in jobs:
            jobs = list(self.ORDER)
        jobs = [self.ALIASES.get(j, j) or j for j in jobs]
        if jobs == ["PASS2"]:
            # resume an interrupted second pass: start from the pass-1 SCALE checkpoint
            f = self._state_file("SCALE")
            if not f.exists():
                raise RuntimeError("PASS2: no state_SCALE.pkl in the output directory (run SCALE first)")
            with open(f, "rb") as fh:
                for k, v in pickle.load(fh).items():
                    setattr(self.st, k, v)
            self.st.params = self.p
            self.log("xpg: resumed from checkpoint after SCALE (pass 1); running the second pass")
            t0 = t = time.time()
            self.log("\n=== second pass (re-integration with post-refined geometry and constrained cell) ===")
            self.job_integrate(second=True)
            self._checkpoint("INTEGRATE")
            self.job_scale(final=True)
            self.timing["PASS2"] = time.time() - t
            self.log(f"\nxpg: finished in {time.time() - t0:.1f}s  (PASS2 {self.timing['PASS2']:.1f}s)")
            self._write_summary()
            return self.st
        first = min((self.ORDER.index(j) for j in jobs if j in self.ORDER), default=0)
        if first > 0:
            self.resume(self.ORDER[first])
        t0 = time.time()
        self.events.emit("run_start", jobs=jobs, template=self.p.name_template, frames=list(self.p.data_range),
                         second_pass=bool(self.second_pass))
        self.log(f"xpg: processing {self.p.name_template} frames {self.p.data_range[0]}-{self.p.data_range[1]}")
        from .backend import describe
        self.log(f"xpg: compute backend {describe()}")
        for job in self.ORDER:
            if job in jobs:
                t = time.time()
                getattr(self, "job_" + job.lower())()
                self.timing[job] = time.time() - t
                self._checkpoint(job)
        if self.second_pass == "auto" and self._fm_will_run():
            self.second_pass = True            # the forward model replaces the FINAL integration: keep a defined one
        if "SCALE" in jobs and self.second_pass == "auto" and not self._pass2_needed():
            self.second_pass = False
        if "SCALE" in jobs and self.second_pass:
            t = time.time()
            self.log("\n=== second pass (re-integration with post-refined geometry and constrained cell) ===")
            self.job_integrate(second=True)
            self._checkpoint("INTEGRATE")     # later SCALE-only runs start from the pass-2 integration
            self.job_scale(final=True)
            self.timing["PASS2"] = time.time() - t
        self.log(f"\nxpg: finished in {time.time() - t0:.1f}s  ({', '.join(f'{k} {v:.1f}s' for k, v in self.timing.items())})")
        self.events.end_stage()
        self._write_audit_merges()
        self._write_summary()
        self.events.emit("run_end", seconds=round(time.time() - t0, 1), timing=self.timing)
        return self.st

    def _write_audit_merges(self):
        """Decision audit: a low-margin point-group call (posterior < 0.9) also gets merged data in each alternative
        point group, reindexed to its conventional cell (xpg_<SG>_merged_iso.mtz; e.g. P 3 against P 3 2 1, or C 2 on
        the orthohexagonal cell of a P 3 1 lattice), so the close call can be checked by refinement or MR without
        reprocessing. Never fails the run."""
        rec = next((d for d in self.st.results.get("decisions", []) if d["name"] == "point group"), None)
        if rec is None or not rec.get("low_margin") or self.st.space_group is None \
                or not (self.wd / "xpg_unmerged.mtz").exists():
            return
        from .mr import merge_in
        Ast = self.st.exp.crystal.Astar
        done, written = set(), []
        for a in self.st.results.get("point_group_alternatives", []):
            hm = gemmi.SpaceGroup(a["space_group"]).hm
            lab = hm.replace(" ", "")
            k = 2
            while lab in done:                       # another setting of the same group (e.g. the 3 tP axes of cP)
                lab = f"{hm.replace(' ', '')}_{k}"
                k += 1
            done.add(lab)
            try:
                # final indices -> the alternative's conventional basis (the same lattice: integer up to refinement)
                T = np.linalg.solve(a["Astar"], Ast)
                Ti = np.round(T)
                if np.abs(T - Ti).max() > 0.05:
                    self.log(f"  AUDIT: {a['label']} not merged (no integer index transformation)")
                    continue
                # cell of the refined final crystal on the alternative's axes (merge_in applies its constraints)
                Ast_alt = Ast @ np.linalg.inv(Ti)
                G = np.linalg.inv(Ast_alt.T @ Ast_alt)
                ln = np.sqrt(np.diag(G))
                ang = np.degrees(np.arccos([G[1, 2] / (ln[1] * ln[2]), G[0, 2] / (ln[0] * ln[2]),
                                            G[0, 1] / (ln[0] * ln[1])]))
                merge_in(self.wd, hm, self.wd / f"xpg_{lab}", log=lambda *x: None, reindex=Ti.astype(int),
                         cell=np.r_[ln, ang])
                self.log(f"  AUDIT: close point-group call ({rec['choice']}, posterior {rec['margin']:.2f}): data also "
                         f"merged as {a['label']} ({a['prob']:.2f}) in {hm} -> xpg_{lab}_merged_iso.mtz")
                written.append(dict(label=a["label"], space_group=hm, posterior=a["prob"],
                                    file=f"xpg_{lab}_merged_iso.mtz", reindex=Ti.astype(int).tolist()))
            except Exception as e:  # pragma: no cover
                self.log(f"  AUDIT: {a['label']} not merged ({type(e).__name__}: {e})")
        summ = self.wd / "xpg_summary.json"
        if written and summ.exists():           # for xpg mr (searches every alternative)
            out = json.loads(summ.read_text())
            out["audit_merges"] = written
            summ.write_text(json.dumps(out, indent=1, default=float))

    def _write_summary(self):
        """xpg_deposit.cif, xpg_summary.md / .html (report.py) and xpg_report.pdf (pdf_report.py); never fails the run."""
        if self.st.results.get("overall") is None:
            return
        try:
            from .deposit import write_deposition_cif
            write_deposition_cif(self.wd, log=self.log)
        except Exception as e:  # pragma: no cover
            self.log(f"xpg: deposition mmCIF not written ({type(e).__name__}: {e})")
        try:
            from .report import write_summary
            write_summary(self)
            self.log("xpg: summary in xpg_summary.md / xpg_summary.html")
        except Exception as e:  # pragma: no cover
            self.log(f"xpg: summary not written ({type(e).__name__}: {e})")
        from .pdf_report import write_pdf_report
        try:
            img = self.images.get(self.p.data_range[0])
        except Exception:  # noqa: BLE001
            img = None
        write_pdf_report(self.wd, image=img, log=self.log)

    def _checkpoint(self, job):
        try:
            with open(self.wd / f"state_{job}.pkl", "wb") as f:
                pickle.dump({k: v for k, v in self.st.__dict__.items() if k != "background"}, f)
        except Exception as e:  # pragma: no cover
            self.log(f"  (checkpoint failed: {e})")

    # ------------------------------------------------------------------
    def job_detector(self):
        self.log("\n=== DETECTOR ===")
        self.st.mask = static_mask(self.p, self.st.geom)
        self.log(f"  static mask: {100 * (1 - self.st.mask.mean()):.2f}% untrusted (keywords, trusted region)")

    def job_background(self):
        self.log("\n=== BACKGROUND ===")
        if self.st.mask is None:
            self.job_detector()
        f0, f1 = self.p.data_range
        n = min(15, f1 - f0 + 1)
        allowed = np.array(self.p.frames())
        frames = np.unique(allowed[np.linspace(0, len(allowed) - 1, n).round().astype(int)])
        ds = analyse_background(self.images, frames, self.st.geom, self.st.mask, self.p.value_range_trusted,
                                log=self.log)
        self.st.mask = ds.mask
        self.st.gain = self.p.gain if self.p.gain else ds.gain
        self.st.background = ds.background
        self.st.bkg_template = None
        self._background_template()
        try:
            from .io.corrimages import write_init_images
            write_init_images(self.wd, self.st.geom, ds.background, ds.mask, self.st.gain)
        except Exception as e:  # pragma: no cover
            self.log(f"  BACKGROUND: control images not written ({e})")

    def job_spots(self):
        self.log("\n=== SPOTS ===")
        frames = sorted({f for a, b in self.p.spot_range for f in self.p.frames(a, b)})
        sp = find_spots(self.images, frames, gain=self.st.gain, mask=self.st.mask, strong_pixel=self.p.strong_pixel,
                        background_pixel=self.p.background_pixel, min_pixels=self.p.min_pixels_per_spot,
                        max_centroid_dist=self.p.spot_maximum_centroid, cluster_radius=self.p.cluster_radius,
                        log=self.log)
        # resolution filter
        d = self.st.geom.resolution_of_pixel(sp.x, sp.y)
        dmax, dmin = self.p.include_resolution
        keep = (d <= dmax) & (d >= dmin)
        for lo, hi in self.p.exclude_resolution:
            keep &= ~((d <= max(lo, hi)) & (d >= min(lo, hi)))
        ice = detect_ice_rings(d[keep], log=self.log)
        self.st.results["ice_rings"] = ice
        for lo, hi in ice:
            keep &= ~((d >= lo) & (d <= hi))
        self.st.spots = sp.subset(keep)
        self._write_spots()
        try:
            zf = np.asarray(self.st.spots.z)
            nb = max(1, int(np.ceil((zf.max() - zf.min() + 1) / 200))) if zf.size else 1
            h, e = np.histogram(zf, bins=np.arange(np.floor(zf.min()), np.ceil(zf.max()) + nb, nb)) if zf.size else ([], [0])
            self.events.emit("metric", name="spots_per_frame", title=f"Spots per frame ({nb}-frame bins)",
                             x=[float(v) for v in e[:-1]], xlabel="frame", series={"spots": [float(v) / nb for v in h]})
        except Exception:  # pragma: no cover
            pass

    def job_mask(self):
        self.log("\n=== MASK ===")
        g = self.st.exp.geom if self.st.exp else self.st.geom
        dmap = g.resolution_map()
        dmax, dmin = self.p.include_resolution
        m = self.st.mask & (dmap <= dmax) & (dmap >= dmin)
        for lo, hi in self.p.exclude_resolution:
            m &= ~((dmap <= max(lo, hi)) & (dmap >= min(lo, hi)))
        self.st.mask = m
        self.log(f"  MASK: {100 * (1 - m.mean()):.2f}% of pixels untrusted; resolution at edge "
                 f"{g.max_resolution_at_edge():.2f} A (corner), {g.resolution_at_edge_inscribed():.2f} A (inscribed)")
