"""Frame filters of SCALE: image groups (Delta-CC1/2), chunks and low batch fitness, each followed by a scale refit."""
from __future__ import annotations

import time

import numpy as np
from .models import d_spacing_from_exp

FILTER_CYCLES = 1       # scaling cycles of each refit after a frame filter

class FilterStage:
    """Pipeline mixin (see pipeline.Pipeline)."""

    def _refit(self, sc):
        """Scale refit after a filter removed observations: warm-started from the current scales, FILTER_CYCLES cycles
        of fit / error model / outlier rejection and a final fit."""
        t0 = time.time()
        sc.run(log=lambda *a: None, select_prior=False, n_cycles=FILTER_CYCLES)
        self._refit_stats = (getattr(self, "_refit_stats", (0, 0.0))[0] + 1,
                             getattr(self, "_refit_stats", (0, 0.0))[1] + time.time() - t0)

    def _prune_image_groups(self, sc, r, uid, stdcutoff=4.0, max_cycles=6, max_fraction=0.2):
        """Removal of bad image ranges: Delta-CC1/2 per image group (~1 degree);
        groups below mean - stdcutoff * sd are removed and the data re-scaled (crystal leaving the beam,
        radiation damage at the end of a sweep, ice). At most ``max_fraction`` of the observations."""
        from .merging import delta_cc_half_groups
        g = self.st.exp.geom
        gsize = max(10, int(round(1.0 / max(abs(g.osc), 1e-6))))
        self.st.results["image_group_size"] = gsize
        frame = np.floor(r["z"]).astype(int) + 1
        group = (frame - self.p.data_range[0]) // gsize
        removed_groups = []
        n0 = sc.active.sum()
        n1 = sc.active.sum()
        for cyc in range(max_cycles):
            Is, ss, act = sc.scaled()
            if act.sum() < 1000:
                break
            gr, dcc, cc_all = delta_cc_half_groups(uid[act], Is[act], group[act])
            ok = np.isfinite(dcc)
            if ok.sum() < 5:
                break
            # robust centre/spread (median, MAD): with a sizeable fraction of bad groups (crystal out of the beam)
            # the mean and sd are dominated by the bad groups themselves and nothing would be flagged
            mu = np.median(dcc[ok])
            sd = 1.4826 * np.median(np.abs(dcc[ok] - mu))
            # absolute floor: on clean data the spread is ~1e-6 and harmless groups would be flagged
            bad = gr[ok & (dcc < mu - stdcutoff * sd) & (dcc < -1e-4)]
            if bad.size == 0:
                break
            drop = act & np.isin(group, bad)
            if (n1 - sc.active.sum() + drop.sum()) > max_fraction * n1:
                break
            sc.active &= ~drop
            removed_groups += bad.tolist()
            self.log(f"  SCALE: image-group filter cycle {cyc + 1}: removed {bad.size} group(s) of {gsize} frames "
                     f"(Delta-CC1/2 < median - {stdcutoff:g} robust sd): frames " +
                     ", ".join(f"{self.p.data_range[0] + b * gsize}-{self.p.data_range[0] + (b + 1) * gsize - 1}"
                               for b in sorted(bad)[:12]) + (" ..." if bad.size > 12 else ""))
            self._refit(sc)
        removed_groups += self._prune_low_fitness(sc, r, uid, group, gsize,
                                                  thr=self.p.prune_fitness_threshold)
        removed_groups += self._prune_chunks(sc, r, uid, group, gsize)
        # short islands left between removed ranges (< 3 groups) carry little data, cannot be scaled stably
        # and fragment the sweep into tiny runs (other programs refuse them): remove them as well
        if removed_groups:
            act_g = np.unique(group[sc.active])
            gaps = np.flatnonzero(np.diff(act_g) > 1)
            starts = np.r_[0, gaps + 1]; ends = np.r_[gaps, len(act_g) - 1]
            isl = [act_g[a_:b_ + 1] for a_, b_ in zip(starts, ends)
                   if b_ - a_ + 1 < 3 and 0 < a_ and b_ < len(act_g) - 1]
            if isl:
                bad = np.concatenate(isl)
                drop = sc.active & np.isin(group, bad)
                if drop.sum() < 0.03 * n0:
                    sc.active &= ~drop
                    removed_groups += bad.tolist()
                    self.log(f"  SCALE: removed {len(isl)} isolated short image range(s) between excluded ranges "
                             f"({drop.sum()} observations)")
                    self._refit(sc)
        if removed_groups:
            self.st.results["removed_image_groups"] = sorted(removed_groups)
            self.log(f"  SCALE: image-group filter removed {100 * (1 - sc.active.sum() / n0):.1f}% of observations")
            fr_ = 100 * (1 - sc.active.sum() / n0)
            self.decide("frame filter", f"{fr_:.1f}% of observations removed", low=fr_ > 20.0,
                        evidence=dict(removed_fraction=fr_ / 100))

    def _prune_chunks(self, sc, r, uid, group, gsize, nchunks=20, max_cycles=2, max_fraction=0.4,
                      max_compl_loss=0.02):
        """Large-chunk Delta-CC1/2 on the high-resolution half of the data (~5% of the sweep per chunk).
        Overall CC1/2 is dominated by strong low-resolution reflections, so stretches that only degrade the
        high-resolution data (crystal moving out of the beam, other crystals entering it, damage) do not show
        in 1-degree groups. A chunk is removed when it lowers the high-resolution CC1/2 (Delta-CC1/2 < 0) and lies > 1 robust sd below the median chunk."""
        from .merging import delta_cc_half_groups
        if not self.p.prune_image_groups or not getattr(self.p, "prune_chunks", True):
            return []
        s2 = 1.0 / np.maximum(d_spacing_from_exp(r["hkl"], self.st.exp), 1e-6) ** 2
        removed = []
        n0 = sc.active.sum()
        n_u0 = np.unique(uid[sc.active]).size
        for cyc in range(max_cycles):
            Is, ss, act = sc.scaled()
            gact = np.unique(group[act])
            if gact.size < 3 * nchunks:
                break
            per = max(int(np.ceil(gact.size / nchunks)), 1)
            chunk_of_group = {gg: i // per for i, gg in enumerate(gact)}
            chunk = np.array([chunk_of_group.get(gg, -1) for gg in group])
            hi = act & (s2 > np.median(s2[act]))
            ch, dcc, cc = delta_cc_half_groups(uid[hi], Is[hi], chunk[hi])
            ok = np.isfinite(dcc)
            if ok.sum() < 8:
                break
            # Outlier test among the contiguous chunks: > 3 robust sd below the median chunk. The sd is floored by the
            # spread of chunks made of randomly drawn image groups (pure noise), so that near-identical chunks are not
            # split by tiny differences. (The permutation null itself is not used as the reference: removing a
            # contiguous chunk takes all equivalents of some reflections and is systematically worse.)
            mu = np.median(dcc[ok])
            sd = 1.4826 * np.median(np.abs(dcc[ok] - mu))
            rng = np.random.default_rng(1234 + cyc)
            pos = np.clip(np.searchsorted(gact, group[hi]), 0, gact.size - 1)
            null = []
            for _ in range(8):
                rank = rng.permutation(gact.size)
                chp = np.where(gact[pos] == group[hi], rank[pos] // per, -1)
                _, d0, _ = delta_cc_half_groups(uid[hi], Is[hi], chp)
                null.append(d0[np.isfinite(d0)])
            null = np.concatenate(null)
            sd0 = 1.4826 * np.median(np.abs(null - np.median(null)))
            sd_eff = max(sd, sd0)
            # floor scaled by how far CC1/2(hi) is from 1: on near-perfect data the differences are noise
            floor = max(1e-4, 0.01 * (1.0 - cc))
            bad = ch[ok & (dcc < -floor) & (dcc < mu - 3.0 * sd_eff)]
            self.log(f"  SCALE: chunk Delta-CC1/2(hi): median {mu:+.5f}, robust sd {sd:.5f} (noise {sd0:.5f}), "
                     f"min {np.min(dcc[ok]):+.5f}; {bad.size} chunk(s) > 3 sd below the median")
            if bad.size == 0:
                break
            drop = act & np.isin(chunk, bad)
            n_u1 = np.unique(uid[act & ~drop]).size
            if (n0 - act.sum() + drop.sum()) > max_fraction * n0 or (n_u0 - n_u1) > max_compl_loss * n_u0:
                self.log(f"  SCALE: chunks degrading high-resolution CC1/2 found but not removed (would lose "
                         f"{100 * (n_u0 - n_u1) / n_u0:.1f}% of unique reflections / {100 * drop.sum() / n0:.0f}% of observations)")
                break
            bad_groups = gact[np.isin(np.array([chunk_of_group[gg] for gg in gact]), bad)]
            f0 = self.p.data_range[0]
            runs, i = [], 0
            bg = np.sort(bad_groups)
            while i < bg.size:
                j = i
                while j + 1 < bg.size and bg[j + 1] == bg[j] + 1:
                    j += 1
                runs.append((bg[i], bg[j])); i = j + 1
            sc.active &= ~drop
            removed += bad_groups.tolist()
            self.log(f"  SCALE: chunk filter (high-resolution Delta-CC1/2 < 0, CC1/2(hi) {cc:.3f}): removed frames "
                     + ", ".join(f"{f0 + a_ * gsize}-{f0 + (b_ + 1) * gsize - 1}" for a_, b_ in runs)
                     + f" ({100 * drop.sum() / n0:.1f}% of observations)")
            self._refit(sc)
        return removed

    def _prune_low_fitness(self, sc, r, uid, group, gsize, thr=0.6, smooth=11, min_run=3, max_compl_loss=0.02,
                           max_fraction=0.4):
        """Exclusion of image ranges with low 'fitness' (crystal moving out of the beam, damage):
        fitness = <I/sigma> of each image group relative to the median group, smoothed by a running median;
        contiguous runs (>= min_run groups) below ``thr`` are removed if completeness drops < max_compl_loss."""
        from scipy.ndimage import median_filter
        if thr <= 0:
            return []
        Is, ss, act = sc.scaled()
        gr = np.unique(group[act])
        f = np.array([np.mean(Is[act & (group == k)] / ss[act & (group == k)]) for k in gr])
        fs = median_filter(f, size=min(smooth, max(len(f) // 2 * 2 - 1, 1)), mode="nearest")
        fs = fs / max(np.median(fs), 1e-9)
        low = fs < thr
        runs, i = [], 0
        while i < len(gr):
            if low[i]:
                j = i
                while j + 1 < len(gr) and low[j + 1]:
                    j += 1
                if j - i + 1 >= min_run:
                    runs.append((i, j))
                i = j + 1
            else:
                i += 1
        if not runs:
            return []
        # stretches whose smoothed mean I/sigma is negative are systematically wrong (real intensities do not
        # average below zero): removed even when the completeness guard would keep weak-but-valid data
        fs_raw = median_filter(f, size=min(smooth, max(len(f) // 2 * 2 - 1, 1)), mode="nearest")
        neg_runs = [(a_, b_) for a_, b_ in runs if np.median(fs_raw[a_:b_ + 1]) < 0]
        if neg_runs:
            badn_ = np.concatenate([gr[a_:b_ + 1] for a_, b_ in neg_runs])
            frac_ = (act & np.isin(group, badn_)).sum() / max(act.sum(), 1)
            if frac_ > max_fraction:
                # a 'stretch' covering most of the sweep is not a bad stretch: the intensities themselves are
                # biased (e.g. an integration model error); removing them would leave nothing to scale
                self.log(f"  SCALE WARNING: mean I/sigma is negative over {100 * frac_:.0f}% of the sweep: not removed "
                         f"(systematic intensity bias, not bad image ranges)")
                neg_runs = []
        if neg_runs:
            badn = np.concatenate([gr[a_:b_ + 1] for a_, b_ in neg_runs])
            dropn = act & np.isin(group, badn)
            sc.active &= ~dropn
            f0 = self.p.data_range[0]
            self.log("  SCALE: image ranges with negative mean I/sigma removed (systematic error): " +
                     ", ".join(f"{f0 + gr[a_] * gsize}-{f0 + (gr[b_] + 1) * gsize - 1}" for a_, b_ in neg_runs) +
                     f" ({100 * dropn.sum() / act.sum():.1f}% of observations)")
            runs = [rn for rn in runs if rn not in neg_runs]
            act = act & ~dropn
            neg_removed = badn.tolist()
            if not runs:
                self._refit(sc)
                return neg_removed
        else:
            neg_removed = []
        bad = np.concatenate([gr[a_:b_ + 1] for a_, b_ in runs])
        drop = act & np.isin(group, bad)
        n_u0 = np.unique(uid[act]).size
        n_u1 = np.unique(uid[act & ~drop]).size
        if drop.sum() > max_fraction * act.sum() or (n_u0 - n_u1) > max_compl_loss * n_u0:
            self.log(f"  SCALE: low-fitness ranges found but not removed (would lose {100 * (n_u0 - n_u1) / n_u0:.1f}% "
                     f"of unique reflections / {100 * drop.sum() / act.sum():.0f}% of observations)")
            if neg_removed:
                self._refit(sc)
            return neg_removed
        sc.active &= ~drop
        f0 = self.p.data_range[0]
        self.log("  SCALE: low-fitness image ranges removed (<I/sigma> < " + f"{thr:g} x median): " +
                 ", ".join(f"{f0 + gr[a_] * gsize}-{f0 + (gr[b_] + 1) * gsize - 1}" for a_, b_ in runs) +
                 f" ({100 * drop.sum() / act.sum():.1f}% of observations)")
        self._refit(sc)
        return neg_removed + bad.tolist()
