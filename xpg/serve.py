"""``xpg serve [RUN_DIR] [--port N]``: live dashboard of a processing run (standard library only).

Polls ``RUN_DIR/events.jsonl`` (written by the pipeline, see events.py): the two passes of the pipeline with stage
timings, live integration progress and the integration mode; the result (space group, cell, resolution, ISa) and the
statistics table from xpg_summary.json; the decisions with their margins, alternatives (with the alternative merges
of a close call) and hints; output files for download; the MR results (xpg_mr.json) when present; plots, flagged
problems, and the log filtered by stage. The summary report (xpg_summary.html) is linked when present. Binds to 127.0.0.1; from the cluster use an SSH tunnel
(ssh -L 8765:localhost:8765 host, then xpg serve there).

Reruns: from a stage whose preceding checkpoint (state_<STAGE>.pkl) exists, with edits to the run's xpg.toml
(TOML tables, e.g. ``[crystal] space_group = 4``), in RUN_DIR/reruns/r<N>_<stage>/ (the earlier checkpoints linked,
``xpg xpg.toml -o . --jobs <stage> ...`` in the background). Each rerun has its own dashboard (?run=<name>) and the
key results of all runs are listed side by side.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .events import read_events

STAGES = ("DETECTOR", "BACKGROUND", "SPOTS", "INDEX", "MASK", "INTEGRATE", "SCALE")     # Pipeline.ORDER


def _dump_toml(doc):
    from .inputs import _toml_value
    out = [f"{k} = {_toml_value(v)}" for k, v in doc.items() if not isinstance(v, dict)]
    for sec, tab in doc.items():
        if isinstance(tab, dict):
            out += ["", f"[{sec}]"] + [f"{k} = {_toml_value(v)}" for k, v in tab.items()]
    return "# XPG input (rerun from xpg serve)\n" + "\n".join(out) + "\n"


def rerun_stages(base):
    """Stages a rerun of the base run can start from (the preceding stage's checkpoint exists)."""
    return [s for i, s in enumerate(STAGES) if i and (base / f"state_{STAGES[i - 1]}.pkl").exists()]


def start_rerun(base, start, edits="", single_pass=False):
    """New rerun directory with the base run's checkpoints before ``start`` and its xpg.toml with ``edits`` (TOML
    tables) applied; the pipeline runs from ``start`` in the background. Returns the rerun's name."""
    import tomllib
    base = Path(base)
    if start not in rerun_stages(base):
        raise ValueError(f"no checkpoint to start {start} from (available: {', '.join(rerun_stages(base))})")
    if not (base / "xpg.toml").exists():
        raise ValueError("the run has no xpg.toml")
    doc = tomllib.loads((base / "xpg.toml").read_text())
    ed = tomllib.loads(edits or "")
    for sec, tab in ed.items():
        if not isinstance(tab, dict):
            raise ValueError(f"edit '{sec}' is not in a [section] table")
        doc.setdefault(sec, {}).update(tab)
    img = doc.get("data", {}).get("images")
    if img and not Path(img).is_absolute():            # relative to the base run's input
        doc["data"]["images"] = str((base / img).resolve())
    rr = base / "reruns"
    rr.mkdir(exist_ok=True)
    n = 1 + max([int(d.name[1:].split("_")[0]) for d in rr.glob("r*_*") if d.name[1:].split("_")[0].isdigit()] or [0])
    d = rr / f"r{n}_{start.lower()}"
    d.mkdir()
    (d / "xpg.toml").write_text(_dump_toml(doc))
    try:
        from .inputs import read_toml
        read_toml(d / "xpg.toml")                     # unknown keys / bad values: refused before anything runs
    except Exception:
        shutil.rmtree(d)
        raise
    i = STAGES.index(start)
    for s in STAGES[:i]:
        f = base / f"state_{s}.pkl"
        if f.exists():
            (d / f.name).symlink_to(f)
    for f in list(base.glob("xpg_*geometry*.json")) + list(base.glob("xpg_*.cbf")):
        shutil.copy2(f, d / f.name)
    cmd = [sys.executable, "-m", "xpg.cli", "xpg.toml", "-o", ".", "--jobs", *STAGES[i:]]
    if single_pass:
        cmd.append("--single-pass")
    with open(d / "rerun.log", "w") as log:
        proc = subprocess.Popen(cmd, cwd=d, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (d / "rerun.json").write_text(json.dumps(dict(start=start, edits=edits, single_pass=bool(single_pass),
                                                  pid=proc.pid, t=time.time())))
    _PROCS[d.name] = proc
    return d.name


_PROCS = {}     # reruns started by this server (status without a pid probe)


def _summary(d):
    p = d / "xpg_summary.json"
    if not p.exists():
        return {}
    try:
        s = json.loads(p.read_text())
    except ValueError:
        return {}
    o = s.get("overall") or {}
    return dict(space_group=s.get("space_group"), d_cc_half=s.get("cc_half_cutoff"), isa=s.get("isa"),
                cc_half=o.get("cc_half"), rmeas=o.get("rmeas"), completeness=o.get("completeness"),
                i_over_sigma=o.get("i_over_sigma"), n_uniq=o.get("n_uniq"), primary=s.get("primary_mtz"))


def list_runs(base):
    """The base run and its reruns: settings, status and key results."""
    out = [dict(name="", start="(base run)", edits="", status="", summary=_summary(base))]
    for d in sorted((base / "reruns").glob("r*_*"), key=lambda d: (len(d.name), d.name)):
        try:
            meta = json.loads((d / "rerun.json").read_text())
        except (OSError, ValueError):
            meta = {}
        log = (d / "rerun.log").read_text(errors="ignore") if (d / "rerun.log").exists() else ""
        proc = _PROCS.get(d.name)
        if "xpg: finished in" in log:
            status = "finished " + log.split("xpg: finished in", 1)[1].split("(")[0].strip()
        elif "Traceback" in log or (proc is not None and proc.poll() not in (None, 0)):
            status = "failed: " + (log.strip().splitlines() or ["?"])[-1][:160]
        elif proc is not None and proc.poll() is None:
            status = "running"
        else:
            status = "stopped"
        out.append(dict(name=d.name, start=meta.get("start", ""), edits=meta.get("edits", ""),
                        single_pass=meta.get("single_pass", False), status=status, summary=_summary(d)))
    return out


def _run_dir(base, name):
    """The base run (name empty) or one of its reruns; None for anything else."""
    if not name:
        return base
    d = (base / "reruns" / name).resolve()
    return d if d.parent == (base / "reruns").resolve() and d.is_dir() else None

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>XPG Monitor</title>
<style>
:root{--bg:#f6f6f3;--card:#fff;--ink:#1d1d1f;--mute:#6b6b70;--line:#e2e2de;--acc:#2f6fdd;--ok:#2e8b57;--warn:#b8740a;--bad:#c0392b;--fm:#7a4fd0;--pend:#c9c9c4}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#131315;--card:#1c1c20;--ink:#ececf0;--mute:#9a9aa3;--line:#2c2c31;--acc:#6ea0ff;--ok:#5cc28a;--warn:#e0a640;--bad:#ff6b5e;--fm:#b18cff;--pend:#45454b}}
:root[data-theme="dark"]{--bg:#131315;--card:#1c1c20;--ink:#ececf0;--mute:#9a9aa3;--line:#2c2c31;--acc:#6ea0ff;--ok:#5cc28a;--warn:#e0a640;--bad:#ff6b5e;--fm:#b18cff;--pend:#45454b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;gap:14px;align-items:baseline;flex-wrap:wrap}
h1{font-size:18px;margin:0;letter-spacing:.02em}.sub{color:var(--mute);font-size:13px;word-break:break-all}
main{padding:16px 20px;display:grid;gap:16px;grid-template-columns:minmax(0,1fr) minmax(0,1fr)}
@media (max-width:820px){main{grid-template-columns:1fr;padding:16px}header{padding:14px 16px}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;min-width:0}
.card h2{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--mute);margin:0 0 10px}
.wide{grid-column:1/-1}
.track{display:grid;grid-template-columns:64px minmax(0,1fr);gap:10px 12px;align-items:center}
.pass{font-size:12px;color:var(--mute);font-weight:600}
.row{display:flex;gap:6px;align-items:stretch;overflow-x:auto;padding-bottom:2px}
.arr{align-self:center;color:var(--pend);font-size:12px}
.node{border:1px solid var(--line);border-left:4px solid var(--pend);border-radius:8px;padding:6px 9px;min-width:96px;cursor:pointer;background:var(--card)}
.node .n{font-weight:600;font-size:12px;white-space:nowrap}.node .s{font-size:12px;color:var(--mute);white-space:nowrap}
.node.done{border-left-color:var(--ok)}.node.run{border-left-color:var(--acc);box-shadow:0 0 0 1px var(--acc) inset}
.node.pend{opacity:.55;cursor:default}.node.sel{background:color-mix(in srgb,var(--acc) 10%,var(--card))}
.tag{display:inline-block;font-size:11px;padding:0 6px;border-radius:9px;border:1px solid var(--line);color:var(--mute);margin-top:3px}
.tag.fm{border-color:var(--fm);color:var(--fm)}.tag.warn{border-color:var(--warn);color:var(--warn)}
.skip{font-size:13px;color:var(--mute);padding:6px 0}
.bar{height:4px;background:var(--line);border-radius:2px;overflow:hidden;margin-top:5px}.bar>i{display:block;height:100%;background:var(--acc)}
table{width:100%;border-collapse:collapse}td,th{padding:4px 8px 4px 0;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}
th{font-size:12px;color:var(--mute);font-weight:600}.kv td:first-child{color:var(--mute);width:40%}
.num td:not(:first-child),.num th:not(:first-child){text-align:right;font-variant-numeric:tabular-nums}
.flag{padding:6px 0;border-bottom:1px solid var(--line);font-size:13px}.flag b{color:var(--warn);font-weight:600;margin-right:6px}.flag.bad b{color:var(--bad)}
.close{color:var(--warn);font-weight:600}
pre{margin:0;max-height:420px;overflow:auto;font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre-wrap;word-break:break-word}
.plots{display:grid;gap:14px;grid-template-columns:repeat(auto-fill,minmax(260px,1fr))}
.plot h3{font-size:12px;font-weight:600;margin:0 0 4px}.plot svg{width:100%;height:120px;display:block}
.plot .ax{font-size:10px;fill:var(--mute)}.plot .ln{fill:none;stroke:var(--acc);stroke-width:1.5}.plot .grid{stroke:var(--line);stroke-width:1}
textarea{width:100%;margin-top:8px;font:12px/1.4 ui-monospace,Menlo,monospace;background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:6px}
select{font:inherit;background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:6px}code{font:12px ui-monospace,Menlo,monospace;white-space:pre-wrap}
a{color:var(--acc)}.muted{color:var(--mute)}button{font:inherit;background:none;border:1px solid var(--line);color:var(--ink);border-radius:6px;padding:2px 8px;cursor:pointer}
</style></head><body>
<header><h1>XPG</h1><span class="sub" id="what">waiting for events…</span><span class="sub" id="status"></span>
<span class="sub" id="which"></span><span style="margin-left:auto" class="sub"><a id="report" href="/report" target="_blank" style="display:none">summary report</a></span></header>
<main>
<section class="card wide"><h2>Pipeline</h2><div class="track" id="track"></div></section>
<section class="card"><h2>Result</h2><table class="kv" id="result"><tr><td colspan="2" class="muted">none yet</td></tr></table></section>
<section class="card"><h2>Statistics <span class="muted" id="statwhat"></span></h2><table class="num" id="stats"><tr><td class="muted">after SCALE</td></tr></table></section>
<section class="card wide" id="dcard" style="display:none"><h2>Decisions</h2><table id="decisions"></table></section>
<section class="card" id="ocard" style="display:none"><h2>Output files</h2><table id="files"></table></section>
<section class="card"><h2>Flagged problems <span id="nflags" class="muted"></span></h2><div id="flags"><div class="muted">none so far</div></div></section>
<section class="card wide" id="mrcard" style="display:none"><h2>Molecular replacement</h2><div id="mr"></div></section>
<section class="card wide" id="mcard" style="display:none"><h2>Plots</h2><div id="plots" class="plots"></div></section>
<section class="card wide"><h2>Runs and reruns</h2><table id="runs"></table>
<div id="rform" style="margin-top:12px"><div class="sub" style="margin-bottom:6px">Rerun from a checkpoint with edited
settings (TOML tables, as in xpg.toml); the decisions' hints suggest what to change.</div>
<div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center"><label>from <select id="rstage"></select></label>
<label><input type="checkbox" id="rsp"> single pass</label><button id="rgo">start rerun</button><span id="rmsg" class="muted"></span></div>
<textarea id="redits" rows="4" placeholder="[crystal]&#10;space_group = 4&#10;&#10;[processing]&#10;prune_image_groups = false"></textarea></div></section>
<section class="card wide"><h2>Log <span id="logwhat" class="muted"></span> <button id="all" style="display:none">all stages</button></h2><pre id="log"></pre></section>
</main>
<script>
const P1=["DETECTOR","BACKGROUND","SPOTS","INDEX","MASK","INTEGRATE","SCALE"],P2=["INTEGRATE","SCALE"];
const RUN=new URLSearchParams(location.search).get("run")||"",RQ=RUN?"&run="+encodeURIComponent(RUN):"";
let off=0,ev=[],nodes={},pass=1,cur=null,sel=null,run=null,frames=null,p2=null,prog=null,summ={},files=[],mr=null;
const metrics=new Map(),plots=new Map(),decisions=new Map();
const METRICS=[
 [/compute backend (.+)$/,m=>["Device",m[1]]],
 [/SPOTS: (\d+) spots from (\d+) strong pixels/,m=>["Spots",(+m[1]).toLocaleString()]],
 [/INDEX: (\d+)\/(\d+) spots indexed \(([\d.]+)%.*cell \[([^\]]+)\]/,m=>["Indexed",m[3]+"% of the spots"]],
 [/estimated sigma_D ([\d.]+) deg, sigma_M ([\d.]+) deg/,m=>["σD / σM",(+m[1]).toFixed(3)+"° / "+(+m[2]).toFixed(3)+"°"]],
 [/(\d+) reflections integrated in ([\d.]+)s/,m=>["Integrated",(+m[1]).toLocaleString()+" reflections"]],
 [/selected (\S+) point group (\S+) -> space group (.+?) \(#/,m=>["Space group",m[3]]],
 [/isotropic resolution estimate \(CC1\/2 = 0\.3 fit\): ([\d.]+) A/,m=>["Resolution",m[1]+" Å (CC½ = 0.3)"]],
 [/ISa = ([\d.]+) \(two-parameter error model/,m=>["ISa",m[1]]],
 [/TWINNING: twin law (\S+), alpha ([\d.]+)/,m=>["Twinning","law "+m[1]+", α "+m[2]]],
 [/tNCS peak (\d+)%/,m=>["tNCS","Patterson peak "+m[1]+"%"]],
];
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
function fmt(s){return s<90?s.toFixed(1)+" s":s<5400?(s/60).toFixed(1)+" min":(s/3600).toFixed(2)+" h"}
function nf(v,d){return v===null||v===undefined||!isFinite(v)?"–":(+v).toFixed(d)}
function nodeOf(name){if(/^second pass/i.test(name)){pass=2;return null}if(/\(pass 2\)/.test(name))pass=2;
 return pass+"|"+name.split(" ")[0].toUpperCase()}
function add(e){
 ev.push(e);
 if(e.type==="run_start"){run=e;frames=e.frames;document.getElementById("what").textContent=e.template+"  ·  frames "+e.frames.join("–");if(!e.second_pass)p2="off (single pass)"}
 if(e.type==="stage_start"){cur=nodeOf(e.stage);if(cur){const n=nodes[cur]||(nodes[cur]={sec:0});n.t0=e.t;n.running=true}}
 if(e.type==="stage_end"&&cur&&nodes[cur]){const n=nodes[cur];n.sec+=e.seconds||0;n.running=false;n.done=true}
 if(e.type==="run_end"){run.end=e;for(const n of Object.values(nodes))n.running=false;if(!Object.keys(nodes).some(k=>k[0]==="2")&&!p2)p2="not run";document.getElementById("report").style.display=""}
 if(e.type==="metric")plots.set(e.name,Object.assign({pass},e));
 if(e.type==="decision"){decisions.set(e.name,Object.assign({pass},e));if(e.name==="integration"&&cur&&nodes[cur])nodes[cur].mode=/^forward/.test(e.choice)?"forward model":"shoebox"}
 if(e.type==="log"){e.nk=cur;for(const[re,f]of METRICS){const m=e.text.match(re);if(m){const[k,v]=f(m);metrics.set(k,v)}}
  const pm=e.text.match(/INTEGRATE: frame (\d+), ([\d.]+)s/);if(pm)prog={k:cur,f:+pm[1],t:+pm[2]};
  const sk=e.text.match(/DECISION second pass: skipped(.*)$/);if(sk)p2="skipped"+sk[1]}
}
function node(k,label){const n=nodes[k],now=Date.now()/1000,st=!n?"pend":n.running?"run":"done";
 const t=!n?"":n.running?fmt(n.sec+now-n.t0):fmt(n.sec);let extra="";
 if(n&&n.mode)extra+=`<div class="tag ${n.mode==="forward model"?"fm":""}">${n.mode}</div>`;
 if(n&&n.running&&prog&&prog.k===k&&frames){const fr=Math.min(1,(prog.f-frames[0]+1)/(frames[1]-frames[0]+1));
  extra+=`<div class="bar"><i style="width:${(100*fr).toFixed(1)}%"></i></div><div class="s">frame ${prog.f}${fr>0.02?" · ~"+fmt(prog.t/fr-prog.t)+" left":""}</div>`}
 return `<div class="node ${st} ${sel===k?"sel":""}" data-k="${k}"><div class="n">${label}</div><div class="s">${st==="run"?"running · ":""}${t}</div>${extra}</div>`}
function track(){const r1=P1.map(s=>node("1|"+s,s)).join('<span class="arr">▸</span>');
 const has2=Object.keys(nodes).some(k=>k[0]==="2");
 const r2=has2||!p2?P2.map(s=>node("2|"+s,s)).join('<span class="arr">▸</span>'):`<div class="skip">${esc(p2)}</div>`;
 return `<div class="pass">pass 1</div><div class="row">${r1}</div><div class="pass">pass 2</div><div class="row">${r2}</div>`}
function chart(x,y,ylab,xlab){const W=300,H=120,L=36,B=18,T=6,R=6;const pts=x.map((v,i)=>[v,y[i]]).filter(p=>p[1]!==null&&isFinite(p[1]));
 if(pts.length<2)return"";const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);let x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
 if(y1===y0){y1+=1;y0-=1}if(x1===x0)x1+=1;const sx=v=>L+(W-L-R)*(v-x0)/(x1-x0),sy=v=>T+(H-T-B)*(1-(v-y0)/(y1-y0));
 const f=v=>Math.abs(v)>=1000||Math.abs(v)<0.01&&v!==0?v.toExponential(1):(+v.toPrecision(3)).toString();
 return `<div class="plot"><h3>${esc(ylab)}</h3><svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"><line class="grid" x1="${L}" y1="${H-B}" x2="${W-R}" y2="${H-B}"/><line class="grid" x1="${L}" y1="${T}" x2="${L}" y2="${H-B}"/>`+
 `<polyline class="ln" points="${pts.map(p=>sx(p[0]).toFixed(1)+","+sy(p[1]).toFixed(1)).join(" ")}"/>`+
 `<text class="ax" x="${L-3}" y="${T+8}" text-anchor="end">${f(y1)}</text><text class="ax" x="${L-3}" y="${H-B}" text-anchor="end">${f(y0)}</text>`+
 `<text class="ax" x="${L}" y="${H-4}">${f(x0)}</text><text class="ax" x="${W-R}" y="${H-4}" text-anchor="end">${f(x1)} ${esc(xlab||"")}</text></svg></div>`}
function result(){const r=new Map(metrics),s=summ||{};
 if(s.space_group)r.set("Space group",s.space_group+(s.space_group_alternatives&&s.space_group_alternatives.length>1?" (also: "+s.space_group_alternatives.filter(g=>g!==s.space_group).join(", ")+")":""));
 if(s.cell)r.set("Cell",s.cell.map((v,i)=>(+v).toFixed(i<3?2:1)).join("  "));
 if(s.cc_half_cutoff)r.set("Resolution",(+s.cc_half_cutoff).toFixed(2)+" Å (CC½ = 0.3)"+(s.aniso&&s.aniso.limits?" · anisotropic "+s.aniso.limits.map(v=>(+v).toFixed(2)).join(" / ")+" Å":""));
 if(s.isa)r.set("ISa",(+s.isa).toFixed(1));if(s.sigma_d)r.set("σD / σM",(+s.sigma_d).toFixed(3)+"° / "+(+s.sigma_m).toFixed(3)+"°");
 if(s.primary_mtz)r.set("Primary output",s.primary_mtz);
 const order=["Space group","Cell","Resolution","ISa","Twinning","tNCS","Primary output","σD / σM","Spots","Indexed","Integrated","Device"];
 const rows=[...r].sort((a,b)=>(order.indexOf(a[0])+99)%99-(order.indexOf(b[0])+99)%99);
 return rows.length?rows.map(([k,v])=>`<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join(""):'<tr><td colspan="2" class="muted">none yet</td></tr>'}
function stats(){const st=summ&&summ.statistics,sa=summ&&summ.statistics_aniso;if(!st)return'<tr><td class="muted">after SCALE</td></tr>';
 const c=[st.overall,st.inner,st.outer].concat(sa?[sa.overall,sa.outer]:[]);
 const pc=v=>v<=1.5?100*v:v,R=[["Resolution (Å)",s=>nf(s.dmax,2)+"–"+nf(s.dmin,2)],["Unique",s=>(+s.n_uniq).toLocaleString()],["Completeness %",s=>nf(pc(s.completeness),1)],
  ["Multiplicity",s=>nf(s.multiplicity,1)],["I/σ",s=>nf(s.i_over_sigma,1)],["Rmeas",s=>nf(s.rmeas,3)],["Rpim",s=>nf(s.rpim,3)],["CC½",s=>nf(s.cc_half,3)],["CCano",s=>nf(s.cc_ano,3)]];
 const grp='<tr><th></th><th colspan="3" style="text-align:center">isotropic</th>'+(sa?'<th colspan="2" style="text-align:center">anisotropic</th>':"")+'</tr>';
 const ell=sa&&sa.ellipsoidal_completeness!=null?`<tr><td>Ellipsoidal compl. %</td><td></td><td></td><td></td><td>${nf(100*sa.ellipsoidal_completeness,1)}</td><td></td></tr>`:"";
 return grp+'<tr><th></th><th>overall</th><th>inner</th><th>outer</th>'+(sa?'<th>overall</th><th>outer</th>':"")+'</tr>'+R.map(([k,f])=>`<tr><td>${k}</td>${c.map(s=>`<td>${f(s)}</td>`).join("")}</tr>`).join("")+ell}
function decis(){const audit=new Map(((summ&&summ.audit_merges)||[]).map(a=>[a.label,a.file]));
 return [...decisions.values()].sort((a,b)=>(b.low_margin?1:0)-(a.low_margin?1:0)).map(d=>{
  const alts=(d.alternatives||[]).map(a=>{const lab=a.replace(/ \(.*$/,""),f=audit.get(lab);return f?`${esc(a)} <a href="/file?name=${encodeURIComponent(f)}${RQ}">${esc(f)}</a>`:esc(a)});
  return `<tr><td style="width:24%" class="muted">${esc(d.name)}${d.pass===2?" (pass 2)":""}</td><td>${d.low_margin?'<span class="close">close call · </span>':""}${esc(d.choice)}`+
  `${d.margin!==null&&d.margin!==undefined?` <span class="muted">(margin ${(+d.margin).toPrecision(3)} ${esc(d.unit||"")})</span>`:""}`+
  `${alts.length?`<div class="muted">alternatives: ${alts.join("; ")}</div>`:""}${d.hint?`<div class="muted">try: ${esc(d.hint)}</div>`:""}</td></tr>`}).join("")}
function size(b){return b<1e6?(b/1e3).toFixed(0)+" kB":b<1e9?(b/1e6).toFixed(1)+" MB":(b/1e9).toFixed(2)+" GB"}
function mrhtml(){if(!mr)return"";const rows=Object.entries(mr).filter(([k,v])=>v&&typeof v==="object"&&"space_group_data"in v);
 const ch=mr.choice&&mr.choice.data;
 return `<table class="num"><tr><th>data</th><th>space group</th><th>copies</th><th>TFZ</th><th>LLG</th><th>R / R-free (placed model)</th></tr>`+rows.map(([k,v])=>
  `<tr><td>${esc(k)}${k===ch?" ◂":""}</td><td>${esc(v.sg||v.space_group_data)}</td><td>${v.placed??"–"}</td><td>${nf(v.TFZ,1)}${v.solved?" ✓":""}</td><td>${nf(v.LLG,0)}</td><td>${nf(v.Rwork,3)} / ${nf(v.Rfree,3)}</td></tr>`).join("")+
  `</table>${mr.choice?`<div class="sub" style="margin-top:6px">${esc(mr.choice.reason)}</div>`:""}`}
function render(){
 document.getElementById("track").innerHTML=track();
 document.querySelectorAll(".node:not(.pend)").forEach(d=>d.onclick=()=>{sel=sel===d.dataset.k?null:d.dataset.k;render()});
 const now=Date.now()/1000,st=document.getElementById("status");
 st.textContent=run&&run.end?`finished in ${fmt(run.end.seconds)}`:run?`running for ${fmt(now-run.t)}`:"";
 document.getElementById("result").innerHTML=result();
 document.getElementById("stats").innerHTML=stats();document.getElementById("statwhat").textContent=summ&&summ.statistics?"(isotropic: to the CC½ = 0.3 limit; anisotropic: inside the cut-off surface)":"";
 const dl=decis();document.getElementById("dcard").style.display=dl?"":"none";document.getElementById("decisions").innerHTML=dl;
 document.getElementById("ocard").style.display=files.length?"":"none";
 document.getElementById("files").innerHTML=files.map(f=>`<tr><td><a href="/file?name=${encodeURIComponent(f.name)}${RQ}">${esc(f.name)}</a></td><td class="muted" style="text-align:right">${size(f.size)}</td></tr>`).join("");
 document.getElementById("mrcard").style.display=mr?"":"none";document.getElementById("mr").innerHTML=mrhtml();
 const fm=new Map();ev.filter(e=>e.type==="flag").forEach(e=>{const k=(e.stage||"")+"|"+e.text.replace(/[\d.]+%?/g,"#");const o=fm.get(k);if(o){o.n++;o.e=e}else fm.set(k,{e,n:1})});
 const fl=[...fm.values()];document.getElementById("nflags").textContent=fl.length?`(${fl.length})`:"";
 document.getElementById("flags").innerHTML=fl.length?fl.slice(-50).reverse().map(({e:f,n})=>`<div class="flag ${/ERROR|failed/i.test(f.text)?"bad":""}"><b>${esc(f.stage||"")}</b>${esc(f.text)}${n>1?` <span class="muted">×${n}</span>`:""}</div>`).join(""):'<div class="muted">none so far</div>';
 const lg=ev.filter(e=>e.type==="log"&&(!sel||e.nk===sel)).slice(-400),pre=document.getElementById("log");
 const atEnd=pre.scrollTop+pre.clientHeight>=pre.scrollHeight-20;pre.textContent=lg.map(e=>e.text).join("\n");if(atEnd)pre.scrollTop=pre.scrollHeight;
 const pl=[...plots.values()];document.getElementById("mcard").style.display=pl.length?"":"none";
 const ph=pl.map(m=>Object.entries(m.series||{}).map(([k,v])=>chart(m.x,v,`${m.title} · ${k}${m.pass===2?" (pass 2)":""}`,m.xlabel)).join("")).join("");
 const pe=document.getElementById("plots");if(pe.dataset.h!==String(ph.length)){pe.innerHTML=ph;pe.dataset.h=String(ph.length)}
 document.getElementById("logwhat").textContent=sel?`(pass ${sel[0]} ${sel.slice(2)})`:"(latest)";document.getElementById("all").style.display=sel?"":"none";
}
document.getElementById("all").onclick=()=>{sel=null;render()};
function sumtxt(m){const f=(v,d)=>v===null||v===undefined?null:(+v).toFixed(d);
 return [m.space_group,m.d_cc_half&&f(m.d_cc_half,2)+" Å",m.isa&&"ISa "+f(m.isa,1),m.cc_half!=null&&"CC½ "+f(m.cc_half,3),
  m.rmeas!=null&&"Rmeas "+f(m.rmeas,3),m.completeness!=null&&f(m.completeness<=1.5?100*m.completeness:m.completeness,1)+"%",m.primary&&m.primary!=="first lattice"&&m.primary].filter(Boolean).join(" · ")}
let stagesKey="";
async function runs(){try{const j=await (await fetch("/api/runs")).json();
 document.getElementById("runs").innerHTML=j.runs.map(r=>`<tr><td><a href="${r.name?"/?run="+encodeURIComponent(r.name):"/"}">${r.name?esc(r.name):"base run"}</a>${(r.name||"")===RUN?" ◂":""}</td>`+
  `<td>${r.name?"from "+esc(r.start)+(r.single_pass?" (single pass)":""):""}${r.edits?`<div><code>${esc(r.edits)}</code></div>`:""}</td><td>${esc(r.status||"")}</td><td>${esc(sumtxt(r.summary||{}))}</td></tr>`).join("");
 const k=j.stages.join(",");if(k!==stagesKey){stagesKey=k;document.getElementById("rstage").innerHTML=j.stages.map(s=>`<option${s==="SCALE"?" selected":""}>${s}</option>`).join("")}
 document.getElementById("rform").style.display=RUN||!j.stages.length?"none":""}catch(e){}}
async function results(){try{summ=await (await fetch("/api/summary?x=1"+RQ)).json();files=(await (await fetch("/api/files?x=1"+RQ)).json()).files;
 const m=await (await fetch("/api/mr?x=1"+RQ)).json();mr=Object.keys(m).length?m:null}catch(e){}}
document.getElementById("rgo").onclick=async()=>{const m=document.getElementById("rmsg");m.textContent="starting…";
 try{const r=await fetch("/api/rerun",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({start:document.getElementById("rstage").value,
  edits:document.getElementById("redits").value,single_pass:document.getElementById("rsp").checked})});const j=await r.json();
  m.textContent=r.ok?`started ${j.name}`:j.error;runs()}catch(e){m.textContent=String(e)}};
if(RUN){document.getElementById("which").innerHTML=`rerun ${esc(RUN)} · <a href="/">base run</a>`;document.getElementById("report").href="/report?run="+encodeURIComponent(RUN)}
runs();setInterval(runs,5000);results();setInterval(results,10000);
async function poll(){try{const r=await fetch("/api/events?offset="+off+RQ);const j=await r.json();j.events.forEach(add);off=j.offset;render()}catch(e){}
 setTimeout(poll,run&&run.end?10000:1500)}
poll();setInterval(render,1000);
</script></body></html>"""


OUTPUT_FILE = re.compile(r"(xpg_[A-Za-z0-9_.+-]+\.(mtz|cif|html|md|json|log|pdf)|xpg\.(log|toml)|XDS_ASCII\.HKL|INTEGRATE\.HKL)")


def output_files(d):
    """The run's output files the dashboard lists for download (no subdirectories, nothing else)."""
    return [dict(name=f.name, size=f.stat().st_size) for f in sorted(d.iterdir())
            if f.is_file() and OUTPUT_FILE.fullmatch(f.name)]


def make_handler(run_dir: Path):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            b = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            u = urlparse(self.path)
            qs = parse_qs(u.query)
            if u.path == "/":
                return self._send(200, PAGE, "text/html; charset=utf-8")
            if u.path == "/api/runs":
                return self._send(200, json.dumps({"runs": list_runs(run_dir), "stages": rerun_stages(run_dir)},
                                                  default=float), "application/json")
            d = _run_dir(run_dir, qs.get("run", [""])[0])
            if d is None:
                return self._send(404, "no such run", "text/plain")
            if u.path == "/api/events":
                try:
                    off = int(qs.get("offset", ["0"])[0])
                except ValueError:
                    off = 0
                evs, new = read_events(d, off)
                return self._send(200, json.dumps({"events": evs, "offset": new}), "application/json")
            if u.path in ("/api/summary", "/api/mr"):
                p = d / ("xpg_summary.json" if u.path == "/api/summary" else "xpg_mr.json")
                return self._send(200, p.read_text() if p.exists() else "{}", "application/json")
            if u.path == "/api/files":
                return self._send(200, json.dumps({"files": output_files(d)}), "application/json")
            if u.path == "/file":
                name = qs.get("name", [""])[0]
                if name in {f["name"] for f in output_files(d)}:
                    b = (d / name).read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Disposition", f'attachment; filename="{name}"')
                    self.send_header("Content-Length", str(len(b)))
                    self.end_headers()
                    self.wfile.write(b)
                    return
            if u.path == "/report":
                p = d / "xpg_summary.html"
                if p.exists():
                    return self._send(200, p.read_bytes(), "text/html; charset=utf-8")
            return self._send(404, "not found", "text/plain")

        def do_POST(self):
            if urlparse(self.path).path != "/api/rerun":
                return self._send(404, "not found", "text/plain")
            # only the dashboard itself may start reruns: a JSON body (a cross-site page cannot send one without a
            # CORS preflight, which is never granted) and no foreign Origin
            origin = self.headers.get("Origin")
            host = self.headers.get("Host", "")
            if not (self.headers.get("Content-Type", "").split(";")[0].strip() == "application/json") or \
                    (origin and urlparse(origin).netloc != host):
                return self._send(403, json.dumps({"error": "forbidden"}), "application/json")
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                name = start_rerun(run_dir, str(req.get("start", "")), str(req.get("edits", "")),
                                   bool(req.get("single_pass")))
            except Exception as e:       # bad stage / edits: reported to the page, nothing started
                return self._send(400, json.dumps({"error": f"{type(e).__name__}: {e}"}), "application/json")
            return self._send(200, json.dumps({"name": name}), "application/json")
    return H


def serve(run_dir=".", port=8765, host="127.0.0.1"):
    run_dir = Path(run_dir).resolve()
    srv = ThreadingHTTPServer((host, port), make_handler(run_dir))
    print(f"xpg serve: monitoring {run_dir} at http://{host}:{port}/  (Ctrl-C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
