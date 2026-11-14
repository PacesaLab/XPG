"""The XPG logo ships with the package and appears in the HTML summary, the dashboard and the PDF report."""
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer


def test_logo_assets():
    from xpg.branding import LOGO_PNG, LOGO_SVG, logo_bytes, logo_data_uri
    assert LOGO_PNG.exists() and LOGO_SVG.exists()
    assert logo_bytes("png")[:8] == b"\x89PNG\r\n\x1a\n"
    assert b"<svg" in logo_bytes("svg")[:200]
    assert logo_data_uri("png").startswith("data:image/png;base64,")
    assert logo_data_uri("svg").startswith("data:image/svg+xml;base64,")


def test_summary_html_has_logo():
    from xpg.report import _md_to_html
    html = _md_to_html("# Run\n\nsome text\n")
    assert "class='logo'" in html and "data:image/png;base64," in html
    assert html.index("class='logo'") < html.index("<h1>")


def test_dashboard_serves_logo(tmp_path):
    from xpg.serve import PAGE, make_handler
    assert 'src="/logo.svg"' in PAGE and 'rel="icon"' in PAGE
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        for path, ctype, head in (("/logo.svg", "image/svg+xml", b"<svg"), ("/logo.png", "image/png", b"\x89PNG")):
            with urllib.request.urlopen(base + path) as r:
                assert r.status == 200 and r.headers["Content-Type"] == ctype
                assert head in r.read()[:200]
    finally:
        srv.shutdown()
        srv.server_close()


def test_pdf_report_with_logo(tmp_path):
    from xpg.pdf_report import write_pdf_report
    ev = [dict(t=0, type="run_start", template="x_????.cbf", frames=[1, 90]),
          dict(t=1, type="run_end", seconds=60.0, timing={"SPOTS": 10.0, "INTEGRATE": 40.0, "SCALE": 10.0})]
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev))
    (tmp_path / "xpg_summary.json").write_text(json.dumps(dict(space_group="P 1 21 1", cc_half_cutoff=2.0)))
    out = write_pdf_report(tmp_path, log=lambda *a: None)
    assert out is not None and out.read_bytes()[:4] == b"%PDF" and b"/Subtype /Image" in out.read_bytes()
