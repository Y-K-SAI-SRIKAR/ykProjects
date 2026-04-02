import { useState, useRef, useCallback } from "react";

const API = "http://localhost:5000";

const styles = `
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #0a0c0f;
    --surface:  #111318;
    --border:   #1e2229;
    --border2:  #2a2f38;
    --text:     #e8eaed;
    --muted:    #5a6070;
    --accent:   #ff4444;
    --accent2:  #ff7a00;
    --green:    #00d084;
    --yellow:   #ffc947;
    --blue:     #4d9fff;
    --red:      #ff2d2d;
    --mono:     'DM Mono', monospace;
    --sans:     'Syne', sans-serif;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    overflow-x: hidden;
  }

  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(255,68,68,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255,68,68,0.03) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
  }

  .app {
    position: relative;
    z-index: 1;
    min-height: 100vh;
    display: grid;
    grid-template-rows: auto 1fr auto;
  }

  /* ── HEADER ── */
  header {
    padding: 28px 48px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid var(--border);
  }
  .logo { display: flex; align-items: center; gap: 12px; }
  .logo-icon {
    width: 36px; height: 36px;
    background: var(--accent);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
    box-shadow: 0 0 20px rgba(255,68,68,0.4);
  }
  .logo-text { font-size: 15px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text); }
  .logo-sub  { font-family: var(--mono); font-size: 10px; color: var(--muted); letter-spacing: 0.12em; text-transform: uppercase; }

  .status-pill {
    font-family: var(--mono); font-size: 11px;
    padding: 6px 14px; border-radius: 100px; border: 1px solid;
    letter-spacing: 0.08em; display: flex; align-items: center; gap: 6px;
  }
  .status-pill.online  { color: var(--green);  border-color: rgba(0,208,132,0.3);  background: rgba(0,208,132,0.07); }
  .status-pill.offline { color: var(--accent); border-color: rgba(255,68,68,0.3);  background: rgba(255,68,68,0.07); }
  .dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
  .dot.pulse { animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100% { opacity:1; transform:scale(1); } 50% { opacity:0.4; transform:scale(0.8); } }

  /* ── MAIN ── */
  main {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
    max-width: 1300px;
    width: 100%;
    margin: 0 auto;
    padding: 48px;
    align-items: start;
  }

  /* ── UPLOAD PANEL ── */
  .upload-panel { padding-right: 48px; border-right: 1px solid var(--border); }
  .section-label {
    font-family: var(--mono); font-size: 10px; color: var(--muted);
    letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 20px;
    display: flex; align-items: center; gap: 10px;
  }
  .section-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }

  .drop-zone {
    border: 1px dashed var(--border2); border-radius: 12px;
    padding: 52px 32px; text-align: center; cursor: pointer;
    transition: all 0.2s; background: rgba(255,255,255,0.01);
    position: relative; overflow: hidden;
  }
  .drop-zone::before {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(255,68,68,0.06) 0%, transparent 65%);
    pointer-events: none;
  }
  .drop-zone:hover, .drop-zone.dragging { border-color: var(--accent); background: rgba(255,68,68,0.04); }
  .drop-zone:hover::before, .drop-zone.dragging::before {
    background: radial-gradient(ellipse at 50% 0%, rgba(255,68,68,0.12) 0%, transparent 65%);
  }
  .drop-icon { font-size: 40px; margin-bottom: 16px; }
  .drop-title { font-size: 16px; font-weight: 700; margin-bottom: 8px; color: var(--text); }
  .drop-sub { font-family: var(--mono); font-size: 12px; color: var(--muted); line-height: 1.7; }
  .drop-browse { color: var(--accent); text-decoration: underline; cursor: pointer; }

  .upload-btn {
    width: 100%; margin-top: 16px; padding: 14px;
    background: var(--accent); color: #fff; border: none; border-radius: 8px;
    font-family: var(--sans); font-size: 14px; font-weight: 700;
    letter-spacing: 0.06em; cursor: pointer; transition: all 0.2s; text-transform: uppercase;
  }
  .upload-btn:hover:not(:disabled) { background: #ff2020; box-shadow: 0 4px 20px rgba(255,68,68,0.4); }
  .upload-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  .preview-box { margin-top: 24px; border-radius: 10px; overflow: hidden; border: 1px solid var(--border); background: var(--surface); }
  .preview-box img { width: 100%; height: 220px; object-fit: contain; background: #0d0f12; display: block; }
  .preview-name { padding: 10px 14px; font-family: var(--mono); font-size: 11px; color: var(--muted); border-top: 1px solid var(--border); display: flex; align-items: center; gap: 8px; }
  .preview-name span { color: var(--text); }

  .info-cards { margin-top: 28px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .info-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px; }
  .info-card-label { font-family: var(--mono); font-size: 9px; color: var(--muted); letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 6px; }
  .info-card-val { font-size: 13px; font-weight: 600; color: var(--text); }

  /* ── RESULTS PANEL ── */
  .results-panel { padding-left: 48px; }

  .loading-state { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 80px 0; gap: 20px; }
  .spinner { width: 44px; height: 44px; border: 2px solid var(--border2); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .loading-text { font-family: var(--mono); font-size: 12px; color: var(--muted); letter-spacing: 0.1em; }

  .empty-state { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 80px 0; gap: 12px; text-align: center; }
  .empty-icon { font-size: 52px; opacity: 0.2; }
  .empty-text { font-family: var(--mono); font-size: 12px; color: var(--muted); letter-spacing: 0.08em; line-height: 1.8; }

  /* Verdict */
  .verdict-card {
    border-radius: 12px; padding: 28px; margin-bottom: 20px;
    border: 1px solid; position: relative; overflow: hidden;
    animation: fadeUp 0.4s ease;
  }
  @keyframes fadeUp { from { opacity:0; transform:translateY(16px); } to { opacity:1; transform:translateY(0); } }
  .verdict-card.accident     { background: rgba(255,68,68,0.06);   border-color: rgba(255,68,68,0.25); }
  .verdict-card.no-accident  { background: rgba(0,208,132,0.05);   border-color: rgba(0,208,132,0.2); }
  .verdict-card.rejected     { background: rgba(255,122,0,0.05);   border-color: rgba(255,122,0,0.25); }
  .verdict-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; }
  .verdict-card.accident::before    { background: var(--accent); }
  .verdict-card.no-accident::before { background: var(--green); }
  .verdict-card.rejected::before    { background: var(--accent2); }

  .verdict-emoji { font-size: 44px; margin-bottom: 12px; display: block; }
  .verdict-label { font-family: var(--mono); font-size: 10px; color: var(--muted); letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 6px; }
  .verdict-result { font-size: 26px; font-weight: 800; line-height: 1; margin-bottom: 8px; }
  .verdict-card.accident    .verdict-result { color: var(--accent); }
  .verdict-card.no-accident .verdict-result { color: var(--green); }
  .verdict-card.rejected    .verdict-result { color: var(--accent2); }

  /* Severity badge */
  .severity-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 100px;
    font-family: var(--mono); font-size: 11px; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 8px;
  }
  .severity-badge.severe   { background: rgba(255,45,45,0.15);  border: 1px solid rgba(255,45,45,0.4);  color: #ff6060; }
  .severity-badge.moderate { background: rgba(255,165,0,0.12);  border: 1px solid rgba(255,165,0,0.35); color: var(--yellow); }
  .severity-badge.mild     { background: rgba(255,215,0,0.10);  border: 1px solid rgba(255,215,0,0.3);  color: #ffd700; }

  .verdict-filename { font-family: var(--mono); font-size: 11px; color: var(--muted); }

  /* Meters */
  .meters { display: flex; flex-direction: column; gap: 12px; margin-bottom: 20px; }
  .meter-header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; }
  .meter-label  { font-size: 13px; font-weight: 600; color: var(--text); }
  .meter-pct    { font-family: var(--mono); font-size: 13px; font-weight: 500; }
  .meter-track  { height: 6px; background: var(--border); border-radius: 100px; overflow: hidden; }
  .meter-fill   { height: 100%; border-radius: 100px; transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1); }
  .meter-fill.accident    { background: linear-gradient(90deg, #ff2020, #ff6060); }
  .meter-fill.no-accident { background: linear-gradient(90deg, #00a060, #00d084); }

  /* Photo score */
  .photo-score-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 18px 20px; margin-bottom: 20px;
    animation: fadeUp 0.4s ease 0.1s both;
  }
  .photo-score-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
  .photo-score-title  { font-family: var(--mono); font-size: 10px; color: var(--muted); letter-spacing: 0.18em; text-transform: uppercase; }
  .photo-score-badge  { font-family: var(--mono); font-size: 11px; padding: 3px 10px; border-radius: 100px; border: 1px solid; font-weight: 500; }
  .badge-real      { color: var(--green);  border-color: rgba(0,208,132,0.3);  background: rgba(0,208,132,0.08); }
  .badge-maybe     { color: var(--yellow); border-color: rgba(255,201,71,0.3); background: rgba(255,201,71,0.08); }
  .badge-uncertain { color: var(--accent2);border-color: rgba(255,122,0,0.3);  background: rgba(255,122,0,0.08); }
  .badge-fake      { color: var(--accent); border-color: rgba(255,68,68,0.3);  background: rgba(255,68,68,0.08); }

  .photo-score-val { font-size: 32px; font-weight: 800; margin-bottom: 8px; line-height: 1; }
  .photo-score-track { height: 4px; background: var(--border); border-radius: 100px; overflow: hidden; margin-bottom: 10px; }
  .photo-score-fill  { height: 100%; border-radius: 100px; transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1); }
  .photo-score-fill.real      { background: var(--green); }
  .photo-score-fill.maybe     { background: var(--yellow); }
  .photo-score-fill.uncertain { background: var(--accent2); }
  .photo-score-fill.fake      { background: var(--accent); }
  .photo-type-text { font-family: var(--mono); font-size: 11px; color: var(--muted); }

  .error-box {
    background: rgba(255,68,68,0.07); border: 1px solid rgba(255,68,68,0.25);
    border-radius: 8px; padding: 16px 18px;
    font-family: var(--mono); font-size: 12px; color: #ff8080;
    animation: fadeUp 0.3s ease;
  }

  footer {
    padding: 20px 48px; border-top: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center;
  }
  .footer-text { font-family: var(--mono); font-size: 11px; color: var(--muted); letter-spacing: 0.06em; }

  @media (max-width: 900px) {
    main { grid-template-columns: 1fr; padding: 24px; }
    .upload-panel { padding-right: 0; border-right: none; border-bottom: 1px solid var(--border); padding-bottom: 32px; margin-bottom: 32px; }
    .results-panel { padding-left: 0; }
    header { padding: 20px 24px; }
    footer { padding: 16px 24px; }
  }
`;

const SEVERITY_META = {
  severe:   { emoji: "🔴", label: "Severe",   cls: "severe"   },
  moderate: { emoji: "🟡", label: "Moderate", cls: "moderate" },
  mild:     { emoji: "🟢", label: "Mild",     cls: "mild"     },
  none:     null,
};

export default function App() {
  const [dragging,  setDragging]  = useState(false);
  const [file,      setFile]      = useState(null);
  const [preview,   setPreview]   = useState(null);
  const [loading,   setLoading]   = useState(false);
  const [result,    setResult]    = useState(null);
  const [error,     setError]     = useState(null);
  const [apiOnline, setApiOnline] = useState(null);
  const inputRef = useRef();

  useState(() => {
    fetch(`${API}/health`)
      .then(r => r.json())
      .then(d => setApiOnline(d.model_loaded))
      .catch(() => setApiOnline(false));
  }, []);

  const handleFile = (f) => {
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);
    setPreview(URL.createObjectURL(f));
  };

  const onDrop = useCallback((e) => {
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, []);

  const onDragOver  = (e) => { e.preventDefault(); setDragging(true); };
  const onDragLeave = () => setDragging(false);
  const onInputChange = (e) => { const f = e.target.files[0]; if (f) handleFile(f); };

  const uploadAndEvaluate = async () => {
    if (!file) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const fd = new FormData();
      fd.append("image", file);
      const res  = await fetch(`${API}/upload`, { method: "POST", body: fd });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setResult(data);
    } catch (err) {
      setError(err.message || "Upload failed. Is the server running?");
    } finally {
      setLoading(false);
    }
  };

  const verdictClass = result
    ? result.rejected ? "rejected" : result.is_accident ? "accident" : "no-accident"
    : "";

  const badgeClass = result
    ? { real:"badge-real", maybe:"badge-maybe", uncertain:"badge-uncertain", fake:"badge-fake" }[result.realness?.type] || "badge-fake"
    : "";

  const severityMeta = result ? SEVERITY_META[result.severity] : null;

  return (
    <>
      <style>{styles}</style>
      <div className="app">

        {/* HEADER */}
        <header>
          <div className="logo">
            <div className="logo-icon">🚨</div>
            <div>
              <div className="logo-text">AccidentAI</div>
              <div className="logo-sub">Detection System</div>
            </div>
          </div>
          <div className={`status-pill ${apiOnline ? "online" : "offline"}`}>
            <div className={`dot ${apiOnline ? "pulse" : ""}`} />
            {apiOnline === null ? "Checking..." : apiOnline ? "Model Online" : "Server Offline"}
          </div>
        </header>

        {/* MAIN */}
        <main>

          {/* LEFT — Upload */}
          <div className="upload-panel">
            <div className="section-label">Upload Image</div>

            <div
              className={`drop-zone ${dragging ? "dragging" : ""}`}
              onDrop={onDrop} onDragOver={onDragOver} onDragLeave={onDragLeave}
              onClick={() => inputRef.current.click()}
            >
              <div className="drop-icon">📁</div>
              <div className="drop-title">Drop image here</div>
              <div className="drop-sub">
                or <span className="drop-browse">browse files</span><br />
                JPG, PNG, BMP, WEBP supported<br />
                Real photos only — drawings will be flagged
              </div>
              <input ref={inputRef} type="file" accept="image/*"
                style={{ display: "none" }} onChange={onInputChange} />
            </div>

            {preview && (
              <div className="preview-box">
                <img src={preview} alt="preview" />
                <div className="preview-name">Selected: <span>{file?.name}</span></div>
              </div>
            )}

            <button
              className="upload-btn"
              onClick={uploadAndEvaluate}
              disabled={!file || loading || apiOnline === false}
            >
              {loading ? "Evaluating..." : "Evaluate Image"}
            </button>

            <div className="info-cards">
              <div className="info-card">
                <div className="info-card-label">Detection</div>
                <div className="info-card-val">Accident / No Accident</div>
              </div>
              <div className="info-card">
                <div className="info-card-label">Photo Filter</div>
                <div className="info-card-val">8-Signal Authenticity Check</div>
              </div>
              <div className="info-card">
                <div className="info-card-label">Severity</div>
                <div className="info-card-val">Severe / Moderate / Mild</div>
              </div>
              <div className="info-card">
                <div className="info-card-label">Input Size</div>
                <div className="info-card-val">224 × 224 px</div>
              </div>
            </div>
          </div>

          {/* RIGHT — Results */}
          <div className="results-panel">
            <div className="section-label">Evaluation Result</div>

            {loading && (
              <div className="loading-state">
                <div className="spinner" />
                <div className="loading-text">Analysing image...</div>
              </div>
            )}

            {!loading && !result && !error && (
              <div className="empty-state">
                <div className="empty-icon">🔍</div>
                <div className="empty-text">
                  Upload an image to begin<br />
                  evaluation. Results will<br />
                  appear here instantly.
                </div>
              </div>
            )}

            {error && <div className="error-box">⚠ {error}</div>}

            {result && !loading && (
              <>
                {/* Verdict */}
                <div className={`verdict-card ${verdictClass}`}>
                  <span className="verdict-emoji">
                    {result.rejected ? "🚫" : result.is_accident ? "🚨" : "✅"}
                  </span>
                  <div className="verdict-label">Detection Result</div>
                  <div className="verdict-result">
                    {result.rejected ? "Image Rejected" : result.result}
                  </div>

                  {/* Severity badge — only shows when accident detected */}
                  {severityMeta && (
                    <div className={`severity-badge ${severityMeta.cls}`}>
                      {severityMeta.emoji} {severityMeta.label} Severity
                    </div>
                  )}

                  <div className="verdict-filename">{result.filename}</div>
                </div>

                {/* Photo authenticity score */}
                <div className="photo-score-card">
                  <div className="photo-score-header">
                    <div className="photo-score-title">Photo Authenticity</div>
                    <div className={`photo-score-badge ${badgeClass}`}>
                      {result.realness?.type_label}
                    </div>
                  </div>
                  <div className="photo-score-val"
                    style={{ color:
                      result.realness?.type === "real"      ? "var(--green)"   :
                      result.realness?.type === "maybe"     ? "var(--yellow)"  :
                      result.realness?.type === "uncertain" ? "var(--accent2)" :
                      "var(--accent)" }}>
                    {result.realness?.score.toFixed(1)}%
                  </div>
                  <div className="photo-score-track">
                    <div
                      className={`photo-score-fill ${result.realness?.type}`}
                      style={{ width: `${result.realness?.score}%` }}
                    />
                  </div>
                  <div className="photo-type-text">
                    {result.realness?.type === "fake"
                      ? "✗ Not a real photograph — confidence penalised"
                      : result.realness?.type === "uncertain"
                        ? "⚠ Image authenticity uncertain"
                        : result.realness?.type === "maybe"
                          ? "~ Possibly a real photo"
                          : "✓ Real photograph confirmed"}
                  </div>
                </div>

                {/* Confidence meters */}
                <div className="meters">
                  {[
                    { label: "Accident Happened",    val: result.scores.accident,    cls: "accident" },
                    { label: "No Accident Happened", val: result.scores.no_accident, cls: "no-accident" },
                  ].map(({ label, val, cls }) => (
                    <div className="meter-row" key={label}>
                      <div className="meter-header">
                        <span className="meter-label">{label}</span>
                        <span className="meter-pct"
                          style={{ color: cls === "accident" ? "var(--accent)" : "var(--green)" }}>
                          {val.toFixed(2)}%
                        </span>
                      </div>
                      <div className="meter-track">
                        <div className={`meter-fill ${cls}`} style={{ width: `${val}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </main>

        {/* FOOTER */}
        <footer>
          <div className="footer-text">Accident Severity Detection System ·</div>
          <div className="footer-text">Made with Image Perception Concepts</div>
        </footer>

      </div>
    </>
  );
}
