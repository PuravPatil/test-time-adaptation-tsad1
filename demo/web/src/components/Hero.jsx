const WORDS = ["shifting,", "drifting,", "changing,", "moving,"];

export default function Hero({ onUpload }) {
  return (
    <header className="hero" id="top">
      <span className="eyebrow">Test-time adaptation · M2N2 audit</span>
      <h1>
        <span className="sr-only">When normal keeps shifting, catch what isn't.</span>
        <span className="line" aria-hidden="true">
          <span className="rise">When normal</span>
        </span>
        <span className="line" aria-hidden="true">
          <span className="rise" style={{ animationDelay: "0.12s" }}>keeps</span>
          <span className="rotor">
            <span className="rot">
              {[...WORDS, WORDS[0]].map((w, i) => (
                <span key={i}>{w}</span>
              ))}
            </span>
          </span>
        </span>
        <span className="line" aria-hidden="true">
          <span className="rise dim" style={{ animationDelay: "0.24s" }}>catch what isn't.</span>
        </span>
      </h1>
      <p className="hero-lead">
        Upload any numeric time series. A detector learns its normal part, then adapts as the rest streams in —
        and you see every score, alarm and update.
      </p>
      <div className="cta-row">
        <button type="button" className="btn primary big" onClick={onUpload}>
          Upload a dataset
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4"
            strokeLinecap="round" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6" /></svg>
        </button>
        <button type="button" className="btn big" onClick={onUpload}>Try an SMD sample</button>
      </div>
    </header>
  );
}
