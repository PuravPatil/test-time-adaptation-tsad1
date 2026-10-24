export function Logo({ size = 28 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="#C6F432" strokeWidth="2.4"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M2 12h4l3-7 5 14 3-7h5" />
    </svg>
  );
}

const LABEL = { checking: "Checking API…", online: "API online", offline: "API offline" };

export default function Nav({ health }) {
  return (
    <nav className="nav" aria-label="Main">
      <a className="brand" href="#top">
        <Logo />
        Drift Studio
      </a>
      <div className="nav-links">
        <a href="#upload">Upload</a>
        <a href="#upload">Results</a>
        <a href="#audit">The audit</a>
        <span className={`status-chip ${health.state}`} role="status">
          <span className={`dot ${health.state === "online" ? "pulse" : ""}`} />
          {LABEL[health.state]}
        </span>
      </div>
    </nav>
  );
}
