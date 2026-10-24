import { useEffect, useRef, useState } from "react";
import { api } from "./api.js";
import Nav from "./components/Nav.jsx";
import Hero from "./components/Hero.jsx";
import UploadStep from "./components/UploadStep.jsx";
import MapStep from "./components/MapStep.jsx";
import ResultsStep from "./components/ResultsStep.jsx";

function useHealth() {
  const [health, setHealth] = useState({ state: "checking" });
  useEffect(() => {
    let alive = true;
    const tick = () =>
      api.health().then(
        (d) => alive && setHealth({ state: "online", ...d }),
        () => alive && setHealth({ state: "offline" })
      );
    tick();
    const t = setInterval(tick, 10000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);
  return health;
}

const STEPS = ["Upload", "Map columns", "Results"];

export default function App() {
  const health = useHealth();
  const [step, setStep] = useState(0);
  const [upload, setUpload] = useState(null);
  const [jobId, setJobId] = useState(null);
  const flowRef = useRef(null);

  const go = (s) => {
    setStep(s);
    flowRef.current?.scrollIntoView({ block: "start" });
  };
  const reachable = [true, Boolean(upload), Boolean(jobId)];

  return (
    <>
      <div className="wrap">
        <Nav health={health} />
        <Hero onUpload={() => go(0)} />
      </div>

      <section id="upload" ref={flowRef} className="wrap flow" aria-labelledby="flow-title">
        <div className="flow-head">
          <h2 id="flow-title" className="section-title">
            Bring your
            <br />
            own stream.
          </h2>
          <nav className="tabs" aria-label="Steps">
            {STEPS.map((label, i) => (
              <button
                key={label}
                type="button"
                className="tab"
                aria-current={step === i ? "step" : undefined}
                disabled={!reachable[i]}
                onClick={() => go(i)}
              >
                <span className="mono">{String(i + 1).padStart(2, "0")}</span> {label}
              </button>
            ))}
          </nav>
        </div>

        {step === 0 && (
          <UploadStep
            maxMb={health.max_upload_mb}
            onUploaded={(meta) => {
              setUpload(meta);
              go(1);
            }}
            onJob={(id) => {
              setJobId(id);
              go(2);
            }}
          />
        )}
        {step === 1 && upload && (
          <MapStep
            upload={upload}
            onBack={() => go(0)}
            onJob={(id) => {
              setJobId(id);
              go(2);
            }}
          />
        )}
        {step === 2 && jobId && (
          <ResultsStep
            key={jobId}
            jobId={jobId}
            onRestart={() => {
              setJobId(null);
              setUpload(null);
              go(0);
            }}
          />
        )}
      </section>

      <div className="wrap">
        <footer className="footer">
          <span>TSAD audit · M2N2 (Kim, Park, Choo, AAAI 2024) re-implemented and audited on SMD.</span>
          <span>Runs on uploaded data are exploratory: settings were tuned on SMD.</span>
        </footer>
      </div>
    </>
  );
}
