# /brag plan: Test-Time Adaptation for TSAD (M2N2 reproduction + Drift Studio)

**Requested:** a full video describing the project from the paper and the code, so it runs 48s instead of the usual ~20s.
**Format:** landscape, 1920×1080, 30fps.
**Tone:** `cinematic` mixed with `polished`: big type, confident claims backed by real numbers, no jokes.

## Rubric answers
- **What it is:** a from-scratch reproduction of M2N2 (AAAI 2024), a time-series anomaly detector that keeps adapting at test time, wrapped in a web app where anyone can run it on their own data.
- **Who it's for:** ML students, reviewers and engineers watching drifting sensor or server metrics. It gives them a detector that doesn't drown in false alarms when "normal" shifts, plus evidence that the paper's numbers hold up.
- **What sets it apart:**
  - It doesn't just claim to match the paper; it shows it. 12 of 14 paper numbers land within ±0.01, over 5 seeds.
  - It isolates where the gain comes from (detrending vs adaptation).
- **Most impressive claim:** SMD 1-4: ours 0.462 / 0.846 / 0.354 vs the paper's 0.463 / 0.845 / 0.354.
- **Visual hook:** a metric line jumps to a new level and a flood of coral alarms sweeps the screen, with "Normal drifts." slamming in on the downbeat.
- **Real UI:**
  - Drift Studio's hero, then the Upload step, then Map columns, then a live SMD machine 1-4 run.
  - That run shows the progress stages, the metric cards and the full metrics panel.
  - All captured from the actual app (FastAPI + React) running locally.
- **Share caption:** "We rebuilt M2N2 (AAAI 2024) from the paper and hit 12 of 14 reported numbers within ±0.01. Then we made it a web app you can stream your own data through."

## Privacy
- Map step uses a synthetic demo file (`plant.csv`, fictional "plant-7" site).
- No secrets, hosts or personal data. Team names and the public repo URL are shown at the user's request.

## Music
- 120 BPM (one beat = 0.5s, one bar = 2s), A minor, progression Am–F–C–G.
- Kick, clap on beats 2 and 4, hats, sub bass, pad and a pluck arpeggio.
- Every scene cut falls on a bar line.
- SFX are tuned to A minor, sit under the music, and play on the grid:
  - a soft whoosh into each cut;
  - blips for counters and bars;
  - soft clicks for the cursor.

## Storyboard (48s = 24 bars)
| # | Time | Scene | On screen |
|---|---|---|---|
| 1 | 0–4 | Hook | A drifting line steps up and coral alarms flood the screen. "Normal drifts." then "Frozen detectors panic." |
| 2 | 4–10 | The paper | "M2N2 · AAAI 2024 · When Model Meets New Normals". Two cards appear one by one: 01 Detrend (EMA equation), 02 Adapt (masked SGD equation). |
| 3 | 10–16 | Rebuilt | "We rebuilt it from the paper." Counters: 76 tests, 30 paper runs, 5 seeds, 38 sensors. Chips: official MLP, one τ, Docker. |
| 4 | 16–24 | Proof | Scoreboard for SMD 1-4, ours vs paper, with bars growing on beats. Then "12 of 14 numbers within ±0.01". |
| 5 | 24–30 | Why it works | AUPRC: 0.173, then 0.336 with Detrend, then 0.354 with Adapt. "Detrending does the heavy lifting." |
| 6 | 30–42 | The app (real UI) | Browser frame: hero, Try sample, Upload, Map columns, Train and stream. A live run moves through Loading → Training → Streaming → Evaluating, then the metric cards and the full metrics panel. |
| 7 | 42–48 | Outro | "Test-Time Adaptation for TSAD." Team, guide, repo, `docker compose up`. |

Transitions: every scene dips through the background (old content out, then new content in). There are no crossfades between busy layouts, except inside the browser frame, where the screens share the same chrome.
