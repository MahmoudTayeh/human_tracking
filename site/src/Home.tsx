/*
 * Design reminder — «مرصد عمليات هادئ»: editorial technical layout, archival ivory,
 * ink navy, restrained Signal Orange, and motion that explains rather than decorates.
 */
import { useState } from "react";
import {
  ArrowUpRight,
  Check,
  ChevronRight,
  CircleDot,
  Code2,
  Cpu,
  GitBranch,
  LockKeyhole,
  Menu,
  MoveRight,
  Play,
  ScanLine,
  ShieldCheck,
  Sparkles,
  X,
} from "lucide-react";

const githubUrl = "https://github.com/MahmoudTayeh/human_tracking";

const pipeline = [
  {
    number: "01",
    label: "Detect",
    detail: "YOLOv8 locates people in each frame with configurable confidence thresholds.",
  },
  {
    number: "02",
    label: "Track",
    detail: "DeepSort preserves identity across movement, occlusion, and camera time.",
  },
  {
    number: "03",
    label: "Recognize",
    detail: "InsightFace compares embeddings only when the recognition layer is enabled.",
  },
  {
    number: "04",
    label: "Explain",
    detail: "Trajectories, heatmaps, and statistics turn frames into reviewable context.",
  },
];

const signals = [
  {
    icon: ScanLine,
    eyebrow: "VISUAL SIGNAL",
    title: "Bounded, not buried",
    detail: "Every detection gets a visible frame, a track ID, and a place in the story of the scene.",
  },
  {
    icon: Cpu,
    eyebrow: "COMPUTE LAYER",
    title: "Ready for the edge",
    detail: "CUDA support keeps the pipeline practical for live analysis without hiding the trade-offs.",
  },
  {
    icon: ShieldCheck,
    eyebrow: "HUMAN OVERSIGHT",
    title: "Context before action",
    detail: "The system surfaces evidence for review; it does not turn an automated label into a verdict.",
  },
];

function SectionLabel({ children }: { children: string }) {
  return (
    <div className="section-label">
      <span className="section-label__line" />
      <span>{children}</span>
      <span className="section-label__ticks" aria-hidden="true">···</span>
    </div>
  );
}

export default function Home() {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <div className="site-shell">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Human Tracking home">
          <span className="brand-mark" aria-hidden="true">
            <span className="brand-mark__corner brand-mark__corner--tl" />
            <span className="brand-mark__corner brand-mark__corner--br" />
            <span className="brand-mark__dot" />
            <span className="brand-mark__path" />
          </span>
          <span className="brand__name">human tracking</span>
          <span className="brand__edition">/ vision system</span>
        </a>

        <button
          className="menu-toggle"
          type="button"
          aria-label={menuOpen ? "Close navigation" : "Open navigation"}
          aria-expanded={menuOpen}
          onClick={() => setMenuOpen((open) => !open)}
        >
          {menuOpen ? <X size={19} /> : <Menu size={19} />}
        </button>

        <nav className={`site-nav ${menuOpen ? "site-nav--open" : ""}`} aria-label="Primary navigation">
          <a href="#pipeline" onClick={() => setMenuOpen(false)}>Pipeline</a>
          <a href="#signals" onClick={() => setMenuOpen(false)}>Capabilities</a>
          <a href="#responsibility" onClick={() => setMenuOpen(false)}>Responsible use</a>
          <a className="nav-repo" href={githubUrl} target="_blank" rel="noreferrer">
            <GitBranch size={15} />
            <span>Repository</span>
            <ArrowUpRight size={14} />
          </a>
        </nav>
      </header>

      <main id="top">
        <section className="hero-section">
          <div className="hero-copy">
            <div className="eyebrow-row">
              <span className="status-dot" />
              <span>REAL-TIME COMPUTER VISION</span>
              <span className="eyebrow-row__rule" />
              <span>BUILD 01.24</span>
            </div>
            <h1>See movement<br /><em>become context.</em></h1>
            <p className="hero-lede">
              A practical vision pipeline for detecting people, preserving their track across frames, and turning raw movement into evidence a human can review.
            </p>
            <div className="hero-actions">
              <a className="button button--primary" href="#pipeline">
                Review the pipeline <MoveRight size={17} />
              </a>
              <a className="button button--text" href={githubUrl} target="_blank" rel="noreferrer">
                <GitBranch size={17} /> Read the code
              </a>
            </div>
            <div className="hero-footnote">
              <CircleDot size={14} />
              <span>Open-source foundation · Python · GPU-aware</span>
            </div>
          </div>

          <div className="hero-visual" aria-label="Annotated computer vision preview">
            <div className="hero-visual__masthead">
              <span>FIELD FRAME / 01482</span>
              <span>09:42:16 UTC</span>
            </div>
            <div className="tracking-stage">
              <img src="./assets/human-tracking-hero-reference.png" alt="Abstract security scene with computer vision tracking overlays" />
              <div className="tracking-stage__wash" />
              <div className="tracking-box tracking-box--one"><span>ID 06 / 0.94</span></div>
              <div className="tracking-box tracking-box--two"><span>ID 03 / 0.87</span></div>
              <div className="tracking-path tracking-path--one" />
              <div className="tracking-path tracking-path--two" />
              <div className="tracking-stage__axis tracking-stage__axis--x" />
              <div className="tracking-stage__axis tracking-stage__axis--y" />
              <div className="tracking-stage__caption"><span className="live-pip" /> ACTIVE TRACKS <b>06</b></div>
            </div>
            <div className="hero-visual__footer">
              <span>LAT 31.95° N / LON 35.91° E</span>
              <span>FRAME QUALITY <b>98.2%</b></span>
            </div>
          </div>
        </section>

        <section className="signal-strip" aria-label="Project facts">
          <div><span className="signal-strip__value">YOLOv8</span><span className="signal-strip__label">Detection engine</span></div>
          <div><span className="signal-strip__value">DeepSort</span><span className="signal-strip__label">Track continuity</span></div>
          <div><span className="signal-strip__value">InsightFace</span><span className="signal-strip__label">Recognition layer</span></div>
          <div><span className="signal-strip__value">25 FPS</span><span className="signal-strip__label">GPU reference</span></div>
        </section>

        <section className="intro-section section-frame">
          <div className="section-index">01 <span>/</span> 04</div>
          <div className="intro-section__copy">
            <SectionLabel>THE OPERATING IDEA</SectionLabel>
            <h2>A camera sees a frame.<br /><span>The system preserves the thread.</span></h2>
          </div>
          <div className="intro-section__aside">
            <p>Human Tracking is a modular research and engineering project for real-time detection, multi-object tracking, face recognition, and movement analytics.</p>
            <a className="inline-link" href="#responsibility">Understand the guardrails <ChevronRight size={16} /></a>
          </div>
        </section>

        <section id="pipeline" className="pipeline-section section-frame">
          <div className="section-index">02 <span>/</span> 04</div>
          <div className="pipeline-section__heading">
            <SectionLabel>THE PROCESS</SectionLabel>
            <h2>Four layers.<br /><span>One reviewable signal.</span></h2>
          </div>
          <div className="pipeline-rail">
            <div className="pipeline-rail__line" />
            {pipeline.map((step) => (
              <article className="pipeline-step" key={step.number}>
                <div className="pipeline-step__number">{step.number}<span>.</span></div>
                <div className="pipeline-step__marker"><span /></div>
                <h3>{step.label}</h3>
                <p>{step.detail}</p>
              </article>
            ))}
          </div>
        </section>

        <section id="signals" className="signals-section section-frame">
          <div className="section-index">03 <span>/</span> 04</div>
          <div className="signals-section__heading">
            <SectionLabel>WHAT THE SYSTEM READS</SectionLabel>
            <h2>Useful detail,<br /><span>without the fog.</span></h2>
          </div>
          <div className="signals-grid">
            {signals.map(({ icon: Icon, eyebrow, title, detail }) => (
              <article className="signal-card" key={title}>
                <div className="signal-card__meta"><span>OBS / 0{signals.findIndex((signal) => signal.title === title) + 1}</span><span className="signal-card__status">verified</span></div>
                <div className="signal-card__top"><Icon size={20} strokeWidth={1.5} /><span>{eyebrow}</span></div>
                <h3>{title}</h3>
                <p>{detail}</p>
                <span className="signal-card__arrow"><ArrowUpRight size={17} /></span>
              </article>
            ))}
          </div>
        </section>

        <section className="evidence-section section-frame">
          <div className="evidence-visual">
            <img src="./assets/track-example-1.jpg" alt="Computer vision frame with tracking boxes" />
            <div className="evidence-visual__label"><span>ANNOTATION LAYER</span><b>TRACK / 06</b></div>
          </div>
          <div className="evidence-copy">
            <SectionLabel>THE EVIDENCE LAYER</SectionLabel>
            <h2>From “someone moved”<br /><span>to “here is the path.”</span></h2>
            <p>Visual overlays are only the beginning. The project writes the ingredients for reviewable output: tracked videos, statistics, heatmaps, and trajectories that help a team inspect what happened inside a scene.</p>
            <div className="evidence-list">
              <div><Check size={16} /><span>Tracked videos with bounding boxes and names</span></div>
              <div><Check size={16} /><span>Distance, duration, and speed by person</span></div>
              <div><Check size={16} /><span>Movement density and path visualizations</span></div>
            </div>
          </div>
        </section>

        <section id="responsibility" className="responsibility-section">
          <div className="responsibility-section__inner section-frame">
            <div className="section-index">04 <span>/</span> 04</div>
            <div className="responsibility-copy">
              <SectionLabel>THE HUMAN CHECK</SectionLabel>
              <h2>Capability is not<br /><em>permission.</em></h2>
              <p>Face recognition and biometric data require a lawful purpose, informed consent where required, secure handling, limited retention, and human review. This project is a technical foundation—not a substitute for governance.</p>
              <div className="responsibility-tags"><span><LockKeyhole size={14} /> Privacy by design</span><span><ShieldCheck size={14} /> Human oversight</span><span><Sparkles size={14} /> Responsible use</span></div>
            </div>
            <div className="responsibility-card">
              <div className="responsibility-card__top"><span className="stamp">FIELD NOTE / 04</span><GitBranch size={20} /></div>
              <h3>Make the next decision<br />with more context.</h3>
              <p>Read the implementation, run it locally, and adapt the pipeline to a documented use case.</p>
              <a className="button button--light" href={githubUrl} target="_blank" rel="noreferrer">
                Open on GitHub <ArrowUpRight size={16} />
              </a>
            </div>
          </div>
        </section>

        <section className="developer-section section-frame">
          <div className="developer-section__copy">
            <SectionLabel>FOR DEVELOPERS</SectionLabel>
            <h2>Run the line.<br /><span>Inspect the result.</span></h2>
            <p>The project keeps the core path visible: configuration in YAML, the processing loop in Python, and output artifacts that can be inspected after a run.</p>
          </div>
          <div className="code-card">
            <div className="code-card__bar"><span><i /> <i /> <i /></span><span>quick-start.sh</span><Code2 size={15} /></div>
            <pre><code><span className="code-comment"># create a clean environment</span>{"\n"}<span className="code-command">python</span> -m venv venv{"\n"}<span className="code-command">source</span> venv/bin/activate{"\n\n"}<span className="code-comment"># install the vision stack</span>{"\n"}<span className="code-command">pip</span> install -r requirements.txt{"\n\n"}<span className="code-comment"># process a configured input</span>{"\n"}<span className="code-command">python</span> main.py --config configs/config.yaml</code></pre>
          </div>
        </section>

        <section className="closing-section section-frame">
          <div className="closing-section__ornament"><span /><span /><span /></div>
          <p>HUMAN TRACKING / COMPUTER VISION SHOWCASE</p>
          <h2>Make motion<br /><em>legible.</em></h2>
          <a className="button button--primary" href={githubUrl} target="_blank" rel="noreferrer">View the repository <ArrowUpRight size={17} /></a>
        </section>
      </main>

      <footer className="site-footer section-frame">
        <div className="footer-brand"><span className="brand-mark brand-mark--small" aria-hidden="true"><span className="brand-mark__corner brand-mark__corner--tl" /><span className="brand-mark__corner brand-mark__corner--br" /><span className="brand-mark__dot" /><span className="brand-mark__path" /></span><span>Human Tracking</span></div>
        <span>Built for careful eyes · 2026</span>
        <a href={githubUrl} target="_blank" rel="noreferrer">GitHub <ArrowUpRight size={14} /></a>
      </footer>
    </div>
  );
}
