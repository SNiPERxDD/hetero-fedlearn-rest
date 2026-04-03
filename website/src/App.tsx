type Metric = {
  label: string;
  value: string;
  detail: string;
};

type FeatureCard = {
  title: string;
  description: string;
  bullets: string[];
};

const heroMetrics: Metric[] = [
  { label: "Modes", value: "2", detail: "Baseline runtime plus DFS-lite extension." },
  { label: "Validation", value: "0.9737", detail: "Verified end-to-end accuracy on the default extended run." },
  { label: "Tests", value: "18", detail: "Backend, browser, onboarding, control-plane, Docker-backed, and smoke coverage." },
  { label: "Workers", value: "HTTP", detail: "Heterogeneous workers communicate through retry-aware REST calls." },
];

const overviewSteps = [
  "Partition the training data once and initialize workers with local shards.",
  "Transmit only model parameters during each federated round.",
  "Aggregate worker updates with weighted FedAvg on the master.",
  "Evaluate the global model on the validation split after every round."
];

const dfsCards: FeatureCard[] = [
  {
    title: "NameNode-style Master",
    description: "The DFS-lite master tracks block placement, cluster health, validation history, and asynchronous round execution.",
    bullets: ["Block metadata map", "Background training thread", "Round telemetry APIs"]
  },
  {
    title: "DataNode-style Worker",
    description: "Each worker persists its assigned block to local disk and reloads it during training instead of receiving raw data again.",
    bullets: ["CSV block persistence", "Local storage accounting", "Round-local metrics"]
  },
  {
    title: "Live Dashboards",
    description: "The system exposes a master control plane and worker dashboards with visible proof of locality, progress, runtime health, and cluster registration.",
    bullets: ["Cluster state", "Worker health table", "Storage usage and local loss"]
  },
  {
    title: "Bootstrap Path",
    description: "The repo now ships explicit operator launchers for macOS/Linux, Windows batch startup, and Windows firewall onboarding.",
    bullets: ["start_dashboard.py", "start_master.sh", "start_worker.bat"]
  }
];

const validationItems = [
  "Browser-level Chromium flow against the live DFS-lite dashboards",
  "Baseline worker API and end-to-end HTTP federated training tests",
  "DFS-lite persistence, failover, and idempotent start control coverage",
  "Master-side runtime config mutation and CSV dataset upload tests",
  "Worker self-registration into the master control plane",
  "Onboarding script verification for Bash, batch, and PowerShell paths",
  "Docker-backed worker validation and live master orchestration smoke tests"
];

const technicalCards: FeatureCard[] = [
  {
    title: "Weighted FedAvg",
    description: "The master combines worker updates in proportion to sample counts so the global model respects shard sizes.",
    bullets: ["Sample-weighted aggregation", "Deterministic defaults", "Validation after every round"]
  },
  {
    title: "Retry-aware HTTP",
    description: "Worker calls are executed with explicit timeout and retry settings to handle slow or transiently unavailable nodes.",
    bullets: ["Config-driven timeouts", "Retry backoff", "Graceful worker drop behavior"]
  },
  {
    title: "Model State Preservation",
    description: "Workers use incremental `SGDClassifier` training with warm state so training rounds update a persistent model instead of reinitializing it.",
    bullets: ["partial_fit", "warm_start", "JSON-safe NumPy transport"]
  },
  {
    title: "Disk-backed Reuse",
    description: "DFS-lite physically writes training blocks to disk and reopens them for each round, proving data-local execution.",
    bullets: ["CSV-backed blocks", "Storage bytes tracked", "Round-by-round reuse"]
  }
];

const repoTree = [
  "master/master.py",
  "worker/worker.py",
  "master/master_dfs.py",
  "worker/worker_dfs.py",
  "config.json",
  "config_extended.json",
  "start_dashboard.py",
  "start_master.sh",
  "start_worker.bat",
  "scripts/windows/onboard_worker.ps1",
  "tests/test_federated_workflow.py",
  "tests/test_dfs_lite_workflow.py",
  "tests/test_dashboard_ui.py",
  "tests/test_onboarding_scripts.py"
];

const commands = [
  {
    title: "Baseline Local Runtime",
    command: "python3 -m master.master --config config.json --log-level INFO"
  },
  {
    title: "DFS-lite Master",
    command: "python3 -m master.master_dfs --config config_extended.json --host 127.0.0.1 --port 8080 --auto-start"
  },
  {
    title: "DFS-lite Worker",
    command: "python3 -m worker.worker_dfs --port 5001 --worker-id worker_1 --storage-dir /tmp/hetero-fedlearn-worker-1"
  },
  {
    title: "Python Quick Start",
    command: "python3 start_dashboard.py --allow-unsupported-python --master-port 18080"
  }
];

const dashboardPanels = [
  { title: "Cluster State", value: "Completed", accent: "teal" },
  { title: "Current Round", value: "10 / 10", accent: "rust" },
  { title: "Validation Accuracy", value: "0.9737", accent: "slate" },
  { title: "Validation Loss", value: "0.1225", accent: "teal" },
];

function App() {
  return (
    <div className="site-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">HF</span>
          <span>Hetero FedLearn REST</span>
        </div>
        <nav className="topnav">
          <a href="#architecture">Architecture</a>
          <a href="#dfs-lite">DFS-lite</a>
          <a href="#quick-start">Quick Start</a>
          <a href="#validation">Validation</a>
        </nav>
      </header>

      <main>
        <section className="hero panel-grid">
          <div className="hero-copy">
            <p className="eyebrow">Distributed Federated Learning Over HTTP</p>
            <h1>Preserved baseline runtime. Disk-local DFS-lite extension. Real verification.</h1>
            <p className="lede">
              Hetero FedLearn REST is a production-minded federated learning system for heterogeneous worker clusters.
              The baseline path keeps raw partitions fixed at workers after initialization, while the DFS-lite extension
              adds disk-backed block locality, live dashboards, browser-driven worker registration, dataset upload, operator launchers, and stronger verification.
            </p>
            <div className="hero-actions">
              <a className="button button-primary" href="#architecture">View Architecture</a>
              <a className="button button-secondary" href="#quick-start">Quick Start</a>
            </div>
          </div>
          <div className="hero-aside">
            <div className="signal-card">
              <div className="signal-header">
                <span>System Snapshot</span>
                <span className="status-dot">Verified</span>
              </div>
              <div className="metric-grid">
                {heroMetrics.map((metric) => (
                  <article key={metric.label} className="metric-card">
                    <p>{metric.label}</p>
                    <strong>{metric.value}</strong>
                    <span>{metric.detail}</span>
                  </article>
                ))}
              </div>
            </div>
          </div>
        </section>

        <section id="architecture" className="section">
          <div className="section-heading">
            <p className="eyebrow">System Overview</p>
            <h2>Train across distributed workers without moving raw data every round.</h2>
          </div>
          <div className="architecture-layout">
            <div className="diagram">
              <div className="diagram-node master-node">
                <span>Master</span>
                <strong>Partition, aggregate, validate</strong>
              </div>
              <div className="diagram-link">Initialize once</div>
              <div className="worker-row">
                <div className="diagram-node worker-node">
                  <span>Worker 1</span>
                  <strong>Shard A</strong>
                </div>
                <div className="diagram-node worker-node">
                  <span>Worker 2</span>
                  <strong>Shard B</strong>
                </div>
              </div>
              <div className="diagram-link">Model parameters only</div>
              <div className="diagram-node aggregate-node">
                <span>FedAvg</span>
                <strong>Weighted aggregation + validation</strong>
              </div>
            </div>
            <div className="overview-notes">
              <ol>
                {overviewSteps.map((step) => (
                  <li key={step}>{step}</li>
                ))}
              </ol>
            </div>
          </div>
        </section>

        <section id="dfs-lite" className="section">
          <div className="section-heading">
            <p className="eyebrow">DFS-lite Extension</p>
            <h2>Move training to the blocks, not the blocks to training.</h2>
          </div>
          <div className="card-grid">
            {dfsCards.map((card) => (
              <article key={card.title} className="feature-card">
                <h3>{card.title}</h3>
                <p>{card.description}</p>
                <ul>
                  {card.bullets.map((bullet) => (
                    <li key={bullet}>{bullet}</li>
                  ))}
                </ul>
              </article>
            ))}
          </div>
        </section>

        <section className="section dashboard-section">
          <div className="section-heading">
            <p className="eyebrow">Dashboard Surface</p>
            <h2>Live telemetry for cluster state, block placement, and worker storage.</h2>
          </div>
          <div className="dashboard-shell">
            <div className="dashboard-window">
              <div className="window-topline">
                <span>Master Dashboard</span>
                <span>http://127.0.0.1:8080/</span>
              </div>
              <div className="dashboard-metrics">
                {dashboardPanels.map((panel) => (
                  <div key={panel.title} className={`dashboard-panel ${panel.accent}`}>
                    <span>{panel.title}</span>
                    <strong>{panel.value}</strong>
                  </div>
                ))}
              </div>
              <div className="telemetry-grid">
                <div className="telemetry-block chart-block">
                  <span>Validation Trajectory</span>
                  <div className="sparkline">
                    <i style={{ height: "32%" }} />
                    <i style={{ height: "48%" }} />
                    <i style={{ height: "63%" }} />
                    <i style={{ height: "74%" }} />
                    <i style={{ height: "88%" }} />
                    <i style={{ height: "96%" }} />
                  </div>
                </div>
                <div className="telemetry-block">
                  <span>Block Map</span>
                  <p>blk_92de6034 → worker_1</p>
                  <p>blk_7dc6ded8 → worker_2</p>
                </div>
                <div className="telemetry-block">
                  <span>Worker Health</span>
                  <p>worker_1 · ready · 181195 B</p>
                  <p>worker_2 · ready · 180310 B</p>
                </div>
              </div>
            </div>
            <div className="dashboard-window worker-window">
              <div className="window-topline">
                <span>Worker Dashboard</span>
                <span>http://127.0.0.1:5001/</span>
              </div>
              <div className="worker-stack">
                <div className="worker-strip">
                  <span>Worker State</span>
                  <strong>Ready</strong>
                </div>
                <div className="worker-strip">
                  <span>Storage Usage</span>
                  <strong>181195 B</strong>
                </div>
                <div className="worker-strip">
                  <span>Last Local Loss</span>
                  <strong>0.0758</strong>
                </div>
                <div className="worker-strip">
                  <span>Block Count</span>
                  <strong>1</strong>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section id="quick-start" className="section">
          <div className="section-heading">
            <p className="eyebrow">Quick Start</p>
            <h2>Operator paths for local simulation, DFS-lite orchestration, and one-command launch.</h2>
          </div>
          <div className="command-grid">
            {commands.map((item) => (
              <article key={item.title} className="command-card">
                <h3>{item.title}</h3>
                <pre><code>{item.command}</code></pre>
              </article>
            ))}
          </div>
        </section>

        <section className="section">
          <div className="section-heading">
            <p className="eyebrow">Platform Onboarding</p>
            <h2>Three operator paths, each mapped to the actual repo entry points.</h2>
          </div>
          <div className="operator-grid">
            <article className="operator-card">
              <h3>Localhost Demo</h3>
              <p>`start_dashboard.py` launches local worker containers, waits for health, forwards the master config, and removes containers on exit.</p>
            </article>
            <article className="operator-card">
              <h3>Master Control Plane</h3>
              <p>The master dashboard can now register workers, upload a CSV dataset, update training settings, and start the asynchronous federated loop directly from the browser.</p>
            </article>
            <article className="operator-card">
              <h3>Worker Self-Registration</h3>
              <p>The worker dashboard can register itself with the master by posting its advertised endpoint, while Windows onboarding still handles firewall and container startup.</p>
            </article>
          </div>
        </section>

        <section id="validation" className="section validation-section">
          <div className="section-heading">
            <p className="eyebrow">Validation</p>
            <h2>This project is presented with evidence, not decorative badges.</h2>
          </div>
          <div className="validation-layout">
            <div className="verification-panel">
              <span className="panel-tag">Verification Summary</span>
              <strong>18 passed</strong>
              <p>Backend, browser, onboarding, Docker-backed, and smoke-run coverage are all included in the current repository validation path.</p>
              <ul>
                {validationItems.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
            <div className="proof-matrix">
              <article>
                <span>UI</span>
                <strong>Chromium flow</strong>
                <p>Master and worker dashboards are exercised in a browser against live in-process services.</p>
              </article>
              <article>
                <span>Docker</span>
                <strong>Worker health passed</strong>
                <p>DFS-lite worker containers build and answer `/health` before the master starts.</p>
              </article>
              <article>
                <span>Locality</span>
                <strong>Persisted blocks confirmed</strong>
                <p>CSV blocks are written to disk and reused for worker-local rounds.</p>
              </article>
              <article>
                <span>Outcome</span>
                <strong>0.9737 accuracy</strong>
                <p>The default DFS-lite smoke run converges to the verified validation target.</p>
              </article>
            </div>
          </div>
        </section>

        <section className="section">
          <div className="section-heading">
            <p className="eyebrow">Technical Details</p>
            <h2>Production-minded implementation decisions that shape the runtime.</h2>
          </div>
          <div className="card-grid">
            {technicalCards.map((card) => (
              <article key={card.title} className="feature-card">
                <h3>{card.title}</h3>
                <p>{card.description}</p>
                <ul>
                  {card.bullets.map((bullet) => (
                    <li key={bullet}>{bullet}</li>
                  ))}
                </ul>
              </article>
            ))}
          </div>
        </section>

        <section className="section">
          <div className="section-heading">
            <p className="eyebrow">Repository Layout</p>
            <h2>The preserved baseline and the extension variants coexist cleanly in the same repo.</h2>
          </div>
          <div className="repo-tree">
            {repoTree.map((entry) => (
              <div key={entry} className="repo-entry">
                <span className="repo-bullet" />
                <code>{entry}</code>
              </div>
            ))}
          </div>
        </section>
      </main>

      <footer className="footer">
        <div>
          <p className="eyebrow">Hetero FedLearn REST</p>
          <h2>Federated learning over HTTP, with a preserved baseline and a disk-local extension.</h2>
        </div>
        <div className="hero-actions">
          <a className="button button-primary" href="#quick-start">Run Local Demo</a>
          <a className="button button-secondary" href="#validation">Review Validation</a>
        </div>
      </footer>
    </div>
  );
}

export default App;
