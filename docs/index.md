---
hide:
  - navigation
  - toc
---

<style>
  .md-content { max-width: 100% !important; padding: 0 !important; }
  .md-main__inner { margin: 0 !important; max-width: 100% !important; }
  .tx-hero {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 80vh;
    padding: 2rem;
    background: transparent;
  }
  .tx-hero-header {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: center;
    gap: 3rem;
    margin-bottom: 3rem;
    flex-wrap: wrap;
  }
  .tx-hero-logo {
    width: 250px;
    height: 250px;
    border-radius: 20%;
  }
  .tx-hero-text {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    text-align: left;
    max-width: 600px;
  }
  .tx-hero h1 {
    font-size: 6rem;
    font-weight: 700;
    letter-spacing: -0.05em;
    margin-bottom: 0.5rem;
    background: linear-gradient(135deg, #1d1d1f 0%, #434353 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
  }
  [data-md-color-scheme="slate"] .tx-hero h1 {
    background: linear-gradient(135deg, #ffffff 0%, #a1a1a6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .tx-hero p {
    font-size: 1.75rem;
    color: var(--md-default-fg-color--light);
    margin: 0;
    line-height: 1.4;
    font-weight: 400;
  }
  .tx-buttons {
    display: flex;
    gap: 1.5rem;
    justify-content: center;
  }
  .tx-btn {
    display: inline-flex;
    align-items: center;
    padding: 0.8rem 2rem;
    font-size: 1.25rem;
    font-weight: 500;
    border-radius: 999px;
    text-decoration: none !important;
    transition: all 0.2s ease;
  }
  .tx-btn-primary {
    background-color: var(--md-default-fg-color);
    color: var(--md-default-bg-color) !important;
  }
  .tx-btn-primary:hover {
    transform: scale(1.02);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  }
  .tx-btn-secondary {
    background-color: transparent;
    border: 1px solid var(--md-default-fg-color);
    color: var(--md-default-fg-color) !important;
  }
  .tx-btn-secondary:hover {
    background-color: var(--md-default-fg-color);
    color: var(--md-default-bg-color) !important;
  }
  
  .tx-features {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 3rem;
    max-width: 1200px;
    margin: 4rem auto;
    padding: 0 2rem;
  }
  .tx-feature {
    text-align: center;
    padding: 3rem;
    background: rgba(0,0,0,0.02);
    border-radius: 24px;
    border: 1px solid rgba(0,0,0,0.06);
    transition: transform 0.3s ease, background 0.3s ease;
  }
  .tx-feature:hover {
    transform: translateY(-5px);
    background: rgba(0,0,0,0.04);
  }
  [data-md-color-scheme="slate"] .tx-feature {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.04);
  }
  [data-md-color-scheme="slate"] .tx-feature:hover {
    background: rgba(255,255,255,0.04);
  }
  .tx-feature h3 {
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 1rem;
  }
  .tx-feature p {
    color: var(--md-default-fg-color--light);
    line-height: 1.6;
  }
  .tx-feature-icon {
    font-size: 3rem;
    margin-bottom: 1.5rem;
  }
</style>

<div class="tx-hero">
  <div class="tx-hero-header">
    <img src="assets/logo.png" alt="TriBench Logo" class="tx-hero-logo">
    <div class="tx-hero-text">
      <h1>TriBench.</h1>
      <p>The standard for Triton kernel benchmarking. Fast, reproducible, and beautifully extensible.</p>
    </div>
  </div>
  <div class="tx-buttons">
    <a href="getting-started/" class="tx-btn tx-btn-primary">Get Started</a>
    <a href="commands/" class="tx-btn tx-btn-secondary">Commands</a>
    <a href="https://github.com/H1yori233/TriBench" class="tx-btn tx-btn-secondary">View on GitHub</a>
  </div>
</div>

<div class="tx-features">
  <div class="tx-feature">
    <div class="tx-feature-icon">🚀</div>
    <h3>Zero Overhead</h3>
    <p>Compile and warmup phases are strictly isolated from the measurement phase to eliminate JIT overhead.</p>
  </div>
  <div class="tx-feature">
    <div class="tx-feature-icon">🔬</div>
    <h3>Reproducible</h3>
    <p>Environment capture and strict random seed control ensure stable benchmark numbers across different runs.</p>
  </div>
  <div class="tx-feature">
    <div class="tx-feature-icon">🧩</div>
    <h3>Composable</h3>
    <p>Easily benchmark and compare multiple kernel variants alongside your main implementation.</p>
  </div>
</div>