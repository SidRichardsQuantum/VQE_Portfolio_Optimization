# VQE Portfolio Optimization

```{raw} html
<div class="portfolio-page" id="top">
  <header class="site-header">
    <a class="brand" href="#top" aria-label="VQE Portfolio Optimization home">
      <span class="brand-mark">VQE</span>
      <span>VQE Portfolio Optimization</span>
    </a>
    <nav class="nav-links" aria-label="Primary navigation">
      <a href="#methods">Methods</a>
      <a href="#package">Package</a>
      <a href="#examples">Examples</a>
      <a href="#docs">Docs</a>
    </nav>
  </header>

  <main>
    <section class="hero section">
      <div class="hero-copy">
        <p class="eyebrow">PennyLane quantum optimization</p>
        <h1>Portfolio Optimization via VQE</h1>
        <p class="hero-text">
          A modular research toolkit for portfolio selection and allocation with
          binary VQE, QAOA, QUBO/Ising mappings, lambda sweeps, and fractional
          simplex-constrained VQE.
        </p>
        <div class="badges" aria-label="Project badges">
          <a href="https://pypi.org/project/vqe-portfolio/">
            <img src="https://img.shields.io/pypi/v/vqe-portfolio?style=flat-square" alt="PyPI version">
          </a>
          <a href="https://pypi.org/project/vqe-portfolio/">
            <img src="https://img.shields.io/pypi/pyversions/vqe-portfolio?style=flat-square" alt="Python versions">
          </a>
          <a href="https://github.com/SidRichardsQuantum/VQE_Portfolio_Optimization/actions/workflows/ci.yml">
            <img src="https://img.shields.io/github/actions/workflow/status/SidRichardsQuantum/VQE_Portfolio_Optimization/ci.yml?label=tests&style=flat-square" alt="Tests">
          </a>
          <a href="https://github.com/SidRichardsQuantum/VQE_Portfolio_Optimization/blob/main/LICENSE">
            <img src="https://img.shields.io/github/license/SidRichardsQuantum/VQE_Portfolio_Optimization?style=flat-square" alt="License">
          </a>
        </div>
        <div class="hero-actions" aria-label="Project links">
          <a class="button primary" href="https://github.com/SidRichardsQuantum/VQE_Portfolio_Optimization">GitHub</a>
          <a class="button" href="https://pypi.org/project/vqe-portfolio/">PyPI</a>
          <a class="button" href="#docs">Documentation</a>
        </div>
      </div>

      <aside class="focus-panel" aria-label="Project focus">
        <h2>Project Scope</h2>
        <ul>
          <li>Binary asset selection through QUBO, Ising Hamiltonians, and VQE</li>
          <li>QAOA workflows with standard X and constraint-aware XY mixers</li>
          <li>Fractional long-only allocations constrained to the simplex</li>
          <li>CLI/API usage, real-data utilities, lambda sweeps, and efficient frontiers</li>
        </ul>
      </aside>
    </section>

    <section id="methods" class="section">
      <div class="section-heading">
        <p class="eyebrow">Research workflows</p>
        <h2>Quantum-native formulations for portfolio optimization</h2>
        <p>
          The package keeps binary, QAOA, and fractional workflows on a common
          data model so experiments can compare selection probabilities,
          feasible candidates, allocations, costs, and efficient-frontier traces.
        </p>
      </div>

      <div class="project-grid">
        <article class="project-card">
          <div>
            <h3>Binary VQE</h3>
            <p>
              Select exactly K assets by mapping a cardinality-constrained
              mean-variance objective to a QUBO and Ising Hamiltonian.
            </p>
          </div>
          <div class="tags">
            <span>VQE</span>
            <span>QUBO</span>
            <span>Ising</span>
            <span>PennyLane</span>
          </div>
          <div class="card-links">
            <a href="user/readme.html#binary-vqe-asset-selection">Overview</a>
            <a href="user/usage.html#binary-vqe-asset-selection">Usage</a>
          </div>
        </article>

        <article class="project-card">
          <div>
            <h3>QAOA</h3>
            <p>
              Solve the same binary objective with alternating cost and mixer
              Hamiltonians, samples, marginal probabilities, and feasible picks.
            </p>
          </div>
          <div class="tags">
            <span>QAOA</span>
            <span>X mixer</span>
            <span>XY mixer</span>
            <span>Sampling</span>
          </div>
          <div class="card-links">
            <a href="user/readme.html#qaoa-binary-asset-selection">Overview</a>
            <a href="user/usage.html#qaoa-binary-asset-selection">Usage</a>
          </div>
        </article>

        <article class="project-card">
          <div>
            <h3>Fractional VQE</h3>
            <p>
              Optimize long-only continuous allocations with the simplex
              constraint enforced by construction rather than by penalties.
            </p>
          </div>
          <div class="tags">
            <span>Simplex</span>
            <span>Allocation</span>
            <span>Frontiers</span>
            <span>Warm starts</span>
          </div>
          <div class="card-links">
            <a href="user/readme.html#fractional-vqe-continuous-allocation">Overview</a>
            <a href="user/usage.html#fractional-vqe-continuous-allocation">Usage</a>
          </div>
        </article>
      </div>
    </section>

    <section id="package" class="section">
      <div class="section-heading">
        <p class="eyebrow">Published package</p>
        <h2>Installable Python tooling</h2>
        <p>
          The PyPI package exposes reusable APIs and a first-class command-line
          interface for running experiments without notebooks.
        </p>
      </div>

      <div class="package-list">
        <article class="package-row">
          <div>
            <h3>vqe-portfolio</h3>
            <p>Quantum portfolio optimization with PennyLane.</p>
          </div>
          <code>pip install vqe-portfolio</code>
          <a href="https://pypi.org/project/vqe-portfolio/">PyPI</a>
        </article>

        <article class="package-row">
          <div>
            <h3>CLI entrypoint</h3>
            <p>Run binary, QAOA, and fractional workflows from the terminal.</p>
          </div>
          <code>vqe-portfolio --help</code>
          <a href="user/usage.html#command-line-interface-cli">Usage</a>
        </article>

        <article class="package-row">
          <div>
            <h3>Python APIs</h3>
            <p>Import solver helpers from scripts, notebooks, and tests.</p>
          </div>
          <code>import vqe_portfolio</code>
          <a href="user/readme.html#usage">Overview</a>
        </article>
      </div>
    </section>

    <section id="examples" class="section split-section">
      <div class="section-heading">
        <p class="eyebrow">Notebook clients</p>
        <h2>Examples and real-data workflows</h2>
      </div>
      <div class="about-copy">
        <p>
          Notebooks in this repository are thin clients around the public API.
          Start with synthetic binary, QAOA, and fractional examples, then move
          into the real market-data examples and lambda-sweep workflows.
        </p>
        <div class="link-stack">
          <a href="user/usage.html">Usage guide</a>
          <a href="user/theory.html">Theory notes</a>
          <a href="https://github.com/SidRichardsQuantum/VQE_Portfolio_Optimization/tree/main/notebooks">Notebook source tree</a>
        </div>
      </div>
    </section>

    <section id="docs" class="section contact-section">
      <div>
        <p class="eyebrow">Documentation and source</p>
        <h2>Read the full project materials</h2>
        <p>
          This page is generated by the repository's Sphinx Pages workflow.
          The deeper project documentation remains available as generated HTML
          from the repository Markdown files.
        </p>
      </div>
      <div class="contact-actions">
        <a class="button primary" href="user/readme.html">Overview</a>
        <a class="button" href="user/usage.html">Usage</a>
        <a class="button" href="user/theory.html">Theory</a>
        <a class="button" href="https://sidrichardsquantum.github.io/">Main portfolio</a>
      </div>
    </section>
  </main>

  <footer class="site-footer">
    <span>&copy; 2026 Sid Richards</span>
    <a href="https://sidrichardsquantum.github.io/">Main portfolio</a>
    <a href="#top">Back to top</a>
  </footer>
</div>
```

```{toctree}
:hidden:
:maxdepth: 1
:titlesonly:
:caption: User Guide

Overview <user/readme>
Usage <user/usage>
Theory <user/theory>
```
