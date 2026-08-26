/* ==========================================================================
   Rikken AI Research Platform — Interactive Documentation App
   ========================================================================== */

(function () {
  'use strict';

  const docs = window.DOCS_DATA || [];
  let currentDocId = null;
  let searchQuery = '';

  // DOM Elements
  const appContainer = document.getElementById('app');
  const brandBtn = document.getElementById('brandBtn');

  // Initialize marked parser options
  if (window.marked) {
    marked.setOptions({
      gfm: true,
      breaks: true,
      headerIds: true,
      highlight: function (code, lang) {
        if (window.hljs && hljs.getLanguage(lang)) {
          return hljs.highlight(code, { language: lang }).value;
        }
        return code;
      }
    });
  }

  // -------------------------------------------------------------------------
  // Routing & View Management
  // -------------------------------------------------------------------------
  function initRouter() {
    window.addEventListener('hashchange', handleRoute);
    if (brandBtn) {
      brandBtn.addEventListener('click', () => {
        window.location.hash = '';
      });
    }
    handleRoute();
  }

  function handleRoute() {
    const hash = window.location.hash.replace(/^#\/?/, '');
    if (!hash || hash === 'home' || hash === 'landing') {
      currentDocId = null;
      renderLandingPage();
      window.scrollTo(0, 0);
    } else {
      const matched = docs.find(d => d.id === hash || d.filename === hash || d.filename.replace('.md', '') === hash);
      if (matched) {
        currentDocId = matched.id;
        renderReaderPage(matched);
      } else {
        // Fallback to first doc
        currentDocId = docs[0] ? docs[0].id : null;
        if (currentDocId) {
          renderReaderPage(docs[0]);
        } else {
          renderLandingPage();
        }
      }
      window.scrollTo(0, 0);
    }
  }

  function navigateTo(docId) {
    window.location.hash = '#' + docId;
  }

  // -------------------------------------------------------------------------
  // Landing Page View
  // -------------------------------------------------------------------------
  function renderLandingPage() {
    const cardsHtml = docs.map((doc, idx) => `
      <div class="doc-card" onclick="window.location.hash='#${doc.id}'">
        <div>
          <span class="doc-card-num">${String(idx + 1).padStart(2, '0')} // MODULE</span>
          <h3 class="doc-card-title">${escapeHtml(doc.title)}</h3>
          <p class="doc-card-desc">${escapeHtml(doc.description)}</p>
        </div>
        <div class="doc-card-footer">
          <span>Read Specification</span>
          <span>→</span>
        </div>
      </div>
    `).join('');

    appContainer.innerHTML = `
      <div class="landing-view">
        <!-- Hero Section -->
        <section class="hero">
          <div class="hero-pill">
            <span class="dot"></span>
            <span>Phase 1 Dataset Verified & Ready</span>
          </div>
          <h1 class="hero-title">
            The <span>Rikken AI</span> Research Platform
          </h1>
          <p class="hero-subtitle">
            A state-of-the-art hybrid AI architecture combining Transformer Bidding Value Networks, ResNet Belief Networks, and Information Set Monte Carlo Tree Search (ISMCTS) for imperfect information trick-taking games.
          </p>
          <div class="hero-cta-group">
            <button class="btn btn-primary" onclick="window.location.hash='#rules'">
              Explore Game Rules & Engine
            </button>
            <button class="btn btn-secondary" onclick="window.location.hash='#dataset'">
              View 1M Dataset Analysis
            </button>
          </div>
        </section>

        <!-- Live Empirical Metrics -->
        <section class="metrics-grid">
          <div class="metric-card">
            <div class="metric-val cyan">1,000,000</div>
            <div class="metric-label">Bootstrapped Games</div>
            <div class="metric-sub">100 compressed .npz shards</div>
          </div>
          <div class="metric-card">
            <div class="metric-val emerald">4.63M</div>
            <div class="metric-label">Bidding Decisions</div>
            <div class="metric-sub">Transformer BVN training samples</div>
          </div>
          <div class="metric-card">
            <div class="metric-val amber">28.7M</div>
            <div class="metric-label">Trick Transitions</div>
            <div class="metric-sub">Belief Network state-action pairs</div>
          </div>
          <div class="metric-card">
            <div class="metric-val violet">96.5%</div>
            <div class="metric-label">Early Stopping Rate</div>
            <div class="metric-sub">Sound lead-dependent pruning</div>
          </div>
        </section>

        <!-- Documentation Index Grid -->
        <div class="section-head">
          <h2 class="section-title">Documentation & System Architecture</h2>
          <p class="section-desc">Comprehensive technical blueprints, empirical benchmarks, and game theory specifications.</p>
        </div>

        <div class="docs-grid">
          ${cardsHtml}
        </div>
      </div>
    `;
  }

  // -------------------------------------------------------------------------
  // Reader View (Sidebar + Content)
  // -------------------------------------------------------------------------
  function renderReaderPage(doc) {
    const currentIndex = docs.findIndex(d => d.id === doc.id);
    const prevDoc = currentIndex > 0 ? docs[currentIndex - 1] : null;
    const nextDoc = currentIndex < docs.length - 1 ? docs[currentIndex + 1] : null;

    // Filter sidebar list by search
    const filteredDocs = docs.filter(d => 
      !searchQuery || 
      d.title.toLowerCase().includes(searchQuery.toLowerCase()) || 
      d.description.toLowerCase().includes(searchQuery.toLowerCase())
    );

    const sidebarItemsHtml = filteredDocs.map((d, i) => `
      <div class="nav-item ${d.id === doc.id ? 'active' : ''}" onclick="window.location.hash='#${d.id}'">
        <span class="nav-item-badge">${String(i + 1).padStart(2, '0')}</span>
        <span>${escapeHtml(d.title)}</span>
      </div>
    `).join('');

    // Render markdown content
    let parsedContent = '';
    if (window.marked) {
      parsedContent = marked.parse(doc.content);
    } else {
      parsedContent = `<pre>${escapeHtml(doc.content)}</pre>`;
    }

    appContainer.innerHTML = `
      <div class="reader-view">
        <!-- Sidebar Navigation -->
        <aside class="sidebar">
          <div class="sidebar-search">
            <span class="search-icon">🔍</span>
            <input type="text" class="search-input" id="sidebarSearch" placeholder="Search documentation..." value="${escapeHtml(searchQuery)}">
          </div>
          <div class="sidebar-nav" id="sidebarNav">
            ${sidebarItemsHtml}
          </div>
        </aside>

        <!-- Main Document Area -->
        <main class="reader-content">
          <article class="doc-article">
            <div class="markdown-body" id="markdownContent">
              ${parsedContent}
            </div>

            <!-- Pagination Controls -->
            <nav class="doc-pagination">
              ${prevDoc ? `
                <div class="page-btn" onclick="window.location.hash='#${prevDoc.id}'">
                  <span class="page-btn-dir">← Previous Chapter</span>
                  <span class="page-btn-title">${escapeHtml(prevDoc.title)}</span>
                </div>
              ` : '<div></div>'}

              ${nextDoc ? `
                <div class="page-btn" style="text-align: right;" onclick="window.location.hash='#${nextDoc.id}'">
                  <span class="page-btn-dir">Next Chapter →</span>
                  <span class="page-btn-title">${escapeHtml(nextDoc.title)}</span>
                </div>
              ` : '<div></div>'}
            </nav>
          </article>
        </main>
      </div>
    `;

    // Bind search input
    const searchEl = document.getElementById('sidebarSearch');
    if (searchEl) {
      searchEl.addEventListener('input', (e) => {
        searchQuery = e.target.value;
        const navEl = document.getElementById('sidebarNav');
        const reFiltered = docs.filter(d => 
          !searchQuery || 
          d.title.toLowerCase().includes(searchQuery.toLowerCase()) || 
          d.description.toLowerCase().includes(searchQuery.toLowerCase())
        );
        navEl.innerHTML = reFiltered.map((d, i) => `
          <div class="nav-item ${d.id === doc.id ? 'active' : ''}" onclick="window.location.hash='#${d.id}'">
            <span class="nav-item-badge">${String(i + 1).padStart(2, '0')}</span>
            <span>${escapeHtml(d.title)}</span>
          </div>
        `).join('');
      });
    }

    // Highlight code blocks
    if (window.hljs) {
      document.querySelectorAll('#markdownContent pre code').forEach((block) => {
        hljs.highlightElement(block);
      });
    }
  }

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------
  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // Start the application
  window.addEventListener('DOMContentLoaded', initRouter);
})();
