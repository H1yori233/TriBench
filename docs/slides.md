---
hide:
  - navigation
  - toc
  - header
  - footer
---

<style>
  /* Override MkDocs Material layout to make the iframe completely fullscreen */
  .md-header, .md-footer, .md-sidebar, .md-tabs { display: none !important; }
  .md-main__inner { margin: 0 !important; max-width: 100% !important; padding: 0 !important; }
  .md-content { max-width: 100% !important; margin: 0 !important; padding: 0 !important; }
  .md-typeset h1, .md-typeset h2, .md-typeset h3 { display: none !important; }
  body, html { margin: 0; padding: 0; overflow: hidden; width: 100vw; height: 100vh; }
  
  .presentation-iframe {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    border: none;
    z-index: 999999;
    background: #0a0a0c;
  }
</style>

<iframe src="../slides.html" class="presentation-iframe" allowfullscreen></iframe>
