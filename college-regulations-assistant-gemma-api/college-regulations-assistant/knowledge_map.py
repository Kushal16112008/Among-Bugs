
import html
import streamlit as st
import streamlit.components.v1 as components


def render_knowledge_map(question, passages, max_nodes=5):
    """Draw a 2D animated map from a question to retrieved PDF passages."""

    passages = list(passages or [])[:max_nodes]

    if not passages:
        st.info("Upload a PDF and ask a question to generate the knowledge map.")
        return

    # Supports common RAG chunk field names.
    def fields(p):
        if isinstance(p, dict):
            text = p.get("text") or p.get("content") or p.get("page_content") or ""
            page = p.get("page") or p.get("page_number") or "?"
            filename = p.get("filename") or p.get("source") or "Regulations PDF"
        else:
            text, page, filename = str(p), "?", "Regulations PDF"
        return str(text), str(page), str(filename).replace("\\", "/").split("/")[-1]

    q = html.escape(str(question)[:70])
    positions = [(345, 65), (545, 125), (535, 260), (345, 310), (150, 220)]

    edges, nodes = [], []
    for i, p in enumerate(passages):
        text, page, filename = fields(p)
        x, y = positions[i]
        name = html.escape(filename[:22])
        page_label = html.escape(f"PAGE {page}")
        excerpt = html.escape(text[:180])

        dx, dy = x - 345, y - 185
        edges.append(f"""
        <path class="edge" d="M345 185 Q{345+dx*.5} {185+dy*.35} {x} {y}"/>
        <circle class="particle" r="4">
          <animateMotion dur="{3+i*.5}s" repeatCount="indefinite"
            path="M345 185 Q{345+dx*.5} {185+dy*.35} {x} {y}"/>
        </circle>
        """)

        nodes.append(f"""
        <g class="source-node" tabindex="0">
          <circle class="halo" cx="{x}" cy="{y}" r="29"/>
          <circle class="node" cx="{x}" cy="{y}" r="21"/>
          <path class="doc" d="M{x-7} {y-10} h10 l5 5 v16 h-15 z
             M{x+3} {y-10} v5 h5
             M{x-4} {y+1} h9 M{x-4} {y+5} h9"/>
          <text class="filename" x="{x}" y="{y+41}" text-anchor="middle">{name}</text>
          <text class="page" x="{x}" y="{y+54}" text-anchor="middle">{page_label}</text>
          <title>{excerpt}</title>
        </g>
        """)

    svg = f"""
    <svg viewBox="0 0 690 365" role="img"
         aria-label="Question linked to retrieved regulations evidence">
      <defs>
        <pattern id="grid" width="22" height="22" patternUnits="userSpaceOnUse">
          <circle cx="1" cy="1" r="1" fill="#a5b4fc" opacity=".4"/>
        </pattern>
        <radialGradient id="glow">
          <stop stop-color="#8b5cf6" stop-opacity=".25"/>
          <stop offset="1" stop-color="#8b5cf6" stop-opacity="0"/>
        </radialGradient>
      </defs>
      <rect width="690" height="365" fill="url(#grid)"/>
      <circle cx="345" cy="185" r="100" fill="url(#glow)"/>
      {''.join(edges)}
      <circle class="ring ring1" cx="345" cy="185" r="48"/>
      <circle class="ring ring2" cx="345" cy="185" r="39"/>
      <circle class="core" cx="345" cy="185" r="31"/>
      <text class="qicon" x="345" y="191" text-anchor="middle">?</text>
      <text class="qtitle" x="345" y="249" text-anchor="middle">YOUR QUESTION</text>
      <text class="qtext" x="345" y="266" text-anchor="middle">{q}</text>
      {''.join(nodes)}
    </svg>
    """

    css = """
    <style>
      body { margin:0; background:transparent; font-family:system-ui,sans-serif; }
      .panel { border:1px solid #ddd6fe; border-radius:18px; overflow:hidden;
        background:radial-gradient(ellipse at center,#f5f3ff,#fff 70%); }
      .heading { padding:16px 20px 0; display:flex; justify-content:space-between;
        color:#3730a3; font-size:14px; font-weight:650; }
      svg { display:block; width:100%; height:auto; }
      .edge { fill:none; stroke:#8b5cf6; stroke-width:1.6; stroke-dasharray:5 7;
        animation:dash 10s linear infinite; opacity:.7; }
      .particle { fill:#06b6d4; }
      .ring { fill:none; stroke:#8b5cf6; stroke-dasharray:4 6; transform-origin:345px 185px; }
      .ring1 { animation:spin 10s linear infinite; }
      .ring2 { animation:spin 14s linear infinite reverse; opacity:.5; }
      .core { fill:white; stroke:#8b5cf6; stroke-width:2; }
      .qicon { fill:#7c3aed; font-size:27px; font-weight:700; }
      .qtitle { fill:#7c3aed; font-size:9px; font-weight:800; letter-spacing:2px; }
      .qtext { fill:#312e81; font-size:10px; font-weight:600; }
      .halo { fill:#c4b5fd22; stroke:#a78bfa88; animation:pulse 3s ease-in-out infinite;
        transform-box:fill-box; transform-origin:center; }
      .node { fill:white; stroke:#c4b5fd; stroke-width:1.5; }
      .doc { fill:none; stroke:#6366f1; stroke-width:1.5; stroke-linejoin:round; }
      .filename { fill:#3730a3; font-size:10px; font-weight:650; }
      .page { fill:#64748b; font-size:9px; letter-spacing:1px; }
      .source-node { cursor:help; outline:none; }
      .source-node:hover .node,.source-node:focus .node { stroke:#06b6d4; stroke-width:3; }
      .source-node:hover .halo,.source-node:focus .halo { stroke:#06b6d4; }
      @keyframes dash { to { stroke-dashoffset:-150; } }
      @keyframes spin { to { transform:rotate(360deg); } }
      @keyframes pulse { 0%,100% { transform:scale(.92); opacity:.6; }
        50% { transform:scale(1.12); opacity:1; } }
      @media(prefers-reduced-motion:reduce) {
        * { animation:none !important; }
      }
    </style>
    """

    st.markdown(
        f'<div class="panel"><div class="heading">'
        f'<span>Knowledge map</span>'
        f'<span>{len(passages)} evidence connections</span></div></div>',
        unsafe_allow_html=True,
    )
    components.html(css + svg, height=385, scrolling=False)

    with st.expander("Inspect retrieved evidence"):
        for i, p in enumerate(passages, 1):
            text, page, filename = fields(p)
            st.markdown(f"**{i}. {filename} — page {page}**")
            st.caption(text[:500])
