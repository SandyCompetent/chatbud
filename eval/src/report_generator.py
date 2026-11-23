"""Report generator for ChatBud Automated Evaluation Suite."""

import os
import json
from typing import Dict, Any


DOMAIN_TITLES = {
    "live_public_api": "Live Public APIs (Frankfurter, Open-Meteo, OpenAlex, OSV)",
    "exeter_academic": "Exeter Academic & Campus (Forum, Guild, Navigation)",
    "memory_vault": "Two-Layer Memory Vault (Episodic Facts & Preferences)",
    "persona_specialist": "Specialist Personas (Athena, Cipher, Harper, Leo)",
}

DATASET_CHECK = "dataset_check"

LIMITATIONS = (
    "Metrics are lexical heuristics (token overlap with a hand-written synonym map), "
    "not LLM-judged Ragas/DeepEval scores. Golden context is supplied to the model, so "
    "retrieval quality is not measured."
)


def _fmt(value, digits: int = 3) -> str:
    """Format a metric; unmeasured metrics are shown as n/a, never as a number."""
    return "n/a" if value is None else f"{value:.{digits}f}"


def _is_dataset_check(summary: Dict[str, Any]) -> bool:
    return summary.get("mode", DATASET_CHECK) == DATASET_CHECK


def _status_text(summary: Dict[str, Any]) -> str:
    status = summary.get("overall_status", "UNKNOWN")
    total = summary.get("total_queries", 0)
    passed = summary.get("passed_queries", 0)
    if _is_dataset_check(summary):
        return f"DATASET CHECK: {status.replace('DATASET_', '')} ({passed} / {total}). Not a model benchmark"
    return f"{status} ({passed} / {total} queries)"


def _banner_text(summary: Dict[str, Any]) -> str:
    if _is_dataset_check(summary):
        return (
            "Dataset check only. No model responses were scored, so this report says nothing about "
            "ChatBud's answer quality. Generate responses with eval/generate_responses.py and re-run "
            "with --responses to get real metrics."
        )
    meta = summary.get("responses_metadata") or {}
    model = meta.get("model", "unknown model")
    generated = meta.get("generated_at", "unknown date")
    return f"Scored real responses from {model} (generated {generated}). {LIMITATIONS}"


def generate_markdown_report(data: Dict[str, Any]) -> str:
    """Generate GitHub-compatible Markdown evaluation summary."""
    summary = data.get("summary", {})
    metrics = summary.get("metrics", {})
    thresholds = summary.get("thresholds", {})
    domains = data.get("domains", {})
    dataset_check = _is_dataset_check(summary)

    lines = [
        "# ChatBud Evaluation Report",
        "",
        f"**Mode**: `{summary.get('mode', DATASET_CHECK)}` | **Status**: {_status_text(summary)} | "
        f"**Queries**: {summary.get('total_queries', 0)} | **Execution Time**: {summary.get('elapsed_seconds', 0)}s",
        "",
        f"> {_banner_text(summary)}",
        "",
    ]

    if dataset_check:
        lines += [
            "## Dataset sanity check",
            "",
            "| Check | Result | Requirement |",
            "| :--- | :---: | :---: |",
            f"| Reference answers supported by their own context | **{_fmt(metrics.get('reference_support'))}** | >= {thresholds.get('reference_support', 0):.2f} |",
            f"| Dataset problems found | **{len(summary.get('dataset_problems', []))}** | 0 |",
            "",
            "## Domain breakdown",
            "",
            "| Domain / Category | Queries | Reference support |",
            "| :--- | :---: | :---: |",
        ]
        for cat, dstats in domains.items():
            title = DOMAIN_TITLES.get(cat, cat.replace("_", " ").title())
            lines.append(f"| {title} | {dstats['total']} | {_fmt(dstats.get('reference_support'))} |")
    else:
        def verdict(value, limit, upper=False):
            if value is None:
                return "not measured"
            ok = value <= limit if upper else value >= limit
            return "PASS" if ok else "FAIL"

        lines += [
            "## Model response metrics",
            "",
            "| Metric | Score | Gate | Result |",
            "| :--- | :---: | :---: | :---: |",
            f"| Faithfulness | **{_fmt(metrics.get('faithfulness'))}** | >= {thresholds.get('faithfulness', 0):.2f} | {verdict(metrics.get('faithfulness'), thresholds.get('faithfulness', 0))} |",
            f"| Answer Relevance | **{_fmt(metrics.get('answer_relevance'))}** | >= {thresholds.get('answer_relevance', 0):.2f} | {verdict(metrics.get('answer_relevance'), thresholds.get('answer_relevance', 0))} |",
            f"| Context Precision | **{_fmt(metrics.get('context_precision'))}** | >= {thresholds.get('context_precision', 0):.2f} | {verdict(metrics.get('context_precision'), thresholds.get('context_precision', 0))} |",
            f"| Hallucination Rate | **{_fmt(metrics.get('hallucination_rate'))}** | <= {thresholds.get('hallucination_rate', 0):.2f} | {verdict(metrics.get('hallucination_rate'), thresholds.get('hallucination_rate', 0), upper=True)} |",
            "",
            "## Domain breakdown",
            "",
            "| Domain / Category | Queries | Items passing | Faithfulness | Relevance | Context Precision | Hallucination Rate |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for cat, dstats in domains.items():
            title = DOMAIN_TITLES.get(cat, cat.replace("_", " ").title())
            lines.append(
                f"| {title} | {dstats['total']} | {dstats['pass_rate']}% | "
                f"{_fmt(dstats.get('faithfulness'))} | {_fmt(dstats.get('answer_relevance'))} | "
                f"{_fmt(dstats.get('context_precision'))} | {_fmt(dstats.get('hallucination_rate'))} |"
            )

    problems = summary.get("dataset_problems", [])
    if problems:
        lines += ["", "## Dataset problems", ""] + [f"- {p}" for p in problems]

    lines += ["", f"> **Limitations**: {LIMITATIONS}"]
    return "\n".join(lines)


def _kpi_html(summary: Dict[str, Any]) -> str:
    metrics = summary.get("metrics", {})
    thresholds = summary.get("thresholds", {})

    def card(title, value, badge, caption):
        return (
            '      <div class="kpi-card">\n'
            f'        <div class="kpi-title">{title}</div>\n'
            f'        <div class="kpi-val">{value}</div>\n'
            '        <div class="kpi-target">\n'
            f'          <span class="kpi-badge">{badge}</span>\n'
            f'          <span>{caption}</span>\n'
            '        </div>\n'
            '      </div>'
        )

    if _is_dataset_check(summary):
        return "\n".join([
            card("Reference support", _fmt(metrics.get("reference_support")),
                 f"Needs &ge; {thresholds.get('reference_support', 0):.2f}", "Reference answer vs its own context"),
            card("Dataset problems", str(len(summary.get("dataset_problems", []))), "Needs 0", "Schema and balance"),
            card("Model metrics", "n/a", "Not measured", "No responses were scored"),
        ])

    return "\n".join([
        card("Faithfulness", _fmt(metrics.get("faithfulness")),
             f"Target &ge; {thresholds.get('faithfulness', 0):.2f}", "Claims supported by context"),
        card("Answer Relevance", _fmt(metrics.get("answer_relevance")),
             f"Target &ge; {thresholds.get('answer_relevance', 0):.2f}", "Query intent match"),
        card("Context Precision", _fmt(metrics.get("context_precision")),
             f"Target &ge; {thresholds.get('context_precision', 0):.2f}", "Needs retrieved chunks"),
        card("Hallucination Rate", _fmt(metrics.get("hallucination_rate")),
             f"Target &le; {thresholds.get('hallucination_rate', 0):.2f}", "1 - faithfulness"),
    ])


def _radar_html(summary: Dict[str, Any]) -> str:
    """Radar built from the real scores. Unmeasured axes sit at the centre and say n/a."""
    if _is_dataset_check(summary):
        return (
            '<div style="display:flex;flex-direction:column;align-items:center;max-width:280px;'
            'color:var(--text-muted);font-size:13px;">No radar: no model responses were scored.</div>'
        )

    metrics = summary.get("metrics", {})
    thresholds = summary.get("thresholds", {})
    pass_fraction = summary.get("pass_rate_percentage", 0) / 100.0

    def axis(value):
        return 0.0 if value is None else max(0.0, min(1.0, value))

    f = axis(metrics.get("faithfulness"))
    p = axis(metrics.get("context_precision"))
    r = axis(metrics.get("answer_relevance"))
    q = axis(pass_fraction)
    radius = 120

    def points(top, right, bottom, left):
        return (
            f"150,{150 - radius * top:.0f} {150 + radius * right:.0f},150 "
            f"150,{150 + radius * bottom:.0f} {150 - radius * left:.0f},150"
        )

    actual = points(f, p, r, q)
    target = points(
        thresholds.get("faithfulness", 0.85),
        thresholds.get("context_precision", 0.80),
        thresholds.get("answer_relevance", 0.85),
        0.80,
    )
    return f"""<div style="display: flex; flex-direction: column; align-items: center;">
          <svg width="280" height="280" viewBox="0 0 300 300">
            <polygon points="150,30 270,150 150,270 30,150" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="1"/>
            <polygon points="150,90 210,150 150,210 90,150" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="1"/>
            <polygon points="{target}" fill="none" stroke="#F59E0B" stroke-dasharray="4,4" stroke-width="1.5"/>
            <polygon points="{actual}" fill="rgba(99, 102, 241, 0.35)" stroke="#6366F1" stroke-width="2.5"/>
            <line x1="150" y1="30" x2="150" y2="270" stroke="rgba(255,255,255,0.1)"/>
            <line x1="30" y1="150" x2="270" y2="150" stroke="rgba(255,255,255,0.1)"/>
            <text x="150" y="22" fill="#A5B4FC" font-size="11" font-weight="600" text-anchor="middle">Faithfulness ({_fmt(metrics.get('faithfulness'))})</text>
            <text x="278" y="154" fill="#A5B4FC" font-size="11" font-weight="600" text-anchor="start">Precision ({_fmt(metrics.get('context_precision'))})</text>
            <text x="150" y="290" fill="#A5B4FC" font-size="11" font-weight="600" text-anchor="middle">Relevance ({_fmt(metrics.get('answer_relevance'))})</text>
            <text x="22" y="154" fill="#34D399" font-size="11" font-weight="600" text-anchor="end">Items passing ({summary.get('pass_rate_percentage', 0)}%)</text>
          </svg>
          <div style="font-size: 12px; color: var(--text-muted); margin-top: 8px;">
            <span style="color: #6366F1;">&#9632;</span> Measured &nbsp;&nbsp;
            <span style="color: #F59E0B;">--</span> Gate (unmeasured axes sit at the centre)
          </div>
        </div>"""


def _domain_rows_html(data: Dict[str, Any]) -> str:
    rows = []
    for cat, d in data.get("domains", {}).items():
        title = DOMAIN_TITLES.get(cat, cat.replace("_", " ").title())
        rows.append(
            "              <tr>\n"
            f"                <td><strong>{title}</strong></td>\n"
            f"                <td>{d['total']}</td>\n"
            f'                <td><span class="kpi-badge">{d["pass_rate"]}%</span></td>\n'
            f'                <td class="score-chip">{_fmt(d.get("faithfulness"))}</td>\n'
            f'                <td class="score-chip">{_fmt(d.get("answer_relevance"))}</td>\n'
            f'                <td class="score-chip">{_fmt(d.get("context_precision"))}</td>\n'
            f'                <td class="mono">{_fmt(d.get("hallucination_rate"))}</td>\n'
            f'                <td class="mono">{_fmt(d.get("reference_support"))}</td>\n'
            "              </tr>"
        )
    return "\n".join(rows)


def generate_html_report(data: Dict[str, Any]) -> str:
    """Generate standalone interactive HTML dashboard."""
    summary = data.get("summary", {})
    metrics = summary.get("metrics", {})
    thresholds = summary.get("thresholds", {})
    domains = data.get("domains", {})
    results = data.get("results", [])

    data_json_str = json.dumps(data)
    status_text = _status_text(summary)
    banner_html = (
        '<div style="margin:0 0 20px;padding:12px 16px;border-radius:10px;'
        'background:rgba(245,158,11,0.12);border:1px solid rgba(245,158,11,0.35);'
        'color:#FCD34D;font-size:13px;line-height:1.5;">' + _banner_text(summary) + '</div>'
    )
    kpi_html = _kpi_html(summary)
    radar_html = _radar_html(summary)
    domain_rows_html = _domain_rows_html(data)
    total_queries = summary.get('total_queries', 0)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ChatBud — Automated Evaluation Suite & RAG Benchmarks</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #0B101A;
      --card-bg: rgba(18, 25, 38, 0.7);
      --card-border: rgba(255, 255, 255, 0.08);
      --text-main: #F3F4F6;
      --text-muted: #94A3B8;
      --accent: #6366F1;
      --accent-glow: rgba(99, 102, 241, 0.25);
      --success: #10B981;
      --success-glow: rgba(16, 185, 129, 0.2);
      --danger: #EF4444;
      --warning: #F59E0B;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      background: radial-gradient(circle at 50% 0%, #151d2f 0%, #0B101A 70%);
      color: var(--text-main);
      min-height: 100vh;
      padding: 32px 20px;
    }}
    .container {{
      max-width: 1200px;
      margin: 0 auto;
    }}
    header {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 32px;
      padding-bottom: 24px;
      border-bottom: 1px solid var(--card-border);
      gap: 16px;
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 14px;
    }}
    .logo-badge {{
      width: 44px;
      height: 44px;
      border-radius: 12px;
      background: linear-gradient(135deg, #6366F1, #8B5CF6);
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 20px;
      color: #fff;
      box-shadow: 0 4px 14px var(--accent-glow);
    }}
    h1 {{
      font-size: 24px;
      font-weight: 700;
      letter-spacing: -0.02em;
    }}
    .subtitle {{
      color: var(--text-muted);
      font-size: 14px;
      margin-top: 2px;
    }}
    .status-pill {{
      padding: 8px 16px;
      border-radius: 9999px;
      font-weight: 600;
      font-size: 14px;
      display: flex;
      align-items: center;
      gap: 8px;
      background: rgba(16, 185, 129, 0.15);
      color: #34D399;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }}
    .status-pill::before {{
      content: '';
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #10B981;
      box-shadow: 0 0 10px #10B981;
    }}
    .grid-kpi {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 20px;
      margin-bottom: 32px;
    }}
    .kpi-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 24px;
      position: relative;
      overflow: hidden;
      backdrop-filter: blur(12px);
    }}
    .kpi-title {{
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      font-weight: 600;
      margin-bottom: 8px;
    }}
    .kpi-val {{
      font-size: 36px;
      font-weight: 800;
      letter-spacing: -0.02em;
      color: #fff;
      margin-bottom: 6px;
    }}
    .kpi-target {{
      font-size: 13px;
      color: #94A3B8;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .kpi-badge {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 600;
      background: rgba(16, 185, 129, 0.15);
      color: #34D399;
    }}
    .section-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 24px;
      margin-bottom: 32px;
      backdrop-filter: blur(12px);
    }}
    .section-title {{
      font-size: 18px;
      font-weight: 700;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .viz-container {{
      display: grid;
      grid-template-columns: 340px 1fr;
      gap: 24px;
      align-items: center;
    }}
    @media (max-width: 860px) {{
      .viz-container {{ grid-template-columns: 1fr; }}
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th {{
      text-align: left;
      padding: 12px 14px;
      color: var(--text-muted);
      font-weight: 600;
      border-bottom: 1px solid var(--card-border);
      background: rgba(255, 255, 255, 0.02);
    }}
    td {{
      padding: 14px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      color: var(--text-main);
    }}
    tr:hover td {{
      background: rgba(255, 255, 255, 0.02);
    }}
    .search-bar {{
      width: 100%;
      padding: 12px 16px;
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      color: #fff;
      font-size: 14px;
      margin-bottom: 16px;
      outline: none;
      font-family: inherit;
    }}
    .search-bar:focus {{
      border-color: var(--accent);
      box-shadow: 0 0 10px var(--accent-glow);
    }}
    .tag {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 600;
      background: rgba(99, 102, 241, 0.15);
      color: #A5B4FC;
    }}
    .mono {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 13px;
    }}
    .score-chip {{
      font-family: 'JetBrains Mono', monospace;
      font-weight: 600;
      color: #34D399;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="brand">
        <div class="logo-badge">CB</div>
        <div>
          <h1>ChatBud Automated Evaluation Suite</h1>
          <div class="subtitle">Golden-dataset evaluation of Grounded RAG, Memory Vault &amp; Persona prompts</div>
        </div>
      </div>
      <div class="status-pill">{status_text}</div>
    </header>

    {banner_html}

    <!-- Top KPIs -->
    <div class="grid-kpi">
{kpi_html}
    </div>

    <!-- Domain Breakdown & Visual Radar -->
    <div class="section-card">
      <div class="section-title">Domain Performance Breakdown & Radar Analysis</div>
      <div class="viz-container">
        {radar_html}

        <!-- Breakdown Table -->
        <div style="overflow-x: auto;">
          <table>
            <thead>
              <tr>
                <th>Domain Category</th>
                <th>Queries</th>
                <th>Pass Rate</th>
                <th>Faithfulness</th>
                <th>Relevance</th>
                <th>Context Precision</th>
                <th>Hallucination</th>
                <th>Reference support</th>
              </tr>
            </thead>
            <tbody>
{domain_rows_html}
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- Query Drill-Down -->
    <div class="section-card">
      <div class="section-title">
        <span>Detailed Query Audit Log ({total_queries} golden test cases)</span>
      </div>
      <input type="text" id="searchInput" class="search-bar" placeholder="Filter queries by keywords (e.g. 'ECB', 'Forum', 'memory', 'Flutter', 'CVE')...">
      <div style="overflow-x: auto; max-height: 520px; overflow-y: auto;">
        <table id="queriesTable">
          <thead>
            <tr>
              <th>ID</th>
              <th>Category</th>
              <th>Query</th>
              <th>Faithfulness</th>
              <th>Relevance</th>
              <th>Precision</th>
              <th>Ref. support</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody id="tableBody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    const data = {data_json_str};
    const tableBody = document.getElementById('tableBody');
    const searchInput = document.getElementById('searchInput');

    const fmt = v => (v === null || v === undefined) ? 'n/a' : v.toFixed(3);

    function renderTable(filter = '') {{
      tableBody.innerHTML = '';
      const filterLower = filter.toLowerCase();
      const filtered = data.results.filter(r => 
        r.query.toLowerCase().includes(filterLower) ||
        r.category.toLowerCase().includes(filterLower) ||
        r.id.toLowerCase().includes(filterLower)
      );

      filtered.forEach(r => {{
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td class="mono" style="color: var(--text-muted);">${{r.id}}</td>
          <td><span class="tag">${{r.category}}</span></td>
          <td>${{r.query}}</td>
          <td class="mono">${{fmt(r.metrics.faithfulness)}}</td>
          <td class="mono">${{fmt(r.metrics.answer_relevance)}}</td>
          <td class="mono">${{fmt(r.metrics.context_precision)}}</td>
          <td class="mono">${{fmt(r.metrics.reference_support)}}</td>
          <td><span class="kpi-badge" style="${{r.passed ? '' : 'background:rgba(239,68,68,0.15);color:#F87171;'}}">${{r.passed ? 'PASS' : 'FAIL'}}</span></td>
        `;
        tableBody.appendChild(tr);
      }});
    }}

    searchInput.addEventListener('input', (e) => renderTable(e.target.value));
    renderTable();
  </script>
</body>
</html>
"""


def save_reports(data: Dict[str, Any], output_dir: str):
    """Write all report files to the target directory."""
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "eval_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    md_path = os.path.join(output_dir, "eval_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(generate_markdown_report(data))

    html_path = os.path.join(output_dir, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(generate_html_report(data))

    print(f"Reports successfully generated in '{output_dir}':")
    print(f"  - JSON: {json_path}")
    print(f"  - Markdown: {md_path}")
    print(f"  - Interactive Dashboard: {html_path}")
