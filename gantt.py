import plotly.express as px
import pandas as pd

data = [
    {"Phase": "Phase 1: Foundation & Defense", "Start": "2025-08-01", "Finish": "2025-09-30", "Category": "Planning"},
    {"Phase": "Phase 2: Data & CSER Proposal", "Start": "2025-10-01", "Finish": "2025-11-30", "Category": "Data"},
    {"Phase": "Phase 3: Architecture Pivot", "Start": "2025-12-01", "Finish": "2026-01-31", "Category": "Architecture"},
    {"Phase": "Phase 4: RAG Expansion", "Start": "2026-02-01", "Finish": "2026-03-31", "Category": "Backend"},
    {"Phase": "Phase 5: Multi-Agent Migration", "Start": "2026-04-01", "Finish": "2026-05-31", "Category": "Agents"},
    {"Phase": "Phase 6: Tool Calling & DURS", "Start": "2026-06-01", "Finish": "2026-06-30", "Category": "Integration"},
    {"Phase": "Phase 7: Voice Integration", "Start": "2026-06-15", "Finish": "2026-06-30", "Category": "Voice"},
    {"Phase": "Phase 8: Final Delivery", "Start": "2026-07-01", "Finish": "2026-07-31", "Category": "Delivery"}
]

df = pd.DataFrame(data)

fig = px.timeline(
    df, 
    x_start="Start", 
    x_end="Finish", 
    y="Phase", 
    color="Category",
    title="FYP Timeline (Aug 2025 - Jul 2026)"
)

fig.update_yaxes(autorange="reversed") 
fig.update_layout(
    template="plotly_white",
    xaxis_title="Timeline",
    yaxis_title="",
    showlegend=False,
    height=500
)

# Adds rounded edges to the bars to match your image
fig.update_traces(marker_line_width=0, opacity=0.9, width=0.6)

fig.show()