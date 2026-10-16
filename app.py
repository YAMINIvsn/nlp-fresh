import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent

COMPARISON_FILE = ROOT / "eval" / "results" / "dart_selector_comparison_v2.json"
PIPELINE_FILE = ROOT / "data" / "dart" / "end_to_end_simulation.json"

st.set_page_config(
    page_title="DART | Regulatory Intelligence",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("""
<style>
.stApp {
    background: #0b1120;
}
[data-testid="stMetric"] {
    background: #151f32;
    border: 1px solid #2b3b55;
    padding: 18px;
    border-radius: 12px;
}
h1, h2, h3 {
    color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)


def load_json(path):
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Could not read {path.name}: {exc}")
        return None


def find_values(obj, names):
    """Find values by key anywhere inside a nested JSON object."""
    found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.lower() in names and isinstance(value, (int, float)):
                found.append(value)
            found.extend(find_values(value, names))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(find_values(item, names))
    return found


comparison = load_json(COMPARISON_FILE)
pipeline = load_json(PIPELINE_FILE)

with st.sidebar:
    st.title("🛡️ DART")
    st.caption("Dependency-Aware Regulatory Transition")
    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Selector Comparison",
            "Pipeline Results",
            "Raw JSON Data",
        ],
    )
    st.divider()
    st.caption("Research prototype • Human review required")

st.title("DART Regulatory Intelligence Dashboard")
st.caption(
    "Analyze regulatory changes, trace affected controls, "
    "and prioritize compliance re-audits."
)

if page == "Overview":
    st.subheader("Project Overview")

    a, b, c = st.columns(3)

    selected = find_values(
        comparison or {},
        {"selected_cases", "selected_count"},
    )
    reductions = find_values(
        comparison or {},
        {"review_reduction", "reaudit_reduction"},
    )
    affected = find_values(
        pipeline or {},
        {"potentially_affected_cases"},
    )

    with a:
        st.metric(
            "Test Suite",
            "21 / 21 passed",
            help="Latest test run reported by the developer.",
        )
    with b:
        st.metric(
            "Affected Cases",
            str(max(affected)) if affected else "See pipeline results",
        )
    with c:
        st.metric(
            "Selector Results",
            f"{len(selected)} values" if selected else (
                "Available" if comparison else "Not loaded"
            ),
        )

    st.info(
        "This dashboard visualizes saved project outputs. "
        "The displayed case data is simulated and is not evidence "
        "of performance on real regulatory cases."
    )

    st.subheader("Workflow")

    stages = [
        ("1. Regulation Input", "Load old and amended provisions"),
        ("2. Change Detection", "Extract and compare obligations"),
        ("3. Dependency Analysis", "Trace connected controls and cases"),
        ("4. Risk-Based Selection", "Prioritize cases for re-audit"),
        ("5. Human Validation", "Review evidence and approve decisions"),
    ]

    for heading, description in stages:
        with st.container(border=True):
            st.markdown(f"**{heading}**")
            st.write(description)

elif page == "Selector Comparison":
    st.subheader("Re-audit Selector Comparison")

    if comparison is None:
        st.error(f"Results file not found: {COMPARISON_FILE}")
        st.code("python scripts/compare_dart_selectors.py")
    else:
        st.write(
            "Explore the saved comparison results for the baseline, "
            "impact-first, and hybrid selectors."
        )

        # JSON structure may vary as the experiment evolves.
        # Display it without assuming specific internal field names.
        st.json(comparison, expanded=True)

        st.subheader("Numeric Summary")
        numeric_rows = []

        def collect_numeric(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    path = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        numeric_rows.append(
                            {"Metric": path, "Value": value}
                        )
                    elif isinstance(value, (dict, list)):
                        collect_numeric(value, path)
            elif isinstance(obj, list):
                for index, value in enumerate(obj):
                    collect_numeric(value, f"{prefix}[{index}]")

        collect_numeric(comparison)

        if numeric_rows:
            df = pd.DataFrame(numeric_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            chart_df = df[
                df["Metric"].str.contains(
                    "selected_cases|skipped_cases|reduction",
                    case=False,
                    regex=True,
                )
            ]

            if not chart_df.empty:
                st.bar_chart(
                    chart_df.set_index("Metric")["Value"]
                )

elif page == "Pipeline Results":
    st.subheader("End-to-End Impact Analysis")

    if pipeline is None:
        st.warning(f"Pipeline output not found: {PIPELINE_FILE}")
        st.write(
            "Generate the end-to-end simulation output first, "
            "or select the correct JSON file in the source code."
        )
    else:
        st.success("Pipeline output loaded successfully.")

        for key in [
            "metadata",
            "change_analysis",
            "obligation_mapping_audit",
            "impact_analysis",
        ]:
            if key in pipeline:
                with st.expander(key.replace("_", " ").title(), expanded=True):
                    st.json(pipeline[key])

        st.subheader("Complete Pipeline Output")
        st.json(pipeline)

elif page == "Raw JSON Data":
    st.subheader("Saved Experiment Files")

    for label, path, data in [
        ("Selector Comparison", COMPARISON_FILE, comparison),
        ("End-to-End Pipeline", PIPELINE_FILE, pipeline),
    ]:
        with st.expander(label, expanded=True):
            st.caption(str(path))
            if data is None:
                st.warning("File not found or could not be loaded.")
            else:
                st.json(data)

st.divider()
st.caption(
    "DART research prototype | Outputs require independent validation "
    "before any real compliance decision."
)