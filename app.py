import json, ast, pandas as pd, streamlit as st, requests
from openai import OpenAI

st.set_page_config(page_title="LessonLab Prototype", layout="wide")

CSV_URL = st.secrets["SHEET_CSV_URL"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

@st.cache_data(ttl=300)
def load_matrix(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    required = [
        "domain","reasoning_type","ai_role","prompt_function",
        "control_min","control_max","activity_menu","literacy_checks",
        "verify_methods","assessment_evidence","advance_rule"
    ]
    for col in required:
        if col not in df.columns:
            df[col] = ""
    return df

df = load_matrix(CSV_URL)

st.sidebar.header("LessonLab Controls")
domains = sorted([d for d in df["domain"].dropna().astype(str).unique() if d.strip()!=""])
domain = st.sidebar.selectbox("Domain", domains)

filtered = df[df["domain"].astype(str)==domain]
types = sorted([t for t in filtered["reasoning_type"].dropna().astype(str).unique() if t.strip()!=""])
rtype = st.sidebar.selectbox("Reasoning type", types)

subset = filtered[filtered["reasoning_type"].astype(str)==rtype]
if subset.empty:
    st.error("No rows match this selection.")
    st.stop()

cmin = int(pd.to_numeric(subset.iloc[0]["control_min"], errors="coerce") if "control_min" in subset.columns else 1)
cmax = int(pd.to_numeric(subset.iloc[0]["control_max"], errors="coerce") if "control_max" in subset.columns else 5)
control = st.sidebar.slider("Control level (L1–L5)", min_value=max(1,cmin), max_value=min(5,cmax), value=max(1,cmin), step=1)

subject = st.sidebar.text_input("Subject/KLA", value="")
topic = st.sidebar.text_input("Topic or concept", value="")
year_stage = st.sidebar.text_input("Year/Stage", value="")
duration = st.sidebar.text_input("Duration (e.g., 60 min)", value="")

row = subset.iloc[0]

def parse_json_field(x):
    if pd.isna(x): return []
    s = str(x).strip()
    if s.startswith("[") or s.startswith("{"):
        try: return json.loads(s)
        except json.JSONDecodeError:
            try: return ast.literal_eval(s)
            except Exception: return []
    return []

activities = parse_json_field(row["activity_menu"])
literacy_checks = parse_json_field(row["literacy_checks"])
verify_methods = parse_json_field(row["verify_methods"])
advance_rule = str(row["advance_rule"]) if "advance_rule" in row else ""

task = None
for item in activities:
    try:
        if int(item.get("lvl")) == int(control):
            task = item.get("task"); break
    except Exception: pass
if not task and activities:
    task = activities[0].get("task")

prompt_function = str(row["prompt_function"])
ai_role = str(row["ai_role"])
literacy = literacy_checks[0] if literacy_checks else "verification"
verify = verify_methods[0] if verify_methods else "source_crosscheck"

st.title("LessonLab — Graduated Control Prototype")
st.caption("Deterministic selection from your Expanded Matrix. GPT verbalises the script.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Selection")
    st.markdown(f"**Domain:** {domain}")
    st.markdown(f"**Reasoning type:** {rtype}")
    st.markdown(f"**AI role:** {ai_role}")
    st.markdown(f"**Control level:** L{control}")
with col2:
    st.subheader("From Matrix")
    st.markdown(f"**Activity:** {task or '—'}")
    st.markdown(f"**AI-literacy check:** {literacy}")
    st.markdown(f"**Verify method:** {verify}")
    st.markdown(f"**Advance rule:** {advance_rule}")

client = OpenAI(api_key=OPENAI_API_KEY)

def build_system():
    return ("You are LESSONLAB. Write concise, classroom-ready scripts that show how AI "
            "amplifies reasoning while the teacher stays in control. Use a 5-phase structure. "
            "Phases: 1) Teacher Model, 2) Shared Reasoning, 3) Independent (constrained), "
            "4) Meta-Reasoning, 5) Consolidate/Transfer. Plain Australian English.")

def build_user_prompt():
    return f"""
Subject/KLA: {subject or '—'}
Year/Stage: {year_stage or '—'}
Duration: {duration or '—'}
Topic: {topic or '—'}

Domain: {domain}
Reasoning type: {rtype}
AI role: {ai_role}
Control level: L{control}

Selected task: {task or '—'}
AI-literacy check focus: {literacy}
Verification method: {verify}

Write:
- Teacher script lines
- Student action lines
- One explicit AI-literacy check step
- One verification step using the method
- A one-line suggestion whether to advance or hold control next lesson based on observed evidence
"""

if st.button("Generate 5-Phase Lesson"):
    with st.spinner("Generating…"):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content": build_system()},
                      {"role":"user","content": build_user_prompt()}],
            temperature=0.3,
        )
    text = resp.choices[0].message.content
    st.markdown("### 5-Phase Lesson Output")
    st.write(text)

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("Hold at this level next time"):
        st.info("Noted: Hold. Use the same control level in the next lesson.")
with c2:
    if st.button("Advance next time"):
        st.success(f"Recommendation set: advance to L{min(control+1,5)} next lesson.")
with c3:
    if st.button("Downgrade next time"):
        st.warning(f"Recommendation set: downgrade to L{max(control-1,1)} next lesson.")
