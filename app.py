"""
app.py
======
Streamlit UI for the Sarcasm Detection System.
All ML logic lives in sarcasm_core.py – this file contains ONLY UI code.

Run:
    streamlit run app.py
"""

import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sarcasm_core import (
    preprocess,
    preprocess_trace,
    BernoulliNB,
    log_softmax,
    run_pipeline,
)

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Sarcasm Detection",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
[data-testid="stAppViewContainer"]      { background:#f4f6fb; }
[data-testid="stSidebar"]               { background:#ffffff; border-right:1px solid #e3e8f0; }
[data-testid="stSidebar"] > div:first-child { padding-top:0; }
[data-testid="stSidebarCollapseButton"] { display:none; }
header[data-testid="stHeader"]          { background:transparent; }
[data-testid="stMain"]                  { background:#f4f6fb; }
.block-container { padding:2rem 2.5rem !important; max-width:100% !important; }

.sidebar-logo { padding:22px 20px 18px; border-bottom:1px solid #e8ecf3; margin-bottom:10px; }
.sidebar-logo .logo-name { font-size:1.15rem; font-weight:700; color:#1a2340; }

.nav-section-label {
    font-size:.68rem; font-weight:700; color:#aab4c4;
    text-transform:uppercase; letter-spacing:1px; padding:10px 24px 4px;
}

.sidebar-footer { padding:14px 20px; border-top:1px solid #e8ecf3; }
.sidebar-footer .user-label { font-size:.75rem; color:#8a94a8; }
.sidebar-footer .user-name  { font-size:.85rem; font-weight:600; color:#2d3a52; }
.sidebar-footer .version    { font-size:.72rem; color:#aab4c4; margin-top:4px; }

.card        { background:#fff; border:1px solid #e3e8f0; border-radius:14px;
               padding:20px; margin-bottom:14px; box-shadow:0 1px 4px rgba(0,0,0,.05); }
.metric-card { background:#fff; border:1px solid #e3e8f0; border-radius:10px;
               padding:14px; text-align:center; margin:4px;
               box-shadow:0 1px 3px rgba(0,0,0,.04); }

.sarc-badge { background:linear-gradient(135deg,#ff6b6b,#cc3333); color:#fff;
              border-radius:28px; padding:7px 22px; font-size:1.15rem;
              font-weight:700; display:inline-block; }
.not-badge  { background:linear-gradient(135deg,#3ecf8e,#1a9e6a); color:#fff;
              border-radius:28px; padding:7px 22px; font-size:1.15rem;
              font-weight:700; display:inline-block; }

.chip     { display:inline-block; border-radius:5px; padding:2px 7px;
            margin:2px; font-size:.76rem; font-family:monospace; }
.chip-uni { background:#eef0ff; border:1px solid #c5caee; color:#3d52d5; }
.chip-neg { background:#fff0f0; border:1px solid #f5b8b8; color:#c0392b; }
.chip-bi  { background:#e8f4ff; border:1px solid #b0d4f0; color:#1a6898; }
.chip-unk { background:#fff8e8; border:1px solid #f0d890; color:#8a6200; }

.code-block { background:#1e2235; border:1px solid #2e3455; border-radius:8px;
              padding:14px; font-family:monospace; font-size:.78rem;
              color:#c8d8ff; white-space:pre; overflow-x:auto; line-height:1.55; }

.step-box { border-left:3px solid #3d52d5; padding:8px 14px; margin:6px 0;
            background:#f0f3ff; border-radius:0 8px 8px 0; color:#2d3a52; }

.page-title    { font-size:1.7rem; font-weight:700; color:#1a2340; margin-bottom:4px; }
.page-subtitle { font-size:.88rem; color:#7a86a0; margin-bottom:1.5rem; }

.config-banner { background:#fff; border:1px solid #e3e8f0; border-radius:10px;
                 padding:10px 18px; font-size:.82rem; color:#5a6478;
                 box-shadow:0 1px 3px rgba(0,0,0,.04); margin-bottom:1.2rem; }
.config-banner b { color:#1a2340; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def page_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def metric_cards(m: dict,
                 keys=('accuracy','precision','recall','f1_macro','f1_sarc'),
                 labels=('Accuracy','Precision','Recall','F1 Macro','F1 Sarcastic'),
                 colors=('#3d52d5','#1a9e6a','#1a6898','#8a6200','#c0392b')):
    cols = st.columns(len(keys))
    for col, (lbl, key, clr) in zip(cols, zip(labels, keys, colors)):
        col.markdown(
            f'<div class="metric-card">'
            f'<div style="color:#7a86a0;font-size:.78rem;">{lbl}</div>'
            f'<div style="color:{clr};font-size:1.5rem;font-weight:700;">{m[key]:.3f}</div>'
            f'</div>', unsafe_allow_html=True)


def plot_cm(cm: np.ndarray, title: str = 'Confusion Matrix'):
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    fig.patch.set_facecolor('#ffffff'); ax.set_facecolor('#ffffff')
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred: Not Sarc','Pred: Sarcastic'], color='#2d3a52', fontsize=9)
    ax.set_yticklabels(['True: Not Sarc','True: Sarcastic'], color='#2d3a52', fontsize=9)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i,j], ha='center', va='center', fontsize=14, fontweight='bold',
                    color='white' if cm[i,j] > cm.max()/2 else '#333')
    ax.set_title(title, color='#1a2340', fontsize=11)
    plt.tight_layout()
    return fig


def chips(tokens: list) -> str:
    html = ''
    for t in tokens:
        if t == '<UNK>':
            html += f'<span class="chip chip-unk">{t}</span>'
        elif t.startswith('NOT_'):
            html += f'<span class="chip chip-neg">{t}</span>'
        elif '_' in t:
            html += f'<span class="chip chip-bi">{t}</span>'
        else:
            html += f'<span class="chip chip-uni">{t}</span>'
    return html


def code_expander(title: str, code_str: str, explanation: str):
    with st.expander(f"📖 {title} – Code & Explanation", expanded=False):
        c1, c2 = st.columns([1,1])
        with c1:
            st.markdown("**Code**")
            st.markdown(f'<div class="code-block">{code_str}</div>', unsafe_allow_html=True)
        with c2:
            st.markdown("**Explanation**")
            st.markdown(explanation)


def light_chart(fig, ax):
    fig.patch.set_facecolor('#ffffff'); ax.set_facecolor('#f8f9ff')
    ax.tick_params(colors='#2d3a52'); ax.spines[:].set_color('#e3e8f0')
    return fig, ax


def build_vec(tokens: list, S: dict) -> np.ndarray:
    vec = np.zeros((1, S['n_feat']), dtype=np.float32)
    for t in tokens:
        if t in S['tok2col']:            vec[0, S['tok2col'][t]] += 1
        elif t in S['non_prime_set']:    vec[0, 0] += 1
    return vec


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

NAV_ITEMS = [
    ("⚙️",  "Setup",                     "setup"),
    ("📝",  "Task 1 – Preprocessing",    "task1"),
    ("📊",  "Task 2 – Features",         "task2"),
    ("🤖",  "Task 3 – Naïve Bayes",      "task3"),
    ("🔢",  "Task 4 – Numeric Features", "task4"),
    ("📉",  "Task 5 – Outlier Detection","task5"),
    ("⚖️",  "Task 6 – Scaling",          "task6"),
    ("⚡",  "Task 7 – Imbalance & Eval", "task7"),
    ("❌",  "Task 8 – Error Analysis",   "task8"),
    ("🔍",  "Live Predictor",            "live"),
]

if "active_page" not in st.session_state:
    st.session_state.active_page = "setup"

with st.sidebar:
    st.markdown(
        '<div class="sidebar-logo">'
        '<span style="font-size:1.4rem;">🎭</span>&nbsp;&nbsp;'
        '<span class="logo-name">Sarcasm Detector</span>'
        '</div>', unsafe_allow_html=True)

    st.markdown('<div class="nav-section-label">Configuration</div>', unsafe_allow_html=True)

    student_id_input = st.number_input(
        "🔑 Student ID", min_value=1, max_value=99999999,
        value=277211, step=1, help="seed = Student ID mod 100 000")
    derived_seed = student_id_input % 100000
    st.markdown(
        f'<div class="step-box" style="margin:4px 0 10px 0;">'
        f'Seed: <b style="color:#3d52d5">{derived_seed}</b></div>',
        unsafe_allow_html=True)
    dataset_path = st.text_input(
        "📂 Dataset path", value="Sarcasm_Headlines_Dataset.json",
        help="Path to the JSON-lines dataset file")

    st.markdown("---")
    st.markdown('<div class="nav-section-label">Navigation</div>', unsafe_allow_html=True)

    for icon, label, key in NAV_ITEMS:
        if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
            st.session_state.active_page = key
            st.rerun()

    st.markdown("---")
    st.markdown(
        f'<div class="sidebar-footer">'
        f'<div class="user-label">Student ID</div>'
        f'<div class="user-name">{student_id_input}</div>'
        f'<div class="version">Version 1.0.0 · Sarcasm Detection</div>'
        f'</div>', unsafe_allow_html=True)

# Active nav button highlight
_ai = next((i for i,(_, _, k) in enumerate(NAV_ITEMS)
            if k == st.session_state.active_page), 0)
st.markdown(f"""
<style>
section[data-testid="stSidebar"] div.stButton button {{
    background:transparent !important; color:#5a6478 !important;
    border:none !important; text-align:left !important;
    justify-content:flex-start !important; padding:9px 14px !important;
    border-radius:9px !important; font-size:.88rem !important;
    font-weight:500 !important; margin:1px 0 !important;
}}
section[data-testid="stSidebar"] div.stButton button:hover {{
    background:#f0f3ff !important; color:#3d52d5 !important;
}}
section[data-testid="stSidebar"] div.stButton:nth-of-type({_ai+3}) button {{
    background:linear-gradient(135deg,#3d52d5,#5468ff) !important;
    color:#ffffff !important; font-weight:600 !important;
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD & TRAIN
# ══════════════════════════════════════════════════════════════════════════════

if not Path(dataset_path).exists():
    st.error(f"❌ Dataset not found at `{dataset_path}`. "
             "Place **Sarcasm_Headlines_Dataset.json** in the same folder as app.py.")
    st.stop()


@st.cache_resource(show_spinner="⚙️ Training all models – please wait…")
def _cached_pipeline(path: str, sid: int) -> dict:
    return run_pipeline(path, sid)


S     = _cached_pipeline(dataset_path, int(student_id_input))
_page = st.session_state.active_page

st.markdown(
    f'<div class="config-banner">'
    f'<b>seed</b> = {S["seed"]} &nbsp;│&nbsp; '
    f'<b>α set</b> = {S["alphas"]} &nbsp;│&nbsp; '
    f'<b>k</b> = {S["k"]} &nbsp;│&nbsp; '
    f'<b>t</b> = {S["t"]} &nbsp;│&nbsp; '
    f'<b>Train</b> {S["n_train"]:,} &nbsp;'
    f'<b>Val</b> {S["n_val"]:,} &nbsp;'
    f'<b>Test</b> {S["n_test"]:,}'
    f'</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SETUP PAGE
# ══════════════════════════════════════════════════════════════════════════════

if _page == "setup":
    page_header("⚙️ Setup", "Seed configuration & dataset split")

    code_expander("Seed Configuration  →  sarcasm_core.derive_seed_config()",
        """seed         = student_id % 100000

smoothing_mod = seed % 3
alphas = [0, 0.1, 1]   # mod==0
       | [0, 0.5, 2]   # mod==1
       | [0, 1,   5]   # mod==2

k = seed % 4           # ablation index
t = (seed % 6) + 2    # rare-word threshold""",
        """**Why a seed?**
Every parameter is derived deterministically from your Student ID — same ID
always gives identical splits, vocabulary, and results.

- `alphas` – which Laplace smoothing values to compare  
- `k` – feature ablation index (0–3)  
- `t` – minimum document frequency; controls vocab size""")

    c1,c2,c3,c4 = st.columns(4)
    for col,(lbl,val,clr) in zip([c1,c2,c3,c4],[
        ("Seed",S['seed'],"#3d52d5"),("Alpha set",str(S['alphas']),"#1a9e6a"),
        ("k (ablation)",S['k'],"#ff9f43"),("t (rare-word)",S['t'],"#c0392b")]):
        col.markdown(f'<div class="metric-card"><div style="color:#7a86a0;font-size:.8rem;">'
                     f'{lbl}</div><div style="color:{clr};font-size:1.6rem;font-weight:700;">'
                     f'{val}</div></div>', unsafe_allow_html=True)

    st.markdown("### Dataset Split")
    code_expander("70 / 15 / 15 Split  →  sarcasm_core.split_dataset()",
        """df = df.sample(frac=1, random_state=seed)  # deterministic
n_train = int(0.70 * n)
n_val   = int(0.15 * n)
# remainder → test (≈15%)""",
        """- **Train (70%)** – fits all model parameters  
- **Validation (15%)** – hyper-parameter tuning without touching test  
- **Test (15%)** – final honest evaluation, used once only""")

    c1,c2,c3 = st.columns(3)
    c1.metric("Train", f"{S['n_train']:,}")
    c2.metric("Validation", f"{S['n_val']:,}")
    c3.metric("Test", f"{S['n_test']:,}")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 – PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

elif _page == "task1":
    page_header("📝 Task 1 – Tokenisation & Preprocessing",
                "5-step pipeline: clean → tokenise → negate → stopwords → n-grams → OOV")

    code_expander("Step 1 – Basic Cleaning  →  sarcasm_core.basic_clean()",
        """text = text.lower()
text = re.sub(r'<[^>]+>', ' ', text)             # strip HTML
# Tonal marker substitution
text = re.sub(r'[?!]*[!][?!]*[!][?!]*', '[EXC_SEQ]', text)
text = re.sub(r'[?][?]+', '[QUE_SEQ]', text)
text = re.sub(r'([?][!]+|[!][?]+)', '[INT_MARK]', text)
text = re.sub(r'\\.{3,}', '[ELLIP]', text)
# Remove punctuation, keep apostrophes in contractions
text = re.sub(r"[^\\w\\s'\\[\\]]", ' ', text)""",
        """1. **Lowercase** – unifies "Trump" and "trump"  
2. **HTML removal** – strips markup artefacts  
3. **Tonal markers** – `!!`, `??`, `?!`, `...` carry sarcasm signal;
   converting them to named tokens lets the model learn from them  
4. **Punctuation removal** – apostrophes inside contractions preserved for negation handling""")

    code_expander("Step 2 – Negation Handling  →  sarcasm_core.apply_negation()",
        """# Vowel count of negation word → window size
# "not"   (1 vowel) → prefix next 1 token  with NOT_
# "never" (2 vowels) → prefix next 2 tokens with NOT_
for tok in tokens:
    if tok in NEGATION_WORDS:
        n_vowels = count_vowels(tok)
        # prefix next n_vowels tokens""",
        """| Word | Vowels | Window |
|------|--------|--------|
| `not` | 1 | 1 token |
| `no` | 1 | 1 token |
| `never` | 2 | 2 tokens |
| `didn't` | 2 | 2 tokens |

`NOT_` tokens are **immune to stopword removal** to preserve the negation signal.""")

    code_expander("Steps 3–5 – Stopwords, N-grams, OOV  →  sarcasm_core.remove_stopwords / generate_ngrams / build_vocabulary()",
        """# Step 3 – NOT_ tokens always preserved
def remove_stopwords(tokens):
    return [t for t in tokens
            if t.startswith('NOT_') or t not in STOPWORDS]

# Step 4 – unigrams + bigrams
bigrams = ['_'.join(tokens[i:i+2]) for i in range(len(tokens)-1)]

# Step 5 – vocab from train only; unseen → <UNK>
vocab = build_vocabulary(train_tokens)
val_tokens = [preprocess(h, vocab) for h in val_headlines]""",
        """**Stopwords** – 60-word custom list reduces noise; `NOT_` tokens survive.  
**Bigrams** – capture phrases like `area_man` (strong Onion sarcasm marker).  
**OOV** – vocabulary built from training data only; new words in val/test → `<UNK>`.""")

    st.markdown("### 🔬 Interactive Preprocessing Trace")
    hl_input = st.text_input("Enter a headline to trace:",
        value="I can't believe they didn't announce another pointless committee!!!")
    if hl_input:
        trace = preprocess_trace(hl_input, S['vocab'])
        for title, content, desc in [
            ("Cleaned text",       trace['cleaned'],
             "Lowercase, HTML stripped, tonal markers substituted, punctuation removed."),
            ("Raw tokens",         trace['raw'],
             f"{len(trace['raw'])} tokens from regex tokenisation."),
            ("After negation",     trace['neg'],
             f"NOT_ prefixes applied – {sum(1 for t in trace['neg'] if t.startswith('NOT_'))} negated token(s)."),
            ("After stopwords",    trace['sw'],
             f"{len(trace['neg'])-len(trace['sw'])} stopword(s) removed."),
            ("After N-grams",      trace['ng'],
             f"{len(trace['ng'])-len(trace['sw'])} bigram(s) added."),
            ("Final (OOV mapped)", trace['final'],
             f"{sum(1 for t in trace['final'] if t=='<UNK>')} <UNK> token(s)."),
        ]:
            with st.expander(f"🔹 {title}", expanded=(title == "Cleaned text")):
                st.caption(desc)
                if isinstance(content, list):
                    st.markdown(chips(content), unsafe_allow_html=True)
                    c1,c2,c3 = st.columns(3)
                    c1.metric("Tokens", len(content))
                    c2.metric("NOT_ tokens", sum(1 for t in content if t.startswith('NOT_')))
                    c3.metric("Bigrams", sum(1 for t in content if '_' in t and not t.startswith('NOT_')))
                else:
                    st.code(content)
        st.markdown(
            "**Legend:** "
            '<span class="chip chip-uni">unigram</span> '
            '<span class="chip chip-neg">NOT_ token</span> '
            '<span class="chip chip-bi">bigram</span> '
            '<span class="chip chip-unk">&lt;UNK&gt;</span>',
            unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    c1.metric("Full vocabulary",           f"{S['vocab_size']:,}")
    c2.metric(f"After filtering (DF≥{S['t']})", f"{S['filtered_size']:,}")
    c3.metric("Prime-indexed features",    f"{S['n_feat']:,}")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 – FEATURES
# ══════════════════════════════════════════════════════════════════════════════

elif _page == "task2":
    import pandas as pd
    page_header("📊 Task 2 – Manual Feature Representation",
                "Count Vectoriser (prime-index scheme) + TF-IDF")

    code_expander("Count Vectoriser  →  sarcasm_core.build_count_matrix()",
        """# 1. Keep tokens with DF ≥ t (rare-word filter)
filtered = sorted([tok for tok,cnt in df_counts.items() if cnt >= t])

# 2. Prime positions → individual columns; others → col 0
prime_toks = [filtered[i-1] for i in range(1,N+1) if is_prime(i)]
tok2col    = {tok: idx+1 for idx,tok in enumerate(prime_toks)}

# 3. Build matrix
mat[row, tok2col[tok]] += 1   # prime feature
mat[row, 0]            += 1   # non-prime aggregate bucket""",
        """Tokens at **prime positions** (2,3,5,7,11…) each get a dedicated column;
all others share column 0.  
**Rare-word filtering** (DF < `t`) removes low-frequency noise before
prime assignment, so `t` directly controls feature matrix width.""")

    code_expander("TF-IDF  →  sarcasm_core.compute_tfidf()",
        """# Smoothed IDF: log((N+1)/(df+1)) + 1
idf[col] = log((N_train+1) / (df_counts[tok]+1)) + 1
# Normalised TF
tf = count_mat / row_sums
tfidf = tf * idf  # element-wise""",
        """- **TF** rewards tokens frequent *in this document*  
- **IDF** rewards tokens appearing in *few documents* (distinctive words)  
- **Smoothing** prevents log(0) and down-weights overly common terms""")

    st.markdown("### Top 10 Tokens by Document Frequency")
    top10_df = pd.DataFrame(S['top10'], columns=['Token','Document Frequency'])
    c1,c2 = st.columns([1,1.4])
    with c1:
        st.dataframe(top10_df, hide_index=True, use_container_width=True)
    with c2:
        fig, ax = plt.subplots(figsize=(6,3))
        fig,ax  = light_chart(fig,ax)
        toks = [r[0] for r in S['top10']]; cnts = [r[1] for r in S['top10']]
        bars = ax.barh(toks[::-1], cnts[::-1], color='#3d52d5', alpha=.85)
        ax.set_xlabel('Document Frequency', color='#2d3a52')
        ax.set_title('Top 10 Tokens', color='#1a2340')
        for b in bars:
            ax.text(b.get_width()+5, b.get_y()+b.get_height()/2,
                    str(int(b.get_width())), va='center', color='#7a86a0', fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Vocab size",             f"{S['vocab_size']:,}")
    c2.metric(f"After DF≥{S['t']} filter", f"{S['filtered_size']:,}")
    c3.metric("Prime-column features",  f"{len(S['prime_toks']):,}")
    c4.metric("Non-prime (col 0)",      f"{S['non_prime_set_len']:,}")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 3 – NAÏVE BAYES
# ══════════════════════════════════════════════════════════════════════════════

elif _page == "task3":
    import pandas as pd
    page_header("🤖 Task 3 – Naïve Bayes Variants",
                "MultinomialNB · BernoulliNB · log-space · Laplace smoothing")

    code_expander("Multinomial NB  →  sarcasm_core.MultinomialNB",
        """# fit()
log_priors[c]      = log(n_c / n)
tc                 = X_c.sum(axis=0) + alpha
log_likelihoods[c] = log(tc / tc.sum())

# predict_log_proba()
# log P(c|x) ∝ log P(c) + Σ x_i · log P(w_i|c)
scores = log_prior + X @ log_likelihoods""",
        """Models word *count frequencies*. Laplace smoothing (α > 0) assigns
non-zero probability to every word, preventing zero-probability collapse.
Everything computed in **log-space** to avoid numerical underflow.""")

    code_expander("Bernoulli NB  →  sarcasm_core.BernoulliNB",
        """# fit() – binarise counts first
X_bin = (X > 0)
p       = (X_c.sum(0) + alpha) / (n_c + 2*alpha)
log_p   = log(p);  log_1p = log(1 - p)

# predict_log_proba() – models ABSENT features too
scores = log_prior + X_bin @ log_p + (1-X_bin) @ log_1p""",
        """Uses **binary presence** (0 or 1) instead of counts.
Explicitly penalises *absent* features — every word NOT in the document
contributes `log(1 − P(w|c))`.
For short headlines this distinction matters because Bernoulli captures
the absence of "serious news" vocabulary.""")

    st.markdown("### Results – All Models × All Alpha Values (Validation)")
    rows = []
    for a in S['alphas']:
        for mk,lbl in [('mnb_count','MultNB Count'),('bnb_count','BernNB Count'),('mnb_tfidf','MultNB TF-IDF')]:
            m = S['nb_results'][a][mk]
            rows.append({'Model':lbl,'Alpha':a,
                'Accuracy':round(m['accuracy'],4),'Precision':round(m['precision'],4),
                'Recall':round(m['recall'],4),'F1 Macro':round(m['f1_macro'],4),
                'F1 Sarcastic':round(m['f1_sarc'],4)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    fig, axes = plt.subplots(1,3, figsize=(14,3.5))
    fig.patch.set_facecolor('#ffffff')
    for ax, mk, mname in zip(axes,
        ['mnb_count','bnb_count','mnb_tfidf'],
        ['Multinomial (Count)','Bernoulli (Count)','Multinomial (TF-IDF)']):
        ax.set_facecolor('#f8f9ff')
        x = np.arange(len(S['alphas'])); w = 0.35
        for i,(met,clr) in enumerate([('accuracy','#3d52d5'),('f1_sarc','#c0392b')]):
            vals = [S['nb_results'][a][mk][met] for a in S['alphas']]
            ax.bar(x+(i-.5)*w, vals, w, label=met.replace('_',' '), color=clr, alpha=.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f'α={a}' for a in S['alphas']], color='#2d3a52', fontsize=8)
        ax.tick_params(colors='#2d3a52'); ax.spines[:].set_color('#e3e8f0')
        ax.set_ylim(0,.85); ax.set_title(mname, color='#1a2340', fontsize=9)
        ax.legend(fontsize=7, facecolor='#fff', labelcolor='#2d3a52')
    plt.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TASK 4 – NUMERIC FEATURES
# ══════════════════════════════════════════════════════════════════════════════

elif _page == "task4":
    import pandas as pd
    page_header("🔢 Task 4 – Numeric Meta-Features & Imputation",
                "9 surface features · 5% missing injection · mean vs class-conditional imputation")

    code_expander("Feature Extraction  →  sarcasm_core.extract_numeric_features()",
        """word_count, char_length, uppercase_words,
excl_count, ques_count, negation_count,
digit_count, uppercase_ratio

# Feature 9 – Sarcasm Ratio (Ambiguity Constant = 0.5)
sarcasm_ratio = 0.5 if word_count == 0
                else uppercase_words / word_count""",
        """9 surface-level features that don't require grammar parsing.
`sarcasm_ratio` is the key derived feature: proportion of ALL-CAPS words.
**Ambiguity Constant 0.5** used when word_count = 0 (neutral/unknown).
Domain name (`theonion`, `huffpost`) is **prohibited**.""")

    code_expander("Imputation  →  sarcasm_core.mean_impute / class_conditional_impute()",
        """# Inject 5% NaN using seed → inject_missing_values()

# Method A – Global mean (fit on train only)
col_means = num_train_miss.mean()
imputed   = df_missing.fillna(col_means)

# Method B – Class-conditional mean
class_means = {c: num_train_miss[y_train==c].mean()
               for c in [0, 1]}""",
        """**Global mean** – simple; ignores class differences.
**Class-conditional** – computes separate means per class, preserving
the class distribution. Usually gives better discriminative performance.
Both strategies fit on **training data only** to avoid data leakage.""")

    st.markdown("### Sample Numeric Features (first 5 training rows)")
    st.dataframe(S['num_tr_sample'].round(4), use_container_width=True)

    na_df = pd.DataFrame(S['num_tr_miss_na'].items(), columns=['Feature','NaN Count'])
    c1,c2 = st.columns([1,1.5])
    with c1:
        st.markdown("### Missing Values per Column")
        st.dataframe(na_df, hide_index=True, use_container_width=True)
    with c2:
        fig,ax = plt.subplots(figsize=(6,3))
        fig,ax = light_chart(fig,ax)
        ax.bar(na_df['Feature'], na_df['NaN Count'], color='#ff9f43', alpha=.85)
        ax.tick_params(axis='x', rotation=35, colors='#2d3a52')
        ax.set_title('Missing Values per Feature', color='#1a2340')
        plt.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TASK 5 – OUTLIER DETECTION
# ══════════════════════════════════════════════════════════════════════════════

elif _page == "task5":
    page_header("📉 Task 5 – Outlier Detection & Treatment",
                "IQR method · 1.5× fence rule · clamping (Winsorisation)")

    code_expander("IQR Detection & Clamping  →  sarcasm_core.detect_outliers_iqr / clamp_outliers()",
        """# detect_outliers_iqr()
Q1 = df_train.quantile(0.25); Q3 = df_train.quantile(0.75)
IQR   = Q3 - Q1
lower = Q1 - 1.5 * IQR;  upper = Q3 + 1.5 * IQR
mask  = ((df < lower) | (df > upper)).any(axis=1)

# clamp_outliers()
df_clamped = df.clip(lower=lower, upper=upper, axis=1)""",
        """**IQR** = spread of the middle 50% of data.
Values beyond `Q1 − 1.5·IQR` or `Q3 + 1.5·IQR` are flagged.

**Clamping** chosen over removal:
- Keeps all rows → class distribution unchanged  
- Replaces extremes with fence value  
- Fences computed on **training data only**""")

    c1,c2 = st.columns(2)
    with c1:
        st.metric("Outlier rows detected (training)", f"{S['outlier_count']:,}")
        st.progress(min(S['outlier_count']/S['n_train'], 1.0))
        st.caption(f"{100*S['outlier_count']/S['n_train']:.1f}% of training rows")
    with c2:
        st.markdown('<div class="step-box">✅ <b>Treatment:</b> Clamping (Winsorisation)<br>'
                    'Extreme values clipped to the 1.5·IQR fence.<br>'
                    'Class distribution is unchanged.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TASK 6 – SCALING
# ══════════════════════════════════════════════════════════════════════════════

elif _page == "task6":
    page_header("⚖️ Task 6 – Scaling Study",
                "MinMax vs Standardisation · Gaussian NB on numeric features")

    code_expander("Scalers + GaussianNB  →  sarcasm_core.MinMaxScaler / StandardScaler / GaussianNB",
        """# MinMaxScaler → [0, 1]
transform(X) = (X - min_) / range_

# StandardScaler → N(0, 1)
transform(X) = (X - mean_) / std_

# ALWAYS fit on train only
mm_scaler  = MinMaxScaler().fit(X_train)
std_scaler = StandardScaler().fit(X_train)

# GaussianNB: assumes features ~ Normal(μ_c, σ²_c)
log P(x|c) = -0.5 Σ [ log(2πσ²) + (x-μ)²/σ² ]""",
        """Gaussian NB assumes normally distributed features — without scaling,
large-magnitude features (e.g. `char_length`) dominate the likelihood.

**MinMax** → [0,1]; sensitive to outliers (handled in Task 5).  
**Standardisation** → zero mean, unit variance; more robust and directly
satisfies the normality assumption.

Scalers **fitted on training data only** to prevent leakage.""")

    c1,c2 = st.columns(2)
    with c1:
        st.caption("MinMax Scaling"); metric_cards(S['scale_results']['minmax'])
    with c2:
        st.caption("Standardisation"); metric_cards(S['scale_results']['stdize'])

    fig, axes = plt.subplots(1,2, figsize=(9,3.5)); fig.patch.set_facecolor('#fff')
    for ax,(lbl,mk) in zip(axes,[('MinMax','minmax'),('Standardisation','stdize')]):
        ax.set_facecolor('#f8f9ff')
        m = S['scale_results'][mk]
        names = ['Accuracy','Precision','Recall','F1 Macro','F1 Sarcastic']
        vals  = [m['accuracy'],m['precision'],m['recall'],m['f1_macro'],m['f1_sarc']]
        clrs  = ['#3d52d5','#1a9e6a','#1a6898','#8a6200','#c0392b']
        ax.bar(names, vals, color=clrs, alpha=.85)
        ax.set_ylim(0,1); ax.set_title(f'GaussianNB – {lbl}', color='#1a2340')
        ax.tick_params(axis='x', rotation=20, colors='#2d3a52')
        ax.tick_params(axis='y', colors='#2d3a52')
        ax.spines[:].set_color('#e3e8f0')
        for i,(v,c) in enumerate(zip(vals,clrs)):
            ax.text(i, v+.01, f'{v:.3f}', ha='center', color=c, fontsize=8)
    plt.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TASK 7 – CLASS IMBALANCE & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

elif _page == "task7":
    page_header("⚡ Task 7 – Class Imbalance & Performance Evaluation",
                "Measure imbalance · random oversampling · threshold tuning · full metrics")

    code_expander("Random Oversampling  →  sarcasm_core.random_oversample()",
        """for c, cnt in zip(classes, counts):
    if cnt < max_count:
        n_extra = max_count - cnt
        idx = rng.choice(cnt, n_extra, replace=True)
        X_c = vstack([X_c, X_c[idx]])  # duplicate minority rows""",
        """If 60% of headlines are non-sarcastic, a trivial model gets 60% accuracy
by always predicting the majority class.
Random oversampling duplicates minority rows until both classes are equal.
Simple and effective — no information is discarded.""")

    code_expander("Threshold Tuning  →  sarcasm_core.tune_threshold()",
        """# Always tune on VALIDATION set, never test!
for thresh in arange(0.30, 0.81, 0.05):
    preds = (val_proba[:,1] >= thresh).astype(int)
    f1    = f1_score(y_val, preds)
    if f1 > best_f1:
        best_thresh = thresh""",
        """Default 0.5 assumes equal cost for FP and FN.
Sweeping thresholds on validation lets us tune the precision/recall trade-off:
- **Lower threshold** → higher recall (catch more sarcasm)  
- **Higher threshold** → higher precision (fewer false alarms)""")

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("### Class Distribution (Training)")
        cd = S['class_dist']
        fig,ax = plt.subplots(figsize=(5,3)); fig,ax = light_chart(fig,ax)
        bars = ax.bar(['Not Sarcastic (0)','Sarcastic (1)'],
                      [cd.get(0,0),cd.get(1,0)], color=['#1a6898','#c0392b'], width=.5, alpha=.85)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+30,
                    f'{int(b.get_height()):,}', ha='center', color='#2d3a52', fontsize=9)
        ax.set_ylabel('Count', color='#2d3a52'); plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.info(f"Imbalance ratio (0:1) = **{cd.get(0,0)/max(cd.get(1,0),1):.3f}**")
    with c2:
        st.markdown("### Threshold Tuning Curve")
        fig,ax = plt.subplots(figsize=(5,3)); fig,ax = light_chart(fig,ax)
        ths=[x[0] for x in S['thresh_curve']]; f1s=[x[1] for x in S['thresh_curve']]
        ax.plot(ths, f1s, color='#3d52d5', linewidth=2, marker='o', markersize=5)
        ax.axvline(S['best_thresh'], color='#c0392b', linestyle='--',
                   linewidth=1.5, label=f"Best = {S['best_thresh']:.2f}")
        ax.set_xlabel('Threshold', color='#2d3a52'); ax.set_ylabel('F1', color='#2d3a52')
        ax.set_title('F1 vs Threshold', color='#1a2340')
        ax.legend(facecolor='#fff', labelcolor='#2d3a52', fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.success(f"Best threshold: **{S['best_thresh']:.2f}** → F1 = **{S['best_f1']:.4f}**")

    st.markdown("### Validation Metrics")
    metric_cards(S['val_met']); st.markdown("")
    c_cm,c_exp = st.columns([1,1.5])
    with c_cm:
        fig = plot_cm(S['val_met']['cm'], 'Confusion Matrix – Validation')
        st.pyplot(fig); plt.close()
    with c_exp:
        cm = S['val_met']['cm']; tn,fp,fn,tp = cm[0,0],cm[0,1],cm[1,0],cm[1,1]
        st.markdown(f"""
**Matrix breakdown:**
- **TN = {tn}** correctly predicted non-sarcastic  
- **FP = {fp}** non-sarcastic → predicted sarcastic  
- **FN = {fn}** sarcastic → missed  
- **TP = {tp}** correctly detected sarcastic  

**Imbalance impact:** Accuracy is misleading when one class dominates —
F1 for the minority class is the more honest metric.
Threshold tuning recovers recall on the sarcastic class at the cost of
some precision.
""")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 8 – ERROR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

elif _page == "task8":
    import pandas as pd
    page_header("❌ Task 8 – Error Analysis",
                "8 misclassified headlines · 3 custom headlines · final test metrics")

    code_expander("Error Analysis  →  sarcasm_core.get_misclassified()",
        """test_proba = log_softmax(model.predict_log_proba(X_test))
y_pred     = (test_proba[:,1] >= threshold).astype(int)
mis_idx    = where(y_pred != y_test)[:8]
# type = 'FP' if y_pred==1 and y_test==0 else 'FN'""",
        """**FP** – genuine headline predicted as sarcastic (dramatic language).  
**FN** – sarcastic headline missed (subtle dry humour, normal vocabulary).  
Analysing these reveals blind spots and motivates richer features.""")

    st.markdown("### 8 Misclassified Test Headlines")
    for item in S['misclassified']:
        icon  = "🔴" if item['type']=='FP' else "🔵"
        label = "False Positive" if item['type']=='FP' else "False Negative"
        with st.expander(f"{icon} {label} — {item['headline'][:72]}…"):
            c1,c2,c3 = st.columns(3)
            c1.metric("Actual",      "Sarcastic" if item['actual']==1 else "Not Sarcastic")
            c2.metric("Predicted",   "Sarcastic" if item['pred']==1   else "Not Sarcastic")
            c3.metric("P(Sarcastic)",f"{item['prob']:.3f}")
            st.markdown(f'> *"{item["headline"]}"*')
            if item['type']=='FP':
                st.warning("Dramatic/unusual language the model associates with sarcasm.")
            else:
                st.info("Subtle sarcasm — vocabulary overlaps with real news headlines.")

    st.markdown("---")
    st.markdown("### ✍️ Test Your Own Headlines (Task 8B)")
    cols = st.columns(3)
    for col,(ltype,default) in zip(cols,[
        ("🎭 Sarcastic","area man solves climate change between breakfast and gym session"),
        ("📰 Serious",  "scientists develop new vaccine for common cold"),
        ("🤔 Mixed",    "government announces yet another task force to study housing costs"),
    ]):
        with col:
            st.markdown(f"**{ltype}**")
            hl = st.text_area("", value=default, height=85, key=f"t8_{ltype}")
            if hl.strip():
                toks = preprocess(hl, S['vocab'])
                vec  = build_vec(toks, S)
                prob = log_softmax(S['best_mnb'].predict_log_proba(vec))[0,1]
                is_s = prob >= S['best_thresh']
                badge = "sarc-badge" if is_s else "not-badge"
                txt   = f"🎭 Sarcastic ({prob:.1%})" if is_s else f"📰 Not Sarcastic ({1-prob:.1%})"
                st.markdown(f'<div class="{badge}">{txt}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Final Test Set Metrics")
    metric_cards(S['test_met']); st.markdown("")
    c_cm,c_tbl = st.columns([1,1.3])
    with c_cm:
        fig = plot_cm(S['test_met']['cm'], 'Confusion Matrix – Test Set')
        st.pyplot(fig); plt.close()
    with c_tbl:
        m = S['test_met']
        st.dataframe(pd.DataFrame({
            'Metric':['Accuracy','Precision (macro)','Recall (macro)','F1 (macro)','F1 (sarcastic)'],
            'Value': [f"{m['accuracy']:.4f}",f"{m['precision']:.4f}",
                      f"{m['recall']:.4f}",f"{m['f1_macro']:.4f}",f"{m['f1_sarc']:.4f}"],
        }), hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# LIVE PREDICTOR PAGE
# ══════════════════════════════════════════════════════════════════════════════

elif _page == "live":
    page_header("🔍 Live Predictor",
                "Type any headline — the trained model classifies it instantly")

    model_choice = st.radio("Model", ["Multinomial NB","Bernoulli NB"], horizontal=True)
    headline = st.text_area("Enter headline:", height=90,
        placeholder="e.g.  Nation proudly elects first president who openly admits he has no idea what he's doing")

    st.markdown("**Quick examples:**")
    examples = [
        "area man solves climate change between breakfast and gym session",
        "scientists develop new vaccine for common cold",
        "nation's dog owners demand answers on where all the good boys went",
        "fed raises interest rates for third consecutive time this year",
        "report: majority of office workers now just describing memes to each other",
    ]
    ex_cols = st.columns(len(examples))
    for col,ex in zip(ex_cols, examples):
        if col.button(ex[:38]+"…", use_container_width=True, key=ex[:18]):
            headline = ex

    if headline and headline.strip():
        toks = preprocess(headline, S['vocab'])
        vec  = build_vec(toks, S)
        mdl  = S['best_mnb'] if model_choice == "Multinomial NB" \
               else BernoulliNB(S['alphas'][-1]).fit(S['_Xtr_os'], S['_ytr_os'])
        prob = log_softmax(mdl.predict_log_proba(vec))[0,1]
        is_s = prob >= S['best_thresh']

        st.markdown("---")
        rc1,rc2,rc3 = st.columns([2.5,1,1])
        with rc1:
            badge = "sarc-badge" if is_s else "not-badge"
            lbl   = "🎭 SARCASTIC" if is_s else "📰 NOT SARCASTIC"
            st.markdown(f'<div class="card"><div class="{badge}">{lbl}</div>'
                        f'<br><p style="color:#5a6478;margin-top:10px;">"{headline}"</p></div>',
                        unsafe_allow_html=True)
        with rc2:
            st.markdown(f'<div class="metric-card"><div style="color:#7a86a0;font-size:.8rem;">'
                        f'P(Sarcastic)</div><div style="color:#c0392b;font-size:1.7rem;'
                        f'font-weight:700;">{prob:.1%}</div></div>', unsafe_allow_html=True)
        with rc3:
            st.markdown(f'<div class="metric-card"><div style="color:#7a86a0;font-size:.8rem;">'
                        f'P(Serious)</div><div style="color:#1a9e6a;font-size:1.7rem;'
                        f'font-weight:700;">{1-prob:.1%}</div></div>', unsafe_allow_html=True)

        fig,ax = plt.subplots(figsize=(8,.55)); fig.patch.set_alpha(0); ax.set_facecolor('none')
        ax.barh([0],[prob], color='#ff6b6b', height=.5)
        ax.barh([0],[1-prob], left=[prob], color='#3ecf8e', height=.5)
        ax.axvline(S['best_thresh'], color='#1a2340', linewidth=1.5, linestyle='--', alpha=.6)
        ax.text(S['best_thresh']+.01, .38, f"threshold={S['best_thresh']:.2f}",
                color='#1a2340', fontsize=8)
        ax.set_xlim(0,1); ax.set_yticks([]); ax.spines[:].set_visible(False)
        plt.tight_layout(pad=0); st.pyplot(fig); plt.close()

        with st.expander("🔬 Preprocessing trace"):
            trace = preprocess_trace(headline, S['vocab'])
            st.caption(f"Cleaned: `{trace['cleaned']}`")
            st.markdown(chips(trace['final']), unsafe_allow_html=True)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total tokens", len(trace['final']))
            c2.metric("Unigrams",  sum(1 for t in trace['final'] if '_' not in t and not t.startswith('NOT_')))
            c3.metric("Bigrams",   sum(1 for t in trace['final'] if '_' in t and not t.startswith('NOT_')))
            c4.metric("NOT_ tokens", sum(1 for t in trace['final'] if t.startswith('NOT_')))
            st.markdown(
                '<span class="chip chip-uni">unigram</span> '
                '<span class="chip chip-neg">NOT_ token</span> '
                '<span class="chip chip-bi">bigram</span> '
                '<span class="chip chip-unk">&lt;UNK&gt;</span>',
                unsafe_allow_html=True)
