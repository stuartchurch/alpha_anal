import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import scipy.stats as stats
from statsmodels.stats.contingency_tables import mcnemar
import io

# ==========================================
# PAGE CONFIG & THEME SETUP
# ==========================================
st.set_page_config(page_title="UX Prototype Analysis", layout="wide")

# NHS Corporate Identity Colors
NHS_COLORS = {
    "blue": "#005EB8",
    "green": "#009639",
    "red": "#DA291C",
    "yellow": "#FAE100",
    "dark_grey": "#425563",
    "black": "#231f20",
    "white": "#FFFFFF"
}

def apply_theme(theme_choice, custom_colors=None, typography=None, borders=None):
    plt.rcParams.update(plt.rcParamsDefault)
    
    if theme_choice == "NHS":
        sns.set_theme(style="whitegrid", rc={"axes.edgecolor": NHS_COLORS["dark_grey"]})
        return [NHS_COLORS["blue"], NHS_COLORS["green"], NHS_COLORS["red"], NHS_COLORS["yellow"], NHS_COLORS["dark_grey"]]
    
    elif theme_choice == "FiveThirtyEight":
        plt.style.use('fivethirtyeight')
        return ['#008fd5', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b']
    
    else: # Custom
        sns.set_theme(style="white") 
        if typography:
            plt.rcParams.update({
                'font.family': typography['font'],
                'axes.titlesize': typography['title_size'],
                'axes.labelsize': typography['label_size'],
                'xtick.labelsize': typography['tick_size'],
                'ytick.labelsize': typography['tick_size']
            })
        if borders:
            plt.rcParams.update({
                'axes.grid': borders['show_grid'],
                'axes.grid.axis': borders['grid_axis'],
                'axes.spines.top': borders['show_top'],
                'axes.spines.right': borders['show_right'],
                'axes.spines.left': borders['show_left'],
                'axes.spines.bottom': True
            })
        return custom_colors if custom_colors else ["#4c72b0", "#55a868", "#c44e52"]

# Helpers for UI, Formatting, and Downloading
def download_plot(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    buf.seek(0)
    return buf

def chart_header(default_title, default_ylabel, default_xlabel, key_prefix):
    """Generates the Edit popover and Download button above charts, split evenly to prevent overlap"""
    c1, c2 = st.columns(2)
    with c1:
        with st.popover("Edit titles & axes", use_container_width=True):
            t = st.text_input("Chart Title", default_title, key=f"{key_prefix}_t")
            x = st.text_input("X-Axis Label", default_xlabel, key=f"{key_prefix}_x")
            y = st.text_input("Y-Axis Label", default_ylabel, key=f"{key_prefix}_y")
    return t, y, x, c2

def format_axes(ax):
    """Helper to strictly enforce grid settings against pandas/seaborn defaults on categorical charts"""
    if theme_choice == "Custom" and border_settings:
        if not border_settings['show_grid']:
            ax.grid(False)
        else:
            ax.grid(True, axis=border_settings['grid_axis'])
            if border_settings['grid_axis'] == 'x':
                ax.grid(False, axis='y')
            elif border_settings['grid_axis'] == 'y':
                ax.grid(False, axis='x')
    elif theme_choice == "NHS":
        # Pandas auto-draws x-grids on bar charts. We aggressively remove them here.
        ax.grid(False, axis='x')
        ax.grid(True, axis='y', color='#e6e6e6')

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("Dashboard Settings")
theme_choice = st.sidebar.selectbox("Choose Chart Theme", ["NHS", "FiveThirtyEight", "Custom"])

custom_palette, typography_settings, border_settings = None, None, None

if theme_choice == "Custom":
    st.sidebar.markdown("### 🎨 Custom Theme Editor")
    with st.sidebar.expander("1. Colors", expanded=False):
        c1 = st.color_picker("Base Chart Color", "#4c72b0")
        c2 = st.color_picker("Success Color", "#55a868")
        c3 = st.color_picker("Failure/Friction Color", "#c44e52")
        custom_palette = [c1, c2, c3]
        
    with st.sidebar.expander("2. Typography", expanded=True):
        custom_font = st.selectbox("Font Family", ["sans-serif", "serif", "monospace"])
        typography_settings = {
            'font': custom_font,
            'title_size': st.slider("Chart Title Size", 10, 24, 16),
            'label_size': st.slider("Axis Label Size", 8, 20, 12),
            'tick_size': st.slider("Tick Label Size", 6, 16, 11)
        }

    with st.sidebar.expander("3. Grid & Borders", expanded=True):
        border_settings = {
            'show_grid': st.checkbox("Show Gridlines", value=True),
            'grid_axis': st.selectbox("Gridline Direction", ["y", "x", "both"]),
            'show_top': st.checkbox("Show Top Border", value=False),
            'show_right': st.checkbox("Show Right Border", value=False),
            'show_left': st.checkbox("Show Left Border", value=False)
        }

palette = apply_theme(theme_choice, custom_palette, typography_settings, border_settings)

# ==========================================
# DATA LOADING & TOP MENU BUTTONS
# ==========================================
st.title("UX Prototype Analysis: Baseline vs. Treatment")

if 'use_demo' not in st.session_state:
    st.session_state.use_demo = False

# Try to load the demo CSV into memory for the download button
try:
    with open('nhs_app_usability_test_data.csv', 'r') as f:
        demo_csv_data = f.read()
except FileNotFoundError:
    demo_csv_data = ""

# Main Control Buttons - tightly grouped in equal columns and left aligned
col_btn1, col_btn2, col_btn3, col_pad = st.columns([1, 1, 1, 3])
with col_btn1:
    if st.button("Run demo with simulated data", use_container_width=True):
        st.session_state.use_demo = True

with col_btn2:
    st.download_button(
        label="Download simulated data", 
        data=demo_csv_data, 
        file_name="nhs_app_usability_test_data.csv", 
        mime="text/csv", 
        use_container_width=True
    )

with col_btn3:
    with st.popover("Data notes", use_container_width=True):
        st.markdown("""
        **Participant Demographics**
        * **participant_no**: Participant number (16 in total, P01 to P16).
        * **order**: Order in which participants receive the prototypes (8 participants in each group). Half of the users will have the baseline (existing design) prototype first, the other half will have the new design first. BT = Baseline then treatment, TB = treatment then baseline.
        * **nhs_app_use**: Frequency of NHS App use (never / weekly / monthly / rarely / na).
        * **digital_literacy**: Level of digital literacy (0 = lowest, 5 = highest, or na).
        * **age**: Age group (18 - 24 / 25 - 34 / 35 - 44 / 45 - 54 / 55 - 64).
        
        **Navigation & First Clicks**
        * **baseline_firstchoice** / **treatment_firstchoice**: Records whether they choose to follow the 'Test Results' or 'Messages' path first (TR = Test Results first, M = Messages first).
        
        **Task Success Measures**
        * **baseline_user_success** / **treatment_user_success**: Recorded when the user considers that they have found the test result (0 = No/not found, 1 = Yes, with some friction, 2 = Yes, easily).
        * **baseline_system_success** / **treatment_system_success**: Recorded when the user locates a screen with the test result on it (0 = No/not found, 1 = Yes, with some friction, 2 = Yes, easily).
        
        **Single Ease Questions (SEQ)**
        * **baseline_seq_find** / **treatment_seq_find**: Participants rated how easy or difficult it was to locate where the test result was (1 = very difficult, 7 = very easy).
        * **baseline_seq_understand** / **treatment_seq_understand**: Participants rated how easy or difficult it was to understand what the result meant (1 = very difficult, 7 = very easy).
        
        **Preferences**
        * **easier_design**: Which design was easier to use? (T = Treatment easiest, B = Baseline easiest).
        * **preferred_realworld**: Which version would you rather use to receive a real result? (T = Treatment preferred, B = Baseline preferred).
        """)

st.write("") # Spacer

uploaded_file = st.file_uploader("Upload CSV File", type="csv")
if uploaded_file is not None:
    st.session_state.use_demo = False
    data_source = uploaded_file
elif st.session_state.use_demo:
    data_source = 'nhs_app_usability_test_data.csv'
else:
    data_source = None

if not data_source:
    st.info("Please upload a CSV file or run the demo to view the analysis.")
    st.stop()

@st.cache_data
def load_data(source):
    df = pd.read_csv(source)
    df.columns = df.columns.str.strip().str.replace(' ', '')
    return df

df = load_data(data_source)
n_total = len(df)

# ==========================================
# SECTION 1: PARTICIPANT PROFILES
# ==========================================
st.header("1. Participant Profiles")
st.markdown("Demographic distributions by testing order (BT = Baseline First, TB = Treatment First)")

with st.expander("What it does & How to interpret"):
    st.write("Displays the demographic breakdown of participants, split by the order in which they tested the prototypes. Ensure that the blue and green bars are relatively balanced across categories.")

col1, col2, col3 = st.columns(3)

with col1:
    t, y, x, dl_col = chart_header("NHS App Use", "Participant Count", "Usage Frequency", "use")
    fig_use, ax_use = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x='nhs_app_use', hue='order', ax=ax_use, palette=palette[:2])
    ax_use.set_title(t); ax_use.set_ylabel(y); ax_use.set_xlabel(x)
    format_axes(ax_use)
    with dl_col: st.download_button("Download as PNG", download_plot(fig_use), "nhs_app_use.png", key="dl_use", use_container_width=True)
    st.pyplot(fig_use)

with col2:
    t, y, x, dl_col = chart_header("Digital Literacy", "Participant Count", "Score (0=Low, 5=High)", "lit")
    fig_lit, ax_lit = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x='digital_literacy', hue='order', ax=ax_lit, palette=palette[:2])
    ax_lit.set_title(t); ax_lit.set_ylabel(y); ax_lit.set_xlabel(x)
    format_axes(ax_lit)
    with dl_col: st.download_button("Download as PNG", download_plot(fig_lit), "digital_literacy.png", key="dl_lit", use_container_width=True)
    st.pyplot(fig_lit)

with col3:
    t, y, x, dl_col = chart_header("Age Group", "Participant Count", "Age", "age")
    fig_age, ax_age = plt.subplots(figsize=(5, 4))
    age_order = sorted(df['age'].dropna().unique())
    sns.countplot(data=df, x='age', hue='order', ax=ax_age, palette=palette[:2], order=age_order)
    ax_age.set_title(t); ax_age.set_ylabel(y); ax_age.set_xlabel(x)
    format_axes(ax_age)
    with dl_col: st.download_button("Download as PNG", download_plot(fig_age), "age_group.png", key="dl_age", use_container_width=True)
    st.pyplot(fig_age)

st.divider()

# ==========================================
# SECTION 2: FIRST CHOICES
# ==========================================
st.header("2. First Navigational Choices")

with st.expander("What it does & How to interpret (McNemar's Test)"):
    st.write("McNemar's test checks if a design change caused users to significantly shift their categorical choice. A p-value < 0.05 indicates the change in behavior is statistically significant.")

crosstab = pd.crosstab(df['baseline_firstchoice'], df['treatment_firstchoice'])
mac_result = mcnemar(crosstab, exact=True)

col1, col2 = st.columns([2, 1])
with col1:
    t, y, x, dl_col = chart_header("First Navigational Choice: Baseline vs Treatment", "Number of Participants", "Prototype Option", "fc")
    fig_fc, ax_fc = plt.subplots(figsize=(8, 5))
    choices = pd.DataFrame({'Baseline': df['baseline_firstchoice'].value_counts(), 'Treatment': df['treatment_firstchoice'].value_counts()}).fillna(0)
    choices.T.plot(kind='bar', stacked=False, ax=ax_fc, color=palette[:2])
    ax_fc.set_title(t); ax_fc.set_ylabel(y); ax_fc.set_xlabel(x)
    ax_fc.set_xticklabels(ax_fc.get_xticklabels(), rotation=0)
    format_axes(ax_fc)
    with dl_col: st.download_button("Download as PNG", download_plot(fig_fc), "first_choices.png", key="dl_fc", use_container_width=True)
    st.pyplot(fig_fc)

with col2:
    st.metric("McNemar's P-Value", f"{mac_result.pvalue:.4f}")
    if mac_result.pvalue < 0.05: st.success("Statistically significant shift in behavior.")
    else: st.info("No statistically significant difference.")

st.divider()

# ==========================================
# SECTION 3: TASK SUCCESS
# ==========================================
st.header("3. Task Success")

with st.expander("What it does & How to interpret (Wilcoxon & Mann-Whitney)"):
    st.write("**Wilcoxon Signed-Rank:** Compares paired ordinal success severity scores. P-value < 0.05 means one prototype is significantly better.\n\n**Mann-Whitney U:** Checks for order effects by comparing the improvement scores of the BT group against the TB group. P-value < 0.05 indicates a learning effect biased the results.")

user_stat, user_p = stats.wilcoxon(df['baseline_user_success'], df['treatment_user_success'])
sys_stat, sys_p = stats.wilcoxon(df['baseline_system_success'], df['treatment_system_success'])

m1, m2 = st.columns(2)
m1.metric("User Success Wilcoxon P-Value", f"{user_p:.4f}")
m2.metric("System Success Wilcoxon P-Value", f"{sys_p:.4f}")

# --- 3a. Task Success Distributions & Transitions (Individual Charts) ---
st.subheader("Task Success Distribution & Transitions")
def get_counts(series):
    return series.value_counts().reindex([0, 1, 2], fill_value=0)

success_labels = ['0: Not Found', '1: Found (Friction)', '2: Found (Easily)']
user_df = pd.DataFrame({'Baseline': get_counts(df['baseline_user_success']), 'Treatment': get_counts(df['treatment_user_success'])})
sys_df = pd.DataFrame({'Baseline': get_counts(df['baseline_system_success']), 'Treatment': get_counts(df['treatment_system_success'])})

# Row 1: Distributions
col_sd1, col_sd2 = st.columns(2)
with col_sd1:
    t, y, x, dl_col = chart_header("User-Perceived Success Distribution", "Number of Participants", "Success Level", "user_dist")
    fig, ax = plt.subplots(figsize=(6, 4))
    user_df.plot(kind='bar', ax=ax, color=palette[:2])
    ax.set_title(t); ax.set_ylabel(y); ax.set_xlabel(x)
    ax.set_xticklabels(success_labels, rotation=0)
    format_axes(ax)
    with dl_col: st.download_button("Download as PNG", download_plot(fig), "user_success_dist.png", key="dl_user_dist", use_container_width=True)
    st.pyplot(fig)

with col_sd2:
    t, y, x, dl_col = chart_header("System-Measured Success Distribution", "Number of Participants", "Success Level", "sys_dist")
    fig, ax = plt.subplots(figsize=(6, 4))
    sys_df.plot(kind='bar', ax=ax, color=palette[:2])
    ax.set_title(t); ax.set_ylabel(y); ax.set_xlabel(x)
    ax.set_xticklabels(success_labels, rotation=0)
    format_axes(ax)
    with dl_col: st.download_button("Download as PNG", download_plot(fig), "sys_success_dist.png", key="dl_sys_dist", use_container_width=True)
    st.pyplot(fig)

# Row 2: Transitions
col_st1, col_st2 = st.columns(2)
with col_st1:
    t, y, x, dl_col = chart_header("User Success Transitions", "Baseline", "Treatment", "user_trans")
    fig, ax = plt.subplots(figsize=(6, 4))
    user_crosstab = pd.crosstab(df['baseline_user_success'], df['treatment_user_success'], rownames=['Baseline'], colnames=['Treatment']).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)
    sns.heatmap(user_crosstab, annot=True, cmap='Greens', fmt='g', ax=ax, cbar=False)
    ax.set_title(t); ax.set_ylabel(y); ax.set_xlabel(x)
    # Heatmaps deliberately omit format_axes() to prevent ugly lines drawing across the colored cells
    with dl_col: st.download_button("Download as PNG", download_plot(fig), "user_success_trans.png", key="dl_user_trans", use_container_width=True)
    st.pyplot(fig)

with col_st2:
    t, y, x, dl_col = chart_header("System Success Transitions", "Baseline", "Treatment", "sys_trans")
    fig, ax = plt.subplots(figsize=(6, 4))
    sys_crosstab = pd.crosstab(df['baseline_system_success'], df['treatment_system_success'], rownames=['Baseline'], colnames=['Treatment']).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)
    sns.heatmap(sys_crosstab, annot=True, cmap='Greens', fmt='g', ax=ax, cbar=False)
    ax.set_title(t); ax.set_ylabel(y); ax.set_xlabel(x)
    # Heatmaps deliberately omit format_axes()
    with dl_col: st.download_button("Download as PNG", download_plot(fig), "sys_success_trans.png", key="dl_sys_trans", use_container_width=True)
    st.pyplot(fig)


# --- 3b. Order Effects Boxplots (Individual Charts) ---
st.subheader("Order Effect Analysis (Mann-Whitney U)")
df['user_success_diff'] = df['treatment_user_success'] - df['baseline_user_success']
df['sys_success_diff'] = df['treatment_system_success'] - df['baseline_system_success']

u_stat_user, p_user = stats.mannwhitneyu(df[df['order'] == 'BT']['user_success_diff'], df[df['order'] == 'TB']['user_success_diff'])
u_stat_sys, p_sys = stats.mannwhitneyu(df[df['order'] == 'BT']['sys_success_diff'], df[df['order'] == 'TB']['sys_success_diff'])

st.write(f"**User Success diff by Order p-value:** {p_user:.4f} | **System Success diff by Order p-value:** {p_sys:.4f}")

col_oe1, col_oe2 = st.columns(2)
with col_oe1:
    t, y, x, dl_col = chart_header("Improvement in User Success", "Diff Score (Treatment - Baseline)", "Testing Order", "order_user")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='order', y='user_success_diff', ax=ax, palette=palette[:2], showfliers=False)
    ax.set_title(t); ax.set_ylabel(y); ax.set_xlabel(x)
    format_axes(ax)
    with dl_col: st.download_button("Download as PNG", download_plot(fig), "order_effects_user.png", key="dl_order_user", use_container_width=True)
    st.pyplot(fig)

with col_oe2:
    t, y, x, dl_col = chart_header("Improvement in System Success", "Diff Score (Treatment - Baseline)", "Testing Order", "order_sys")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='order', y='sys_success_diff', ax=ax, palette=palette[:2], showfliers=False)
    ax.set_title(t); ax.set_ylabel(y); ax.set_xlabel(x)
    format_axes(ax)
    with dl_col: st.download_button("Download as PNG", download_plot(fig), "order_effects_sys.png", key="dl_order_sys", use_container_width=True)
    st.pyplot(fig)


# --- 3c. Sankey Plots ---
st.subheader("Success Flow (Sankey Diagrams)")
def build_sankey(df, base_col, treat_col, title):
    transitions = df.groupby([base_col, treat_col]).size().reset_index(name='value')
    node_colors = [NHS_COLORS.get("red", "red"), NHS_COLORS.get("yellow", "orange"), NHS_COLORS.get("green", "green")] * 2
    nodes = ['Base 0: Fail', 'Base 1: Friction', 'Base 2: Easy', 'Treat 0: Fail', 'Treat 1: Friction', 'Treat 2: Easy']
    links = [{'source': int(row[base_col]), 'target': int(row[treat_col]) + 3, 'value': row['value']} for _, row in transitions.iterrows()]
    fig = go.Figure(data=[go.Sankey(node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=nodes, color=node_colors),
                                    link=dict(source=[l['source'] for l in links], target=[l['target'] for l in links], value=[l['value'] for l in links]))])
    fig.update_layout(title_text=title, font_size=12, height=400, margin=dict(t=40, l=0, r=0, b=0))
    return fig

sk1, sk2 = st.columns(2)
with sk1:
    t_sk1, _, _, dl_sk1 = chart_header("User Success Transitions", "", "", "sankey_user")
    fig_sk1 = build_sankey(df, 'baseline_user_success', 'treatment_user_success', t_sk1)
    with dl_sk1: st.download_button("Download HTML", fig_sk1.to_html(), "user_sankey.html", key="dl_sk1", use_container_width=True)
    st.plotly_chart(fig_sk1, use_container_width=True)

with sk2:
    t_sk2, _, _, dl_sk2 = chart_header("System Success Transitions", "", "", "sankey_sys")
    fig_sk2 = build_sankey(df, 'baseline_system_success', 'treatment_system_success', t_sk2)
    with dl_sk2: st.download_button("Download HTML", fig_sk2.to_html(), "system_sankey.html", key="dl_sk2", use_container_width=True)
    st.plotly_chart(fig_sk2, use_container_width=True)

st.divider()

# ==========================================
# SECTION 4: SEQ SCORES
# ==========================================
st.header("4. Single Ease Question (SEQ)")

with st.expander("What it does & How to interpret (Boxplots & Wilcoxon)"):
    st.write("SEQ measures perceived ease of a task on a 1-7 scale. The Wilcoxon test compares each user's score on the Baseline vs Treatment. Look for the median line inside the box to move upwards (closer to 7).")

find_stat, find_p = stats.wilcoxon(df['baseline_seq_find'], df['treatment_seq_find'])
understand_stat, understand_p = stats.wilcoxon(df['baseline_seq_understand'], df['treatment_seq_understand'])

st.write(f"**SEQ Find P-Value:** {find_p:.4f} | **SEQ Understand P-Value:** {understand_p:.4f}")

df_find = pd.melt(df[['baseline_seq_find', 'treatment_seq_find']], var_name='Prototype', value_name='SEQ_Score')
df_find['Prototype'] = df_find['Prototype'].map({'baseline_seq_find': 'Baseline', 'treatment_seq_find': 'Treatment'})

df_understand = pd.melt(df[['baseline_seq_understand', 'treatment_seq_understand']], var_name='Prototype', value_name='SEQ_Score')
df_understand['Prototype'] = df_understand['Prototype'].map({'baseline_seq_understand': 'Baseline', 'treatment_seq_understand': 'Treatment'})

col_seq1, col_seq2 = st.columns(2)
with col_seq1:
    t, y, x, dl_col = chart_header("Ease of Finding Results", "SEQ Score (1-7)", "Prototype", "seq_find")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df_find, x='Prototype', y='SEQ_Score', ax=ax, palette=palette[:2], width=0.5, showfliers=False)
    sns.stripplot(data=df_find, x='Prototype', y='SEQ_Score', ax=ax, color='black', alpha=0.6, jitter=True)
    ax.set_title(t); ax.set_ylabel(y); ax.set_xlabel(x)
    ax.set_ylim(0.5, 7.5)
    format_axes(ax)
    with dl_col: st.download_button("Download as PNG", download_plot(fig), "seq_finding.png", key="dl_seq_find", use_container_width=True)
    st.pyplot(fig)

with col_seq2:
    t, y, x, dl_col = chart_header("Ease of Understanding Results", "SEQ Score (1-7)", "Prototype", "seq_und")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df_understand, x='Prototype', y='SEQ_Score', ax=ax, palette=palette[:2], width=0.5, showfliers=False)
    sns.stripplot(data=df_understand, x='Prototype', y='SEQ_Score', ax=ax, color='black', alpha=0.6, jitter=True)
    ax.set_title(t); ax.set_ylabel(y); ax.set_xlabel(x)
    ax.set_ylim(0.5, 7.5)
    format_axes(ax)
    with dl_col: st.download_button("Download as PNG", download_plot(fig), "seq_understanding.png", key="dl_seq_und", use_container_width=True)
    st.pyplot(fig)

st.divider()

# ==========================================
# SECTION 5: PREFERENCES
# ==========================================
st.header("5. Final Preferences")

with st.expander("What it does & How to interpret (Binomial Test)"):
    st.write("Evaluates binary choices using an exact Binomial test against a 50/50 chance baseline. A p-value < 0.05 proves the group strongly prefers one design over the other.")

easier_t_count = (df['easier_design'] == 'T').sum()
easier_b_count = (df['easier_design'] == 'B').sum()
easier_p = stats.binomtest(easier_t_count, n=n_total, p=0.5).pvalue

pref_t_count = (df['preferred_realworld'] == 'T').sum()
pref_b_count = (df['preferred_realworld'] == 'B').sum()
pref_p = stats.binomtest(pref_t_count, n=n_total, p=0.5).pvalue

st.write(f"**Easier Design P-Value:** {easier_p:.4f} | **Preferred Real World P-Value:** {pref_p:.4f}")

col_pref1, col_pref2 = st.columns([2, 1])
with col_pref1:
    t_pref, y_pref, x_pref, dl_pref = chart_header("Participant Preferences: Baseline vs Treatment", "Number of Participants", "Selected Prototype", "pref")
    fig_pref, ax_pref = plt.subplots(figsize=(8, 5))
    pref_data = pd.DataFrame({'Easier Design': [easier_b_count, easier_t_count], 'Preferred in Real World': [pref_b_count, pref_t_count]}, index=['Baseline (B)', 'Treatment (T)'])

    pref_data.T.plot(kind='bar', stacked=True, ax=ax_pref, color=palette[:2], edgecolor='black')
    ax_pref.axhline(n_total / 2, color=NHS_COLORS.get("red", "red"), linestyle='--', label='50% Threshold')
    ax_pref.set_title(t_pref); ax_pref.set_ylabel(y_pref); ax_pref.set_xlabel(x_pref)
    ax_pref.set_xticklabels(ax_pref.get_xticklabels(), rotation=0)
    format_axes(ax_pref)

    for c in ax_pref.containers:
        ax_pref.bar_label(c, label_type='center', color='white', weight='bold')

    ax_pref.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    with dl_pref: st.download_button("Download as PNG", download_plot(fig_pref), "preferences.png", key="dl_pref_btn", use_container_width=True)
    st.pyplot(fig_pref)
