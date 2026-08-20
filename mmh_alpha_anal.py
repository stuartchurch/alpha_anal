import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import wilcoxon, binomtest
from statsmodels.stats.contingency_tables import mcnemar

# --- PAGE CONFIG ---
st.set_page_config(page_title="UX Analysis Dashboard", layout="wide")

# --- THEME SWITCHER ---
st.sidebar.header("Dashboard Settings")
theme = st.sidebar.selectbox("Choose Chart Theme", ["FiveThirtyEight", "Microsoft", "Excel", "NHS"])

# Apply selected theme
if theme == "FiveThirtyEight":
    plt.style.use("fivethirtyeight")
    sns.set_palette("Set2")
    semantic_colors = ["#fc4f30", "#e5ae38", "#6d904f"]  # 538 Red, Yellow, Green
    
elif theme == "Microsoft":
    plt.style.use("default")
    sns.set_style("whitegrid")
    # Microsoft Brand Colors: Blue, Green, Yellow, Orange/Red, Grey
    sns.set_palette(["#00A4EF", "#7FBA00", "#FFB900", "#F25022", "#737373"])
    semantic_colors = ["#F25022", "#FFB900", "#7FBA00"]  # MS Red, Yellow, Green
    
elif theme == "Excel":
    plt.style.use("default")
    sns.set_style("whitegrid")
    # Standard Office 2016+ Default Palette
    sns.set_palette(["#4472C4", "#ED7D31", "#A5A5A5", "#FFC000", "#5B9BD5", "#70AD47"])
    semantic_colors = ["#ED7D31", "#FFC000", "#70AD47"]  # Excel Orange/Red, Yellow, Green
    
elif theme == "NHS":
    plt.style.use("default")
    sns.set_style("white")
    # NHS Brand Guidelines: NHS Blue, Light Blue, Aqua, Dark Grey, Warm Yellow, Focus Red
    sns.set_palette(["#005EB8", "#41B6E6", "#00A9CE", "#425563", "#FFB81C", "#DA291C"])
    # NHS Emergency Red, Warm Yellow, NHS Green
    semantic_colors = ["#DA291C", "#FFB81C", "#007F3B"]

st.title("UX Prototype Analysis: Baseline vs. Treatment")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    # --- 1. DATA PREP ---
    df = pd.read_csv(uploaded_file)
    
    # Calculate SEQ averages & differences
    df["baseline_overall_seq"] = (df["baseline_seq_find"] + df["baseline_seq_understand"] + df["baseline_seq_nextsteps"]) / 3
    df["treatment_overall_seq"] = (df["treatment_seq_find"] + df["treatment_seq_understand"] + df["treatment_seq_nextsteps"]) / 3
    df["seq_difference"] = df["treatment_overall_seq"] - df["baseline_overall_seq"]
    
    # Create long-form dataframe for mixed modeling
    baseline = pd.DataFrame({"participant": df["participant_id"], "order": df["order"], "prototype": "Baseline", "overall_seq": df["baseline_overall_seq"]})
    treatment = pd.DataFrame({"participant": df["participant_id"], "order": df["order"], "prototype": "Treatment", "overall_seq": df["treatment_overall_seq"]})
    long_df = pd.concat([baseline, treatment])

    # --- 2. VISUALIZATIONS ---
    st.header("1. Visual Analysis")
    
    # Row 1: Demographics
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df, x="nhs_app_use", order=["Never", "Rarely", "Monthly", "Weekly"], ax=ax)
        ax.set_title("Participant NHS App Usage")
        ax.set_xlabel("")
        ax.set_ylabel("Participants")
        st.pyplot(fig)
        
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df, x="screening_knowledge", order=["Low", "Medium", "High"], ax=ax)
        ax.set_title("Screening Knowledge")
        ax.set_xlabel("")
        ax.set_ylabel("")
        st.pyplot(fig)

    # Row 2: Success & Interpretation
    col3, col4 = st.columns(2)
    with col3:
        success_map = {0: "Failure", 1: "Success With Friction", 2: "Success No Friction"}
        b_succ = df["baseline_success"].map(success_map).value_counts()
        t_succ = df["treatment_success"].map(success_map).value_counts()
        plot_df = pd.DataFrame({"Baseline": b_succ, "Treatment": t_succ}).fillna(0)
        plot_df = plot_df.reindex(["Failure", "Success With Friction", "Success No Friction"]).fillna(0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        # Applied dynamic semantic colors here
        plot_df.T.plot(kind="bar", stacked=True, ax=ax, color=semantic_colors)
        ax.set_title("Task Success by Prototype")
        ax.set_ylabel("Participants")
        plt.xticks(rotation=0)
        st.pyplot(fig)
        
    with col4:
        interpret_map = {0: "Incorrect", 1: "Partially Correct", 2: "Correct"}
        base = df["baseline_interpretation"].map(interpret_map).value_counts(normalize=True)
        treat = df["treatment_interpretation"].map(interpret_map).value_counts(normalize=True)
        interp_df = pd.DataFrame({"Baseline": base, "Treatment": treat}).fillna(0)
        interp_df = interp_df.reindex(["Incorrect", "Partially Correct", "Correct"]).fillna(0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        # Applied dynamic semantic colors here
        interp_df.T.plot(kind="bar", stacked=True, ax=ax, color=semantic_colors)
        ax.set_title("Interpretation Accuracy")
        ax.set_ylabel("Proportion")
        plt.xticks(rotation=0)
        st.pyplot(fig)

    # Row 3: SEQ Boxplots
    col5, col6 = st.columns(2)
    with col5:
        plot_df = pd.DataFrame({"Baseline": df["baseline_seq_find"], "Treatment": df["treatment_seq_find"]})
        plot_df = plot_df.melt(var_name="Prototype", value_name="SEQ")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=plot_df, x="Prototype", y="SEQ", ax=ax)
        sns.stripplot(data=plot_df, x="Prototype", y="SEQ", color="#333333", alpha=0.4, ax=ax)
        ax.set_title("Ease of Finding Result")
        st.pyplot(fig)
        
    with col6:
        seq_plot = pd.DataFrame({"Baseline": df["baseline_overall_seq"], "Treatment": df["treatment_overall_seq"]})
        seq_plot = seq_plot.melt(var_name="Prototype", value_name="Overall SEQ")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=seq_plot, x="Prototype", y="Overall SEQ", ax=ax)
        sns.stripplot(data=seq_plot, x="Prototype", y="Overall SEQ", color="#333333", alpha=0.4, ax=ax)
        ax.set_title("Overall SEQ Distribution")
        st.pyplot(fig)

    # Row 4: Differences & Order Effects
    col7, col8 = st.columns(2)
    with col7:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df["seq_difference"], bins=10, ax=ax)
        ax.axvline(df["seq_difference"].mean(), color=semantic_colors[0], linestyle="--")
        ax.set_title("Treatment - Baseline Difference Scores")
        st.pyplot(fig)
        
    with col8:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x="order", y="seq_difference", ax=ax)
        sns.stripplot(data=df, x="order", y="seq_difference", color="#333333", ax=ax)
        ax.set_title("Difference Scores by Order Group")
        ax.set_ylabel("Treatment − Baseline")
        st.pyplot(fig)

    # Row 5: Preference
    st.subheader("Prototype Preference")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df, x="preferred_realworld", ax=ax)
    ax.set_title("Preferred Prototype for Real Results")
    ax.set_xlabel("")
    st.pyplot(fig)

    st.divider()

    # --- 3. STATISTICAL ANALYSIS ---
    st.header("2. Statistical Testing")

    # A. Mixed Modeling
    st.subheader("Mixed Effects Model (order effects) - primary inferential model")
    with st.expander("What it does & How to interpret"):
        st.write("""
        * **What it does:** A Mixed Effects Model (specifically a Linear Mixed Model) is an advanced regression technique that accounts for both fixed effects (the predictable variables we are testing, like the prototype and the testing order) and random effects (the unpredictable, natural variance between individual human participants).
        * **Why we use it here:** In within-subjects testing, the order in which users see the designs (e.g., Baseline first vs. Treatment first) can heavily bias their scores due to learning effects or fatigue. Furthermore, some users are just naturally harsher or more generous graders than others. By setting groups=long_df["participant"], this model gives each user their own personal "baseline" score, allowing us to isolate the true impact of the prototype while mathematically controlling for individual participant quirks and order biases.
        * **How to interpret:** Look at the P>|z| (p-value) column for the following specific rows: 
            * **prototype[T.Treatment] (Main Treatment Effect):** This tells you if the Treatment is genuinely better (or worse) than the Baseline. If the p-value is < 0.05, there is a statistically significant difference in SEQ scores between the prototypes, independent of the order they were shown. 
            * **order[T.TB] (Main Order Effect):** This checks if the sequence itself changed how users scored the session. A significant p-value here means that participants who saw the Treatment then Baseline (TB) rated things systematically higher or lower overall than the other group. 
            * **prototype[T.Treatment]:order[T.TB] (Interaction Effect):** This is the critical test for asymmetric order effects. If this p-value is significant (< 0.05), it means the effectiveness of the prototype depends on the order it was shown (e.g., the Treatment only scores higher if they saw the Baseline first). If this happens, you cannot fully trust a simple main treatment effect because the testing sequence confounded the results.
        """)
    
    model = smf.mixedlm("overall_seq ~ prototype * order", data=long_df, groups=long_df["participant"])
    results = model.fit()
    st.code(results.summary().as_text(), language="text")

    # B. Wilcoxon & Cohen's dz
    st.subheader("Wilcoxon Signed Ranks Tests & Effect Size (Cohen's dz)")
    with st.expander("What it does & How to interpret"):
        st.write("""
        **Wilcoxon Signed Ranks Test:**
        * **What it does:** This is a non-parametric test used to compare two related samples to assess whether their population mean ranks differ. It is the non-parametric equivalent of a paired t-test.
        * **Why we use it here:** The Single Ease Question (SEQ) uses a 7-point Likert scale. Because this data is ordinal (ranked categories) rather than continuous, it often violates the assumption of normal distribution required for a standard t-test. The Wilcoxon test safely handles this ordinal data by comparing the magnitude and direction of the differences in SEQ scores between the baseline and treatment for each participant.
        * **How to interpret:** Significant p-value (< 0.05): Users found a statistically significant difference in the ease of use between the two prototypes. Check the median scores to see which one performed better.
        
        **Effect Size (Cohen's dz):**
        * **What it does:** While p-values tell you if a statistically significant difference exists, effect size tells you how big or meaningful that difference actually is. Cohen’s dz is the specific variation of Cohen's d used for within-subjects (paired) designs.
        * **Why we use it here:** With a large enough sample size, even tiny, practically useless differences can become statistically significant. Calculating dz standardizes the difference so you can understand the true impact of the treatment.
        * **How to interpret:** Standard benchmarks for Cohen's dz (though context always matters in UX): 
            * ~0.2: Small effect (a minor improvement) 
            * ~0.5: Medium effect (a noticeable improvement) 
            * ~0.8 or higher: Large effect (a massive difference in the user experience)
        """)
        
    stat, p_val_wilcoxon = wilcoxon(df["baseline_overall_seq"], df["treatment_overall_seq"])
    std_diff = df["seq_difference"].std(ddof=1)
    cohens_dz = df["seq_difference"].mean() / std_diff if std_diff != 0 else 0
    
    col_w1, col_w2 = st.columns(2)
    col_w1.metric(label="Wilcoxon P-Value", value=f"{p_val_wilcoxon:.5f}")
    col_w2.metric(label="Cohen's dz", value=f"{cohens_dz:.3f}")

    # C. McNemar Test
    st.subheader("McNemar Test - Task Success")
    with st.expander("What it does & How to interpret"):
        st.write("""
        * **What it does:** The McNemar test is used to determine if there is a statistically significant difference in proportions between two paired groups. It is specifically designed for binary, categorical data (e.g., Pass/Fail or Yes/No).
        * **Why we use it here:** Because the same participants completed tasks on both prototypes, their success rates are dependent. The McNemar test ignores the users who had the same outcome on both prototypes (e.g., failed both or passed both). Instead, it looks only at the "discordant pairs"—the users who failed the baseline but passed the treatment, versus those who passed the baseline but failed the treatment.
        * **How to interpret:** Significant p-value (< 0.05): One prototype had a significantly higher success rate than the other. Non-significant p-value: The difference in task success between the baseline and treatment is not large enough to rule out random chance.
        """)
        
    b_binary = (df["baseline_success"] > 0).astype(int)
    t_binary = (df["treatment_success"] > 0).astype(int)
    table = pd.crosstab(b_binary, t_binary)
    
    if table.shape == (2,2):
        mcnemar_result = mcnemar(table, exact=True)
        st.metric(label="McNemar P-Value", value=f"{mcnemar_result.pvalue:.5f}")
    else:
        st.info("Not enough variance in success rates to run McNemar's test.")

    # D. Binomial Test
    st.subheader("Preference binomial test")
    with st.expander("What it does & How to interpret"):
        st.write("""
        * **What it does:** The binomial test compares an observed frequency of two categories against an expected distribution. In most preference testing, the expected baseline distribution is a 50/50 split (random chance).
        * **Why we use it here:** At the end of the session, you likely asked participants, "Which prototype did you prefer?". This test evaluates whether the number of votes for the 'treatment' prototype is statistically meaningful, or if it could have just happened by flipping a coin.
        * **How to interpret:** Significant p-value (< 0.05): There is a clear, statistically significant preference for one prototype over the other. Non-significant p-value: The preference is too evenly split to declare a definitive winner; any slight advantage may just be statistical noise.
        """)
        
    n_treatment = df["preferred_realworld"].eq("Treatment").sum()
    binom_res = binomtest(n_treatment, len(df), p=0.5)
    
    col_b1, col_b2 = st.columns(2)
    col_b1.metric(label="Treatment Preferences", value=f"{n_treatment} / {len(df)}")
    col_b2.metric(label="Binomial P-Value", value=f"{binom_res.pvalue:.5f}")
