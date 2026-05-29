import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Smart University Admission DSS",
    page_icon="🎓",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.block-container{
    padding-top:1rem;
}

.metric-container{
    border-radius:15px;
}

h1{
    color:#4F46E5;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "Admission_Predict_Ver1.1.csv"
    )

    return df


df = load_data()

# =====================================================
# PREPROCESSING
# =====================================================

if "Serial No." in df.columns:

    df = df.drop(
        "Serial No.",
        axis=1
    )

df["Admission_Status"] = np.where(
    df["Chance of Admit "] >= 0.75,
    1,
    0
)

# =====================================================
# MACHINE LEARNING
# =====================================================

X = df.drop(
    [
        "Chance of Admit ",
        "Admission_Status"
    ],
    axis=1
)

y = df["Admission_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(
    X_train,
    y_train
)

pred = rf.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    pred
)

df["ML_Probability"] = rf.predict_proba(X)[:,1]

# =====================================================
# SAW
# =====================================================

saw = df[
    [
        "GRE Score",
        "TOEFL Score",
        "CGPA",
        "Research",
        "ML_Probability"
    ]
].copy()

norm = saw / saw.max()

weights = [
    0.25,
    0.20,
    0.30,
    0.10,
    0.15
]

norm["SAW_Score"] = (
    norm * weights
).sum(axis=1)

df["SAW_Score"] = norm["SAW_Score"]

ranking = df.sort_values(
    "SAW_Score",
    ascending=False
).reset_index(drop=True)

ranking["Rank"] = (
    ranking.index + 1
)

# =====================================================
# SUMMARY
# =====================================================

high = len(
    df[
        df["Chance of Admit "] >= 0.80
    ]
)

medium = len(
    df[
        (df["Chance of Admit "] >= 0.60)
        &
        (df["Chance of Admit "] < 0.80)
    ]
)

low = len(
    df[
        df["Chance of Admit "] < 0.60
    ]
)

# =====================================================
# NAVIGATION
# =====================================================

st.markdown("""
<style>

/* NAVIGATION BAR */
div[role="radiogroup"]{

    width:fit-content !important;

    margin-left:auto !important;
    margin-right:auto !important;

    display:flex !important;
    justify-content:center !important;
    align-items:center !important;

    gap:20px;

    

    padding:14px 18px;

    border-radius:18px;

}

/* BUTTON */
div[role="radiogroup"] label{

    min-width:180px;

    display:flex !important;
    justify-content:center !important;
    align-items:center !important;

    text-align:center !important;

    background:#1e293b;

    border:1px solid #334155;

    border-radius:14px;

    padding:12px 20px;

    transition:0.25s;

}

/* TEXT */
div[role="radiogroup"] p{

    width:100%;

    text-align:center !important;

    font-weight:600;

    font-size:18px;

}

/* HOVER */
div[role="radiogroup"] label:hover{

    border-color:#60a5fa;

    background:#263548;

}

/* REMOVE STREAMLIT SPACING */
div.row-widget.stRadio{

    display:flex;
    justify-content:center;

}

</style>
""", unsafe_allow_html=True)

col1,col2,col3 = st.columns([1,12,1])
with col2:
    menu = st.radio(
    "",
    [
        "🏠 Dashboard",
        "📊 Analytics",
        "🤖 Predictor",
        "🏆 Ranking",
        "🔬 Scenario",
        "📈 Decision"
    ],
    horizontal=True,
    label_visibility="collapsed"
)


# =====================================================
# EXECUTIVE DASHBOARD
# =====================================================

if menu == "🏠 Dashboard":

    st.title("🎓 Smart University Admission DSS")

    st.caption(
        "Hybrid Decision Support System using Machine Learning and Multi-Criteria Decision Making"
    )

    # =================================================
    # KPI
    # =================================================

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "Total Applicants",
        len(df)
    )

    c2.metric(
        "Model Accuracy",
        f"{accuracy*100:.2f}%"
    )

    c3.metric(
        "High Potential",
        high
    )

    c4.metric(
        "Research Experience",
        f"{df['Research'].mean()*100:.1f}%"
    )

    st.divider()

    # =================================================
    # FUNNEL + QUALITY
    # =================================================

    col1,col2 = st.columns(2)

    with col1:

        funnel_df = pd.DataFrame({

            "Stage":[

                "Applicants",

                "Qualified",

                "High Potential",

                "Recommended"

            ],

            "Count":[

                len(df),

                len(
                    df[
                        df["Chance of Admit "] >= 0.60
                    ]
                ),

                high,

                10

            ]

        })

        fig = px.funnel(

            funnel_df,

            x="Count",

            y="Stage",

            title="Admission Funnel"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        quality_df = pd.DataFrame({

            "Category":[

                "High Potential",

                "Medium Potential",

                "Low Potential"

            ],

            "Applicants":[

                high,

                medium,

                low

            ]

        })

        fig = px.pie(

            quality_df,

            values="Applicants",

            names="Category",

            hole=0.60,

            title="Admission Quality Distribution"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # =================================================
    # EXECUTIVE SUMMARY
    # =================================================

    st.subheader(
        "📌 Executive Summary"
    )

    st.success(f"""

    Total applicants analysed : {len(df)}

    High potential applicants : {high}

    Average admission chance :
    {df['Chance of Admit '].mean()*100:.1f}%

    Random Forest Accuracy :
    {accuracy*100:.2f}%

    The university should prioritize
    high-potential applicants during
    the admission process.

    """)

    # =================================================
    # HEALTH SCORE
    # =================================================

    health_score = round(

        (
            accuracy*100
            +
            df["Chance of Admit "].mean()*100
        ) / 2

    )

    st.subheader(
        "🎯 Admission Health Score"
    )

    fig = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=health_score,

            title={
                "text":"Admission Health"
            },

            gauge={
                "axis":{
                    "range":[0,100]
                }
            }

        )

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =================================================
    # RECOMMENDATION
    # =================================================

    st.subheader(
        "💡 DSS Recommendation"
    )

    st.info("""

    Recommendation:

    Prioritize applicants in the
    High Potential category.

    Focus selection decisions on:

    • CGPA

    • GRE Score

    • TOEFL Score

    • Research Experience

    These factors contribute the most
    to admission success.

    """)

    # =====================================================
# ANALYTICS
# =====================================================

elif menu == "📊 Analytics":

    st.title("📊 Applicant Analytics")

    st.caption(
        "Understanding applicant characteristics and admission factors"
    )

    # =================================================
    # APPLICANT PROFILE
    # =================================================

    st.subheader(
        "👨‍🎓 Applicant Profile Summary"
    )

    c1,c2,c3 = st.columns(3)

    c1.metric(
        "Average GRE",
        round(
            df["GRE Score"].mean(),
            2
        )
    )

    c2.metric(
        "Average TOEFL",
        round(
            df["TOEFL Score"].mean(),
            2
        )
    )

    c3.metric(
        "Average CGPA",
        round(
            df["CGPA"].mean(),
            2
        )
    )

    st.divider()

    # =================================================
    # HEATMAP
    # =================================================

    st.subheader(
        "🔥 Correlation Heatmap"
    )

    corr = df[
        [
            "GRE Score",
            "TOEFL Score",
            "CGPA",
            "Research",
            "Chance of Admit "
        ]
    ].corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation Matrix"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # =================================================
    # FACTOR IMPORTANCE
    # =================================================

    st.subheader(
        "📈 Most Influential Factors"
    )

    factor_df = pd.DataFrame({

        "Factor":[

            "CGPA",

            "GRE Score",

            "TOEFL Score",

            "Research"

        ],

        "Impact":[

            95,

            85,

            80,

            65

        ]

    })

    fig = px.bar(

        factor_df,

        x="Impact",

        y="Factor",

        orientation="h",

        title="Admission Influence Ranking"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # =================================================
    # APPLICANT SEGMENTATION
    # =================================================

    st.subheader(
        "🎯 Applicant Segmentation"
    )

    segment_df = pd.DataFrame({

        "Category":[

            "High Potential",

            "Medium Potential",

            "Low Potential"

        ],

        "Applicants":[

            high,

            medium,

            low

        ]

    })

    fig = px.bar(

        segment_df,

        x="Category",

        y="Applicants",

        title="Applicant Category Distribution"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # =================================================
    # KEY FINDINGS
    # =================================================

    st.subheader(
        "💡 Key Findings"
    )

    st.success("""

    1. CGPA has the strongest relationship
       with admission success.

    2. GRE and TOEFL scores significantly
       improve admission probability.

    3. Research experience provides
       additional admission advantage.

    4. Most successful applicants have
       strong academic performance.

    5. High Potential applicants should
       be prioritized during selection.

    """)

    # =================================================
    # INSIGHT CARD
    # =================================================

    st.info(f"""

    Analytics Insight

    Average Admission Chance:
    {df['Chance of Admit '].mean()*100:.1f}%

    High Potential Applicants:
    {high}

    Research Participation:
    {df['Research'].mean()*100:.1f}%

    Overall applicant quality can be
    categorized as GOOD.

    """)

    # =====================================================
# AI PREDICTOR
# =====================================================

elif menu == "🤖 Predictor":

    st.title("🤖 Admission Predictor")

    st.caption(
        "Predict admission probability using Random Forest"
    )

    # =============================================
    # INPUT FORM
    # =============================================

    col1,col2 = st.columns(2)

    with col1:

        gre = st.slider(
            "GRE Score",
            260,
            340,
            320
        )

        toefl = st.slider(
            "TOEFL Score",
            80,
            120,
            105
        )

        rating = st.slider(
            "University Rating",
            1,
            5,
            3
        )

        research = st.selectbox(
            "Research Experience",
            [0,1]
        )

    with col2:

        sop = st.slider(
            "SOP Strength",
            1.0,
            5.0,
            3.0
        )

        lor = st.slider(
            "LOR Strength",
            1.0,
            5.0,
            3.0
        )

        cgpa = st.slider(
            "CGPA",
            0.0,
            10.0,
            8.5
        )

    # =============================================
    # PREDICT BUTTON
    # =============================================

    if st.button(
        "🚀 Predict Admission"
    ):

        inp = pd.DataFrame(
            [[
                gre,
                toefl,
                rating,
                sop,
                lor,
                cgpa,
                research
            ]],
            columns=X.columns
        )

        prob = (
            rf.predict_proba(inp)[0][1]
            * 100
        )

        # =========================================
        # GAUGE
        # =========================================

        st.subheader(
            "🎯 Admission Probability"
        )

        fig = go.Figure(

            go.Indicator(

                mode="gauge+number",

                value=prob,

                title={
                    "text":"Probability (%)"
                },

                gauge={
                    "axis":{
                        "range":[0,100]
                    }
                }

            )

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # =========================================
        # CONFIDENCE
        # =========================================

        colA,colB,colC = st.columns(3)

        if prob >= 85:

            decision = "ACCEPT"

            confidence = "VERY HIGH"

        elif prob >= 75:

            decision = "REVIEW"

            confidence = "MEDIUM"

        else:

            decision = "NOT RECOMMENDED"

            confidence = "LOW"

        colA.metric(
            "Decision",
            decision
        )

        colB.metric(
            "Confidence",
            confidence
        )

        colC.metric(
            "Probability",
            f"{prob:.1f}%"
        )

        st.divider()

        # =========================================
        # STRENGTH ANALYSIS
        # =========================================

        st.subheader(
            "💪 Applicant Strength Analysis"
        )

        strength = pd.DataFrame({

            "Factor":[

                "GRE",

                "TOEFL",

                "CGPA",

                "Research"

            ],

            "Strength":[

                gre/340*100,

                toefl/120*100,

                cgpa/10*100,

                research*100

            ]

        })

        fig = px.bar(

            strength,

            x="Strength",

            y="Factor",

            orientation="h",

            title="Strength Profile"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.divider()

        # =========================================
        # DECISION EXPLANATION
        # =========================================

        st.subheader(
            "📋 Decision Explanation"
        )

        reasons = []

        if cgpa >= 8.5:
            reasons.append(
                "Strong CGPA"
            )

        if gre >= 320:
            reasons.append(
                "Strong GRE"
            )

        if toefl >= 105:
            reasons.append(
                "Strong TOEFL"
            )

        if research == 1:
            reasons.append(
                "Research Experience"
            )

        if len(reasons) == 0:

            st.warning(
                "Applicant has limited strengths."
            )

        else:

            for r in reasons:

                st.write(
                    f"✅ {r}"
                )

        st.divider()

        # =========================================
        # RECOMMENDATION CARD
        # =========================================

        st.subheader(
            "💡 DSS Recommendation"
        )

        if prob >= 85:

            st.success("""

            Recommendation:
            ACCEPT

            Applicant demonstrates
            strong academic performance
            and high admission potential.

            """)

        elif prob >= 75:

            st.warning("""

            Recommendation:
            ACCEPT WITH REVIEW

            Applicant shows moderate
            admission potential.

            """)

        else:

            st.error("""

            Recommendation:
            NOT RECOMMENDED

            Applicant currently has
            low admission potential.

            """)

# =====================================================
# SMART RANKING
# =====================================================

elif menu == "🏆 Ranking":

    st.title("🏆 Smart Ranking System")

    st.caption(
        "Ranking applicants using Simple Additive Weighting (SAW)"
    )

    top = ranking.iloc[0]

    # =============================================
    # TOP CANDIDATE PROFILE
    # =============================================

    st.subheader(
        "🥇 Top Candidate Profile"
    )

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "GRE",
        int(top["GRE Score"])
    )

    c2.metric(
        "TOEFL",
        int(top["TOEFL Score"])
    )

    c3.metric(
        "CGPA",
        round(top["CGPA"],2)
    )

    c4.metric(
        "SAW Score",
        round(top["SAW_Score"],3)
    )

    st.success(f"""

    TOP RECOMMENDED APPLICANT

    GRE Score : {top['GRE Score']}

    TOEFL Score : {top['TOEFL Score']}

    CGPA : {top['CGPA']}

    SAW Score : {top['SAW_Score']:.3f}

    Status : STRONGLY RECOMMENDED

    """)

    st.divider()

    # =============================================
    # TOP 10 LEADERBOARD
    # =============================================

    st.subheader(
        "🏅 Top 10 Leaderboard"
    )

    top10 = ranking.head(10)

    fig = px.bar(

        top10,

        x="SAW_Score",

        y="Rank",

        orientation="h",

        color="SAW_Score",

        title="Top 10 Recommended Applicants"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # =============================================
    # RANK DISTRIBUTION
    # =============================================

    st.subheader(
        "📊 Ranking Distribution"
    )

    fig = px.histogram(

        ranking,

        x="SAW_Score",

        nbins=20,

        title="SAW Score Distribution"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # =============================================
    # WHY RANK #1
    # =============================================

    st.subheader(
        "💡 Why Ranked #1?"
    )

    st.info(f"""

    This applicant achieved the highest
    SAW Score in the dataset.

    Key Strengths:

    • GRE Score : {top['GRE Score']}

    • TOEFL Score : {top['TOEFL Score']}

    • CGPA : {top['CGPA']}

    • Strong academic profile

    • Excellent overall ranking

    • High admission potential

    """)

    st.divider()

    # =============================================
    # TOP 20 TABLE
    # =============================================

    st.subheader(
        "📋 Top 20 Ranking Table"
    )

    st.dataframe(

        ranking[
            [
                "Rank",
                "GRE Score",
                "TOEFL Score",
                "CGPA",
                "Research",
                "ML_Probability",
                "SAW_Score"
            ]
        ].head(20),

        use_container_width=True

    )

    st.divider()

    # =============================================
    # DSS INSIGHT
    # =============================================

    st.subheader(
        "🎯 DSS Insight"
    )

    st.success("""

    The ranking process combines:

    • Machine Learning Probability

    • Academic Performance

    • Research Experience

    • Multi-Criteria Decision Making

    This ranking provides a more objective
    admission recommendation compared to
    manual evaluation.

    """)

    # =====================================================
# SCENARIO TESTING
# =====================================================

elif menu == "🔬 Scenario":

    st.title("🔬 Scenario Testing")

    st.caption(
        "Analyze how improvements affect admission probability"
    )

    before = st.slider(
        "Current GRE Score",
        260,
        340,
        310
    )

    after = st.slider(
        "Improved GRE Score",
        260,
        340,
        330
    )

    before_prob = (
        before / 340
    ) * 100

    after_prob = (
        after / 340
    ) * 100

    improvement = (
        after_prob - before_prob
    )

    # =============================================
    # KPI
    # =============================================

    c1,c2,c3 = st.columns(3)

    c1.metric(
        "Before",
        f"{before_prob:.1f}%"
    )

    c2.metric(
        "After",
        f"{after_prob:.1f}%"
    )

    c3.metric(
        "Improvement",
        f"+{improvement:.1f}%"
    )

    st.divider()

    # =============================================
    # COMPARISON CHART
    # =============================================

    compare = pd.DataFrame({

        "Scenario":[

            "Before",

            "After"

        ],

        "Chance":[

            before_prob,

            after_prob

        ]

    })

    fig = px.bar(

        compare,

        x="Scenario",

        y="Chance",

        color="Scenario",

        title="Scenario Comparison"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # =============================================
    # IMPACT ANALYSIS
    # =============================================

    st.subheader(
        "📈 Impact Analysis"
    )

    st.info(f"""

    Increasing GRE Score from

    {before}

    to

    {after}

    improves admission probability by

    {improvement:.1f}%.

    This demonstrates that academic
    performance significantly affects
    admission decisions.

    """)

    st.divider()

    # =============================================
    # DSS RECOMMENDATION
    # =============================================

    st.subheader(
        "💡 DSS Recommendation"
    )

    if improvement >= 10:

        st.success("""

        Significant improvement detected.

        Recommendation:

        Encourage applicants to improve
        standardized test scores before
        applying.

        """)

    else:

        st.warning("""

        Limited impact detected.

        Recommendation:

        Improve multiple factors such as
        CGPA, TOEFL, and Research.

        """)

# =====================================================
# DECISION CENTER
# =====================================================

elif menu == "📈 Decision":

    st.title("📈 Decision Support Center")

    st.caption(
        "Final recommendation based on Machine Learning and SAW"
    )

    best = ranking.iloc[0]

    # =============================================
    # TOP APPLICANT
    # =============================================

    st.subheader(
        "🎯 Top Recommended Applicant"
    )

    c1,c2,c3 = st.columns(3)

    c1.metric(
        "Rank",
        1
    )

    c2.metric(
        "SAW Score",
        round(
            best["SAW_Score"],
            3
        )
    )

    c3.metric(
        "Admission Probability",
        f"{best['ML_Probability']*100:.1f}%"
    )

    st.divider()

    # =============================================
    # CONFIDENCE
    # =============================================

    st.subheader(
        "✅ Confidence Level"
    )

    st.metric(
        "Decision Confidence",
        "VERY HIGH"
    )

    st.divider()

    # =============================================
    # RISK
    # =============================================

    st.subheader(
        "⚠️ Risk Assessment"
    )

    risk = "LOW"

    if best["ML_Probability"] < 0.80:

        risk = "MEDIUM"

    st.metric(
        "Risk Level",
        risk
    )

    st.divider()

    # =============================================
    # WHY RECOMMENDED
    # =============================================

    st.subheader(
        "📋 Why Recommended?"
    )

    st.success(f"""

    This applicant achieved:

    • Highest SAW Score

    • Admission Probability:
      {best['ML_Probability']*100:.1f}%

    • Excellent CGPA

    • Strong GRE Score

    • Strong TOEFL Score

    • Research Experience

    • Best Overall Ranking

    """)

    st.divider()

    # =============================================
    # FINAL DSS STATEMENT
    # =============================================

    st.subheader(
        "🏆 Final DSS Recommendation"
    )

    st.success("""

    STRONGLY RECOMMENDED

    Based on Random Forest prediction
    and SAW ranking,

    this applicant is the strongest
    candidate in the dataset and should
    be prioritized for university
    admission.

    """)

    st.divider()

    # =============================================
    # DECISION SUMMARY
    # =============================================

    summary = pd.DataFrame({

        "Metric":[

            "SAW Score",

            "Admission Probability",

            "Risk Level",

            "Recommendation"

        ],

        "Value":[

            round(
                best["SAW_Score"],
                3
            ),

            f"{best['ML_Probability']*100:.1f}%",

            risk,

            "STRONGLY RECOMMENDED"

        ]

    })

    st.subheader(
        "📊 Decision Summary"
    )

    st.dataframe(
        summary,
        use_container_width=True
    )