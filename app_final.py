import re
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from features import extract_features


# ============================================================
# IPSA - Intelligent Password Security Analyzer
# Final prototype version
# ============================================================

st.set_page_config(
    page_title="IPSA - Intelligent Password Security Analyzer",
    page_icon="🔐",
    layout="wide"
)

MODEL_PATH = r"model_v3_fast\ipsa_v3_fast_combined.joblib"


# ============================================================
# Model loading
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


bundle = load_model()

tfidf = bundle["tfidf"]
model = bundle["model"]
scaler = bundle["scaler"]
feature_columns = bundle["feature_columns"]


# ============================================================
# Helper functions
# ============================================================

def normalize_prediction(prediction):
    """
    Convert numeric/string model output into a human-readable
    empirical exposure tier without changing the trained model.
    """
    if isinstance(prediction, (int, np.integer)):
        mapping = {
            0: "High Exposure",
            1: "Medium Exposure",
            2: "Lower Exposure"
        }
        return mapping.get(int(prediction), str(prediction))

    text = str(prediction).strip().lower()

    aliases = {
        "0": "High Exposure",
        "1": "Medium Exposure",
        "2": "Lower Exposure",
        "high": "High Exposure",
        "high exposure": "High Exposure",
        "medium": "Medium Exposure",
        "medium exposure": "Medium Exposure",
        "lower": "Lower Exposure",
        "low": "Lower Exposure",
        "lower exposure": "Lower Exposure"
    }

    return aliases.get(text, str(prediction))


def character_category(ch):
    if ch.isupper():
        return "U"
    if ch.islower():
        return "L"
    if ch.isdigit():
        return "D"
    return "S"


def has_repeated_structural_block(password):
    """
    Detects repeated character-category blocks such as:
    LLLDDD LLLDDD -> repeated structural block.

    This is intentionally a diagnostic rule, not a claim that
    every mixed-character structure is insecure.
    """
    if len(password) < 4:
        return False

    signature = "".join(character_category(c) for c in password)

    for block_len in range(2, len(signature) // 2 + 1):
        if len(signature) % block_len != 0:
            continue

        repeats = len(signature) // block_len
        if repeats >= 2:
            block = signature[:block_len]
            if signature == block * repeats:
                return True

    return False


# ============================================================
# Eight deterministic IPSA assessment rules
# ============================================================

def evaluate_rules(features, password):
    structural_pattern = has_repeated_structural_block(password)

    rules = [
        {
            "Rule ID": "R1",
            "Security Rule": "Minimum Length",
            "Condition": "Length >= 12",
            "Pass": features["length"] >= 12,
            "Violation": "Password length is below 12 characters.",
            "Recommendation": "Use a longer password or passphrase."
        },
        {
            "Rule ID": "R2",
            "Security Rule": "Character Diversity",
            "Condition": "Diversity >= 0.60",
            "Pass": features["character_diversity"] >= 0.60,
            "Violation": "Character diversity is below the IPSA threshold of 0.60.",
            "Recommendation": "Use a broader and less predictable character composition."
        },
        {
            "Rule ID": "R3",
            "Security Rule": "Repetition Ratio",
            "Condition": "Repetition <= 0.20",
            "Pass": features["repetition_ratio"] <= 0.20,
            "Violation": "Excessive character repetition was detected.",
            "Recommendation": "Reduce repeated characters or repeated constructions."
        },
        {
            "Rule ID": "R4",
            "Security Rule": "Repeated Pattern",
            "Condition": "No repeated pattern",
            "Pass": features["repeated_pattern"] == 0,
            "Violation": "A repeated character pattern was detected.",
            "Recommendation": "Avoid repeated character sequences or blocks."
        },
        {
            "Rule ID": "R5",
            "Security Rule": "Sequential Pattern",
            "Condition": "No sequential pattern",
            "Pass": features["sequential_pattern"] == 0,
            "Violation": "A predictable sequential pattern was detected.",
            "Recommendation": "Avoid predictable sequences such as 123, 456, abc, or xyz."
        },
        {
            "Rule ID": "R6",
            "Security Rule": "Keyboard Pattern",
            "Condition": "No keyboard pattern",
            "Pass": features["keyboard_pattern"] == 0,
            "Violation": "A common keyboard pattern was detected.",
            "Recommendation": "Avoid recognizable keyboard sequences such as qwerty or asdf."
        },
        {
            "Rule ID": "R7",
            "Security Rule": "Dictionary Pattern",
            "Condition": "No dictionary pattern",
            "Pass": features["dictionary_pattern"] == 0,
            "Violation": "A common dictionary-word component was detected.",
            "Recommendation": "Avoid common words and predictable word-based constructions."
        },
        {
            "Rule ID": "R8",
            "Security Rule": "Structural Pattern",
            "Condition": "No repeated structural block",
            "Pass": not structural_pattern,
            "Violation": "A repeated character-category structure was detected.",
            "Recommendation": "Avoid repeating the same letter/digit/symbol block structure."
        }
    ]

    return rules


# ============================================================
# Actionable recommendation engine
# ============================================================

def generate_recommendations(features):
    recommendations = []

    if features["length"] < 12:
        recommendations.append(
            "Increase the password length to at least 12 characters."
        )

    if features["uppercase"] == 0:
        recommendations.append(
            "Add uppercase characters (A-Z) to increase character-category diversity."
        )

    if features["lowercase"] == 0:
        recommendations.append(
            "Include lowercase characters (a-z) where appropriate."
        )

    if features["digits"] == 0:
        recommendations.append(
            "Consider including numeric characters (0-9)."
        )

    if features["special"] == 0:
        recommendations.append(
            "Consider including special characters such as @, #, $, %, or !."
        )

    if features["character_diversity"] < 0.60:
        recommendations.append(
            "Increase character-category diversity instead of relying on one character type."
        )

    if features["repetition_ratio"] > 0.20:
        recommendations.append(
            "Reduce repeated characters and repeated constructions."
        )

    if features["repeated_pattern"] == 1:
        recommendations.append(
            "Avoid repeated character patterns or repeated blocks."
        )

    if features["sequential_pattern"] == 1:
        recommendations.append(
            "Avoid predictable sequences such as 123, 456, abc, or xyz."
        )

    if features["keyboard_pattern"] == 1:
        recommendations.append(
            "Avoid common keyboard patterns such as qwerty, asdf, or adjacent-key sequences."
        )

    if features["dictionary_pattern"] == 1:
        recommendations.append(
            "Avoid common dictionary words or predictable word-based constructions."
        )

    if not recommendations:
        recommendations.append(
            "No major weakness was detected by the implemented IPSA indicators. "
            "Keep the password unique and avoid reusing it across accounts."
        )
    else:
        recommendations.append(
            "Illustrative format only: prefer a long, unique construction using "
            "unrelated words/characters rather than simply modifying a familiar password."
        )

    return recommendations


def rule_compliance(rules):
    passed = sum(1 for r in rules if r["Pass"])
    total = len(rules)
    percentage = (passed / total) * 100
    return passed, total - passed, percentage


def complementarity_message(compliance, exposure_tier):
    """
    Deliberately avoids claiming a statistically validated
    'rule-ML consistency score'. The two layers are independent
    evidence sources.
    """
    if compliance >= 75 and exposure_tier == "High Exposure":
        return (
            "The deterministic indicators are mostly satisfied, but the ML model "
            "still estimates High Exposure. This is a useful divergence: visible "
            "complexity does not necessarily imply low empirical exposure."
        )

    if compliance < 50 and exposure_tier == "Lower Exposure":
        return (
            "The deterministic indicators show several weaknesses while the ML "
            "model estimates Lower Exposure. The layers provide different evidence "
            "and should not be treated as interchangeable scores."
        )

    return (
        "The deterministic rule layer and ML layer provide complementary evidence. "
        "Rule compliance is a transparent heuristic assessment, while the ML output "
        "is an empirical exposure-tier prediction from the trained model."
    )


# ============================================================
# Header
# ============================================================

st.title("🔐 Intelligent Password Security Analyzer")
st.subheader("IPSA — Password Guessability and Security Assessment")

st.write(
    "Enter a password to analyze measurable security characteristics, "
    "estimate its empirical exposure tier, and receive actionable recommendations."
)

st.warning(
    "For demonstration purposes, use a dummy password. Do not enter a real "
    "personal or account password."
)

st.caption(
    "Note: The eight IPSA rules are project-specific diagnostic heuristics. "
    "They are not presented as universal password-policy standards."
)


# ============================================================
# Input
# ============================================================

password = st.text_input(
    "Enter Password",
    type="password",
    placeholder="Enter a password for analysis..."
)

analyze = st.button("🔍 Analyze Password", type="primary")


# ============================================================
# Analysis
# ============================================================

if analyze:

    if not password:
        st.error("Please enter a password first.")
        st.stop()

    # -------------------------------
    # Feature extraction
    # -------------------------------

    features = extract_features(password)

    feature_vector = np.array(
        [[features[col] for col in feature_columns]],
        dtype=float
    )

    feature_scaled = scaler.transform(feature_vector)
    tfidf_vector = tfidf.transform([password])

    combined_vector = hstack(
        [feature_scaled, tfidf_vector]
    )

    # -------------------------------
    # ML prediction
    # -------------------------------

    raw_prediction = model.predict(combined_vector)[0]
    prediction = normalize_prediction(raw_prediction)

    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(combined_vector)[0]

    # -------------------------------
    # Rules
    # -------------------------------

    rules = evaluate_rules(features, password)
    passed, failed, compliance = rule_compliance(rules)

    # ========================================================
    # Analysis Result
    # ========================================================

    st.divider()
    st.header("📊 Analysis Result")

    st.metric(
        label="Predicted Exposure Tier",
        value=prediction
    )

    if prediction == "High Exposure":
        st.error("⚠️ High Exposure")
    elif prediction == "Medium Exposure":
        st.warning("⚠️ Medium Exposure")
    elif prediction == "Lower Exposure":
        st.success("✅ Lower Exposure")

    # ========================================================
    # Password Characteristics
    # ========================================================

    st.subheader("🔎 Password Characteristics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Length", features["length"])
        st.metric("Uppercase", features["uppercase"])
        st.metric("Lowercase", features["lowercase"])
        st.metric("Digits", features["digits"])

    with col2:
        st.metric("Special Characters", features["special"])
        st.metric("Unique Characters", features["unique_chars"])
        st.metric(
            "Character Diversity",
            f'{features["character_diversity"]:.2f}'
        )
        st.metric(
            "Entropy",
            f'{features["entropy"]:.2f}'
        )

    with col3:
        st.metric(
            "Repetition Ratio",
            f'{features["repetition_ratio"]:.2f}'
        )
        st.metric("Repeated Pattern", features["repeated_pattern"])
        st.metric("Sequential Pattern", features["sequential_pattern"])
        st.metric("Keyboard Pattern", features["keyboard_pattern"])

    # ========================================================
    # Security Indicators
    # ========================================================

    st.subheader("🛡️ Security Indicators")

    indicators = {
        "Dictionary Pattern": features["dictionary_pattern"],
        "Structural Transitions": features["structural_transitions"],
        "Repeated Structural Block": int(
            has_repeated_structural_block(password)
        )
    }

    indicator_df = pd.DataFrame(
        list(indicators.items()),
        columns=["Indicator", "Value"]
    )

    st.table(indicator_df)

    # ========================================================
    # Rule-Based Security Assessment
    # ========================================================

    st.divider()
    st.header("📋 Rule-Based Security Assessment")

    rule_table = pd.DataFrame([
        {
            "Rule ID": r["Rule ID"],
            "Security Rule": r["Security Rule"],
            "Condition": r["Condition"],
            "Status": "PASS" if r["Pass"] else "FAIL"
        }
        for r in rules
    ])

    st.dataframe(
        rule_table,
        use_container_width=True,
        hide_index=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Rules Passed", f"{passed}/8")

    with c2:
        st.metric("Rules Failed", failed)

    with c3:
        st.metric("Rule Compliance", f"{compliance:.2f}%")

    if failed == 0:
        st.success("All predefined IPSA diagnostic rules were satisfied.")
    else:
        st.warning(f"{failed} IPSA diagnostic rule(s) were not satisfied.")

        st.subheader("⚠️ Detected Rule Violations")

        for r in rules:
            if not r["Pass"]:
                st.warning(
                    f'{r["Rule ID"]} — {r["Violation"]}'
                )

    # ========================================================
    # Rule–ML Complementarity
    # ========================================================

    st.divider()
    st.header("🔬 Rule–ML Complementarity Analysis")

    st.info(
        complementarity_message(compliance, prediction)
    )

    st.write(f"**Rule Compliance:** {compliance:.2f}%")
    st.write(f"**ML Exposure Prediction:** {prediction}")

    # ========================================================
    # Model Probabilities
    # ========================================================

    if probabilities is not None:

        st.subheader("🤖 Model Prediction Probabilities")

        classes = list(model.classes_)

        display_classes = [
            normalize_prediction(c)
            for c in classes
        ]

        probability_df = pd.DataFrame({
            "Exposure Tier": display_classes,
            "Probability (%)": (
                np.asarray(probabilities) * 100
            ).round(2)
        })

        st.table(probability_df)

        chart_df = probability_df.set_index("Exposure Tier")
        st.bar_chart(chart_df)

    # ========================================================
    # Recommendations
    # ========================================================

    st.subheader("💡 Security Recommendations")

    recommendations = generate_recommendations(features)

    for recommendation in recommendations:
        st.info("• " + recommendation)

    # ========================================================
    # Technical Analysis
    # ========================================================

    with st.expander("Technical Analysis"):

        st.write(
            "The prediction uses the trained combined IPSA + "
            "character-level TF-IDF representation."
        )

        st.write(f"Model: {type(model).__name__}")
        st.write(f"TF-IDF analyzer: {tfidf.analyzer}")
        st.write(f"TF-IDF n-gram range: {tfidf.ngram_range}")
        st.write(f"Engineered features used: {len(feature_columns)}")
        st.write("Deterministic IPSA assessment rules: 8")

        st.caption(
            "The ML prediction and deterministic rule compliance are "
            "reported as separate evidence layers; rule compliance is "
            "not treated as ML accuracy."
        )
