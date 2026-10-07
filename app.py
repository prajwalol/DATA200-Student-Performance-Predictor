import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

MODEL_PATH = Path("student_score_pipeline.pkl")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


model = load_model()


if model is None:
    st.error(
        "Model file 'student_score_pipeline.pkl' was not found. "
        "Place it in the same folder as app.py."
    )
    st.stop()


# --------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------

st.title("🎓 Student Performance Predictor")

st.write(
    "Predict a student's exam score using academic, family, "
    "and learning-related factors."
)

st.caption(
    "This prediction is an estimate based on the "
    "DATA 200 Multiple Linear Regression model."
)

st.divider()


# --------------------------------------------------
# ACADEMIC INFORMATION
# --------------------------------------------------

st.subheader("📚 Academic Information")


hours_studied = st.slider(
    "Hours Studied per Week",
    min_value=1,
    max_value=44,
    value=20,
    step=1
)


attendance = st.slider(
    "Attendance (%)",
    min_value=60,
    max_value=100,
    value=80,
    step=1
)


previous_scores = st.slider(
    "Previous Score",
    min_value=50,
    max_value=100,
    value=75,
    step=1
)


tutoring_sessions = st.slider(
    "Tutoring Sessions",
    min_value=0,
    max_value=8,
    value=1,
    step=1
)


# --------------------------------------------------
# LEARNING ENVIRONMENT
# --------------------------------------------------

st.subheader("🏫 Learning Environment")


access_to_resources = st.selectbox(
    "Access to Resources",
    options=[
        "Low",
        "Medium",
        "High"
    ],
    index=1
)


teacher_quality = st.selectbox(
    "Teacher Quality",
    options=[
        "Low",
        "Medium",
        "High"
    ],
    index=1
)


distance_from_home = st.selectbox(
    "Distance from Home",
    options=[
        "Near",
        "Moderate",
        "Far"
    ],
    index=0
)


# --------------------------------------------------
# PERSONAL AND FAMILY FACTORS
# --------------------------------------------------

st.subheader("👨‍👩‍👧 Personal and Family Factors")


motivation_level = st.selectbox(
    "Motivation Level",
    options=[
        "Low",
        "Medium",
        "High"
    ],
    index=1
)


parental_involvement = st.selectbox(
    "Parental Involvement",
    options=[
        "Low",
        "Medium",
        "High"
    ],
    index=1
)


parental_education = st.selectbox(
    "Parental Education Level",
    options=[
        "High School",
        "College",
        "Postgraduate"
    ],
    index=1
)


family_income = st.selectbox(
    "Family Income",
    options=[
        "Low",
        "Medium",
        "High"
    ],
    index=1
)


peer_influence = st.selectbox(
    "Peer Influence",
    options=[
        "Negative",
        "Neutral",
        "Positive"
    ],
    index=1
)


st.divider()


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button(
    "Predict Exam Score",
    type="primary",
    width="stretch"
):

    # Create DataFrame using exactly the same
    # feature names used during model training
    student = pd.DataFrame(
        [
            {
                "Hours_Studied": hours_studied,
                "Attendance": attendance,
                "Previous_Scores": previous_scores,
                "Tutoring_Sessions": tutoring_sessions,
                "Access_to_Resources": access_to_resources,
                "Parental_Involvement": parental_involvement,
                "Parental_Education_Level": parental_education,
                "Peer_Influence": peer_influence,
                "Motivation_Level": motivation_level,
                "Family_Income": family_income,
                "Teacher_Quality": teacher_quality,
                "Distance_from_Home": distance_from_home
            }
        ]
    )

    try:

        # Generate prediction
        prediction = float(
            model.predict(student)[0]
        )

        # Keep displayed score within valid range
        displayed_prediction = max(
            0.0,
            min(
                100.0,
                prediction
            )
        )

        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------

        st.subheader("📊 Prediction Result")

        st.metric(
            label="Predicted Exam Score",
            value=f"{displayed_prediction:.2f} / 100"
        )

        # Performance classification
        if displayed_prediction >= 80:
            performance_level = "Excellent"

        elif displayed_prediction >= 70:
            performance_level = "Good"

        elif displayed_prediction >= 60:
            performance_level = "Satisfactory"

        else:
            performance_level = "Needs Improvement"

        st.info(
            f"Performance Level: **{performance_level}**"
        )

        # --------------------------------------------------
        # DISPLAY INPUT DATA
        # --------------------------------------------------

        with st.expander(
            "View Entered Student Data"
        ):

            display_data = pd.DataFrame(
                {
                    "Factor": [
                        "Hours Studied per Week",
                        "Attendance",
                        "Previous Score",
                        "Tutoring Sessions",
                        "Access to Resources",
                        "Parental Involvement",
                        "Parental Education Level",
                        "Peer Influence",
                        "Motivation Level",
                        "Family Income",
                        "Teacher Quality",
                        "Distance from Home"
                    ],

                    "Value": [
                        str(hours_studied),
                        f"{attendance}%",
                        str(previous_scores),
                        str(tutoring_sessions),
                        str(access_to_resources),
                        str(parental_involvement),
                        str(parental_education),
                        str(peer_influence),
                        str(motivation_level),
                        str(family_income),
                        str(teacher_quality),
                        str(distance_from_home)
                    ]
                }
            )

            # All values are strings,
            # preventing PyArrow datatype errors
            display_data["Factor"] = (
                display_data["Factor"]
                .astype(str)
            )

            display_data["Value"] = (
                display_data["Value"]
                .astype(str)
            )

            st.dataframe(
                display_data,
                width="stretch",
                hide_index=True
            )

    except Exception as error:

        st.error(
            "Prediction could not be generated."
        )

        st.exception(error)


# --------------------------------------------------
# ABOUT MODEL
# --------------------------------------------------

st.divider()


with st.expander("ℹ️ About the Model"):

    st.write(
        """
        This application was developed for the DATA 200
        Applied Statistical Analysis project.

        The application uses a Multiple Linear Regression
        model to estimate a student's exam score based on
        academic, family, and learning-related factors.

        The prediction pipeline automatically handles
        numerical and categorical variables before
        generating the estimated exam score.
        """
    )
