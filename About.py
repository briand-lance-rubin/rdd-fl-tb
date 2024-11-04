import streamlit as st

# Set up the page configuration
st.set_page_config(
    page_title="Evaluation Result",
    page_icon="🚧",
)

# Display the banner image
st.image("./resource/road_damage_detection_banner.png", use_column_width="always", width=800)

st.divider()

# Project overview with enhanced formatting
st.markdown(
    """
    <style>
        .project-overview {
            background-color: #f0f8ff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }
        .key-component, .dataset, .damage-type {
            color: black;  /* Set all text to black */
        }
        .bold {
            font-weight: bold;  /* Make text bold */
        }
    </style>
    <div class="project-overview">
        <h2>Project Overview</h2>
        <p><span class="bold">This project focuses on two key components:</span></p>
        <ul>
            <li class="key-component"><strong>Road Damage Detection with YOLO</strong> – Detects road damage efficiently.</li>
            <li class="key-component"><strong>Federated Learning Applications</strong> – Trains models without central data.</li>
        </ul>
        <p><span class="bold">We utilized two datasets for this project:</span></p>
        <ul>
            <li class="dataset"><strong>RDD 2022 Dataset</strong> – Comprehensive training image set.</li>
            <li class="dataset"><strong>Thunder Bay Dataset</strong> – Localized data for accuracy improvement.</li>
        </ul>
        <p><span class="bold">The project aims to classify four types of road damage:</span></p>
        <ul>
            <li class="damage-type"><strong>Longitudinal Cracks</strong> – Cracks parallel to the road.</li>
            <li class="damage-type"><strong>Transverse Cracks</strong> – Cracks crossing the roadway.</li>
            <li class="damage-type"><strong>Alligator Cracks</strong> – Interconnected surface cracks.</li>
            <li class="damage-type"><strong>Potholes</strong> – Depressions caused by erosion.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# Footer with university information
st.markdown(
    """
    <style>
        .footer {
            font-size: 14px;
            color: black;  /* Set footer text color to black */
        }
    </style>
    <div class="footer">
        Lakehead University - Computer Science Department
    </div>
    """,
    unsafe_allow_html=True
)