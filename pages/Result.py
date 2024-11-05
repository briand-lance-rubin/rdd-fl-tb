import streamlit as st

# Set up the page configuration
st.set_page_config(
    page_title="Training Evaluation",
    page_icon="🚧",
)

# Display the banner image
st.image("./resource/road_damage_detection_banner.png", use_column_width="always")

st.divider()
st.title("Training Evaluation")

# Define a list of images with their captions and brief descriptions
images = [
    {
        "path": "./resource/confusion_matrix.png",
        "caption": "Confusion Matrix",
        "description": "This matrix shows the performance of the model in classifying road damage types."
    },
    {
        "path": "./resource/confusion_matrix_normalized.png",
        "caption": "Normalized Confusion Matrix",
        "description": "This version normalizes the confusion matrix to provide a clearer view of the classification performance."
    },
    {
        "path": "./resource/F1_curve.png",
        "caption": "F1 Score Curve",
        "description": "The F1 score curve illustrates the trade-off between precision and recall for the model."
    },
    {
        "path": "./resource/P_curve.png",
        "caption": "Precision Curve",
        "description": "This curve depicts the precision of the model at various thresholds."
    },
    {
        "path": "./resource/R_curve.png",
        "caption": "Recall Curve",
        "description": "The recall curve shows how well the model identifies true positive cases across different thresholds."
    },
]

# Display images in two columns with specified width
for i in range(0, len(images), 2):
    cols = st.columns(2)
    
    for j in range(2):
        if i + j < len(images):
            with cols[j]:
                st.image(images[i + j]["path"], width=500)  # Adjust the width as needed
                st.caption(images[i + j]["caption"])
                st.write(images[i + j]["description"])  # Add a brief description

st.divider()

st.markdown(
    """
    Research Methodology in Computer Science
    """
)
