import streamlit as st

# Set up the page configuration
st.set_page_config(
    page_title="Federated Learning",
    page_icon="🚧",  # Text icon related to road work
)

# Add a title for Federated Learning in Thunder Bay
st.title("Federated Learning in Thunder Bay")

# Add a divider
st.divider()

# Display the image related to federated learning
st.image("./resource/federated_learning_thunder_bay.png", width=700)  # Adjust the width as needed
st.caption("This image illustrates how federated learning can be applied to enhance road damage detection using datasets from Thunder Bay's four major areas.")

# Add a divider for separation
st.divider()

# Footer
st.markdown(
    """
    Research Methodology in Computer Science
    """
)
