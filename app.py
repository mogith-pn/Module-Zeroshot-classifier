# installation:
#   - pip install clarifai streamlit

# Run: streamlit run app.py
#

import streamlit as st
#from PIL import Image
from clarifai.client.model import Model
from clarifai.client.model import Inputs


# Clarifai Configuration

query_params = st.experimental_get_query_params()
USER_ID=query_params.get("user_id", [])[0]
app_id=query_params.get("app_id", [])[0]
PAT = query_params.get("pat",[])[0]

#use multi model - Gpt-4o
model_url="https://clarifai.com/openai/chat-completion/models/gpt-4o"

# Define functions
def get_concepts_list(text_input):
    #text_input = st.text_area("**Enter your concept lists (separated by comma)**")
    return text_input.split(',')

def prompt_template(concepts):
    prompt = f""" You are highly skilled at describing things/person/animal/fruit or anything, 
    your task is to take the list of {concepts} provided to you and return a list of descriptions about that particular entity on how it look like. 
    Generate neat and precise output for the task.
    For example:
      Question : What does a ['lorikeet', 'marimba', 'viaduct', 'papillon'] look like?
      Answer : 
        'lorikeet' : 'A lorikeet is a small to medium-sized parrot with a brightly colored plumage.'
        'marimba' : 'A marimba is a large wooden percussion instrument that looks like a xylophone.'
        'viaduct' : 'A viaduct is a bridge composed of several spans supported by piers or pillars.'
        'papillon' : 'A papillon is a small, spaniel-type dog with a long, silky coat and fringed ears'
    IMPORTANT : Return only the answer
    """
    return prompt

@st.cache_data
def create_prompt_desc_for_concepts(concepts, _model_object):
    try:
        prompt = prompt_template(concepts)
        response = _model_object.predict_by_bytes(prompt.encode(), input_type="text")
        return response.outputs[0].data.text.raw
    except Exception as e:
        return str(e)

def upload_image_func():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        st.image(image_bytes, caption="Uploaded Image", use_column_width=True)
        return image_bytes
    return None

def classification_prompt_template(cupl_prompt, image_bytes):
    prompt = f""" You are good at classifying stuffs based on the descriptions provided to you. Here we will give you objects with description of how it looks and an IMAGE.
    Your task is to read the descriptions of classification entities and classify the given image based on one of the classes mentioned in the descriptions.
    Class descriptions : {cupl_prompt}
    Classify it and return only the classification label
    """
    return prompt

def classify_concept(cupl_prompt, image_bytes, model_obj):
    try:
        prompt = classification_prompt_template(cupl_prompt, image_bytes)
        print(f"Classification Prompt {prompt}")

        response = model_obj.predict(inputs = [Inputs.get_multimodal_input(input_id="", image_bytes = image_bytes, raw_text=prompt)], inference_params=inference_params)

        #response = model_obj.predict_by_bytes(prompt.encode(), input_type="text")
        return response.outputs[0].data.text.raw
    except Exception as e:
        return str(e)


# Title of the app
st.title("Clarify Image Using LLM Prompt")


# Main app logic
if 'reset' not in st.session_state:
    st.session_state.reset = False

if st.session_state.reset:
    st.session_state.reset = False
    st.experimental_rerun()

# Model for Prediction
Model_obj = Model(model_url, pat=PAT)
inference_params = dict(temperature=0.7, max_tokens=2048, top_p=1, top_k=40)

concept_text_input = st.text_area("**Enter your concept lists (separated by comma)**")

#Sample concepts
sample_concept = ["golden retriever", "bull dog", "chihuahua", "poodle"]

img = upload_image_func()


# Place buttons on the same line
col1, col2  = st.columns((4,2))


# Button to display image and run clarifai Image classification
with col1:
    if st.button("Classify Image",use_container_width=True):
        if img and concept_text_input:
            try:
                with st.spinner('Generating LLM Prompt... Please Wait'):
                    concept2 = get_concepts_list(concept_text_input)
                    #print(f"***** Concepts: {concept2} *******")
                    Cupl_prompt = create_prompt_desc_for_concepts(concept2, Model_obj)
                    st.write(f"CUPL Prompt is\n{Cupl_prompt}")

                with st.spinner('Classifying... Please Wait'):
                    ans = classify_concept(Cupl_prompt, img, Model_obj)
                    if ans:
                        st.success(f"Predicted Classification: {ans}")
                    else:
                        st.warning("No concepts predicted.")
            except Exception as e:
                st.error(f"Error: {e}")

with col2:
    if st.button("Reset"):
        st.session_state.reset = True
        img=None
        st.experimental_rerun()

