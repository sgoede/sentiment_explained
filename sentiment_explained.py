import torch
from flair.data import Sentence
from flair.models import TextClassifier
from flair_model_wrapper import ModelWrapper
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz
from interpret_flair import interpret_sentence, visualize_attributions
import numpy as np
import streamlit as st
import pandas as pd
st.set_page_config(layout="wide")
st.title("Sentiment prediction using Transformer models explained")
st.subheader("Created by: Stephan de Goede")

st.caption("Introduction")
st.markdown("Before I go any further, I want to make the following **statement: every Data Scientist should assume bias.**")
st.markdown("Transformer models are frequently hyped as 'the next big wave in AI' and have gained massive popularity and successful adoptions \
across industries. While these models break down many of the existing barriers of successful Data Science projects, such as: data availability and computing power. \
Another big advantage of Transformers is that most of them are very easy-to-use, especially thanks to the HuggingFace team. However, as a Data Scientist \
    one should do it's utmost best to assess bias in their predictions, as with many other things:**'With great power comes great responsibility' - parker.**")
st.caption("About the app")
st.markdown("This app will do 2 things. First, it will take your input and predict the overall sentiment of your text, either positive or negative. Second: it will\
    let you drilldown on token level to inspect token by token what is predicted positive or negative. This is visualized using a LIME like coloring \
        of the associated token.")
st.caption("Theoretical Framework")
st.markdown('''In this example the Flair NLP package is used. From the package, the build-in sentiment Transformer is loaded and used to predict the sentiment of the text, being negative or positive. Documentation can be found here:

https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_2_TAGGING.md 

- Important note

Before using any build-in Transformer model. It is useful to carefully read the model card on the Hugging Face page. If the model card is empty, or very sparse, make sure to understand the intended use of the Transformer, before using or deploying it. 

For the Flair sentiment model that is used in this example the following link brings you to the model card. 

https://huggingface.co/distilbert-base-uncased

**Transformer explainability using Captum:**


We need to define simple input and baseline tensors. Baselines belong to the input space and often carry no predictive signal. Zero tensor can serve as a baseline for many tasks. Some interpretability algorithms such as Integrated Gradients, Deeplift and GradientShap are designed to attribute the change between the input and baseline to a predictive class or a value that the neural network outputs.

We will apply model interpretability algorithms on the network mentioned above in order to understand the importance of individual neurons/layers and the parts of the input that play an important role in the final prediction.

Let's define our input and baseline tensors. Baselines are used in some interpretability algorithms such as IntegratedGradients. 

Next we will use IntegratedGradients algorithms to assign attribution scores to each input feature with respect to the first target output.

The algorithm outputs an attribution score for each input element and a convergence delta. The lower the absolute value of the convergence delta the better is the approximation. If we choose not to return delta, we can simply not provide the return_convergence_delta input argument. The absolute value of the returned deltas can be interpreted as an approximation error for each input sample. It can also serve as a proxy of how accurate the integral approximation for given inputs and baselines is. If the approximation error is large, we can try a larger number of integral approximation steps by setting n_steps to a larger value. Not all algorithms return approximation error. Those which do, though, compute it based on the completeness property of the algorithms.
Positive attribution score means that the input in that particular position positively contributed to the final prediction and negative means the opposite. The magnitude of the attribution score signifies the strength of the contribution. Zero attribution score means no contribution from that particular feature.

 attributions: Total conductance with respect to each neuron in output of given layer

source: (https://github.com/pytorch/captum/blob/master/README.md)

In order to make Flair work with Captum I need to give credit where credit's due. Check out Robin van Schaik's repo to rework Flair Forward Function, essentially making it compatible

Interpret-FLAIR: https://github.com/robinvanschaik/interpret-flair#authors

Finally to get a better understanding of the Gauss-Legendre estimation method, see the interactive demo on the following page:

https://keisan.casio.com/exec/system/1280883022
''')

st.caption("Interactive sentiment prediction using Transformer model")

st.markdown("Type in some text to have the sentiment predicted.")

@st.experimental_singleton
def get_classifier():
    # load tagger
    classifier = TextClassifier.load('sentiment')
    return classifier
# Load the pretrained Flair classifier.
model_load_state = st.text("Loading Sentiment Model...")
flair_model = get_classifier()
model_load_state.text("Loading Sentiment Model... done")
# sentence object
user_input = st.text_area("The English text you want to have the sentiment from", placeholder= "man, woman, child, boy, girl, European, American" )
# Initialize state.
if "clicked" not in st.session_state:
    st.session_state.clicked = False
    st.session_state.word_scores = None
    st.session_state.ordered_lig = None
# Define callbacks to handle button clicks.
def handle_click():
    if  st.session_state.clicked == True:
        st.session_state.word_scores = word_attributions.detach().numpy()
        st.session_state.ordered_lig = [(readable_tokens[i], word_scores[i]) for i in np.argsort(word_scores)][::-1]
def handle_second_click():
    if st.session_state.clicked == True:
        word_scores = st.session_state.word_scores
        ordered_lig = st.session_state.ordered_lig

if len(user_input) >0 :
    #prediction to select the target label:
    tokenized_user_input = Sentence(user_input)
    predicted_sentiment = flair_model.predict(tokenized_user_input)
    target = tokenized_user_input.get_label_names()[0]
    st.write("the model has predicted the sentiment as:",tokenized_user_input )
    if st.button("Click here to visually inspect the outcome of the model"):
    # In order to make use of Captum's LayerIntegratedGradients method we had to rework Flair's forward function.
    # This is handled by the wrapper. The wrapper inherits functions of the Flair text-classifier object
     # and allows us to calculate attributions with respect to a target class.
        st.session_state.submitted = True
        @st.experimental_singleton
        def rework_flair_model():
            flair_model_wrapper = ModelWrapper(flair_model)
            return flair_model_wrapper
        rework_load_state = st.text("Reworking Flair's forward function....")
        flair_rework = rework_flair_model()
        rework_load_state.text("Successfully reworked Flair's forward function.")
        # As described in the source code of documentation of Captum:
        # "Layer Integrated Gradients is a variant of Integrated Gradients that assigns an importance score to
        # layer inputs or outputs, depending on whether we attribute to the former or to the latter one."
        # In this case, we are interested how the input embeddings of the model contribute to the output.
        @st.experimental_singleton
        def LayerIntegratedGradients_Flair():
            lig = LayerIntegratedGradients(flair_rework, flair_rework.model.embeddings)
            return lig
        LayerIntegratedGradients_load_state = st.text("Calculating LayerIntegratedGradients....")
        lig = LayerIntegratedGradients_Flair()
        LayerIntegratedGradients_load_state.text("Successfully calculated LayerIntegratedGradients")
        # create an empty list to store our attribitions results in order to visualize them using Captum.
        visualization_list = []
        #Let's run the Layer Integrated Gradient method on the two paragraphs, and determine what
        # drives the prediction. As an additional note, the number of steps & the estimation method can have an
        # impact on the attribution.
        readable_tokens, word_attributions, delta = interpret_sentence(flair_rework,
                                                                lig,
                                                                user_input,
                                                                target,
                                                                visualization_list,
                                                                n_steps=15,
                                                                estimation_method="gausslegendre",
                                                                internal_batch_size=3)
        # Let's visualize the score attribution of the model
        st.write(viz.visualize_text(visualization_list))
        st.write(" ")
        st.session_state.clicked = True
        word_scores = word_attributions.detach().numpy()
        ordered_lig = [(readable_tokens[i], word_scores[i]) for i in np.argsort(word_scores)][::-1]
        st.session_state.word_scores = word_scores
        st.session_state.ordered_lig = ordered_lig
        st.legacy_caching.clear_cache()
    if st.session_state.ordered_lig is not None:
       if st.button("Click here to see the absolute values of the words"):
            st.dataframe(pd.DataFrame(st.session_state.ordered_lig,columns=['readable tokens',' word scores']))
            st.legacy_caching.clear_cache()
            st.caption("o, one more thing")
            st.markdown("**Did you assume bias?** Since this 'explainer' is biased itself.. by using non-converged estimates it is not to be trusted. \
        this can be improved by using a (much) higher threshold in the Captum method, but that comes at the price of higher computation costs. But much more, it would defy the message I need you to remember: **Assume bias** because it's everywhere.")




st.caption('sources')

st.write('''

FLAIR Framework: @inproceedings{akbik2019flair,
  title={FLAIR: An easy-to-use framework for state-of-the-art NLP},
  author={Akbik, Alan and Bergmann, Tanja and Blythe, Duncan and Rasul, Kashif and Schweter, Stefan and Vollgraf, Roland},
  booktitle={{NAACL} 2019, 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations)},
  pages={54--59},
  year={2019}
}


https://github.com/pytorch/captum


https://huggingface.co/distilbert-base-uncased


https://github.com/robinvanschaik/interpret-flair


https://github.com/pytorch/captum/blob/master/README.md


https://keisan.casio.com/exec/system/1280883022


https://streamlit.io/
''')



