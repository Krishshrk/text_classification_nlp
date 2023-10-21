import streamlit as st
import nltk 
import streamlit.components.v1 as components
import string
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
all_stopwords=stopwords.words('english')
all_stopwords.remove('not') 
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle

messages = pd.read_csv('Restaurant_Reviews.tsv', sep='\t')
food_model=pickle.load(open("food.pkl","rb"))

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    ps=PorterStemmer()
    # Now just remove any stopwords
    return [ps.stem(word) for word in nopunc.split() if word.lower() not in set(all_stopwords)]
    
#preprocessing section of the text

bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['Review'])
messages_bow = bow_transformer.transform(messages['Review'])
tfidf_transformer = TfidfTransformer().fit(messages_bow)
messages_tfidf = tfidf_transformer.transform(messages_bow)

#streamlit section


st.set_page_config(
    page_title="Restaurent review analysis",
    page_icon="data:image/jpeg;base64,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",
    layout="centered",
    initial_sidebar_state="expanded")

page_bg_img='''
    <style>
    .stApp{
    background-size:cover;
    background-image:url('https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhJoDsppA6x2JI7WQnTgfnYaoTFVW4W48Sg-VDTMjM_l5knjx7o0_pURlt0SDAfHU0mk_jZQPCZojgLiNFQYq9ak_2I2WsNwjrQ_c8OBc-w8f4ATsuq0nK3FmY8NfHZrhwPPatB9qM2Ugl4UjFQI4goBPqTTjJECavIYOgNCLDVlWkB4CBvcBeeCoHabQ/s2560/wallpaper-for-setup-gamer-2560x1440.jpg')
    }
    </style>
    '''
st.markdown(page_bg_img, unsafe_allow_html=True)

components.html("""<h1><p class='font-family:Arial' style='color:white; margin-bottom:-40px'><u>Food Review in the restaurent (Liked/Not Liked)</u></p></h1>""")

df_section,pred_section=st.columns(spec=[1,1],gap="large")



with df_section:
    st.subheader("Sample restaurent reviews")
    st.write(messages["Review"].head())
    st.subheader("Review for the restaurent food")
    text=st.text_area("Write the review")
    if text:
        text=tfidf_transformer.transform(bow_transformer.transform([text]))
        button=st.button("Predict",key="df_section")
        if button:
            if food_model.predict(text)[0]==0:
                st.error("Food is not Liked")
            else:
                st.success("Food is Liked")
    

with pred_section:
    st.subheader("Upload the text File")
    uploaded_text=st.file_uploader(label="Text File",type=["txt"])
    
    if uploaded_text is not None:
        text_written=uploaded_text.getvalue().decode("utf-8")
        text_written=tfidf_transformer.transform(bow_transformer.transform([text_written]))
        button=st.button("Predict",key="pred_section")
        if button:
            if food_model.predict(text_written)[0]==0:
                st.error("Food is not Liked")
            else:
                st.success("Food is Liked")
        
        
        
        
    