import streamlit as st
import pickle
import tensorflow as tf
#st.write("hello1234")
file = open("finalized_model.pkl",'rb')
new_model = pickle.load(file)
def main():
 
        page_title="SkimLit",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"

        st.title('SpeedRead📄🔥')
        st.caption('An NLP model to classify abstract sentences into the role they play (e.g. objective, methods, results, etc..) to enable researchers to skim through the literature and dive deeper when necessary.')
        
        col1, col2 = st.columns(2)

        with col1:
           st.write('#### Entre Abstract Here !!')
           abstract = st.text_area(label='', height=50)
       
           predict = st.button('Extract !')


        if predict:
           from spacy.lang.en import English
           nlp = English()
           sentencizer = nlp.create_pipe("sentencizer")
           nlp.add_pipe('sentencizer')
           doc = nlp(abstract) 
           abstract_lines = [str(sent) for sent in list(doc.sents)] 
           #st.write(abstract_lines)
           # Get total number of lines
           total_lines_in_sample = len(abstract_lines)
           sample_lines = []
           for i, line in enumerate(abstract_lines):
              sample_dict = {}
              sample_dict["text"] = str(line)
              sample_dict["line_number"] = i
              sample_dict["total_lines"] = total_lines_in_sample - 1
              sample_lines.append(sample_dict)
           #st.write(sample_lines)
           test_abstract_line_numbers = [line["line_number"] for line in sample_lines]
           test_abstract_line_numbers_one_hot = tf.one_hot(test_abstract_line_numbers, depth=15) 
           test_abstract_total_lines = [line["total_lines"] for line in sample_lines]
           test_abstract_total_lines_one_hot = tf.one_hot(test_abstract_total_lines, depth=20)
           def split_chars(text):
             return ' '.join(list(text))
           abstract_chars = [split_chars(sentence) for sentence in abstract_lines]
           # Make predictions on sample abstract features
           test_abstract_pred_probs = new_model.predict(x=(test_abstract_line_numbers_one_hot,
                                                              test_abstract_total_lines_one_hot,
                                                              tf.constant(abstract_lines),
                                                              tf.constant(abstract_chars)))
           test_abstract_preds = tf.argmax(test_abstract_pred_probs, axis=1)   
           classes = ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']
           test_abstract_pred_classes = [classes[i] for i in test_abstract_preds]
           for i, line in enumerate(abstract_lines):
               st.write(f"{test_abstract_pred_classes[i]}: {line}")                                               
           

    
 
         
    

if __name__=='__main__': 
    main()