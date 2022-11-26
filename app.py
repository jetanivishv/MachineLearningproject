import streamlit as st
import streamlit.components.v1 as stc
import neattext.functions as nfx
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel

import pandas as pd
import pickle
import io

from class_prediction import vectorize_text
from freepaid import converthours,conversion,sub,lev,year

def load_data(data):
    df=pd.read_csv(data)
    return df

def to_cosine_similarity_matrix(data):
    countVector=CountVectorizer()
    countMatrix=countVector.fit_transform(data)
    similarity_matrix=cosine_similarity(countMatrix)
    return similarity_matrix

def recommender(title,similarity_matrix,df,recommendno=5):
    course_indices=pd.Series(df.index,index=df['course_title']).drop_duplicates()
    idx=course_indices[title]
    scores=list(enumerate(similarity_matrix[idx]))
    scores=sorted(scores,key=lambda x:x[1],reverse=True)
    selected_index = [i[0] for i in scores[1:]]
    selected_course_scores = [i[1] for i in scores[1:]]
    recommended_result=df.iloc[selected_index]
    recommended_result['similarity_score'] = selected_course_scores
    final_recommended_courses=recommended_result[['course_title','similarity_score','url','price','num_subscribers']]
    return final_recommended_courses.head(recommendno)

RESULT_FOUND = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;
background-color:white;
  border-left: 10px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">📈Similarity_Score::</span>{}</p>
<p style="color:blue;"><span style="color:black;">🔗</span><a href="{}",target="_blank">Link</a></p>
<p style="color:blue;"><span style="color:black;">💲Price:</span>{}</p>
<p style="color:blue;"><span style="color:black;">🧑‍🎓👨🏽‍🎓 Students:</span>{}</p>
</div>
"""

RESULT_NOT_FOUND = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;
background-color:white;
  border-left: 10px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">🔗</span><a href="{}",target="_blank">Link</a></p>
<p style="color:blue;"><span style="color:black;">💲Price:</span>{}</p>
<p style="color:blue;"><span style="color:black;">🧑‍🎓👨🏽‍🎓 Students:</span>{}</p>
</div>
"""

def search_term_if_not_found(term,df,recommendno=6):
    df1=df[df['course_title'].str.contains(term)]
    df2=df[df['course_title'].str.contains(term.capitalize())]
    df3=df[df['course_title'].str.contains(term.lower())]
    df4=df[df['course_title'].str.contains(term.upper())]
    result=pd.concat([df1,df2,df3,df4])
    final_result=result[['course_title','url','price','num_subscribers']]
    return final_result.head(recommendno)

def main():
    st.title("Udemy_dataset_Project")
    menu=["Home","Recommendation","Predict_Subject","predict_Free_or_Paid","About"]
    choice=st.sidebar.selectbox("Menu",menu)
    df=load_data("DATASET/udemy_courses.csv")
    df['clean_course_title'] = df['course_title'].apply(nfx.remove_stopwords)
    df['clean_course_title'] = df['course_title'].apply(nfx.remove_special_characters)

    if choice=="Home":
        st.subheader("Dataset")
        st.dataframe(df.head(10))
        st.subheader("Dataset Information")
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
    elif choice=="Recommendation":
        st.subheader("Recommend Course")
        similarity_matrix=to_cosine_similarity_matrix(df['clean_course_title'])
        search_term=st.text_input("Search").strip()
        recommendno=st.sidebar.number_input("How Many Recommendation you want",4,30,7)

        if st.button("Recommend"):
            if search_term is not None:
                try:
                    result=recommender(search_term,similarity_matrix,df,recommendno)
                    with st.expander("Results as JSON"):
                        results_json = result.to_dict('index')
                        st.write(results_json)
                    for row in result.iterrows():
                        title = row[1][0]
                        score = row[1][1]
                        url = row[1][2]
                        price = row[1][3]
                        subscriber = row[1][4]

                        stc.html(RESULT_FOUND.format(title,score,url,price,subscriber),height=250)
                except:
                    results="Not Found"
                    st.warning(results)
                    result= search_term_if_not_found(search_term,df,recommendno)
                    print(result)
                    if result.shape[0]:
                        st.info("Suggested Options include")
                        for row in result.iterrows():
                            title = row[1][0]
                            url = row[1][1]
                            price = row[1][2]
                            subscriber = row[1][3]
                            stc.html(RESULT_NOT_FOUND.format(title,url,price,subscriber),height=200)
    elif choice=="Predict_Subject":
        st.subheader("Predict Subject from Course Name")
        search_term=st.text_input("Search").strip()
        if st.button("Predict"):
            pickle_in = open('classifier.pkl', 'rb')
            classifier = pickle.load(pickle_in) 
            result = classifier.predict(vectorize_text(search_term))
            st.success('The output is {}'.format(result[0]))
    elif choice=="predict_Free_or_Paid":
        st.subheader("Predict whether course is free or paid")
        subjectlist=["Business Finance","Graphic Design","Musical Instruments","Web Development"]
        subject=st.selectbox("Subject",subjectlist)
        levellist=["All Levels","Beginner Level","Intermediate Level","Expert Level"]
        level=st.selectbox("Level",levellist)
        duration=st.number_input("Content Duration",0.0,100.0,10.0,format="%.2f")
        lectures=st.number_input("Number of Lectures",0,800,50)
        subscribers=st.number_input("Number of students",0,1000000,5000)
        reviews=st.number_input("Number of Reviews",0,100000,200)
        yearlist=[2012,2013,2014,2015,2016,2017]
        yr=st.selectbox("year",yearlist)
        if st.button("Predict"):
            pickle_in = open('paid_free.pkl', 'rb')
            classifier = pickle.load(pickle_in) 
            level=lev(level)
            subject=sub(subject)
            yr=year(yr)

            result = classifier.predict([[subscribers,reviews,lectures,level,duration,subject,yr]])
            result="It is Paid Course" if result==1 else "It is Free Course"
            st.success(result)
         
    else:
        st.subheader("About")
        st.text("This project includes the machine learning concepts to predict and recommend the course details of udemy")
        st.subheader("Dataset")
        st.text("Udemy Dataset from Kaggle")
        st.subheader("1.Recommendation System")
        st.text("cosine similarity")
        st.subheader("2.Subject Prediction")
        st.text("Logistic Regression")
        st.subheader("3.Whether free or Paid Course")
        st.text("KNN Model")
        st.subheader("Libraries that is used")
        st.write("1. sklearn")
        st.write("2. pandas")
        st.write("3. neattext")
        st.subheader("Application that is used")
        st.write("1. VS Code")
        st.write("2. Jupyter Notebook")
        st.write("3. Excel")
        st.subheader("Deploy")
        st.text("Streamlit Platform")
        st.subheader("By")
        st.write("20BCE095 : JADA URVIK")
        st.write("20BCE104 : JATIN UNDHAD")
        st.write("20BCE106 : JETANI VISHV")
        st.write("20BCE114 : JWAL SHAH")

if __name__=='__main__':
    main()

