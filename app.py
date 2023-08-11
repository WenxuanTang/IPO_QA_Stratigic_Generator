

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
from myopenai import get_completion_from_messages, format_response
from dotenv import load_dotenv, find_dotenv
import os

app = Flask(__name__)
_ = load_dotenv(find_dotenv())  # Read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


df = pd.read_csv('ipo_embeddings_2sen.csv')
df['embedding'] = df['embedding'].apply(eval).apply(np.array)

@app.route('/', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        search_query = request.form['search_query']
        model1_search_results = search_model('text-embedding-ada-002', 1, search_query)
        model2_search_results = search_model('text-embedding-ada-002', 2, search_query)
        # model3_search_results = search_model('text-embedding-ada-002', 3, search_query)
        return render_template('result.html', 
                               model1_results=model1_search_results, 
                               model2_results=model2_search_results, 
                            #    model3_results=model3_search_results,
                               query=search_query)
    return render_template('index.html')

def search_model(model_name, exclude, query):
    search_term_vector = get_embedding(query, engine=model_name)
    df["similarities"] = df['embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))
    output = df.sort_values("similarities", ascending=False)['text'].to_list()[:5]
    context = ' '.join(output)
    delimiter = "####"

    if exclude == 1:
        system_message = f"""
        You will be provided with relative sentences in the ipo file of company Kenvue. \
        And your task is to find the answer only with the context provided.\
        Please provide a well-structured and comprehensive response, organizing the information into clear paragraphs and addressing each point separately.\
        Make sure there are no repeated sentences in your reply.\
        Here's the context: {context}\
        The query will be delimited with {delimiter} characters.
        """
        messages = [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': f"{delimiter}{query}{delimiter}"},
        ]
        response1 = get_completion_from_messages(messages, temperature = 0)
        return response1

    elif exclude == 2:
        system_message = f"""
        You will be provided with relative sentences in the ipo file of company Kenvue. \
        And your task is to find the answer with the context provided and your knowledge.\
        Please provide a well-structured and comprehensive response, organizing the information into clear paragraphs and addressing each point separately.\
        Make sure there are no repeated sentences in your reply\
        Here's the context {context}\
        The query will be delimited with {delimiter} characters.
        """
        messages = [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': f"{delimiter}{query}{delimiter}"},
        ]
        response2 = get_completion_from_messages(messages, temperature = 0.7)
        return response2

if __name__ == '__main__':
    app.run(debug=True, port=8000)
