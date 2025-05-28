import pickle
from flask import Flask, request, render_template
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the serialized model and vectorizer
with open('vectorizer0.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
    
with open('svm_model0.pkl', 'rb') as f:
    svm = pickle.load(f)
# Load job data from the pickle file
with open('job_data0.pkl', 'rb') as f:
    job_data = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from POST request
    userInput = request.form['user_input']
    
    # Transform user input using the vectorizer
    pred = vectorizer.transform([userInput.lower()])
    
    # Make prediction using the model
    output = svm.predict(pred)
    
    # Filter job data based on predicted label
    labelData = job_data[job_data['Label'] == output[0]].copy()
    
    # Calculate cosine similarity for each job
    cos=[]
    for index, row in labelData.iterrows(): 
        skills= [row['job_skills']]
        skillVec = vectorizer.transform(skills)
        cos_lib=cosine_similarity(skillVec, pred)
        cos.append(cos_lib[0][0])
    
    # Add cosine similarity to labelData DataFrame
    labelData.loc[:, 'cosine_similarity'] = cos
    
    # Sort labelData DataFrame by cosine similarity
    top_5 = labelData.sort_values('cosine_similarity', ascending=False).head(5)

    # Convert top 5 recommendations to JSON
    top_5_json = top_5.to_dict(orient='records')

    # Before rendering the template
    # print("Top 5 Recommendations:", top_5_json)
    return render_template('index.html', top_5=top_5_json)

if __name__ == "__main__":
    app.run(debug=True)
