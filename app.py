from flask import Flask, request, jsonify
import pickle
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
app = Flask(__name__)

def load_spam_classifier():
    with open('spam_classifier_model.pkl', 'rb') as model_file:
        cv, classifier = pickle.load(model_file)
    return cv, classifier

def predict_spam(sample_message):
    cv, classifier = load_spam_classifier()
    sample_message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sample_message)
    sample_message = sample_message.lower()
    sample_message_words = sample_message.split()
    sample_message_words = [word for word in sample_message_words if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    final_message = [ps.stem(word) for word in sample_message_words]
    final_message = ' '.join(final_message)
    temp = cv.transform([final_message]).toarray()
    return classifier.predict(temp)

@app.route('/')
def home():
    return "Hello, World!"

@app.route('/predict', methods=['POST'])
def predict():
    msg = request.form.get('msg')
    result = ['Wait a minute, this is a SPAM!', 'Ohhh, this is a normal message.']
    if predict_spam(msg):
        return jsonify({'prediction': result[0]})
    else:
        return jsonify({'prediction': result[1]})

if __name__ == '__main__':
    app.run(debug=True)
