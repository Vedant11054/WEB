from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the pre-trained models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Home route to render the index.html
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the SMS text from the form
        input_sms = request.form['sms']

        # Preprocess the SMS text
        transformed_sms = transform_text(input_sms)

        # Vectorize the input
        vector_input = tfidf.transform([transformed_sms])

        # Predict using the model
        result = model.predict(vector_input)[0]

        # Return the result page with the prediction
        if result == 1:
         result_text = "Spam"
         result_class = "spam"
        else:
         result_text = "Not Spam"
         result_class = "not-spam"

        return render_template('result.html', result_text=result_text, result_class=result_class)

# Function to preprocess the text
def transform_text(text):
    import string
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer

    nltk.download('punkt')
    nltk.download('stopwords')

    ps = PorterStemmer()

    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

if __name__ == "__main__":
    app.run(debug=True)
