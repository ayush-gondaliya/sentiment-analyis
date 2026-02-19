from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO
from nltk.corpus import stopwords, opinion_lexicon
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import base64

STOPWORDS = set(stopwords.words("english"))
MANUAL_NEGATIVE_WORDS = {
    "bad", "poor", "sad", "angry", "annoyed", "bored", "worried", "nervous",
    "hate", "dislike", "disappointed", "regret", "guilty", "shame", "angry",
    "unhappy", "unpleasant", "dreadful", "miserable", "horrible", "terrible",
    "awful", "fearful", "scared", "anxious", "stressful", "uptight", "frustrated",
    "irritated", "enraged", "agitated", "confused", "lonely", "abandoned", "rejected",
    "worthless", "useless", "hopeless", "desperate", "depressed", "gloomy", "sorrow",
    "grief", "pain", "suffering", "heartbroken", "betrayed", "bitter", "disgusted",
    "offended", "jealous", "envy", "resentful", "embarrassed", "ashamed", "awkward",
    "uncomfortable", "awkward", "lazy", "tired", "exhausted", "sleepy", "weak", "sick",
    "ill", "pathetic", "unfortunate", "disastrous", "melancholy", "tragic", "grim",
    "bleak", "gloomy", "dark", "worst", "no", "never", "none", "not"
}

# Add opinion lexicon negative words
OPINION_NEGATIVE_WORDS = set(opinion_lexicon.negative())
NEGATIVE_WORDS = MANUAL_NEGATIVE_WORDS.union(OPINION_NEGATIVE_WORDS)

app = Flask(__name__)

@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")

@app.route("/predict", methods=["POST"])
def predict():
    predictor = pickle.load(open(r"Models/model_xgb.pkl", "rb"))
    scaler = pickle.load(open(r"Models/scaler.pkl", "rb"))
    cv = pickle.load(open(r"Models/countVectorizer.pkl", "rb"))
    try:
        if "file" in request.files:
            file = request.files["file"]
            data = pd.read_csv(file)
            predictions, graph = bulk_prediction(predictor, scaler, cv, data)
            response = send_file(
                predictions,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            )
            response.headers["X-Graph-Exists"] = "true"
            response.headers["X-Graph-Data"] = base64.b64encode(graph.getbuffer()).decode("ascii")
            return response

        elif "text" in request.json:
            text_input = request.json["text"]
            predicted_sentiment, marked_text = single_prediction(predictor, scaler, cv, text_input)
            return jsonify({"prediction": predicted_sentiment, "marked_text": marked_text})

    except Exception as e:
        return jsonify({"error": str(e)})

def preprocess_text(text):
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text)
    review = review.lower().split()
    processed_review = []
    negate = False  # Flag to track negation

    for word in review:
        if word == "not":
            negate = True
            continue
        
        stemmed_word = stemmer.stem(word)
        
        if negate:
            processed_review.append(f"not_{stemmed_word}" or"not")
        else:
            if stemmed_word in NEGATIVE_WORDS:
                processed_review.append(f"<{stemmed_word}>")
            elif stemmed_word not in STOPWORDS:
                processed_review.append(stemmed_word)
        
        negate = false  # Reset negate after handling "not" phrase

    return " ".join(processed_review)

def single_prediction(predictor, scaler, cv, text_input):
    processed_text = preprocess_text(text_input)
    corpus = [processed_text]
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    predicted_class = y_predictions.argmax(axis=1)[0]
    
    # Check if any negative word is present in processed text
    contains_negative_word = any(f"<{word}>" in processed_text for word in NEGATIVE_WORDS)
    
    # Check for negations in processed text
    contains_negation = any("not_" in word for word in processed_text.split())
    
    if contains_negative_word or contains_negation:
        return "Negative", processed_text
    else:
        return "Positive" if predicted_class == 1 else "Negative", processed_text

def bulk_prediction(predictor, scaler, cv, data):
    corpus = [preprocess_text(sentence) for sentence in data["Sentence"]]
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)
    y_predictions = list(map(sentiment_mapping, y_predictions))
    data["Predicted sentiment"] = y_predictions
    predictions_csv = BytesIO()
    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)
    graph = get_distribution_graph(data)
    return predictions_csv, graph

def get_distribution_graph(data):
    fig = plt.figure(figsize=(5, 5))
    colors = ("green", "red")
    wp = {"linewidth": 1, "edgecolor": "black"}
    tags = data["Predicted sentiment"].value_counts()
    explode = (0.01, 0.01)
    tags.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        colors=colors,
        startangle=90,
        wedgeprops=wp,
        explode=explode,
        title="Sentiment Distribution",
        xlabel="",
        ylabel="",
    )
    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close()
    return graph

def sentiment_mapping(x):
    return "Positive" if x == 1 else "Negative"

if __name__ == "__main__":
    app.run(port=5000, debug=True)
