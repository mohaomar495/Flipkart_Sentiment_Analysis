from flask import Flask, render_template, request
import joblib as jb

app = Flask(__name__)

# Load the model and vectorizer
with open('MNB_bow_model.pkl', 'rb') as model:
    model = jb.load(model)

with open('CV_vectorizer.pkl', 'rb') as CV_vect:
    vectorizer = jb.load(CV_vect)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_review = request.form.get("clear_text")

        # Check if input is empty
        if not user_review:
            return render_template("index.html", sentiment="No text provided")
        
        text = vectorizer.transform([user_review])
        pred = model.predict(text)[0]
        return render_template("index.html", user_review=user_review, pred = pred)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
