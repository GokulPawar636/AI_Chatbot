from flask import Flask, render_template, request, jsonify
from rag_pipeline import generate_answer

app = Flask(__name__)

# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------------- CHAT API ----------------
@app.route("/chat", methods=["POST"])
def chat():

    user_message = request.json.get("message")

    response = generate_answer(user_message)

    return jsonify({
        "response": response
    })

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)