from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

responses = {
    "hello": "Hello! How can I help you today?",
    "hi": "Hi! Welcome to CodeAlpha AI Chatbot.",
    "how are you": "I'm doing great! How about you?",
    "what is your name": "I'm a simple AI chatbot created using Flask.",
    "services": "We provide AI, Cloud Computing, Web Development, and Internship support.",
    "internship": "CodeAlpha offers internships in AI, Cloud Computing, Web Development, and more.",
    "bye": "Goodbye! Have a great day!"
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_message = request.json["message"].lower().strip()
    reply = responses.get(
        user_message,
        "Sorry, I don't understand that. Please try another question."
    )
    return jsonify({"response": reply})

if __name__ == "__main__":
    app.run(debug=True)
