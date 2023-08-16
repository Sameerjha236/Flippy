from flask import request, render_template
from flask import Flask
chat_history = []

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    # Initialize an empty query and chat history
    query = None
    # Check if the request method is POST
    if request.method == "POST":
        # Get the user input from the form
        query = request.form.get("query")
        # Pass the query and the chat history to the chain object, and get the result as a dictionary
        answer = 'This is the question ' + query
        # Append the query and the answer to the chat history list for context
        chat_history.append((query, answer))
    # Render the index.html template with the query and chat history variables
    return render_template("blank.html", query=query, chat_history=chat_history)


if __name__ == "__main__":
    chat_history = []  # Empty chat history when re-running the application
    app.run()
