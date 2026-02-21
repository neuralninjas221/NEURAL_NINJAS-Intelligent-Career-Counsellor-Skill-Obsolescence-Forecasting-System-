from flask import Flask, request, jsonify
from flask_cors import CORS
from recommender import recommend

app = Flask(__name__)
CORS(app)

@app.route("/recommend", methods=["POST"])
def recommend_route():
    skills = request.json.get("skills", [])
    results = recommend(skills)
    return jsonify({"career": results.to_dict()})
if __name__ == "__main__":
    app.run(debug=True)