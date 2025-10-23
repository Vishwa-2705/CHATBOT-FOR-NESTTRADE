import json
import random
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import re

app = Flask(__name__)
CORS(app)

# ------------------------
# Load house data
# ------------------------
houses = pd.read_csv("houses.csv")
houses['location'] = houses['location'].str.strip().str.title()

# Load intents
with open("intents.json", encoding="utf-8") as f:
    data = json.load(f)

# ------------------------
# Prepare ML training data for intents
# ------------------------
patterns = []
tags = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern.lower())
        tags.append(intent['tag'])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

encoder = LabelEncoder()
y = encoder.fit_transform(tags)

model = LogisticRegression(max_iter=200)
model.fit(X, y)

# ------------------------
# User session storage
# ------------------------
user_sessions = {}

# ------------------------
# Budget parser
# ------------------------
def parse_budget(budget_str):
    budget_str = budget_str.replace(" ", "").lower()
    if "lakhs" in budget_str or "lakh" in budget_str:
        return int(float(budget_str.replace("lakhs","").replace("lakh","")) * 100000)
    elif "crore" in budget_str:
        return int(float(budget_str.replace("crore","")) * 10000000)
    else:
        budget_str = budget_str.replace(",", "")
        return int(budget_str)

# ------------------------
# Facility normalization
# ------------------------
facility_map = {"park": "Parking", "parking": "Parking",
                "garden": "Garden", "gym": "Gym",
                "pool": "Swimming Pool", "swimming pool": "Swimming Pool"}

def normalize_facilities(fac_list):
    return [facility_map.get(f.strip().lower(), f.strip().title()) for f in fac_list]

# ------------------------
# Recommendation function
# ------------------------
def get_recommendations(session):
    global houses

    min_bhk = max(1, session['bhk'] - 1)
    max_bhk = session['bhk'] + 1
    max_budget = int(session['budget'] * 1.2)

    filtered = houses[
        (houses['location'].str.lower() == session['location'].lower()) &
        (houses['amount'] <= max_budget) &
        (houses['bhk'].between(min_bhk, max_bhk))
    ]

    if filtered.empty:
        filtered = houses[houses['location'].str.lower() == session['location'].lower()]

    # Similarity ML features
    houses_features = filtered[['sqft', 'amount', 'bhk']].copy()
    all_facilities = ['Garden', 'Parking', 'Gym', 'Swimming Pool']
    for f in all_facilities:
        houses_features[f] = filtered['facilities'].apply(lambda x: 1 if f in x else 0)

    scaler = StandardScaler()
    houses_features[['sqft','amount','bhk']] = scaler.fit_transform(houses_features[['sqft','amount','bhk']])

    user_features = pd.DataFrame([{
        'sqft': 0,
        'amount': 0,
        'bhk': session['bhk'],
        'Garden': 1 if 'Garden' in session['facilities'] else 0,
        'Parking': 1 if 'Parking' in session['facilities'] else 0,
        'Gym': 1 if 'Gym' in session['facilities'] else 0,
        'Swimming Pool': 1 if 'Swimming Pool' in session['facilities'] else 0
    }])
    user_features[['sqft','amount','bhk']] = scaler.transform(user_features[['sqft','amount','bhk']])

    sim_scores = cosine_similarity(houses_features.values, user_features.values).flatten()
    filtered['similarity'] = sim_scores
    recommended = filtered.sort_values(by='similarity', ascending=False).head(5)

    response_text = "Here are some houses you might like:\n"
    for i, (_, row) in enumerate(recommended.iterrows(), 1):
        response_text += f"{i}. {row['location']} | {row['sqft']} sq ft | ₹{row['amount']} | {row['bhk']} BHK | {row['facilities']}\n"
    response_text += "\nType the number of the house to select it, or 'recommend' to see more options."

    session["last_filtered"] = recommended.reset_index(drop=True)
    session["selected_house_index"] = None
    return response_text

# ------------------------
# Chat route
# ------------------------
@app.route('/chat', methods=['POST'])
def chat():
    global houses
    user_message = request.json.get('message', '').strip()
    user_message_lower = user_message.lower()
    user_id = "default_user"

    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "step": "welcome",
            "location": None,
            "budget": None,
            "bhk": None,
            "facilities": [],
            "last_filtered": None,
            "selected_house_index": None,
            "sell_step": None,
            "sell_details": {}
        }
    session = user_sessions[user_id]

    # Handle thanks or goodbye
    if any(word in user_message_lower for word in ["thank", "thanks", "thank you"]):
        return jsonify({"response": "You’re welcome! 😊 Hope you find your dream home."})
    if any(word in user_message_lower for word in ["bye", "goodbye"]):
        return jsonify({"response": "Thank you for using NestTrade! We wish you the best in finding your dream home."})

    # ------------------------
    # House selection
    # ------------------------
    if session["last_filtered"] is not None and session["selected_house_index"] is None:
        if user_message.isdigit():
            choice = int(user_message) - 1
            if 0 <= choice < len(session["last_filtered"]):
                session["selected_house_index"] = choice
                house = session["last_filtered"].iloc[choice]
                return jsonify({"response": f"You selected: {house['location']} | {house['sqft']} sq ft | ₹{house['amount']} | {house['bhk']} BHK | {house['facilities']}\nYou can ask about nearby amenities like hospital, market, school, etc."})
            else:
                return jsonify({"response": "Invalid selection. Please type a valid house number."})

    # ------------------------
    # Landmark queries
    # ------------------------
    landmark_keywords = ["school", "hospital", "railway", "station", "market"]
    if any(word in user_message_lower for word in landmark_keywords):
        if session.get("selected_house_index") is not None:
            house = session["last_filtered"].iloc[session["selected_house_index"]]
            for word in landmark_keywords:
                if word in user_message_lower:
                    distance_col = f"{word}_km"
                    distance = house.get(distance_col, "N/A")
                    return jsonify({"response": f"The nearest {word} is {distance} km away from your selected house."})
        else:
            return jsonify({"response": "Please select a house first to check nearby amenities."})

    # ------------------------
    # Welcome step
    # ------------------------
    if session["step"] == "welcome":
        session["step"] = "ask_intent"
        return jsonify({"response": "Welcome to NestTrade! Are you looking to buy or sell?"})

    # ------------------------
    # Intent recognition
    # ------------------------
    if session["step"] in ["ask_intent", "retry"]:
        user_vec = vectorizer.transform([user_message_lower])
        predicted_tag = encoder.inverse_transform(model.predict(user_vec))[0]

        if predicted_tag == "greeting":
            return jsonify({"response": random.choice([r for r in data['intents'] if r['tag']=='greeting'][0]['responses'])})
        elif predicted_tag == "buy":
            session["step"] = "location"
            return jsonify({"response": "Great! Which city are you looking for?"})
        elif predicted_tag == "sell":
            session["step"] = "sell"
            session["sell_step"] = "owner"
            return jsonify({"response": "Sure! Let's register your property. First, please provide the owner's name."})
        elif predicted_tag == "contact":
            return jsonify({"response": "You can contact NestTrade at support@nesttrade.com or call +91 9876543210."})
        elif predicted_tag == "recommend":
            if session["last_filtered"] is not None and not session["last_filtered"].empty:
                recommended = get_recommendations(session)
                return jsonify({"response": recommended})
            else:
                return jsonify({"response": "No previous houses to recommend. Please enter your requirements first."})
        else:
            return jsonify({"response": "I can help you buy or sell houses. What would you like to do?"})

    # ------------------------
    # Buy workflow
    # ------------------------
    if session["step"] == "location":
        session["location"] = user_message.strip().title()
        session["step"] = "budget"
        return jsonify({"response": "What's your budget? (e.g., 40lakhs, 0.5crore, 4000000)"})
    elif session["step"] == "budget":
        try:
            session["budget"] = parse_budget(user_message)
            session["step"] = "bhk"
            return jsonify({"response": "How many BHK do you want?"})
        except:
            return jsonify({"response": "Please enter a valid number for budget (e.g., 35lakhs, 0.5crore, 4000000)."})
    elif session["step"] == "bhk":
        try:
            session["bhk"] = int(user_message)
            session["step"] = "facilities"
            return jsonify({"response": "Any specific facilities? (e.g., Garden, Parking, Gym) or type 'skip'"})
        except ValueError:
            return jsonify({"response": "Please enter a valid number for BHK."})
    elif session["step"] == "facilities":
        if user_message_lower in ["skip", "none", "no"]:
            session["facilities"] = []
        else:
            session["facilities"] = normalize_facilities(user_message.split(","))

        recommended = get_recommendations(session)
        session["step"] = "ask_intent"
        return jsonify({"response": recommended})

    # ------------------------
    # Sell workflow
    # ------------------------
    if session["step"] == "sell":
        if session["sell_step"] == "owner":
            session["sell_details"]["owner"] = user_message.strip()
            session["sell_step"] = "contact"
            return jsonify({"response": "Please provide the owner's contact number."})
        elif session["sell_step"] == "contact":
            session["sell_details"]["contact"] = user_message.strip()
            session["sell_step"] = "location"
            return jsonify({"response": "Which city is the property in?"})
        elif session["sell_step"] == "location":
            session["sell_details"]["location"] = user_message.strip().title()
            session["sell_step"] = "sqft"
            return jsonify({"response": "What is the size of the house in sq ft?"})
        elif session["sell_step"] == "sqft":
            try:
                session["sell_details"]["sqft"] = int(user_message)
                session["sell_step"] = "amount"
                return jsonify({"response": "What is the expected price?"})
            except ValueError:
                return jsonify({"response": "Please enter a valid number for the size in sq ft."})
        elif session["sell_step"] == "amount":
            try:
                session["sell_details"]["amount"] = parse_budget(user_message)
                session["sell_step"] = "bhk"
                return jsonify({"response": "How many BHK does the house have?"})
            except:
                return jsonify({"response": "Enter a valid price (e.g., 50lakhs, 0.5crore, 5000000)."})
        elif session["sell_step"] == "bhk":
            try:
                session["sell_details"]["bhk"] = int(user_message)
                session["sell_step"] = "facilities"
                return jsonify({"response": "Any specific facilities? (e.g., Garden, Parking, Gym) or type 'skip'"})
            except ValueError:
                return jsonify({"response": "Please enter a valid number for BHK."})
        elif session["sell_step"] == "facilities":
            if user_message_lower in ["skip", "none", "no"]:
                session["sell_details"]["facilities"] = []
            else:
                session["sell_details"]["facilities"] = normalize_facilities(user_message.split(","))

            # Add property to houses
            new_house = pd.DataFrame([{
                "location": session["sell_details"]["location"],
                "sqft": session["sell_details"]["sqft"],
                "amount": session["sell_details"]["amount"],
                "bhk": session["sell_details"]["bhk"],
                "facilities": ", ".join(session["sell_details"]["facilities"]),
                "owner": session["sell_details"]["owner"],
                "contact": session["sell_details"]["contact"]
            }])
            houses = pd.concat([houses, new_house], ignore_index=True)
            houses.to_csv("houses.csv", index=False)

            session["step"] = "ask_intent"
            session["sell_step"] = None
            session["sell_details"] = {}

            return jsonify({"response": "Thank you! Your property has been registered successfully. Our team will contact you shortly. Within 24 hours, your property will be added in our app after verification."})

    # ------------------------
    # Fallback
    # ------------------------
    return jsonify({"response": "I didn't understand that. Can you try again?"})

# ------------------------
# Run the app
# ------------------------
from flask import send_from_directory

@app.route("/")
def index():
    return send_from_directory(".", "index.html")
if __name__ == "__main__":
    app.run(debug=True)
