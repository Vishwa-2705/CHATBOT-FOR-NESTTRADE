import requests

# URL of your running Flask app
url = "http://127.0.0.1:5000/chat"

# Message to send
data = {"message": "Show me houses in Chennai"}

# Send POST request
response = requests.post(url, json=data)

# Print chatbot response
print("Chatbot response:", response.json())
