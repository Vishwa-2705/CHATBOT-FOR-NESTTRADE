async function sendMessage() {
  const inputField = document.getElementById("user-input");
  const message = inputField.value.trim();
  if (message === "") return;

  const chatBox = document.getElementById("chat-box");

  // Display user message
  chatBox.innerHTML += `<div class="user"><strong>You:</strong> ${message}</div>`;

  // Send message to Flask backend
  const response = await fetch("http://127.0.0.1:5000/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });

  const data = await response.json();

  // Landmark icons mapping
  const landmarkIcons = {
    "railway": "🚆",
    "school": "🏫",
    "market": "🛒",
    "hospital": "🏥",
    "metro": "🚇"
  };

  let botMessage = data.response;

  // Check if message contains landmark keywords
  Object.keys(landmarkIcons).forEach(keyword => {
    if (message.toLowerCase().includes(keyword)) {
      botMessage = `${landmarkIcons[keyword]} ${data.response}`;
    }
  });

  // Display bot response
  chatBox.innerHTML += `<div class="bot"><strong>Bot:</strong> ${botMessage}</div>`;

  // Scroll down automatically
  chatBox.scrollTop = chatBox.scrollHeight;

  // Clear input field
  inputField.value = "";
}

// Send message on button click
document.getElementById("send-btn").addEventListener("click", sendMessage);

// Send message on Enter key
document.getElementById("user-input").addEventListener("keypress", function(e) {
  if (e.key === "Enter") sendMessage();
});
