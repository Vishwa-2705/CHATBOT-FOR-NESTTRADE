async function sendMessage() {
  const inputField = document.getElementById("user-input");
  const message = inputField.value.trim();

  if (message === "") return;

  // Display user message
  const chatBox = document.getElementById("chat-box");
  chatBox.innerHTML += `<div class="user"><strong>You:</strong> ${message}</div>`;

  // Send message to Flask backend
  const response = await fetch("http://127.0.0.1:5000/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message }),
  });

  const data = await response.json();

  // Display bot response
  chatBox.innerHTML += `<div class="bot"><strong>Bot:</strong> ${data.response}</div>`;

  // Scroll down automatically
  chatBox.scrollTop = chatBox.scrollHeight;

  // Clear input field
  inputField.value = "";
}
