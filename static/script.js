async function sendMessage() {

    const input = document.getElementById("user-input");

    const message = input.value.trim();

    if (!message) return;

    const chatBox = document.getElementById("chat-box");

    // USER MESSAGE
    const userDiv = document.createElement("div");

    userDiv.className = "user-message";

    userDiv.innerText = message;

    chatBox.appendChild(userDiv);

    input.value = "";

    // BOT LOADING
    const botDiv = document.createElement("div");

    botDiv.className = "bot-message";

    botDiv.innerText = "⏳ Thinking...";

    chatBox.appendChild(botDiv);

    chatBox.scrollTop = chatBox.scrollHeight;

    // API CALL
    const response = await fetch("/chat", {

        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify({
            message: message
        })
    });

    const data = await response.json();

    botDiv.innerText = data.response;

    chatBox.scrollTop = chatBox.scrollHeight;
}

// ENTER KEY SUPPORT
document
.getElementById("user-input")
.addEventListener("keypress", function(e) {

    if (e.key === "Enter") {
        sendMessage();
    }
});
