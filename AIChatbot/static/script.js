async function sendMessage() {
    const input = document.getElementById("user-input");
    const message = input.value.trim();
    if (message === "") return;

    const chatBox = document.getElementById("chat-box");
    chatBox.innerHTML += `<div class="user"><b>You:</b> ${message}</div>`;

    const response = await fetch("/get_response", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({message: message})
    });

    const data = await response.json();
    chatBox.innerHTML += `<div class="bot"><b>Bot:</b> ${data.response}</div>`;
    chatBox.scrollTop = chatBox.scrollHeight;
    input.value = "";
}
