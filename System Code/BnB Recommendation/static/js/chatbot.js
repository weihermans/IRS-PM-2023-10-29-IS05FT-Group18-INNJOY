const chatInput = document.querySelector(".chat-input textarea");
const sendChatBtn = document.querySelector(".chat-input span");
const chatbox = document.querySelector(".chatbox");

const inputInitHeight = chatInput.scrollHeight;

const createChatLi = (message, className) => {
    // Create a chat <li> element with passed message and className
    const chatLi = document.createElement("li");
    chatLi.classList.add("chat", `${className}`);
    chatLi.classList.add('animate-box');
    chatLi.classList.add('fadeInUp'); 
    chatLi.classList.add('animated');
    let chatContent = className === "outgoing" ? `<p></p>` : `<span class="material-symbols-outlined">smart_toy</span><p></p>`;
    chatLi.innerHTML = chatContent;
    chatLi.querySelector("p").textContent = message;
    return chatLi; // return chat <li> element
}

const generateResponse = (chatElement, message) => {
    const messageElement = chatElement.querySelector("p");
    messageElement.textContent = message; // Fixed response for now, you can replace this with the actual response from the backend
}

const handleChat = () => {
    const userMessage = chatInput.value.trim(); // Get user entered message and remove extra whitespace
    if (!userMessage) return;

    // Clear the input textarea and set its height to default
    chatInput.value = "";
    chatInput.style.height = `${inputInitHeight}px`;

    // Append the user's message to the chatbox
    chatbox.appendChild(createChatLi(userMessage, "outgoing"));
    chatbox.scrollTo(0, chatbox.scrollHeight);

    // Display "Thinking..." message while waiting for the response
    const incomingChatLi = createChatLi("Thinking...", "incoming");
    chatbox.appendChild(incomingChatLi);
    chatbox.scrollTo(0, chatbox.scrollHeight);

    // Send the user message to the backend
    fetch("/chatbot", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: userMessage })
    })
    .then(response => response.json())
    .then(data => {
        // Handle the response from the backend (if needed)
        console.log("Response from backend:", data.message);
        // Set the response from the backend to the chatbox
        generateResponse(incomingChatLi,data.message);
    })
    .catch(error => {
        // Handle errors (if any)
        console.error("Error:", error);
        // Display error message in the chatbox
        incomingChatLi.querySelector("p").textContent = "Oops! Something went wrong. Please try again.";
    })
    .finally(() => {
        chatbox.scrollTo(0, chatbox.scrollHeight);
    });
}

chatInput.addEventListener("input", () => {
    // Adjust the height of the input textarea based on its content
    chatInput.style.height = `${inputInitHeight}px`;
    chatInput.style.height = `${chatInput.scrollHeight}px`;
});

chatInput.addEventListener("keydown", (e) => {
    // If Enter key is pressed without Shift key, handle the chat
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleChat();
    }
});

sendChatBtn.addEventListener("click", handleChat);


const socket = io.connect('http://localhost:8000');
socket.on('update', function(data) {
    // 在这里执行前端更新逻辑
    const roomContainer = document.getElementById('room-container');
    roomContainer.innerHTML = '';
    data.room_data.forEach(room => {
        const roomDiv = document.createElement('div');
        roomDiv.className = 'col-md-3 col-sm-6 col-padding text-center animate-box fadeInUp animated';
        roomDiv.innerHTML = `
            <a href="${room.listing_url}" target="_blank" class="work image-popup" style="background-image: url(${room.picture_url});">
                <div class="desc">
                    <h3>${room.name}</h3>
                    <span>S$${room.price}</span>
                </div>
            </a>
        `;
        roomContainer.appendChild(roomDiv);
});
})

fetch('/get-initial')
    .then(response => {
        // 请求成功处理
        if (response.ok) {
            console.log('Initial data request successful!');
        } else {
            console.error('Initial data request failed.');
        }
    })
    .catch(error => {
        // 请求失败处理
        console.error('Error:', error);
    });