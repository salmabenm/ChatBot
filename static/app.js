class Chatbox {
    constructor() {
        this.args =  {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button')
        }

        this.state = false;
        this.message = [];
    }

    display() {
        const { openButton, chatBox, sendButton } = this.args;

        openButton.addEventListener('click', () => this.toggleState(chatBox));
        sendButton.addEventListener('click', () => this.onSendButton(chatBox));

        const node = chatBox.querySelector('input');
        node.addEventListener('keyup', ({ key }) => {
            if (key === "Enter") {
                this.onSendButton(chatBox)
            }
        })
    }

    toggleState(chatbox) {
        this.state = !this.state;
    
        if (this.state) {
            chatbox.classList.add('chatbox--active')
        } else {
            chatbox.classList.remove('chatbox--active')
        }
    }

    onSendButton(chatbox) {
        var textField = chatbox.querySelector('input');
        let text1 = textField.value
        if (text1 === "") {
            return;
        }
        
        let msg1 = { name:"User", message: text1}
        this.message.push(msg1);

        // http://127.0.0.1:5000/predict
        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text1 }),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json'
            },
        })
        .then(r => r.json())
        .then(r => {
            let msg2 = { name: "BcS Bot", message: r.answer };
            this.message.push(msg2);
            this.updateChatText(chatbox);
            textField.value = '';
        })
        .catch((error) => {
            console.error('Error:', error);
            this.updateChatText(chatbox);
            textField.value = '';
        });
    }

    updateChatText(chatbox) {
        var html = '';
        this.message.slice().reverse().forEach(function(item, number) {
            if (item.name === "BcS Bot")
            {
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>'
            } 
            else {
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>'
            }
        });
        
        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html
    }
}

// Exemple d'utilisation de la classe Chatbox
const chatbox = new Chatbox();
chatbox.display();