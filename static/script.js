/**
 * script.js - Frontend Logic
 * ===========================
 * Handles all user interactions and API communication.
 * 
 * Flow:
 * 1. User types message and submits
 * 2. Display user message in chat
 * 3. Send POST request to /chat endpoint
 * 4. Display bot response when received
 * 5. Auto-scroll to bottom
 */

// API Configuration
const API_URL = window.location.origin;
const CHAT_ENDPOINT = `${API_URL}/chat`;

// DOM Elements
const messagesContainer = document.getElementById('messages');
const chatForm = document.getElementById('chatForm');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const sendIcon = document.getElementById('sendIcon');
const loadingOverlay = document.getElementById('loadingOverlay');
const quickQuestions = document.getElementById('quickQuestions');

// State
let isProcessing = false;

/**
 * Initialize the application
 */
function init() {
    // Focus on input field when page loads
    userInput.focus();
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Focus input with Ctrl/Cmd + /
        if ((e.ctrlKey || e.metaKey) && e.key === '/') {
            e.preventDefault();
            userInput.focus();
        }
        
        // Clear chat with Ctrl/Cmd + K
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            if (confirm('Clear all messages?')) {
                clearChat();
            }
        }
    });
    
    console.log('🚀 FAST NUCES Chatbot initialized');
}

/**
 * Send user message to the server
 * @param {Event} event - Form submit event
 */
async function sendMessage(event) {
    event.preventDefault();
    
    // Get user input
    const message = userInput.value.trim();
    
    // Validate input
    if (!message || isProcessing) {
        return;
    }
    
    // Hide quick questions after first message
    if (quickQuestions) {
        quickQuestions.style.display = 'none';
    }
    
    // Clear input immediately for better UX
    userInput.value = '';
    
    // Display user message
    appendMessage(message, 'user');
    
    // Show typing indicator
    const typingId = showTypingIndicator();
    
    // Disable input during processing
    setProcessingState(true);
    
    try {
        // Send POST request to backend
        const response = await fetch(CHAT_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message
            })
        });
        
        // Check if request was successful
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        // Parse JSON response
        const data = await response.json();
        
        // Remove typing indicator
        removeTypingIndicator(typingId);
        
        // Display bot response
        appendMessage(data.response, 'bot');
        
        // Log sources for debugging (optional)
        if (data.sources) {
            console.log('📚 Sources used:', data.sources);
        }
        
    } catch (error) {
        // Remove typing indicator
        removeTypingIndicator(typingId);
        
        // Display error message
        appendMessage(
            `⚠️ Sorry, I encountered an error: ${error.message}. Please try again or contact support.`,
            'bot',
            'error'
        );
        
        console.error('❌ Error:', error);
    } finally {
        // Re-enable input
        setProcessingState(false);
        
        // Refocus input for next message
        userInput.focus();
    }
}

/**
 * Send a quick question
 * @param {string} question - Predefined question text
 */
function sendQuickQuestion(question) {
    // Set input value
    userInput.value = question;
    
    // Trigger form submission
    chatForm.dispatchEvent(new Event('submit'));
}

/**
 * Append a message to the chat
 * @param {string} text - Message text
 * @param {string} type - 'user' or 'bot'
 * @param {string} messageType - Optional: 'error', 'success'
 */
function appendMessage(text, type, messageType = 'normal') {
    // Create message container
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    // Add special styling for error/success messages
    if (messageType === 'error') {
        messageDiv.classList.add('error-message');
    } else if (messageType === 'success') {
        messageDiv.classList.add('success-message');
    }
    
    // Create avatar
    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'message-avatar';
    avatarDiv.textContent = type === 'user' ? '👤' : '🤖';
    
    // Create content container
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Create text container
    const textDiv = document.createElement('div');
    textDiv.className = 'message-text';
    
    // Format text (convert newlines to <br>, support markdown-like formatting)
    textDiv.innerHTML = formatMessage(text);
    
    // Assemble message
    contentDiv.appendChild(textDiv);
    messageDiv.appendChild(avatarDiv);
    messageDiv.appendChild(contentDiv);
    
    // Add to messages container
    messagesContainer.appendChild(messageDiv);
    
    // Scroll to bottom
    scrollToBottom();
}

/**
 * Format message text
 * Supports basic markdown-like formatting
 * @param {string} text - Raw text
 * @returns {string} - Formatted HTML
 */
function formatMessage(text) {
    // Escape HTML to prevent XSS
    text = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    
    // Convert newlines to <br>
    text = text.replace(/\n/g, '<br>');
    
    // Bold: **text** or __text__
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/__(.*?)__/g, '<strong>$1</strong>');
    
    // Italic: *text* or _text_
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
    text = text.replace(/_(.*?)_/g, '<em>$1</em>');
    
    // Code: `text`
    text = text.replace(/`(.*?)`/g, '<code>$1</code>');
    
    // Links: [text](url)
    text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
    
    return text;
}

/**
 * Show typing indicator
 * @returns {string} - Unique ID for this indicator
 */
function showTypingIndicator() {
    const id = `typing-${Date.now()}`;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';
    messageDiv.id = id;
    
    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'message-avatar';
    avatarDiv.textContent = '🤖';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const textDiv = document.createElement('div');
    textDiv.className = 'message-text';
    textDiv.innerHTML = `
        <div class="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;
    
    contentDiv.appendChild(textDiv);
    messageDiv.appendChild(avatarDiv);
    messageDiv.appendChild(contentDiv);
    
    messagesContainer.appendChild(messageDiv);
    scrollToBottom();
    
    return id;
}

/**
 * Remove typing indicator
 * @param {string} id - ID of typing indicator to remove
 */
function removeTypingIndicator(id) {
    const element = document.getElementById(id);
    if (element) {
        element.remove();
    }
}

/**
 * Set processing state (disable/enable inputs)
 * @param {boolean} processing - Whether currently processing
 */
function setProcessingState(processing) {
    isProcessing = processing;
    
    // Disable/enable input and button
    userInput.disabled = processing;
    sendBtn.disabled = processing;
    
    // Change button icon
    if (processing) {
        sendIcon.textContent = '⏳';
        loadingOverlay.classList.add('active');
    } else {
        sendIcon.textContent = '📤';
        loadingOverlay.classList.remove('active');
    }
}

/**
 * Scroll chat to bottom
 */
function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    // Smooth scroll animation
    messagesContainer.scrollTo({
        top: messagesContainer.scrollHeight,
        behavior: 'smooth'
    });
}

/**
 * Clear all messages
 */
function clearChat() {
    // Keep only welcome message
    const welcomeMessage = messagesContainer.querySelector('.bot-message');
    messagesContainer.innerHTML = '';
    if (welcomeMessage) {
        messagesContainer.appendChild(welcomeMessage);
    }
    
    // Show quick questions again
    if (quickQuestions) {
        quickQuestions.style.display = 'flex';
    }
    
    console.log('🗑️ Chat cleared');
}

/**
 * Copy message text to clipboard
 * @param {HTMLElement} element - Message element
 */
function copyMessage(element) {
    const text = element.innerText;
    navigator.clipboard.writeText(text).then(() => {
        // Show feedback
        const feedback = document.createElement('div');
        feedback.textContent = '✓ Copied!';
        feedback.style.cssText = 'position: absolute; background: #10b981; color: white; padding: 5px 10px; border-radius: 5px; font-size: 0.8rem;';
        element.appendChild(feedback);
        
        setTimeout(() => feedback.remove(), 2000);
    });
}

// Event Listeners
chatForm.addEventListener('submit', sendMessage);

// Initialize on page load
window.addEventListener('DOMContentLoaded', init);

// Export functions for use in HTML
window.sendQuickQuestion = sendQuickQuestion;
window.clearChat = clearChat;
window.copyMessage = copyMessage;

console.log('✅ Script loaded successfully');