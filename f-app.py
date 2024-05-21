import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from pywebio import start_server, output
from pywebio.input import input
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask

app = Flask(__name__)

# Load tokenizer and model configuration
tokenizer = AutoTokenizer.from_pretrained("fine_tuned_model")
config = AutoConfig.from_pretrained("fine_tuned_model")
# Load model
model = TFAutoModelForSequenceClassification.from_pretrained("fine_tuned_model", config=config)
# Initialize lists to store conversation outputs and distraction levels
conversation_outputs = []
predicted_emotions = []
probability_distributions = []
distraction_levels = []

def handle_input(user_input):
    global probability_distributions
    output.put_text("User:", user_input)
    # Tokenize the text
    input_encoded = tokenizer(user_input, return_tensors='tf')

    # Perform inference
    outputs = model(input_encoded)

    # Get predicted label and probabilities
    logits = outputs.logits
    probabilities = tf.nn.softmax(logits, axis=1).numpy()[0]
    pred = tf.argmax(logits, axis=1).numpy()
    predicted_label = pred[0]
    predicted_emotions.append(predicted_label)
    probability_distributions.append(probabilities)
    # Display predicted emotion and probabilities
    emojis = {
        'sadness': '😔',
        'joy': '🤗',
        'love': '❤️',
        'anger': '😠',
        'fear': '😨😱',
        'surprise': '😮'
    }
    labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    predicted_emoji = emojis[labels[predicted_label]]

    output.put_text("Chatbot:", f"The predicted emotion is: {labels[predicted_label]}{predicted_emoji}")
    # Calculate distraction level (for demonstration purposes, a random value is used)
    distraction_level = np.random.uniform(0, 1)
    distraction_levels.append(distraction_level)

def generate_probability_distribution_graph():
    global probability_distributions
    if not probability_distributions:
        output.put_text("No probability distributions available to generate graph.")
        return

    # Plot probability distribution bar graph
    plt.rcParams['axes.facecolor'] = 'white'
    plt.figure(figsize=(10, 6))
    labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    probs = probability_distributions[-1] 
    colors = ['#9FB9C7', '#3D657A', '#C8E8FA', '#7DCFFB', '#677881', '#5B96A9']
    bars = plt.bar(labels, probs, color=colors)
    emoji_dict = {
        'sadness': '😔',
        'joy': '😀',
        'love': '❤️',
        'anger': '😠',
        'fear': '😨😱',
        'surprise': '😮'
    }

    x_positions = [bar.get_x() + bar.get_width() / 2 for bar in bars]  # Center of each bar
    y_positions = [bar.get_height() + 0.1 for bar in bars]  # Slightly above the bar
    for x, y, label in zip(x_positions, y_positions, labels):
        plt.text(x, y, emoji_dict[label], ha='center', va='bottom', fontsize=20)

    plt.xlabel('Emotion')
    plt.ylabel('Probability')
    plt.title('Probability Distribution of Emotions')
    plt.grid(False)
    plt.tight_layout()

    # Save plot as image
    plt.savefig('probability_distribution_graph.png')
    plt.close()

    # Display the generated image
    output.put_image(open('probability_distribution_graph.png', 'rb').read(), width='80%')

def generate_distraction_graph():
    global distraction_levels
    if not distraction_levels:
        output.put_text("No distraction levels available to generate graph.")
        return

    # Plot distraction graph
    plt.figure(figsize=(10, 6))
    turns = range(1, len(distraction_levels) + 1)
    plt.plot(turns, distraction_levels, marker='o', linestyle='-', color='r')
    plt.xlabel('Turns')
    plt.ylabel('Distraction Level')
    plt.title('Distraction Throughout Conversation')
    plt.grid(False)
    plt.tight_layout()

    # Save plot as image
    plt.savefig('distraction_graph.png')
    plt.close()

    # Display the generated image
    output.put_image(open('distraction_graph.png', 'rb').read(), width='80%')

# PyWebIO chatbot interface
def chatbot_app():
    output.put_html("<div style='background-color: white; padding: 20px;'> <h1>Emotion Depicter</h1> </div>")

    def handle_text_input():
        user_input = input("User:", placeholder="Type here...")
        conversation_outputs.append(user_input)  # Save conversation output
        handle_input(user_input)
        generate_probability_distribution_graph()

    def generate_graph():
        generate_distraction_graph()

    output.put_buttons(['Please Write', 'Generate Distraction Graph'], onclick=[handle_text_input, generate_graph])

if __name__ == "__main__":
    start_server(chatbot_app, port=8082)
