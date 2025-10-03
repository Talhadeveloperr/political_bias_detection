import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load saved model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_encoder.classes_))
model.load_state_dict(torch.load("bert_bias_model.pt", map_location=device))
model.to(device)
model.eval()

def predict_bias(text):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
    
    # Convert back to label
    label = label_encoder.inverse_transform(predictions)[0]
    return label

if __name__ == "__main__":
    while True:
        user_input = input("Enter a political news text (or 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        print("Prediction:", predict_bias(user_input))
