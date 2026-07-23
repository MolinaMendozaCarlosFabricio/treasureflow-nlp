from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained("model_artifacts/modelo_moderacion_final")
tokenizer = AutoTokenizer.from_pretrained("model_artifacts/modelo_moderacion_final")
model.eval()

for texto in ["hola", "Maderita", "Botellas de plástico", "xd", "0w0", "Botellas de plástico", "Este es un texto bastante más largo y normal, con contexto suficiente"]:
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits)[0]
    print(f"{texto!r} -> grosero={probs[0]:.4f}, amenaza={probs[1]:.4f}, inapropiado={probs[2]:.4f}")