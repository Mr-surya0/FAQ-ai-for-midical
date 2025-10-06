from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

app = Flask(__name__)
CORS(app)

# ---- Load MedAlpaca model ----

MODEL_NAME = "medalpaca/medalpaca-7b"

# Only load model once (not on reloader restart)
if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
    print("Loading MedAlpaca model... please wait, it may take a few minutes.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    print("Model loaded successfully!")

# ---- Chat endpoint ----

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("message")

        if not user_input:
            return jsonify({"error": "No input provided"}), 400

        # Format prompt for MedAlpaca (instruction format)
        formatted_prompt = f"### Instruction:\n{user_input}\n\n### Response:"

        # Prepare input
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

        # Generate response
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the response part (remove the prompt)
        response_text = response_text.split("### Response:")[-1].strip()

        return jsonify({"response": response_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---- Run server ----

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)  # Set debug=False