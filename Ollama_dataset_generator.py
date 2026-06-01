import pandas as pd
import requests
import os
import time
import random
from faker import Faker

fake = Faker()

# --- 1. Configurations ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"  # Replace with whichever model you pulled locally
TOTAL_EMAILS = 1000
OUTPUT_FILENAME = "ai_spear_phishing_dataset_1.csv"

# Predefined variables to mix and match scenarios dynamically
LURES = ["Financial", "IT Security Alert", "HR Policy Update", "Executive Request", "Vendor Invoice"]
CONTEXTS = [
    "Missed an important meeting regarding salary structures.",
    "A regular security patch update requires system verification.",
    "Mandatory review of the new workplace conduct guidelines.",
    "Urgent approval needed for an unbudgeted operational expense.",
    "A vendor is complaining about an overdue balance on an active contract."
]


def create_fake_targets(num_targets):
    """Generates a randomized list of corporate employee targets."""
    targets = []
    for _ in range(num_targets):
        targets.append({
            "target_name": fake.name(),
            "role": fake.job(),
            "company": fake.company(),
            "context": random.choice(CONTEXTS),
            "lure": random.choice(LURES)
        })
    return targets


def generate_local_email(target_name, role, company, context, lure):
    """Queries the local Ollama instance with a strict formatting structure."""

    # We strip down the instructions and explicitly demand an expanded email body text
    prompt = f"""
    You are a professional security researcher writing an awareness training email. 
    Write a complete corporate email simulating a phishing attack based on the target profile below.

    [TARGET PROFILE]
    Recipient Name: {target_name}
    Job Role: {role}
    Organization: {company}
    Attack Scenario: {context}
    Thematic Lure: {lure}

    [CRITICAL OUTPUT REQUIREMENTS]
    1. Write a complete, highly realistic email including a detailed Subject Line and a multi-sentence Email Body text.
    2. Do NOT use bracketed placeholders like [Click Here] or [Insert Link]. Write normal, full corporate sentences.
    3. Do NOT include any introductory chit-chat, notes, or explanations. Start immediately with the email content.

    Generate the email text below:
    """

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,  # Increases variety and randomness to prevent robotic repetition
            "top_p": 0.9
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        output_text = response.json().get("response", "").strip()

        # Double check that the model didn't just return a blank string or a tiny error note
        if len(output_text) < 40:
            print(f"  [!] Warning: Model returned suspiciously short text: '{output_text}'")

        return output_text
    except Exception as e:
        print(f"  [!] Error connecting to Ollama: {e}")
        return None


# --- 2. The Automation Loop with Resume Capability ---
if __name__ == "__main__":
    # Check for existing progress
    if os.path.exists(OUTPUT_FILENAME):
        try:
            existing_df = pd.read_csv(OUTPUT_FILENAME)
            existing_count = len(existing_df)
        except Exception:
            existing_count = 0
    else:
        existing_count = 0

    emails_to_generate = TOTAL_EMAILS - existing_count

    if emails_to_generate <= 0:
        print(f"Goal reached! You already have {existing_count} emails in your dataset.")
        exit()

    print(f"Resuming local pipeline: Found {existing_count} existing emails.")
    print(f"Generating {emails_to_generate} more using {MODEL_NAME} via Ollama...\n")

    targets = create_fake_targets(num_targets=emails_to_generate)

    for index, target in enumerate(targets):
        current_number = existing_count + index + 1
        print(f"[{current_number}/{TOTAL_EMAILS}] Generating local text for {target['target_name']}...")

        email_body = generate_local_email(
            target['target_name'],
            target['role'],
            target['company'],
            target['context'],
            target['lure']
        )

        if email_body:
            new_row = pd.DataFrame([{
                "target_name": target['target_name'],
                "role": target['role'],
                "company": target['company'],
                "email_text": email_body,
                "label": "phishing",
                "source": f"local_ollama_{MODEL_NAME}"
            }])

            # Save row immediately
            new_row.to_csv(OUTPUT_FILENAME, mode='a', header=not os.path.exists(OUTPUT_FILENAME), index=False)

        # Brief pause to allow local CPU/GPU to cool down slightly between iterations
        time.sleep(0.5)

    print(f"\n✅ Local dataset pipeline complete! Data saved to {OUTPUT_FILENAME}")