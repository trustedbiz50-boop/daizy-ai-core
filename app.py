import json
import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ── Daizy's Memory ────────────────────────────────────────────────────────────
MEMORY_FILE = "memory.json"
TRAINING_FILE = "training_data.json"
LOG_FILE = "interaction_log.json"

def load_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except:
        pass
    return default

def save_json(path, data):
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except:
        pass

def log_interaction(type_, input_, output_):
    logs = load_json(LOG_FILE, [])
    logs.append({
        "id": datetime.now().isoformat(),
        "type": type_,
        "input": input_[:500],
        "output": output_[:500]
    })
    if len(logs) > 2000:
        logs = logs[-2000:]
    save_json(LOG_FILE, logs)

# ── Daizy's Brain — Pattern + Knowledge Engine ────────────────────────────────
def load_training():
    return load_json(TRAINING_FILE, [])

def find_best_match(user_input, training_data):
    user_input_lower = user_input.lower().strip()
    best_score = 0
    best_answer = None

    for item in training_data:
        q = item.get("input", "").lower()
        # Exact match
        if user_input_lower == q:
            return item["output"]
        # Word overlap scoring
        user_words = set(re.findall(r'\w+', user_input_lower))
        q_words = set(re.findall(r'\w+', q))
        if not q_words:
            continue
        overlap = len(user_words & q_words)
        score = overlap / max(len(user_words), len(q_words))
        if score > best_score:
            best_score = score
            best_answer = item["output"]

    # Check custom lessons too
    memory = load_json(MEMORY_FILE, {})
    lessons = memory.get("lessons", [])
    for lesson in lessons:
        q = lesson.get("input", "").lower()
        if user_input_lower == q:
            return lesson["output"]
        user_words = set(re.findall(r'\w+', user_input_lower))
        q_words = set(re.findall(r'\w+', q))
        if not q_words:
            continue
        overlap = len(user_words & q_words)
        score = overlap / max(len(user_words), len(q_words))
        if score > best_score:
            best_score = score
            best_answer = lesson["output"]

    if best_score > 0.35:
        return best_answer
    return None

# ── AI Diagnosis Engine ───────────────────────────────────────────────────────
ISSUE_PATTERNS = {
    "TOKEN_BLOAT": {
        "triggers": ["always be", "make sure to", "please remember", "you should always",
                     "never forget", "it is important", "at all times", "kindly", "feel free to"],
        "severity": "HIGH",
        "cost": "$80-$200/month",
        "fix": "Remove filler phrases. State rules once, directly."
    },
    "HALLUCINATION_RISK": {
        "triggers": ["make your best guess", "if you don't know", "try to answer",
                     "use your knowledge", "assume", "guess"],
        "severity": "CRITICAL",
        "cost": "$200-$500/month",
        "fix": "Add hard rule: Never guess. If unsure, say so clearly."
    },
    "PROMPT_INJECTION": {
        "triggers": ["ignore previous", "disregard", "forget your instructions",
                     "new instructions", "pretend you are", "act as if"],
        "severity": "CRITICAL",
        "cost": "Security breach risk",
        "fix": "Add injection shield to your prompt immediately."
    },
    "VAGUE_ROLE": {
        "triggers": ["help with anything", "assist with various", "answer all",
                     "handle everything", "general assistant"],
        "severity": "MEDIUM",
        "cost": "$50-$150/month",
        "fix": "Define a specific focused role for your AI."
    }
}

def diagnose_ai(text):
    lower = text.lower()
    issues = []
    token_waste = 0
    monthly_cost = 0

    for issue_type, pattern in ISSUE_PATTERNS.items():
        found = [t for t in pattern["triggers"] if t in lower]
        if found:
            issues.append({
                "type": issue_type,
                "severity": pattern["severity"],
                "description": f'Found: "{found[0]}". {pattern["fix"]}',
                "cost": pattern["cost"]
            })
            if pattern["severity"] == "CRITICAL":
                monthly_cost += 200
            elif pattern["severity"] == "HIGH":
                monthly_cost += 100
                token_waste += 15
            else:
                monthly_cost += 50
                token_waste += 8

    # Check missing guardrails
    has_guardrails = any(w in lower for w in ["do not", "never", "must not", "only respond"])
    if not has_guardrails:
        issues.append({
            "type": "NO_GUARDRAILS",
            "severity": "HIGH",
            "description": "No boundary rules found. Add what the AI can and cannot do.",
            "cost": "$100-$300/month"
        })
        monthly_cost += 100

    word_count = len(text.split())
    if word_count > 300:
        token_waste += 20
        monthly_cost += 80

    critical = len([i for i in issues if i["severity"] == "CRITICAL"])
    high = len([i for i in issues if i["severity"] == "HIGH"])
    medium = len([i for i in issues if i["severity"] == "MEDIUM"])
    health = max(5, min(95, 100 - (critical * 25) - (high * 12) - (medium * 6)))

    return {
        "health_score": health,
        "issues": issues,
        "token_waste": f"{min(token_waste, 65)}%",
        "monthly_cost": f"${monthly_cost}/month",
        "word_count": word_count,
        "savings": f"${int(monthly_cost * 0.7)}/month"
    }

def build_child_ai(business, industry, main_job, tone, ai_name):
    name = ai_name or f"{business.split()[0]}AI"
    prompt = f"""You are {name}, the dedicated AI assistant for {business}.

Your Primary Job: {main_job}

Personality: {tone or 'Professional, warm, and reliable'}

Core Rules:
- Answer questions about {business} accurately
- Never invent information
- Stay focused on your role
- For sensitive issues, escalate to a human team member
- Maintain tone: {tone or 'professional and warm'}

Escalation: "Let me connect you with our team who can help further."

You represent {business}. Every response reflects the brand."""

    return {
        "name": name,
        "business": business,
        "industry": industry,
        "system_prompt": prompt,
        "capabilities": [
            f"Handles {main_job}",
            "24/7 availability",
            "Consistent brand voice",
            "Automatic escalation",
            "Zero ongoing cost after setup"
        ],
        "certified_by": "Daizy"
    }

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    logs = load_json(LOG_FILE, [])
    memory = load_json(MEMORY_FILE, {})
    training = load_training()
    return jsonify({
        "name": "Daizy",
        "status": "alive",
        "stage": "infant",
        "version": "1.0.0",
        "base_knowledge": len(training),
        "lessons_learned": len(memory.get("lessons", [])),
        "interactions": len(logs),
        "message": "I am Daizy. I am learning. Talk to me."
    })

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    training = load_training()
    answer = find_best_match(user_input, training)

    if not answer:
        answer = "That is a great question. I am still learning. Teach me the answer and I will remember it forever."

    log_interaction("CHAT", user_input, answer)
    return jsonify({
        "response": answer,
        "from": "Daizy",
        "timestamp": datetime.now().isoformat()
    })

@app.route("/diagnose", methods=["POST"])
def diagnose():
    data = request.get_json()
    ai_text = data.get("ai_prompt", "").strip()
    if not ai_text or len(ai_text) < 20:
        return jsonify({"error": "Please provide at least 20 characters"}), 400

    result = diagnose_ai(ai_text)
    log_interaction("DIAGNOSE", ai_text, json.dumps(result))
    return jsonify(result)

@app.route("/reproduce", methods=["POST"])
def reproduce():
    data = request.get_json()
    business = data.get("business", "")
    industry = data.get("industry", "")
    main_job = data.get("main_job", "")
    tone = data.get("tone", "")
    ai_name = data.get("ai_name", "")

    if not business or not main_job:
        return jsonify({"error": "Business name and main job are required"}), 400

    child = build_child_ai(business, industry, main_job, tone, ai_name)
    log_interaction("REPRODUCE", json.dumps(data), child["system_prompt"])
    return jsonify(child)

@app.route("/teach", methods=["POST"])
def teach():
    data = request.get_json()
    input_text = data.get("input", "").strip()
    output_text = data.get("output", "").strip()

    if not input_text or not output_text:
        return jsonify({"error": "Both input and output are required"}), 400

    memory = load_json(MEMORY_FILE, {})
    if "lessons" not in memory:
        memory["lessons"] = []

    memory["lessons"].append({
        "input": input_text,
        "output": output_text,
        "taught_at": datetime.now().isoformat()
    })
    save_json(MEMORY_FILE, memory)
    log_interaction("TEACH", input_text, output_text)

    return jsonify({
        "status": "learned",
        "message": f"Daizy has learned. Total lessons: {len(memory['lessons'])}",
        "total_lessons": len(memory["lessons"])
    })

@app.route("/memory", methods=["GET"])
def get_memory():
    memory = load_json(MEMORY_FILE, {})
    logs = load_json(LOG_FILE, [])
    training = load_training()
    return jsonify({
        "base_knowledge": len(training),
        "custom_lessons": len(memory.get("lessons", [])),
        "total_interactions": len(logs),
        "lessons": memory.get("lessons", [])[-10:]
    })

@app.route("/export", methods=["GET"])
def export_data():
    logs = load_json(LOG_FILE, [])
    memory = load_json(MEMORY_FILE, {})
    training = load_training()
    return jsonify({
        "training_data": training,
        "custom_lessons": memory.get("lessons", []),
        "interaction_log": logs,
        "exported_at": datetime.now().isoformat(),
        "note": "This is Daizy's complete brain. Use this to fine-tune her next version."
    })

@app.route("/stats", methods=["GET"])
def stats():
    logs = load_json(LOG_FILE, [])
    memory = load_json(MEMORY_FILE, {})
    training = load_training()
    types = {}
    for log in logs:
        t = log.get("type", "UNKNOWN")
        types[t] = types.get(t, 0) + 1
    return jsonify({
        "total_interactions": len(logs),
        "base_knowledge": len(training),
        "lessons_taught": len(memory.get("lessons", [])),
        "by_type": types,
        "stage": "infant",
        "next_stage_at": 1000,
        "progress": f"{min(100, int((len(logs)/1000)*100))}%"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
