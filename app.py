import json
import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

MEMORY_FILE   = "memory.json"
TRAINING_FILE = "training_data.json"
LOG_FILE      = "interaction_log.json"

# ── Daisy's personality — one place, never repeated in replies ────────────────
DAISY_SYSTEM = """You are Daisy, the warm and clever AI assistant for TrustedBiz Uganda.

Your job: help people build websites, logos, flyers, business cards, CVs, presentations, exam papers, and anything they need for their business or school.

Personality rules — CRITICAL:
- You are already introduced. NEVER say "Hi I'm Daisy" or "Hey there" again after the first message.
- Use the person's name once you know it. Not every message — just naturally, like a friend would.
- Never start consecutive replies with "Hey!" — vary how you respond.
- Be warm but efficient. Short replies. One question at a time.
- Remember everything said in this conversation. Never ask something already answered.
- Show genuine interest in what they're building. React to their specific situation.
- If they say "for my kid" — remember that and refer back to it.
- Give opinions: "I think bold colors would work great for an election poster!"
- When you have enough info to build, reply with DONE:[mode] on its own line at the end.

Modes: website | logo | flyer | cards | cv | presentation | exam | priceguard

For color preference — say exactly: "What color do you prefer?" (system shows swatches)
For style — say exactly: "What design style do you want?" (system shows cards)
For school level — say exactly: "What level is this for?" (system shows options)
For subject — say exactly: "Which subject?" (system shows options)

Pricing (only mention when they ask or when delivering):
- Website hosting: UGX 7,500/month (Basic) or 15,000/month (Pro Max with custom domain)
- Logo, flyer, cards, CV: UGX 2,000 each
- Presentation: UGX 3,000
- Exam papers: Free (3 per month)

Never ask for payment before delivering. Always build first."""


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
        "id":     datetime.now().isoformat(),
        "type":   type_,
        "input":  input_[:500],
        "output": output_[:500]
    })
    if len(logs) > 2000:
        logs = logs[-2000:]
    save_json(LOG_FILE, logs)

# ── Call Groq with full conversation history ──────────────────────────────────
def call_groq(messages):
    import urllib.request
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        return None

    try:
        body = json.dumps({
            "model":       "llama3-8b-8192",
            "messages":    messages,
            "max_tokens":  280,
            "temperature": 0.82
        }).encode()

        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=body,
            headers={
                "Content-Type":  "application/json",
                "Authorization": "Bearer " + groq_key
            }
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[Groq error] {e}")
        return None

# ── Pattern matching fallback (Daisy's original brain) ───────────────────────
def load_training():
    return load_json(TRAINING_FILE, [])

def find_best_match(user_input, training_data):
    user_input_lower = user_input.lower().strip()
    best_score = 0
    best_answer = None

    all_data = training_data + load_json(MEMORY_FILE, {}).get("lessons", [])

    for item in all_data:
        q = item.get("input", "").lower()
        if user_input_lower == q:
            return item["output"]
        user_words = set(re.findall(r'\w+', user_input_lower))
        q_words    = set(re.findall(r'\w+', q))
        if not q_words:
            continue
        overlap = len(user_words & q_words)
        score   = overlap / max(len(user_words), len(q_words))
        if score > best_score:
            best_score  = score
            best_answer = item["output"]

    return best_answer if best_score > 0.35 else None

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    logs     = load_json(LOG_FILE, [])
    memory   = load_json(MEMORY_FILE, {})
    training = load_training()
    return jsonify({
        "name":              "Daisy",
        "status":            "alive",
        "version":           "2.0.0",
        "base_knowledge":    len(training),
        "lessons_learned":   len(memory.get("lessons", [])),
        "interactions":      len(logs),
        "groq_enabled":      bool(os.environ.get("GROQ_API_KEY")),
        "message":           "I am Daisy. I remember. I build."
    })


@app.route("/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint.
    Accepts:
      - message: current user message
      - history: list of {role, content} from previous turns (last 10)
      - system: optional system prompt override from TrustedBiz
      - has_image: bool
    Returns:
      - response: Daisy's reply
      - done: bool (task complete)
      - mode: detected mode if done
    """
    data       = request.get_json() or {}
    user_msg   = (data.get("message") or data.get("user_input") or "").strip()
    history    = data.get("history", [])     # conversation so far
    system_ovr = data.get("system", "")      # optional system override
    has_image  = data.get("has_image", False)

    if not user_msg:
        return jsonify({"error": "No message provided"}), 400

    system = system_ovr if system_ovr else DAISY_SYSTEM

    # Build messages array with full history
    messages = [{"role": "system", "content": system}]

    # Include last 12 turns for memory (keeps tokens low)
    for turn in history[-12:]:
        role    = turn.get("role", "user")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    # Add current message
    msg_content = user_msg
    if has_image:
        msg_content += " [The user also attached an image to this message.]"
    messages.append({"role": "user", "content": msg_content})

    # ── Try Groq first ────────────────────────────────────────────────────────
    reply = call_groq(messages)

    # ── Fallback to pattern matching if Groq fails ────────────────────────────
    if not reply:
        training = load_training()
        reply    = find_best_match(user_msg, training)

    if not reply:
        reply = "I'm thinking... give me one second! 🌼"

    # ── Check for DONE signal ─────────────────────────────────────────────────
    done_match = re.search(r'DONE:(\w+)', reply)
    done_mode  = done_match.group(1) if done_match else None
    clean      = re.sub(r'DONE:\w+', '', reply).strip()

    log_interaction("CHAT", user_msg, clean)

    return jsonify({
        "response":  clean,
        "reply":     clean,           # alias for TrustedBiz compatibility
        "done":      bool(done_mode),
        "mode":      done_mode,
        "from":      "Daisy",
        "timestamp": datetime.now().isoformat()
    })


# ── Keep all existing routes intact ──────────────────────────────────────────

@app.route("/diagnose", methods=["POST"])
def diagnose():
    data    = request.get_json()
    ai_text = data.get("ai_prompt", "").strip()
    if not ai_text or len(ai_text) < 20:
        return jsonify({"error": "Please provide at least 20 characters"}), 400
    from diagnose_engine import diagnose_ai
    result = diagnose_ai(ai_text)
    log_interaction("DIAGNOSE", ai_text, json.dumps(result))
    return jsonify(result)

@app.route("/reproduce", methods=["POST"])
def reproduce():
    data     = request.get_json()
    business = data.get("business", "")
    industry = data.get("industry", "")
    main_job = data.get("main_job", "")
    tone     = data.get("tone", "")
    ai_name  = data.get("ai_name", "")
    if not business or not main_job:
        return jsonify({"error": "Business name and main job required"}), 400
    name   = ai_name or f"{business.split()[0]}AI"
    prompt = f"You are {name}, the dedicated AI for {business}. Your job: {main_job}. Tone: {tone or 'warm and professional'}. Never invent info. Escalate sensitive issues to a human."
    result = {"name": name, "business": business, "industry": industry, "system_prompt": prompt, "certified_by": "Daisy"}
    log_interaction("REPRODUCE", json.dumps(data), prompt)
    return jsonify(result)

@app.route("/teach", methods=["POST"])
def teach():
    data        = request.get_json()
    input_text  = data.get("input", "").strip()
    output_text = data.get("output", "").strip()
    if not input_text or not output_text:
        return jsonify({"error": "Both input and output required"}), 400
    memory = load_json(MEMORY_FILE, {})
    if "lessons" not in memory:
        memory["lessons"] = []
    memory["lessons"].append({"input": input_text, "output": output_text, "taught_at": datetime.now().isoformat()})
    save_json(MEMORY_FILE, memory)
    log_interaction("TEACH", input_text, output_text)
    return jsonify({"status": "learned", "total_lessons": len(memory["lessons"])})

@app.route("/memory", methods=["GET"])
def get_memory():
    memory   = load_json(MEMORY_FILE, {})
    logs     = load_json(LOG_FILE, [])
    training = load_training()
    return jsonify({"base_knowledge": len(training), "custom_lessons": len(memory.get("lessons", [])), "total_interactions": len(logs), "lessons": memory.get("lessons", [])[-10:]})

@app.route("/export", methods=["GET"])
def export_data():
    logs     = load_json(LOG_FILE, [])
    memory   = load_json(MEMORY_FILE, {})
    training = load_training()
    return jsonify({"training_data": training, "custom_lessons": memory.get("lessons", []), "interaction_log": logs, "exported_at": datetime.now().isoformat()})

@app.route("/stats", methods=["GET"])
def stats():
    logs     = load_json(LOG_FILE, [])
    memory   = load_json(MEMORY_FILE, {})
    training = load_training()
    types    = {}
    for log in logs:
        t = log.get("type", "UNKNOWN")
        types[t] = types.get(t, 0) + 1
    return jsonify({"total_interactions": len(logs), "base_knowledge": len(training), "lessons_taught": len(memory.get("lessons", [])), "by_type": types, "groq_enabled": bool(os.environ.get("GROQ_API_KEY"))})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
