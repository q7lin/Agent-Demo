from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os

from TTSAssistant import TTSAssistant, identify_intent, TEACHER_MAP, assistant_cache

app = Flask(__name__)
CORS(app)

@app.route('/ask/', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # 学科识别
    intent = identify_intent(question)
    config = TEACHER_MAP[intent]
    name = config["name"]
    path = config["text_path"]

    # 判断是否已经创建过该学科智能体
    if intent not in assistant_cache:
        assistant = TTSAssistant(name)
        assistant.split_sentence(path)
        assistant_cache[intent] = assistant
    else:
        assistant = assistant_cache[intent]

    # 回答问题
    answer = assistant.selfChain(question)
    return jsonify({"answer": answer})

@app.route('/audio/', methods=['GET'])
def audio():
    # 默认从所有老师中获取最近生成的音频
    latest_audio = None
    for a in assistant_cache.values():
        if os.path.exists(a.audio_path):
            latest_audio = a.audio_path
    if latest_audio:
        return send_file(latest_audio, mimetype="audio/mpeg")
    return "Audio not found", 404

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
