from flask import jsonify, request

app = Flask(__name__)

@app.route('/', methods=['POST'])
def home():
    return jsonify({"message": "Mnemosyne Qwen AI agent is Running!"})


@app.route('/health', methods=['POST'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(
        host:"0.0.0.0",
        port:5000,
        debug=True
    )
