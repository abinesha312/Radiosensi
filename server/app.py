# server/app.py
from flask import Flask, request
from hybrid_trainer import HybridTrainer

app = Flask(__name__)
trainer = HybridTrainer()

@app.route('/api/train', methods=['POST'])
def train_model():
    config = request.json
    trainer.update_parameters(
        lr=config['learningRate'],
        batch_size=config['batchSize']
    )
    results = trainer.run_epoch()
    return jsonify(results)

@app.route('/api/predict', methods=['POST'])
def predict():
    patient_data = request.json
    prediction = trainer.model.predict(patient_data)
    return jsonify({
        'radiosensitivity': prediction,
        'confidence': trainer.get_confidence()
    })
