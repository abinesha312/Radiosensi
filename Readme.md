# Radiosensitivity Prediction System

A machine learning system for predicting radiosensitivity using telomere length data and temporal attention mechanisms.

## Project Overview

This project implements a deep learning-based system for predicting radiosensitivity using temporal telomere length data. The system uses a temporal attention mechanism to analyze patterns in telomere length changes over time and predict radiosensitivity outcomes.

## Features

- Temporal attention-based deep learning model
- Distributed training support
- Docker containerization
- PostgreSQL database integration
- Web-based client interface
- Comprehensive data preprocessing pipeline
- Model checkpointing and visualization

## Project Structure

```
.
├── client/                 # Web client interface
├── data/                   # Data storage and preprocessing
├── database/              # Database configuration and migrations
├── distributed_train/     # Distributed training scripts
├── models/                # Core ML models and training code
├── plots/                 # Visualization outputs
├── server/                # Backend server code
├── checkpoints/           # Model checkpoints
├── docker-compose.yml     # Docker compose configuration
├── requirements.txt       # Python dependencies
└── train_cpu.py          # CPU training script
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/abinesha312/Radiosensi.git
cd Radiosensi
```

2. Create and activate a virtual environment:

```bash
python -m venv radiovenv
source radiovenv/bin/activate  # On Windows: radiovenv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

### Training the Model

For CPU training:

```bash
python train_cpu.py
```

For distributed training:

```bash
docker-compose -f docker-compose-distributed.yml up
```

### Data Preprocessing

To split large telomere data files:

```bash
python split_large_telomere_data.py
```

## Docker Setup

1. Build the containers:

```bash
docker-compose build
```

2. Start the services:

```bash
docker-compose up
```

## Database

The system uses PostgreSQL for data storage. Configuration is handled through environment variables in the `.env` file.

## Model Architecture

The core model uses a temporal attention mechanism to analyze telomere length changes over time. Key components include:

- Temporal attention layers
- Feature extraction
- Classification head
- Loss functions: BCEWithLogitsLoss
- Optimizer: Adam

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Telomere data providers
- Research collaborators
- Open source community

## Contact

For questions and support, please contact the project maintainers.
