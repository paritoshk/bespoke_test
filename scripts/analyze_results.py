import logging
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import re

class ModelLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger("FastTextService")
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(self.log_dir / f"training_{timestamp}.log")
        fh.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # Metrics storage
        self.metrics = {
            'train_loss': [],
            'eval_metrics': {},
            'model_params': None
        }

    def log_training_start(self, model_params):
        """Log training parameters"""
        self.metrics['model_params'] = model_params
        self.logger.info(f"Training started with parameters: {json.dumps(model_params, indent=2)}")
        self.save_metrics()

    def parse_fasttext_progress(self, line: str):
        """Parse FastText's progress output to extract loss"""
        if "Progress" in line and "loss" in line:
            match = re.search(r'avg\.loss:\s+([0-9.]+)', line)
            if match:
                loss = float(match.group(1))
                self.metrics['train_loss'].append(loss)
                self.save_metrics()

    def log_evaluation(self, metrics):
        """Log evaluation metrics"""
        self.metrics['eval_metrics'] = metrics
        self.logger.info(f"Evaluation metrics: {json.dumps(metrics, indent=2)}")
        self.save_metrics()

    def save_metrics(self):
        """Save all metrics to JSON"""
        with open(self.log_dir / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def plot_training_curves(self):
        """Plot and save training curves"""
        if not self.metrics['train_loss']:
            self.logger.warning("No loss data to plot")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['train_loss'], label='Training Loss')
        plt.title('Training Loss Curve')
        plt.xlabel('Progress')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.log_dir / 'training_curve.png')
        plt.close()