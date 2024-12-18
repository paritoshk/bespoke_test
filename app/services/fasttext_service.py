import numpy as np
import os
from typing import List, Tuple
import uuid
import fasttext
import types
import tempfile
from utils.logger import ModelLogger  # Import our logger

# Patch FastText to fix numpy issue
def _patched_predict(self, text, k=1, threshold=0.0, on_unicode_error='strict'):
    def check(entry):
        if entry.find('\n') != -1:
            raise ValueError("predict processes one line at a time (remove '\\n')")
        entry += "\n"
        return entry

    if type(text) == list:
        text = [check(entry) for entry in text]
        all_labels, all_probs = self.f.multilinePredict(text, k, threshold, on_unicode_error)
        return all_labels, np.asarray(all_probs)
    else:
        text = check(text)
        predictions = self.f.predict(text, k, threshold, on_unicode_error)
        if predictions:
            probs, labels = zip(*predictions)
        else:
            probs, labels = ([], ())
        
        return labels, np.asarray(probs)

# Apply the patch to FastText
fasttext.FastText._FastText.predict = _patched_predict

class FastTextService:
    def __init__(self):
        self.models = {}
        self.models_dir = "trained_models"
        self.logger = ModelLogger()  # Initialize logger
        os.makedirs(self.models_dir, exist_ok=True)

    async def train_model(self) -> str:
        """Train a FastText classifier using data from positive/negative examples"""
        try:
            # Log training start
            model_params = {
                'lr': 0.5,
                'epoch': 25,
                'wordNgrams': 2,
                'minCount': 1,
                'loss': 'softmax'
            }
            self.logger.log_training_start(model_params)

            train_data = []
            
            # Load and log data loading
            positive_dir = "data/train/positive"
            self.logger.logger.info(f"Loading positive examples from {positive_dir}")
            for filename in os.listdir(positive_dir):
                with open(os.path.join(positive_dir, filename), 'r') as f:
                    text = f.read().strip()
                    train_data.append(f"__label__positive {text}")
            
            negative_dir = "data/train/negative"
            self.logger.logger.info(f"Loading negative examples from {negative_dir}")
            for filename in os.listdir(negative_dir):
                with open(os.path.join(negative_dir, filename), 'r') as f:
                    text = f.read().strip()
                    train_data.append(f"__label__negative {text}")

            self.logger.logger.info(f"Total training examples: {len(train_data)}")

            # Create temporary training file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:
                for line in train_data:
                    f.write(f"{line}\n")
                training_file = f.name

            try:
                # Train model and log progress
                self.logger.logger.info("Starting model training...")
                model = fasttext.train_supervised(
                    input=training_file,
                    lr=model_params['lr'],
                    epoch=model_params['epoch'],
                    wordNgrams=model_params['wordNgrams'],
                    minCount=model_params['minCount'],
                    loss=model_params['loss'],
                    verbose=2  # Enable verbose output
                )

                # Generate UUID and save model
                model_id = str(uuid.uuid4())
                model_path = os.path.join(self.models_dir, f"{model_id}.bin")
                model.save_model(model_path)
                self.models[model_id] = model

                # Log evaluation metrics
                eval_metrics = self._evaluate_model(model)
                self.logger.log_evaluation(eval_metrics)
                
                # Save training curves
                self.logger.plot_training_curves()
                
                self.logger.logger.info(f"Training completed. Model ID: {model_id}")
                return model_id

            finally:
                if os.path.exists(training_file):
                    os.unlink(training_file)

        except Exception as e:
            self.logger.logger.error(f"Training failed: {str(e)}")
            raise

    def _evaluate_model(self, model):
        """Evaluate model performance"""
        try:
            test_docs = []
            test_labels = []
            
            # Get test documents
            test_positive = "data/train/positive"
            test_negative = "data/train/negative"
            
            # Sample some documents for testing
            num_test_samples = 100
            
            self.logger.logger.info(f"Evaluating on {num_test_samples} samples from each class")
            
            # Get positive test samples
            for filename in list(os.listdir(test_positive))[:num_test_samples]:
                with open(os.path.join(test_positive, filename), 'r') as f:
                    test_docs.append(f.read().strip())
                    test_labels.append("__label__positive")
            
            # Get negative test samples
            for filename in list(os.listdir(test_negative))[:num_test_samples]:
                with open(os.path.join(test_negative, filename), 'r') as f:
                    test_docs.append(f.read().strip())
                    test_labels.append("__label__negative")
            
            # Calculate metrics
            predictions = []
            probabilities = []
            
            for doc in test_docs:
                labels, probs = model.predict(doc)
                predictions.append(labels[0])
                probabilities.append(float(probs[0]))
            
            # Calculate accuracy
            correct = sum(1 for p, t in zip(predictions, test_labels) if p == t)
            accuracy = correct / len(test_docs)
            
            metrics = {
                'accuracy': accuracy,
                'avg_confidence': float(np.mean(probabilities)),
                'num_test_samples': len(test_docs),
                'class_distribution': {
                    'positive': test_labels.count("__label__positive"),
                    'negative': test_labels.count("__label__negative")
                }
            }
            
            return metrics
            
        except Exception as e:
            self.logger.logger.error(f"Evaluation failed: {str(e)}")
            raise

    async def score_documents(self, model_id: str, documents: List[str]) -> List[float]:
        """Score documents using the trained model."""
        try:
            if model_id not in self.models:
                model_path = os.path.join(self.models_dir, f"{model_id}.bin")
                if not os.path.exists(model_path):
                    raise ValueError(f"Model {model_id} not found")
                self.models[model_id] = fasttext.load_model(model_path)

            model = self.models[model_id]
            scores = []
            for doc in documents:
                # Clean the document text
                doc = ' '.join(doc.split())  # normalize whitespace
                labels, probs = model.predict(doc)
                # Get probability for positive class
                score = float(probs[0]) if labels[0] == '__label__positive' else (1.0 - float(probs[0]))
                scores.append(score)

            return scores
        except Exception as e:
            print(f"Scoring failed: {str(e)}")
            raise