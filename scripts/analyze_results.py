from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def analyze_training_results():
    """Analyze and visualize the latest training results"""
    # Find latest log file
    log_dir = Path("logs")
    metrics_file = max(log_dir.glob("metrics*.json"), key=lambda x: x.stat().st_mtime)
    
    with open(metrics_file) as f:
        metrics = json.load(f)
    
    # Create output directory for plots
    plots_dir = log_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # 1. Training Configuration
    print("Training Configuration:")
    print(json.dumps(metrics['model_params'], indent=2))
    
    # 2. Model Performance
    print("\nModel Performance:")
    print(json.dumps(metrics['eval_metrics'], indent=2))
    
    # 3. Create visualizations
    plt.style.use('seaborn')
    
    # Loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['train_loss'], marker='o')
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(plots_dir / 'loss_curve.png')
    plt.close()
    
    # Class distribution
    class_dist = metrics['eval_metrics']['class_distribution']
    plt.figure(figsize=(8, 6))
    plt.bar(class_dist.keys(), class_dist.values())
    plt.title('Test Set Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.savefig(plots_dir / 'class_distribution.png')
    plt.close()
    
    # Create summary report
    report = f"""
    Training Summary Report
    ----------------------
    
    Model Configuration:
    - Learning Rate: {metrics['model_params']['lr']}
    - Epochs: {metrics['model_params']['epoch']}
    - Word N-grams: {metrics['model_params']['wordNgrams']}
    
    Performance Metrics:
    - Accuracy: {metrics['eval_metrics']['accuracy']:.4f}
    - Average Confidence: {metrics['eval_metrics']['avg_confidence']:.4f}
    - Test Samples: {metrics['eval_metrics']['num_test_samples']}
    
    Class Distribution:
    - Positive: {class_dist['positive']}
    - Negative: {class_dist['negative']}
    """
    
    with open(plots_dir / 'summary_report.txt', 'w') as f:
        f.write(report)
    
    print("\nAnalysis completed! Check the 'logs/plots' directory for visualizations.")

if __name__ == "__main__":
    analyze_training_results()