# Evaluate.py
import torch
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import io

def evaluate(model, loader, device, experiment_name="ImageClassificationEval"):
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Create and save confusion matrix plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Save plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Classification report (dict form for MLflow metrics)
    report_dict = classification_report(all_labels, all_preds, output_dict=True)
    report_text = classification_report(all_labels, all_preds)

    # Log everything to MLflow
    with mlflow.start_run():
        # Log metrics (precision, recall, f1-score, etc.)
        for label, metrics in report_dict.items():
            if isinstance(metrics, dict):  # skip 'accuracy' key which is float
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric_name}", value)
            else:
                mlflow.log_metric(label, metrics)

        # Log confusion matrix image
        mlflow.log_figure(plt.gcf(), "confusion_matrix.png")

        # Log full classification report as an artifact text file
        with open("classification_report.txt", "w") as f:
            f.write(report_text)
        mlflow.log_artifact("classification_report.txt")

        # Log model itself
        mlflow.pytorch.log_model(model, artifact_path="model")

    # Show confusion matrix locally too
    plt.show()
    print(report_text)
