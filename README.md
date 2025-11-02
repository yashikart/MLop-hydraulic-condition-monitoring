# MLop-hydraulic-condition-monitoring

Advanced Hydraulic System AI Monitor - A comprehensive predictive maintenance and condition monitoring platform with MLflow integration and Docker support.

## 🚀 Features

- **Real-time Monitoring**: Live system health tracking and alerts
- **Machine Learning Models**: Multiple ML models for condition prediction
- **Explainable AI (XAI)**: Model interpretability and feature importance
- **MLflow Integration**: Complete experiment tracking and model versioning
- **Maintenance Advisor**: AI-powered recommendations
- **Interactive Chatbot**: Maintenance Q&A assistant
- **Docker Support**: Containerized deployment
- **Streamlit Cloud Ready**: Optimized for cloud deployment

## 📋 Requirements

- Python 3.10+
- See `requirements.txt` for all dependencies

## 🛠️ Installation

### Local Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MLop-hydraulic-condition-monitoring
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

### Docker Installation

1. Build the Docker image:
```bash
docker build -t hydraulic-monitor .
```

2. Run the container:
```bash
docker run -p 8501:8501 hydraulic-monitor
```

For persistent MLflow tracking, mount a volume:
```bash
docker run -p 8501:8501 -v $(pwd)/mlruns:/app/mlruns hydraulic-monitor
```

### Docker Compose (Optional)

Create a `docker-compose.yml` file:

```yaml
version: '3.8'
services:
  hydraulic-monitor:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./mlruns:/app/mlruns
    environment:
      - MLFLOW_TRACKING_URI=./mlruns
```

Then run:
```bash
docker-compose up
```

## 📊 MLflow Integration

This application uses MLflow for experiment tracking and model versioning.

### MLflow Features

- **Automatic Experiment Tracking**: All model training runs are automatically logged
- **Model Versioning**: Track different model versions and their performance
- **Metrics Logging**: Accuracy, precision, recall, F1-score, and training time
- **Artifact Storage**: Models and confusion matrices are saved as artifacts
- **Experiment Management**: View and compare experiments in the MLOps Management page

### MLflow Configuration

By default, MLflow uses a local file-based tracking store (`./mlruns`), which works seamlessly on Streamlit Cloud.

To use a remote MLflow tracking server:

```bash
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
```

Or set it in the environment:
```python
# In app.py, update:
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', './mlruns')
```

### Viewing MLflow Experiments

1. **In the App**: Navigate to "⚙️ MLOps Management" page to see recent experiments
2. **MLflow UI**: Run `mlflow ui` in the project directory to access the full MLflow interface:
```bash
mlflow ui --backend-store-uri ./mlruns
```
Then visit `http://localhost:5000`

### What Gets Logged

For each model training run, MLflow logs:
- **Parameters**: Model hyperparameters, data dimensions
- **Metrics**: Training time, test accuracy, precision, recall, F1-score
- **Artifacts**: Trained models (sklearn format), confusion matrices
- **Metadata**: Run name, timestamp, experiment name

## 🐳 Docker Deployment

### Building the Image

```bash
docker build -t hydraulic-monitor:latest .
```

### Running the Container

Basic run:
```bash
docker run -p 8501:8501 hydraulic-monitor:latest
```

With volume mounting for persistent MLflow data:
```bash
docker run -p 8501:8501 \
  -v $(pwd)/mlruns:/app/mlruns \
  -v $(pwd)/pretrained_models:/app/pretrained_models \
  hydraulic-monitor:latest
```

### Environment Variables

You can customize the deployment using environment variables:

```bash
docker run -p 8501:8501 \
  -e MLFLOW_TRACKING_URI=./mlruns \
  -e STREAMLIT_SERVER_PORT=8501 \
  hydraulic-monitor:latest
```

## ☁️ Streamlit Cloud Deployment

This application is optimized for Streamlit Cloud deployment:

1. Push your code to GitHub
2. Connect your repository to Streamlit Cloud
3. The app will automatically:
   - Install dependencies from `requirements.txt`
   - Use file-based MLflow tracking (works without external services)
   - Start the Streamlit app

### Streamlit Cloud Considerations

- MLflow uses local file storage (`./mlruns`) which persists during the session
- For production, consider using a remote MLflow tracking server or cloud storage
- The `.mlflow_temp` directory is used for temporary artifact storage

## 📁 Project Structure

```
MLop-hydraulic-condition-monitoring/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── .dockerignore            # Docker ignore patterns
├── README.md               # This file
├── pretrained_models/      # Pre-trained model files
│   ├── decision_tree.joblib
│   ├── gradient_boosting.joblib
│   ├── logistic_regression.joblib
│   ├── metadata.joblib
│   ├── random_forest.joblib
│   └── svm.joblib
└── mlruns/                 # MLflow tracking data (created at runtime)
```

## 🎯 Usage Guide

### 1. Load Dataset
- Click "📥 Load Dataset" in the sidebar
- Dataset contains 12,000 synthetic hydraulic system records

### 2. Train Models
- Navigate to "🧠 Model Training"
- Select problem type (classification or anomaly detection)
- Click "🚀 Train All Models"
- Models are automatically tracked in MLflow

### 3. View Experiments
- Go to "⚙️ MLOps Management"
- See all MLflow experiments and metrics
- Compare model performance

### 4. Make Predictions
- Navigate to "🔍 Live Monitoring"
- Input system parameters
- Get real-time condition predictions

### 5. Maintenance Planning
- Check "🧾 Maintenance Advisor" for recommendations
- Use "💬 AI Chatbot" for maintenance Q&A

## 🔧 Configuration

### MLflow Settings

Edit MLflow configuration in `app.py`:

```python
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', './mlruns')
mlflow.set_experiment("hydraulic_system_monitoring")
```

### Model Settings

Adjust model parameters in `AdvancedMLSystem` class:

```python
self.classification_models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    # ... other models
}
```

## 📈 MLflow Best Practices

1. **Experiment Organization**: Each training session creates a new run with timestamp
2. **Model Comparison**: Use the MLOps Management page to compare runs
3. **Best Model Selection**: The app automatically identifies the best performing model
4. **Artifact Management**: Models are saved in MLflow format for easy retrieval

## 🐛 Troubleshooting

### MLflow Issues

**Issue**: MLflow tracking not working
- **Solution**: Ensure `mlruns` directory has write permissions
- Check `MLFLOW_TRACKING_URI` environment variable

**Issue**: Cannot view MLflow UI
- **Solution**: Run `mlflow ui --backend-store-uri ./mlruns` in project directory

### Docker Issues

**Issue**: Container won't start
- **Solution**: Check port 8501 is not already in use
- Verify Dockerfile syntax

**Issue**: MLflow data not persisting
- **Solution**: Mount a volume: `-v $(pwd)/mlruns:/app/mlruns`

### Streamlit Cloud Issues

**Issue**: App crashes on Streamlit Cloud
- **Solution**: Check all dependencies in `requirements.txt`
- Ensure MLflow uses relative paths (default behavior)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

See LICENSE file for details.

## 🔗 Links

- **Live App**: https://mlop-hydraulic-condition-monitoring-mrxtbbkuozcqp8yjzkfmvx.streamlit.app/
- **MLflow Docs**: https://www.mlflow.org/docs/latest/index.html
- **Streamlit Docs**: https://docs.streamlit.io/

## 📧 Contact

For issues or questions, please open an issue on GitHub.
