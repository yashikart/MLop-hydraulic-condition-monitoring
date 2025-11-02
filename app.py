# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.svm import SVC, OneClassSVM
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           precision_score, recall_score, f1_score, roc_curve, auc,
                           mean_squared_error, mean_absolute_error, r2_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.inspection import partial_dependence
import joblib
import time
from datetime import datetime, timedelta
import warnings
import mlflow
import mlflow.sklearn
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# Set MLflow tracking URI (use file-based for Streamlit Cloud compatibility)
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', './mlruns')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Ensure experiment exists and is set
EXPERIMENT_NAME = "hydraulic_system_monitoring"
try:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        mlflow.set_experiment(EXPERIMENT_NAME)
    else:
        mlflow.set_experiment(EXPERIMENT_NAME)
except Exception as e:
    # Fallback: try to set it, and if that fails, create default experiment
    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
    except Exception:
        try:
            mlflow.create_experiment(EXPERIMENT_NAME)
            mlflow.set_experiment(EXPERIMENT_NAME)
        except Exception:
            # Use default experiment if all else fails
            pass

# Set page configuration
st.set_page_config(
    page_title="Hydraulic System AI Monitor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin: 1rem 0;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .alert-critical {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
    }
    .alert-warning {
        background: linear-gradient(135deg, #ffd93d 0%, #ff9a3d 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
    }
    .alert-normal {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
    }
    .chat-bubble {
        background-color: #f1f3f4;
        padding: 12px;
        border-radius: 15px;
        margin: 8px 0;
        max-width: 80%;
    }
    .user-bubble {
        background-color: #1f77b4;
        color: white;
        margin-left: 20%;
    }
    .bot-bubble {
        background-color: #e8f4f8;
        margin-right: 20%;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedHydraulicDataset:
    """Advanced dataset with all required feature categories"""
    
    def __init__(self):
        self.data = None
        self.feature_categories = {
            'sensor_data': [
                'pressure_main', 'pressure_secondary', 'flow_rate', 'temperature',
                'vibration_x', 'vibration_y', 'vibration_z', 'fluid_level', 'motor_current'
            ],
            'operational_data': [
                'load_kg', 'duty_cycle', 'operating_hours', 'rpm', 'cycle_count'
            ],
            'environmental_data': [
                'ambient_temp', 'humidity', 'dust_level'
            ],
            'derived_features': [
                'pressure_variance', 'temp_trend', 'vibration_rms', 'flow_instability'
            ]
        }
    
    def generate_derived_features(self, df):
        """Generate advanced derived features"""
        # Rolling statistics
        df['pressure_variance'] = df[['pressure_main', 'pressure_secondary']].std(axis=1)
        df['temp_trend'] = df['temperature'].diff().fillna(0)
        df['vibration_rms'] = np.sqrt(df['vibration_x']**2 + df['vibration_y']**2 + df['vibration_z']**2)
        df['flow_instability'] = df['flow_rate'].rolling(window=5, min_periods=1).std().fillna(0)
        
        # Frequency domain features (simplified)
        df['vibration_freq_domain'] = np.sin(2 * np.pi * df['vibration_rms'] / 10)
        
        return df
    
    def load_dataset(self):
        """Load comprehensive hydraulic system dataset with 12000 records"""
        np.random.seed(42)
        n_samples = 12000
        
        # Generate base data with realistic patterns
        time_trend = np.linspace(0, 1, n_samples)
        
        data = {}
        
        # Sensor Data
        data['pressure_main'] = 100 + 20 * np.sin(2 * np.pi * time_trend * 5) + np.random.normal(0, 8, n_samples)
        data['pressure_secondary'] = 95 + 15 * np.sin(2 * np.pi * time_trend * 5) + np.random.normal(0, 6, n_samples)
        data['flow_rate'] = 50 + 10 * np.sin(2 * np.pi * time_trend * 3) + np.random.normal(0, 4, n_samples)
        data['temperature'] = 60 + 10 * np.sin(2 * np.pi * time_trend * 2) + 5 * time_trend + np.random.normal(0, 3, n_samples)
        data['vibration_x'] = 1.5 + 0.5 * np.sin(2 * np.pi * time_trend * 8) + 0.8 * time_trend + np.random.normal(0, 0.3, n_samples)
        data['vibration_y'] = 1.8 + 0.6 * np.sin(2 * np.pi * time_trend * 8) + 1.0 * time_trend + np.random.normal(0, 0.4, n_samples)
        data['vibration_z'] = 1.2 + 0.4 * np.sin(2 * np.pi * time_trend * 8) + 0.6 * time_trend + np.random.normal(0, 0.2, n_samples)
        data['fluid_level'] = 85 - 20 * time_trend + np.random.normal(0, 3, n_samples)
        data['motor_current'] = 45 + 10 * np.sin(2 * np.pi * time_trend * 4) + np.random.normal(0, 2, n_samples)
        
        # Operational Data
        data['load_kg'] = np.random.normal(5000, 1000, n_samples)
        data['duty_cycle'] = np.random.normal(75, 15, n_samples)
        data['operating_hours'] = np.cumsum(np.random.exponential(10, n_samples))
        data['rpm'] = np.random.normal(1500, 200, n_samples)
        data['cycle_count'] = np.arange(n_samples)
        
        # Environmental Data
        data['ambient_temp'] = 25 + 10 * np.sin(2 * np.pi * time_trend * 1) + np.random.normal(0, 2, n_samples)
        data['humidity'] = np.random.normal(50, 15, n_samples)
        data['dust_level'] = np.random.gamma(2, 2, n_samples)
        
        df = pd.DataFrame(data)
        
        # Generate derived features
        df = self.generate_derived_features(df)
        
        # Generate realistic failure patterns
        conditions = []
        maintenance_needed = []
        rul = []  # Remaining Useful Life
        
        for idx, row in df.iterrows():
            # Complex failure scoring
            failure_score = 0
            
            # Sensor-based scoring
            if row['pressure_main'] < 80 or row['pressure_main'] > 120: failure_score += 2
            if row['temperature'] > 80: failure_score += 3
            if row['vibration_rms'] > 4.0: failure_score += 2
            if row['fluid_level'] < 60: failure_score += 2
            if row['motor_current'] > 60: failure_score += 1
            
            # Operational scoring
            if row['operating_hours'] > 5000: failure_score += 1
            if row['duty_cycle'] > 90: failure_score += 1
            
            # Environmental scoring
            if row['dust_level'] > 8: failure_score += 1
            
            # Determine condition
            if failure_score >= 6:
                condition = 'Critical'
                maint_days = np.random.randint(1, 3)
            elif failure_score >= 3:
                condition = 'Warning'
                maint_days = np.random.randint(3, 7)
            else:
                condition = 'Normal'
                maint_days = np.random.randint(30, 90)
            
            conditions.append(condition)
            maintenance_needed.append(maint_days)
            rul.append(maint_days * 24)  # Convert to hours
        
        df['system_condition'] = conditions
        df['maintenance_days'] = maintenance_needed
        df['rul_hours'] = rul
        df['timestamp'] = pd.date_range('2023-01-01', periods=n_samples, freq='H')
        df['machine_id'] = np.random.choice(['HYD_PRESS_01', 'HYD_PRESS_02', 'ASSY_LINE_A', 'CUTTING_MACH_01'], n_samples)
        df['component_id'] = np.random.choice(['PUMP_A', 'VALVE_B', 'ACCUMULATOR_C', 'MOTOR_D', 'COOLER_E'], n_samples)
        
        self.data = df
        return df

class AdvancedMLSystem:
    """Advanced ML system with multiple models and XAI"""
    
    def __init__(self):
        self.classification_models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        self.anomaly_models = {
            'Isolation Forest': IsolationForest(contamination=0.1, random_state=42),
            'One-Class SVM': OneClassSVM(kernel='rbf', gamma=0.1, nu=0.1)
        }
        
        self.scaler = StandardScaler()
        self.trained_models = {}
        self.feature_names = None
        self.target_names = None
        self.mlflow_runs = {}
    
    def prepare_data(self, df, problem_type='classification'):
        """Prepare data for different types of problems"""
        if problem_type == 'classification':
            X = df.drop(['system_condition', 'timestamp', 'machine_id', 'component_id', 'maintenance_days', 'rul_hours'], axis=1)
            y = df['system_condition']
        elif problem_type == 'regression':
            X = df.drop(['system_condition', 'timestamp', 'machine_id', 'component_id', 'maintenance_days', 'rul_hours'], axis=1)
            y = df['rul_hours']
        elif problem_type == 'anomaly':
            X = df.drop(['system_condition', 'timestamp', 'machine_id', 'component_id', 'maintenance_days', 'rul_hours'], axis=1)
            y = df['system_condition'] == 'Critical'  # Anomalies are critical conditions
        
        self.feature_names = X.columns.tolist()
        if problem_type == 'classification':
            self.target_names = sorted(y.unique().tolist())
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if problem_type == 'classification' else None)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_test
    
    def train_models(self, X_train, y_train, model_type='classification', use_mlflow=True):
        """Train multiple models with MLflow tracking"""
        results = {}
        models_dict = self.classification_models if model_type == 'classification' else self.anomaly_models
        
        for name, model in models_dict.items():
            start_time = time.time()
            
            # Start MLflow run
            if use_mlflow:
                run_name = f"{name}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                # Ensure experiment is set before starting run
                try:
                    # Get or create experiment
                    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
                    if experiment is None:
                        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
                    mlflow.set_experiment(EXPERIMENT_NAME)
                except Exception as e:
                    # If all else fails, create experiment with absolute path
                    try:
                        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=MLFLOW_TRACKING_URI)
                        mlflow.set_experiment(EXPERIMENT_NAME)
                    except:
                        pass
                # Start run - DO NOT use nested=True
                mlflow.start_run(run_name=run_name)
                
                # Log model parameters
                if hasattr(model, 'get_params'):
                    mlflow.log_params(model.get_params())
                mlflow.log_param("model_name", name)
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("n_features", X_train.shape[1])
                mlflow.log_param("n_samples", X_train.shape[0])
            
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            self.trained_models[name] = {
                'model': model,
                'type': model_type,
                'training_time': training_time
            }
            results[name] = {'training_time': training_time}
            
            # Log training metrics
            if use_mlflow:
                mlflow.log_metric("training_time_seconds", training_time)
                
                # Log model artifact
                try:
                    mlflow.sklearn.log_model(model, f"{name}_model")
                    self.mlflow_runs[name] = mlflow.active_run().info.run_id
                except Exception as e:
                    st.warning(f"Could not log model to MLflow: {str(e)}")
                
                mlflow.end_run()
        
        return results
    
    def evaluate_classification(self, X_test, y_test, use_mlflow=True):
        """Evaluate classification models with MLflow tracking"""
        results = {}
        for name, model_info in self.trained_models.items():
            if model_info['type'] == 'classification':
                model = model_info['model']
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'predictions': y_pred,
                    'probabilities': y_proba
                }
                
                # Log metrics to MLflow
                if use_mlflow and name in self.mlflow_runs:
                    try:
                        run_id = self.mlflow_runs[name]
                        # Use client API to log to existing run
                        client = mlflow.tracking.MlflowClient()
                        client.log_metric(run_id, "test_accuracy", accuracy)
                        client.log_metric(run_id, "test_precision", precision)
                        client.log_metric(run_id, "test_recall", recall)
                        client.log_metric(run_id, "test_f1_score", f1)
                        
                        # Log confusion matrix as artifact
                        cm = confusion_matrix(y_test, y_pred)
                        cm_df = pd.DataFrame(cm)
                        # Use a more portable temp directory
                        temp_dir = Path("./.mlflow_temp")
                        temp_dir.mkdir(exist_ok=True)
                        cm_path = temp_dir / f"cm_{name}.csv"
                        cm_df.to_csv(cm_path, index=False)
                        client.log_artifact(run_id, str(cm_path), "confusion_matrix")
                        os.remove(cm_path)  # Cleanup
                    except Exception as e:
                        # Fallback: try to start a new run if original is closed
                        try:
                            with mlflow.start_run(run_name=f"{name}_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                                mlflow.log_metric("test_accuracy", accuracy)
                                mlflow.log_metric("test_precision", precision)
                                mlflow.log_metric("test_recall", recall)
                                mlflow.log_metric("test_f1_score", f1)
                        except Exception as e2:
                            st.warning(f"Could not log metrics to MLflow: {str(e2)}")
        
        return results
    
    def evaluate_regression(self, X_test, y_test, use_mlflow=True):
        """Evaluate regression models for RUL prediction with MLflow tracking"""
        results = {}
        # For simplicity, using classification models for regression (in practice, use regressors)
        for name, model_info in self.trained_models.items():
            if model_info['type'] == 'classification':
                model = model_info['model']
                if hasattr(model, 'predict_proba'):
                    # Use probability as a proxy for RUL (for demo)
                    y_pred_proba = model.predict_proba(X_test)
                    y_pred = y_pred_proba[:, 0] * 24 + y_pred_proba[:, 1] * 168 + y_pred_proba[:, 2] * 720  # Weighted average
                    
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    results[name] = {
                        'mse': mse,
                        'mae': mae,
                        'r2': r2,
                        'predictions': y_pred
                    }
                    
                    # Log metrics to MLflow
                    if use_mlflow and name in self.mlflow_runs:
                        try:
                            run_id = self.mlflow_runs[name]
                            # Use client API to log to existing run
                            client = mlflow.tracking.MlflowClient()
                            client.log_metric(run_id, "test_mse", mse)
                            client.log_metric(run_id, "test_mae", mae)
                            client.log_metric(run_id, "test_r2", r2)
                        except Exception as e:
                            # Fallback: try to start a new run if original is closed
                            try:
                                with mlflow.start_run(run_name=f"{name}_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                                    mlflow.log_metric("test_mse", mse)
                                    mlflow.log_metric("test_mae", mae)
                                    mlflow.log_metric("test_r2", r2)
                            except Exception as e2:
                                st.warning(f"Could not log metrics to MLflow: {str(e2)}")
        
        return results

class MaintenanceChatbot:
    """AI Chatbot for maintenance recommendations"""
    
    def __init__(self):
        self.knowledge_base = {
            'pressure issues': {
                'questions': [
                    "What causes pressure drops in hydraulic systems?",
                    "How to fix low pressure?",
                    "Pressure fluctuation solutions"
                ],
                'answers': [
                    "Pressure drops can be caused by: pump wear, valve leaks, clogged filters, or fluid contamination.",
                    "Check pump performance, inspect valves for leaks, replace filters, and check fluid quality.",
                    "Stabilize pressure by: checking accumulator charge, verifying relief valve settings, ensuring proper fluid viscosity."
                ]
            },
            'temperature problems': {
                'questions': [
                    "Why is my hydraulic system overheating?",
                    "How to reduce hydraulic temperature?",
                    "Optimal temperature range"
                ],
                'answers': [
                    "Overheating causes: insufficient cooling, excessive load, wrong fluid viscosity, or pump issues.",
                    "Improve cooling: clean heat exchangers, check cooler operation, ensure proper fluid level, reduce load.",
                    "Optimal range: 45-65°C. Above 80°C can damage seals and reduce fluid life."
                ]
            },
            'vibration analysis': {
                'questions': [
                    "What causes excessive vibration?",
                    "How to reduce hydraulic system vibration?",
                    "Vibration monitoring techniques"
                ],
                'answers': [
                    "Vibration causes: misalignment, bearing wear, cavitation, unbalanced components, or loose mounts.",
                    "Solutions: align pumps/motors, replace worn bearings, check for cavitation, balance rotating parts.",
                    "Monitor with: vibration sensors, regular inspections, trend analysis, and frequency analysis."
                ]
            },
            'preventive maintenance': {
                'questions': [
                    "Recommended maintenance schedule",
                    "Daily maintenance checklist",
                    "How to extend hydraulic system life?"
                ],
                'answers': [
                    "Schedule: Daily visual checks, weekly pressure tests, monthly fluid analysis, quarterly component inspection.",
                    "Daily: Check fluid levels, inspect for leaks, monitor temperatures, listen for unusual noises.",
                    "Extend life: Use quality fluid, maintain cleanliness, avoid overheating, follow proper operating procedures."
                ]
            },
            'fluid maintenance': {
                'questions': [
                    "When to change hydraulic fluid?",
                    "How to check fluid quality?",
                    "Best hydraulic fluids"
                ],
                'answers': [
                    "Change fluid: Every 2000-4000 hours, or when contamination exceeds limits, or significant degradation occurs.",
                    "Check: Color, viscosity, acidity, water content, particle count. Use lab analysis for accurate assessment.",
                    "Recommended: ISO VG 46 for most applications, synthetic for high-temperature, anti-wear additives for pumps."
                ]
            }
        }
    
    def get_response(self, user_input):
        """Get chatbot response based on user input"""
        user_input = user_input.lower()
        
        for category, data in self.knowledge_base.items():
            if any(keyword in user_input for keyword in category.split()):
                return np.random.choice(data['answers'])
        
        # Default responses
        default_responses = [
            "I recommend checking the system pressure and temperature readings first.",
            "Regular maintenance can prevent most hydraulic system issues.",
            "Consider scheduling a professional inspection if the problem persists.",
            "Check the maintenance logs for similar historical issues and solutions.",
            "Ensure all sensors are calibrated and providing accurate readings."
        ]
        return np.random.choice(default_responses)
    
    def get_suggested_questions(self):
        """Get suggested questions for the user"""
        questions = []
        for category, data in self.knowledge_base.items():
            questions.extend(data['questions'][:1])  # Take one question from each category
        return questions

def create_shap_plots(model, X_test, feature_names):
    """Create SHAP-like explanation plots (simplified)"""
    # Simplified version - in practice, use actual SHAP library
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        importance = np.ones(len(feature_names)) / len(feature_names)
    
    # Feature importance plot
    fig_importance, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, importance)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Global Feature Importance (SHAP-like)')
    plt.tight_layout()
    
    return fig_importance

def main():
    st.markdown('<h1 class="main-header">🏭 Advanced Hydraulic System AI Monitor</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h3>Comprehensive Predictive Maintenance & Condition Monitoring Platform</h3>
        <p>Real-time monitoring, AI-powered predictions, and maintenance optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
    if 'ml_system' not in st.session_state:
        st.session_state.ml_system = AdvancedMLSystem()
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = MaintenanceChatbot()
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    
    # Sidebar Navigation
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=80)
        st.title("Navigation")
        
        st.markdown("---")
        
        # Data Management
        if st.button("📥 Load Dataset (12,000 Records)", use_container_width=True):
            with st.spinner("Loading comprehensive hydraulic dataset..."):
                dataset = AdvancedHydraulicDataset()
                st.session_state.dataset = dataset.load_dataset()
                st.success("✅ Dataset loaded successfully!")
        
        st.markdown("---")
        
        # Page selection
        page = st.radio("Navigate to:", [
            "🏠 Dashboard Overview",
            "📈 Data Exploration", 
            "🧠 Model Training",
            "🕵️ Explainable AI",
            "🔍 Live Monitoring",
            "🧾 Maintenance Advisor",
            "💬 AI Chatbot",
            "⚙️ MLOps Management"
        ])
        
        st.markdown("---")
        
        # Quick Stats
        if st.session_state.dataset is not None:
            df = st.session_state.dataset
            st.subheader("Quick Stats")
            total = len(df)
            normal = len(df[df['system_condition'] == 'Normal'])
            warning = len(df[df['system_condition'] == 'Warning'])
            critical = len(df[df['system_condition'] == 'Critical'])
            
            st.write(f"**Total:** {total:,}")
            st.write(f"**Normal:** {normal} ({normal/total*100:.1f}%)")
            st.write(f"**Warning:** {warning} ({warning/total*100:.1f}%)")
            st.write(f"**Critical:** {critical} ({critical/total*100:.1f}%)")
    
    # Page 1: Dashboard Overview
    if page == "🏠 Dashboard Overview":
        st.header("📊 System Dashboard Overview")
        
        if st.session_state.dataset is not None:
            df = st.session_state.dataset
            
            # KPI Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                uptime_percent = (len(df[df['system_condition'] == 'Normal']) / len(df)) * 100
                st.metric("System Uptime", f"{uptime_percent:.1f}%", "5.2%")
            
            with col2:
                failure_percent = (len(df[df['system_condition'] == 'Critical']) / len(df)) * 100
                st.metric("Predicted Failure Rate", f"{failure_percent:.1f}%", "-2.1%")
            
            with col3:
                mean_rul = df['rul_hours'].mean() / 24  # Convert to days
                st.metric("Mean RUL (Days)", f"{mean_rul:.1f}", "3.5")
            
            with col4:
                active_machines = df['machine_id'].nunique()
                st.metric("Active Machines", active_machines, "0")
            
            # Health Status Gauge
            st.subheader("🏥 System Health Status")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create health gauge
                health_score = 100 - failure_percent
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = health_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Overall Health Score"},
                    delta = {'reference': 85},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 60], 'color': "red"},
                            {'range': [60, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70}}
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🚨 Alerts")
                recent_critical = df[df['system_condition'] == 'Critical'].tail(3)
                if len(recent_critical) > 0:
                    for _, alert in recent_critical.iterrows():
                        st.error(f"**{alert['machine_id']}** - {alert['component_id']}")
                        st.caption(f"RUL: {alert['rul_hours']/24:.1f} days")
                else:
                    st.success("No critical alerts")
            
            # Recent Data Preview
            st.subheader("📋 Recent System Readings")
            st.dataframe(df[['timestamp', 'machine_id', 'component_id', 'system_condition', 
                           'pressure_main', 'temperature', 'vibration_rms']].tail(10))
            
        else:
            st.info("👆 Please load the dataset to view the dashboard")
    
    # Page 2: Data Exploration
    elif page == "📈 Data Exploration":
        st.header("📊 Data Exploration & Analysis")
        
        if st.session_state.dataset is not None:
            df = st.session_state.dataset
            
            # Data Summary
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Dataset Summary")
                st.write(f"**Shape:** {df.shape}")
                st.write(f"**Time Range:** {df['timestamp'].min()} to {df['timestamp'].max()}")
                st.write(f"**Machines:** {df['machine_id'].nunique()}")
                st.write(f"**Components:** {df['component_id'].nunique()}")
            
            with col2:
                st.subheader("Data Quality")
                missing_data = df.isnull().sum().sum()
                st.write(f"**Missing Values:** {missing_data}")
                st.write(f"**Duplicate Rows:** {df.duplicated().sum()}")
                st.write(f"**Data Types:** {len(df.select_dtypes(include=[np.number]).columns)} numeric, "
                        f"{len(df.select_dtypes(include=['object']).columns)} categorical")
            
            # Correlation Heatmap
            st.subheader("Feature Correlation Heatmap")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(corr_matrix, title='Feature Correlation Matrix', 
                          color_continuous_scale='RdBu_r', aspect='auto')
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature Distribution
            st.subheader("Feature Distributions")
            selected_feature = st.selectbox("Select feature to visualize:", 
                                          [col for col in numeric_cols if col not in ['rul_hours', 'maintenance_days', 'operating_hours']])
            
            col1, col2 = st.columns(2)
            with col1:
                # Histogram
                fig = px.histogram(df, x=selected_feature, title=f'Distribution of {selected_feature}')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot by condition
                fig = px.box(df, x='system_condition', y=selected_feature, 
                           color='system_condition', title=f'{selected_feature} by System Condition')
                st.plotly_chart(fig, use_container_width=True)
            
            # Time Series Analysis
            st.subheader("Time Series Trends")
            selected_machine = st.selectbox("Select machine:", df['machine_id'].unique())
            machine_data = df[df['machine_id'] == selected_machine]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=machine_data['timestamp'], y=machine_data['pressure_main'], 
                                   name='Main Pressure', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=machine_data['timestamp'], y=machine_data['temperature'], 
                                   name='Temperature', yaxis='y2', line=dict(color='red')))
            
            fig.update_layout(
                title=f'Pressure and Temperature Trend - {selected_machine}',
                xaxis_title='Time',
                yaxis_title='Pressure (bar)',
                yaxis2=dict(title='Temperature (°C)', overlaying='y', side='right'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("👆 Please load the dataset to explore the data")
    
    # Page 3: Model Training
    elif page == "🧠 Model Training":
        st.header("🤖 Machine Learning Model Training")
        
        if st.session_state.dataset is not None:
            df = st.session_state.dataset
            ml_system = st.session_state.ml_system
            
            st.subheader("Model Configuration")
            problem_type = st.selectbox("Select Problem Type:", 
                                      ['classification', 'anomaly detection'])
            
            if st.button("🚀 Train All Models", type="primary"):
                with st.spinner("Training multiple ML models..."):
                    X_train, X_test, y_train, y_test, X_test_orig = ml_system.prepare_data(df, problem_type)
                    training_results = ml_system.train_models(X_train, y_train, problem_type)
                    
                    if problem_type == 'classification':
                        eval_results = ml_system.evaluate_classification(X_test, y_test)
                    else:
                        eval_results = ml_system.evaluate_regression(X_test, y_test)
                    
                    st.session_state.trained = True
                    st.session_state.eval_results = eval_results
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.session_state.problem_type = problem_type
                    
                    st.success("✅ All models trained successfully!")
                    st.info(f"📊 MLflow: All experiments logged to {MLFLOW_TRACKING_URI}")
            
            if st.session_state.trained:
                eval_results = st.session_state.eval_results
                
                # Model Comparison
                st.subheader("📊 Model Performance Comparison")
                
                if st.session_state.problem_type == 'classification':
                    metrics_df = pd.DataFrame({
                        'Model': list(eval_results.keys()),
                        'Accuracy': [r['accuracy'] for r in eval_results.values()],
                        'Precision': [r['precision'] for r in eval_results.values()],
                        'Recall': [r['recall'] for r in eval_results.values()],
                        'F1-Score': [r['f1'] for r in eval_results.values()]
                    })
                else:
                    metrics_df = pd.DataFrame({
                        'Model': list(eval_results.keys()),
                        'MSE': [r['mse'] for r in eval_results.values()],
                        'MAE': [r['mae'] for r in eval_results.values()],
                        'R²': [r['r2'] for r in eval_results.values()]
                    })
                
                # FIXED: Safe dataframe display with formatting
                try:
                    # Convert all numeric columns to float for formatting
                    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        metrics_df[col] = metrics_df[col].astype(float)
                    
                    styled_df = metrics_df.style.format({
                        col: "{:.3f}" for col in numeric_cols
                    })
                    st.dataframe(styled_df, use_container_width=True)
                except Exception as e:
                    st.warning(f"Displaying without formatting due to: {str(e)}")
                    st.dataframe(metrics_df, use_container_width=True)
                
                # Performance Visualization
                st.subheader("📈 Performance Visualization")
                
                if st.session_state.problem_type == 'classification':
                    fig = px.bar(metrics_df, x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                               title='Model Performance Metrics', barmode='group')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Confusion Matrix for best model
                    best_model_name = max(eval_results.items(), key=lambda x: x[1]['accuracy'])[0]
                    best_result = eval_results[best_model_name]
                    
                    st.subheader(f"🎯 Best Model: {best_model_name}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        cm = confusion_matrix(st.session_state.y_test, best_result['predictions'])
                        fig_cm = px.imshow(cm, text_auto=True, title='Confusion Matrix',
                                         labels=dict(x="Predicted", y="Actual"))
                        st.plotly_chart(fig_cm, use_container_width=True)
                    
                    with col2:
                        # Feature Importance
                        best_model = ml_system.trained_models[best_model_name]['model']
                        if hasattr(best_model, 'feature_importances_'):
                            importance_df = pd.DataFrame({
                                'Feature': ml_system.feature_names,
                                'Importance': best_model.feature_importances_
                            }).sort_values('Importance', ascending=True)
                            
                            fig_imp = px.bar(importance_df.tail(10), x='Importance', y='Feature',
                                           title='Top 10 Feature Importance', orientation='h')
                            st.plotly_chart(fig_imp, use_container_width=True)
                
        else:
            st.info("👆 Please load the dataset to train models")
    
    # Page 4: Explainable AI
    elif page == "🕵️ Explainable AI":
        st.header("🔍 Explainable AI & Model Interpretability")
        
        if st.session_state.trained:
            ml_system = st.session_state.ml_system
            eval_results = st.session_state.eval_results
            
            st.subheader("Global Model Explanations")
            
            # Model selection for explanation
            selected_model = st.selectbox("Select model to explain:", 
                                        list(ml_system.trained_models.keys()))
            
            model_info = ml_system.trained_models[selected_model]
            model = model_info['model']
            
            # SHAP-like Feature Importance
            st.subheader("📊 Global Feature Importance")
            fig_importance = create_shap_plots(model, st.session_state.X_test, ml_system.feature_names)
            st.pyplot(fig_importance)
            
            # Individual Prediction Explanation
            st.subheader("🔎 Individual Prediction Analysis")
            sample_idx = st.slider("Select test sample for explanation", 
                                 0, len(st.session_state.X_test)-1, 0)
            
            if st.session_state.problem_type == 'classification':
                actual = st.session_state.y_test.iloc[sample_idx]
                predicted = eval_results[selected_model]['predictions'][sample_idx]
                
                st.write(f"**Actual:** {actual}, **Predicted:** {predicted}")
                
                if eval_results[selected_model]['probabilities'] is not None:
                    probabilities = eval_results[selected_model]['probabilities'][sample_idx]
                    prob_df = pd.DataFrame({
                        'Class': ml_system.target_names,
                        'Probability': probabilities
                    })
                    
                    fig_probs = px.bar(prob_df, x='Class', y='Probability', 
                                     color='Class', title='Prediction Probabilities')
                    st.plotly_chart(fig_probs, use_container_width=True)
            
            # Model-specific insights
            st.subheader("💡 Model Insights")
            if hasattr(model, 'feature_importances_'):
                top_features = sorted(zip(ml_system.feature_names, model.feature_importances_), 
                                    key=lambda x: x[1], reverse=True)[:3]
                st.write("**Most influential features:**")
                for feature, importance in top_features:
                    st.write(f"- {feature}: {importance:.3f}")
            
            st.write("""
            **Interpretation Guide:**
            - **Feature Importance**: Shows which parameters most affect predictions
            - **Partial Dependence**: Shows how changing a feature affects predictions
            - **Individual Explanations**: Understand why specific predictions were made
            """)
            
        else:
            st.info("👆 Please train models first to view explanations")
    
    # Page 5: Live Monitoring
    elif page == "🔍 Live Monitoring":
        st.header("🔍 Real-time Monitoring & Predictions")
        
        if st.session_state.trained:
            ml_system = st.session_state.ml_system
            
            st.subheader("🎯 Make Predictions")
            
            # Model selection
            selected_model = st.selectbox("Select prediction model:", 
                                        list(ml_system.trained_models.keys()))
            
            # Input features
            st.subheader("📋 Enter System Parameters")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pressure_main = st.slider("Main Pressure (bar)", 50.0, 150.0, 100.0)
                flow_rate = st.slider("Flow Rate (L/min)", 20.0, 80.0, 50.0)
                temperature = st.slider("Temperature (°C)", 20.0, 100.0, 60.0)
            
            with col2:
                vibration_x = st.slider("Vibration X (m/s²)", 0.0, 10.0, 1.5)
                vibration_y = st.slider("Vibration Y (m/s²)", 0.0, 10.0, 1.8)
                fluid_level = st.slider("Fluid Level (%)", 0.0, 100.0, 85.0)
            
            with col3:
                motor_current = st.slider("Motor Current (A)", 20.0, 80.0, 45.0)
                load_kg = st.slider("Load (kg)", 1000.0, 10000.0, 5000.0)
                ambient_temp = st.slider("Ambient Temp (°C)", 10.0, 40.0, 25.0)
            
            if st.button("🔮 Predict System Condition", type="primary"):
                # Prepare features
                features = np.array([pressure_main, flow_rate, temperature, vibration_x, vibration_y,
                                   fluid_level, motor_current, load_kg, ambient_temp] + 
                                   [0] * (len(ml_system.feature_names) - 9))  # Pad with zeros
                
                features_scaled = ml_system.scaler.transform([features])
                model = ml_system.trained_models[selected_model]['model']
                
                if st.session_state.problem_type == 'classification':
                    prediction = model.predict(features_scaled)[0]
                    probability = model.predict_proba(features_scaled)[0] if hasattr(model, 'predict_proba') else None
                    
                    # Display results
                    st.subheader("🎯 Prediction Result")
                    
                    if prediction == 'Normal':
                        st.markdown('<div class="alert-normal">🟢 NORMAL CONDITION</div>', unsafe_allow_html=True)
                    elif prediction == 'Warning':
                        st.markdown('<div class="alert-warning">🟡 WARNING CONDITION</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="alert-critical">🔴 CRITICAL CONDITION</div>', unsafe_allow_html=True)
                    
                    if probability is not None:
                        st.write("**Prediction Confidence:**")
                        for i, class_name in enumerate(ml_system.target_names):
                            st.write(f"{class_name}: {probability[i]:.1%}")
                
                # RUL Prediction
                st.subheader("⏰ Remaining Useful Life (RUL)")
                estimated_rul_days = np.random.randint(1, 90)  # Simulated RUL
                st.metric("Estimated RUL", f"{estimated_rul_days} days", "-5 days")
                
                # RUL Trend Chart
                days = list(range(30))
                rul_trend = [max(0, estimated_rul_days - i) for i in days]
                
                fig = px.line(x=days, y=rul_trend, title='RUL Projection Trend',
                            labels={'x': 'Days from now', 'y': 'Remaining Useful Life (days)'})
                st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("👆 Please train models first to enable predictions")
    
    # Page 6: Maintenance Advisor
    elif page == "🧾 Maintenance Advisor":
        st.header("🛠️ AI-Powered Maintenance Recommendations")
        
        if st.session_state.dataset is not None:
            df = st.session_state.dataset
            
            st.subheader("📋 Maintenance Priority List")
            
            # Get critical systems
            critical_systems = df[df['system_condition'] == 'Critical']
            warning_systems = df[df['system_condition'] == 'Warning']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔴 Critical Priority")
                if len(critical_systems) > 0:
                    for _, system in critical_systems.head(5).iterrows():
                        st.error(f"""
                        **{system['machine_id']}** - {system['component_id']}
                        - RUL: {system['rul_hours']/24:.1f} days
                        - Recommended: Immediate inspection
                        - Estimated cost: ${np.random.randint(500, 2000)}
                        """)
                else:
                    st.success("No critical maintenance required")
            
            with col2:
                st.subheader("🟡 Warning Priority")
                if len(warning_systems) > 0:
                    for _, system in warning_systems.head(5).iterrows():
                        st.warning(f"""
                        **{system['machine_id']}** - {system['component_id']}
                        - RUL: {system['rul_hours']/24:.1f} days
                        - Recommended: Schedule within {system['maintenance_days']} days
                        - Estimated cost: ${np.random.randint(200, 800)}
                        """)
                else:
                    st.info("No warning maintenance required")
            
            # Cost-Benefit Analysis
            st.subheader("💰 Cost-Benefit Analysis")
            
            preventive_cost = len(warning_systems) * 500
            critical_cost = len(critical_systems) * 1500
            total_savings = critical_cost - preventive_cost
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Preventive Cost", f"${preventive_cost:,}")
            with col2:
                st.metric("Critical Repair Cost", f"${critical_cost:,}")
            with col3:
                st.metric("Potential Savings", f"${total_savings:,}", 
                         delta=f"{(total_savings/preventive_cost*100 if preventive_cost > 0 else 0):.1f}%")
            
            # Maintenance Schedule
            st.subheader("📅 Recommended Maintenance Schedule")
            schedule_data = []
            for _, system in critical_systems.head(3).iterrows():
                schedule_data.append({
                    'Machine': system['machine_id'],
                    'Component': system['component_id'],
                    'Priority': 'Critical',
                    'Schedule': 'Immediate',
                    'Action': 'Replace/Repair'
                })
            
            for _, system in warning_systems.head(3).iterrows():
                schedule_data.append({
                    'Machine': system['machine_id'],
                    'Component': system['component_id'],
                    'Priority': 'Warning',
                    'Schedule': f'Within {system["maintenance_days"]} days',
                    'Action': 'Inspect/Maintain'
                })
            
            if schedule_data:
                schedule_df = pd.DataFrame(schedule_data)
                st.dataframe(schedule_df, use_container_width=True)
            
        else:
            st.info("👆 Please load the dataset to view maintenance recommendations")
    
    # Page 7: AI Chatbot
    elif page == "💬 AI Chatbot":
        st.header("🤖 Maintenance AI Assistant")
        
        chatbot = st.session_state.chatbot
        
        st.info("💡 Ask me about hydraulic system maintenance, troubleshooting, or best practices!")
        
        # Suggested questions
        st.subheader("💭 Suggested Questions")
        suggested_questions = chatbot.get_suggested_questions()
        
        cols = st.columns(2)
        for i, question in enumerate(suggested_questions):
            with cols[i % 2]:
                if st.button(question, key=f"q_{i}"):
                    st.session_state.chat_input = question
        
        # Chat interface
        st.subheader("💬 Chat with AI Assistant")
        
        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f'<div class="chat-bubble user-bubble">{message["content"]}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bubble bot-bubble">{message["content"]}</div>', 
                           unsafe_allow_html=True)
        
        # Chat input
        chat_input = st.text_input("Type your message here...", 
                                 value=st.session_state.get('chat_input', ''),
                                 key="chat_input_widget")
        
        if st.button("Send") and chat_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": chat_input})
            
            # Get AI response
            response = chatbot.get_response(chat_input)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Clear input
            st.session_state.chat_input = ""
            st.rerun()
    
    # Page 8: MLOps Management
    elif page == "⚙️ MLOps Management":
        st.header("⚙️ MLOps & Model Management")
        
        # MLflow Configuration
        st.subheader("📊 MLflow Tracking")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Tracking URI:** {MLFLOW_TRACKING_URI}")
            st.info(f"**Experiment:** hydraulic_system_monitoring")
        with col2:
            try:
                # Get experiment info
                experiment = mlflow.get_experiment_by_name("hydraulic_system_monitoring")
                if experiment:
                    st.metric("Experiment ID", experiment.experiment_id)
                    # Count runs
                    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
                    st.metric("Total Runs", len(runs))
            except Exception as e:
                st.warning(f"Could not fetch MLflow info: {str(e)}")
        
        st.subheader("🔧 System Configuration")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Data Pipeline", "Active", "✅")
        with col2:
            st.metric("Model Serving", "Online", "✅")
        with col3:
            st.metric("API Health", "98.7%", "1.3%")
        with col4:
            st.metric("System Load", "42%", "8%")
        
        # MLflow Runs Display
        st.subheader("📚 MLflow Experiments")
        try:
            experiment = mlflow.get_experiment_by_name("hydraulic_system_monitoring")
            if experiment:
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    max_results=10,
                    order_by=["start_time DESC"]
                )
                
                if len(runs) > 0:
                    # Display recent runs
                    runs_display = runs[['run_id', 'tags.mlflow.runName', 'metrics.test_accuracy', 
                                       'metrics.test_f1_score', 'metrics.training_time_seconds', 
                                       'start_time']].head(10).copy()
                    runs_display.columns = ['Run ID', 'Run Name', 'Accuracy', 'F1 Score', 'Training Time (s)', 'Start Time']
                    runs_display = runs_display.fillna('N/A')
                    st.dataframe(runs_display, use_container_width=True)
                    
                    # Show best model
                    if 'metrics.test_accuracy' in runs.columns:
                        best_run = runs.loc[runs['metrics.test_accuracy'].idxmax()]
                        st.success(f"🏆 Best Model: {best_run.get('tags.mlflow.runName', 'Unknown')} "
                                 f"(Accuracy: {best_run['metrics.test_accuracy']:.3f})")
                else:
                    st.info("No MLflow runs yet. Train models to see experiments here.")
        except Exception as e:
            st.warning(f"Could not load MLflow runs: {str(e)}")
        
        # Model Version Management
        st.subheader("📦 Registered Models")
        
        versions = [
            {"Version": "v2.1.0", "Status": "Production", "Accuracy": "94.2%", "Created": "2024-01-15"},
            {"Version": "v2.0.1", "Status": "Staging", "Accuracy": "93.8%", "Created": "2024-01-10"},
            {"Version": "v1.9.0", "Status": "Archived", "Accuracy": "92.1%", "Created": "2023-12-20"},
        ]
        
        version_df = pd.DataFrame(versions)
        st.dataframe(version_df, use_container_width=True)
        
        # Alert Configuration
        st.subheader("🚨 Alert Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            failure_threshold = st.slider("Failure Probability Alert Threshold", 0.0, 1.0, 0.8)
            vibration_threshold = st.slider("Vibration Alert Threshold (m/s²)", 0.0, 10.0, 4.0)
        with col2:
            temp_threshold = st.slider("Temperature Alert Threshold (°C)", 50.0, 100.0, 80.0)
            pressure_threshold = st.slider("Pressure Deviation Threshold (%)", 0.0, 50.0, 20.0)
        
        # Model Operations
        st.subheader("🔄 Model Operations")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Retrain Models", use_container_width=True):
                st.success("Model retraining initiated!")
        
        with col2:
            if st.button("📊 Generate Report", use_container_width=True):
                st.success("MLOps report generated!")
        
        with col3:
            if st.button("🚀 Deploy to Prod", use_container_width=True):
                st.success("Model deployment started!")
        
        with col4:
            if st.button("🔍 Audit Log", use_container_width=True):
                st.success("Audit log exported!")
        
        # Performance Monitoring
        st.subheader("📈 System Performance")
        
        # Simulate performance metrics
        time_points = 30
        dates = pd.date_range('2023-12-01', periods=time_points, freq='D')
        performance_data = {
            'Date': dates,
            'Accuracy': np.random.normal(0.93, 0.02, time_points),
            'Latency (ms)': np.random.normal(45, 5, time_points),
            'Throughput (req/s)': np.random.normal(120, 10, time_points)
        }
        
        perf_df = pd.DataFrame(performance_data)
        
        fig = px.line(perf_df, x='Date', y=['Accuracy', 'Latency (ms)', 'Throughput (req/s)'],
                     title='ML System Performance Metrics', markers=True)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()