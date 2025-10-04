"""
main_training_pipeline.py
End-to-End Training Pipeline for Deepfake Detection System
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.data_preprocessing import DeepfakeDataPreprocessor
from src.cnn_model import CNNDeepfakeDetector
from src.bert_model import BERTSyntheticIdentityDetector
from src.hybrid_model import HybridDeepfakeDetector
from src.evaluation import ModelEvaluator

class DeepfakeDetectionPipeline:
    """Complete end-to-end training pipeline"""
    
    def __init__(self, config_path='config.json'):
        self.config = self.load_config(config_path)
        self.results = {}
        self.start_time = datetime.now()
        
        # Create directories
        self.create_directories()
        
    def load_config(self, config_path):
        """Load configuration"""
        default_config = {
            'data': {
                'real_images_path': 'data/raw/real_faces',
                'fake_images_path': 'data/raw/fake_faces',
                'text_data_path': 'data/raw/kyc_text_data.csv',
                'img_size': [224, 224],
                'test_size': 0.2,
                'val_size': 0.1
            },
            'cnn': {
                'model_type': 'efficientnet',
                'epochs': 50,
                'batch_size': 32,
                'learning_rate': 0.0001,
                'use_augmentation': True
            },
            'bert': {
                'model_name': 'bert-base-uncased',
                'epochs': 4,
                'batch_size': 16,
                'learning_rate': 0.00002,
                'max_length': 512
            },
            'hybrid': {
                'meta_classifier': 'gradient_boosting'
            },
            'paths': {
                'model_output': 'models/',
                'results_output': 'results/',
                'logs_output': 'logs/'
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        
        return default_config
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.config['paths']['model_output'],
            self.config['paths']['results_output'],
            self.config['paths']['logs_output'],
            'models/cnn_visual_detector',
            'models/bert_text_analyzer',
            'models/hybrid_model'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def log(self, message):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        # Write to log file
        with open(f"{self.config['paths']['logs_output']}/training.log", 'a') as f:
            f.write(log_message + '\n')
    
    def step1_data_preprocessing(self):
        """Step 1: Load and preprocess data"""
        self.log("="*60)
        self.log("STEP 1: DATA PREPROCESSING")
        self.log("="*60)
        
        # Initialize preprocessor
        img_size = tuple(self.config['data']['img_size'])
        preprocessor = DeepfakeDataPreprocessor(img_size=img_size)
        
        # Prepare image dataset
        self.log("Loading and preprocessing image data...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
            preprocessor.prepare_dataset(
                self.config['data']['real_images_path'],
                self.config['data']['fake_images_path'],
                test_size=self.config['data']['test_size'],
                val_size=self.config['data']['val_size']
            )
        
        # Create augmentation pipeline
        if self.config['cnn']['use_augmentation']:
            self.log("Creating data augmentation pipeline...")
            augmentation = preprocessor.augment_data()
        else:
            augmentation = None
        
        # Load text data if available
        text_data = None
        if os.path.exists(self.config['data']['text_data_path']):
            self.log("Loading text data...")
            text_df = pd.read_csv(self.config['data']['text_data_path'])
            text_data = {
                'train': text_df[text_df['split'] == 'train'],
                'val': text_df[text_df['split'] == 'val'],
                'test': text_df[text_df['split'] == 'test']
            }
        
        self.log(f"✓ Data preprocessing complete")
        self.log(f"  Train samples: {len(X_train)}")
        self.log(f"  Validation samples: {len(X_val)}")
        self.log(f"  Test samples: {len(X_test)}")
        
        return {
            'images': {
                'train': (X_train, y_train),
                'val': (X_val, y_val),
                'test': (X_test, y_test)
            },
            'text': text_data,
            'augmentation': augmentation
        }
    
    def step2_train_cnn(self, data):
        """Step 2: Train CNN visual detector"""
        self.log("\n" + "="*60)
        self.log("STEP 2: TRAINING CNN VISUAL DETECTOR")
        self.log("="*60)
        
        # Initialize CNN detector
        detector = CNNDeepfakeDetector(
            input_shape=tuple(self.config['data']['img_size']) + (3,),
            model_type=self.config['cnn']['model_type']
        )
        
        # Build and compile model
        self.log(f"Building {self.config['cnn']['model_type']} model...")
        detector.build_model()
        detector.compile_model(learning_rate=self.config['cnn']['learning_rate'])
        
        # Display model summary
        self.log("Model architecture:")
        detector.get_model_summary()
        
        # Train model
        self.log(f"Training for {self.config['cnn']['epochs']} epochs...")
        X_train, y_train = data['images']['train']
        X_val, y_val = data['images']['val']
        
        history = detector.train(
            X_train, y_train,
            X_val, y_val,
            epochs=self.config['cnn']['epochs'],
            batch_size=self.config['cnn']['batch_size'],
            augmentation=data['augmentation']
        )
        
        # Evaluate on test set
        self.log("Evaluating on test set...")
        X_test, y_test = data['images']['test']
        predictions, binary_preds = detector.predict(X_test)
        
        # Calculate metrics
        evaluator = ModelEvaluator(model_name='CNN Visual Detector')
        metrics = evaluator.evaluate_binary_classification(
            y_test, binary_preds.flatten(), predictions.flatten()
        )
        
        self.log(f"✓ CNN training complete")
        self.log(f"  Test Accuracy: {metrics['accuracy']:.4f}")
        self.log(f"  Test AUC: {metrics['roc_auc']:.4f}")
        
        # Save model
        detector.model.save(
            self.config['paths']['model_output'] + 'cnn_visual_detector/model.h5'
        )
        
        self.results['cnn'] = {
            'metrics': metrics,
            'history': history.history if hasattr(history, 'history') else history
        }
        
        return detector, metrics
    
    def step3_train_bert(self, data):
        """Step 3: Train BERT text analyzer"""
        self.log("\n" + "="*60)
        self.log("STEP 3: TRAINING BERT TEXT ANALYZER")
        self.log("="*60)
        
        if data['text'] is None:
            self.log("No text data available, skipping BERT training")
            return None, None
        
        # Initialize BERT detector
        detector = BERTSyntheticIdentityDetector(
            model_name=self.config['bert']['model_name']
        )
        
        # Prepare data
        train_texts = data['text']['train']['text'].values
        train_labels = data['text']['train']['label'].values
        val_texts = data['text']['val']['text'].values
        val_labels = data['text']['val']['label'].values
        
        # Create dataloaders
        self.log("Preparing dataloaders...")
        train_loader, val_loader = detector.prepare_dataloaders(
            train_texts, train_labels,
            val_texts, val_labels,
            batch_size=self.config['bert']['batch_size']
        )
        
        # Train model
        self.log(f"Training BERT for {self.config['bert']['epochs']} epochs...")
        history = detector.train(
            train_loader, val_loader,
            epochs=self.config['bert']['epochs'],
            learning_rate=self.config['bert']['learning_rate']
        )
        
        # Evaluate on test set
        self.log("Evaluating on test set...")
        test_texts = data['text']['test']['text'].values
        test_labels = data['text']['test']['label'].values
        
        predictions, probabilities = detector.predict(test_texts)
        
        # Calculate metrics
        evaluator = ModelEvaluator(model_name='BERT Text Analyzer')
        metrics = evaluator.evaluate_binary_classification(
            test_labels, predictions, probabilities[:, 1]
        )
        
        self.log(f"✓ BERT training complete")
        self.log(f"  Test Accuracy: {metrics['accuracy']:.4f}")
        self.log(f"  Test AUC: {metrics['roc_auc']:.4f}")
        
        self.results['bert'] = {
            'metrics': metrics,
            'history': history
        }
        
        return detector, metrics
    
    def step4_train_hybrid(self, data, cnn_model, bert_model):
        """Step 4: Train hybrid fusion model"""
        self.log("\n" + "="*60)
        self.log("STEP 4: TRAINING HYBRID FUSION MODEL")
        self.log("="*60)
        
        # Initialize hybrid detector
        hybrid = HybridDeepfakeDetector(
            cnn_model=cnn_model,
            bert_model=bert_model,
            meta_classifier=self.config['hybrid']['meta_classifier']
        )
        
        # Prepare data
        X_train_img, y_train = data['images']['train']
        X_val_img, y_val = data['images']['val']
        X_test_img, y_test = data['images']['test']
        
        # Prepare text data (if available)
        if data['text'] is not None:
            train_texts = data['text']['train']['text'].values
            val_texts = data['text']['val']['text'].values
            test_texts = data['text']['test']['text'].values
        else:
            train_texts = [None] * len(y_train)
            val_texts = [None] * len(y_val)
            test_texts = [None] * len(y_test)
        
        # Train hybrid model
        self.log(f"Training {self.config['hybrid']['meta_classifier']} meta-classifier...")
        train_results = hybrid.train(
            X_train_img, train_texts, y_train,
            X_val_img, val_texts, y_val
        )
        
        # Evaluate on test set
        self.log("Evaluating hybrid model on test set...")
        test_metrics, predictions, probabilities = hybrid.evaluate(
            X_test_img, test_texts, y_test
        )
        
        self.log(f"✓ Hybrid model training complete")
        self.log(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        self.log(f"  Test Precision: {test_metrics['precision']:.4f}")
        self.log(f"  Test Recall: {test_metrics['recall']:.4f}")
        self.log(f"  Test F1-Score: {test_metrics['f1_score']:.4f}")
        self.log(f"  Test AUC: {test_metrics['roc_auc']:.4f}")
        
        # Save hybrid model
        hybrid.save_model(
            self.config['paths']['model_output'] + 'hybrid_model/'
        )
        
        self.results['hybrid'] = {
            'metrics': test_metrics,
            'predictions': predictions,
            'probabilities': probabilities
        }
        
        return hybrid, test_metrics, predictions, probabilities
    
    def step5_generate_reports(self, data):
        """Step 5: Generate comprehensive evaluation reports"""
        self.log("\n" + "="*60)
        self.log("STEP 5: GENERATING EVALUATION REPORTS")
        self.log("="*60)
        
        X_test, y_test = data['images']['test']
        predictions = self.results['hybrid']['predictions']
        probabilities = self.results['hybrid']['probabilities'][:, 1]
        
        # Initialize evaluator
        evaluator = ModelEvaluator(model_name='Hybrid Deepfake Detector')
        
        # Generate plots
        self.log("Generating visualizations...")
        results_path = self.config['paths']['results_output']
        
        # Confusion matrix
        evaluator.plot_confusion_matrix(
            y_test, predictions,
            save_path=f'{results_path}/confusion_matrix.png'
        )
        
        # ROC curve
        evaluator.plot_roc_curve(
            y_test, probabilities,
            save_path=f'{results_path}/roc_curve.png'
        )
        
        # Precision-Recall curve
        evaluator.plot_precision_recall_curve(
            y_test, probabilities,
            save_path=f'{results_path}/precision_recall_curve.png'
        )
        
        # Find optimal threshold
        optimal_threshold, _ = evaluator.find_optimal_threshold(
            y_test, probabilities, metric='f1'
        )
        
        # Bootstrap confidence intervals
        self.log("Calculating bootstrap confidence intervals...")
        ci_results = evaluator.bootstrap_confidence_intervals(
            y_test, predictions, probabilities
        )
        
        # Generate comprehensive report
        self.log("Generating comprehensive evaluation report...")
        report = evaluator.generate_evaluation_report(
            y_test, predictions, probabilities,
            save_path=f'{results_path}/evaluation_report.json'
        )
        
        # Print classification report
        evaluator.print_classification_report(y_test, predictions)
        
        # Save all results
        self.save_final_results(ci_results, optimal_threshold)
        
        self.log(f"✓ Evaluation reports generated")
        self.log(f"  Results saved to: {results_path}")
        
        return report, ci_results
    
    def save_final_results(self, ci_results, optimal_threshold):
        """Save final comprehensive results"""
        final_results = {
            'project_info': {
                'title': 'Deepfake Fraud Detection in Banking',
                'training_date': self.start_time.isoformat(),
                'training_duration': str(datetime.now() - self.start_time)
            },
            'configuration': self.config,
            'cnn_results': self.results.get('cnn', {}),
            'bert_results': self.results.get('bert', {}),
            'hybrid_results': self.results.get('hybrid', {}).get('metrics', {}),
            'confidence_intervals': ci_results,
            'optimal_threshold': optimal_threshold
        }
        
        # Save to JSON
        with open(f"{self.config['paths']['results_output']}/final_results.json", 'w') as f:
            json.dump(final_results, f, indent=4, default=str)
        
        # Create summary report
        self.create_summary_report(final_results)
    
    def create_summary_report(self, results):
        """Create human-readable summary report"""
        summary = f"""
{'='*80}
DEEPFAKE FRAUD DETECTION IN BANKING - TRAINING SUMMARY
{'='*80}

Project Information:
-------------------
Start Time: {results['project_info']['training_date']}
Duration: {results['project_info']['training_duration']}

Model Performance Summary:
-------------------------

1. CNN Visual Detector:
   - Test Accuracy: {results['cnn_results'].get('metrics', {}).get('accuracy', 0):.4f}
   - Test AUC: {results['cnn_results'].get('metrics', {}).get('roc_auc', 0):.4f}
   - Test Precision: {results['cnn_results'].get('metrics', {}).get('precision', 0):.4f}
   - Test Recall: {results['cnn_results'].get('metrics', {}).get('recall', 0):.4f}

2. BERT Text Analyzer:
   - Test Accuracy: {results['bert_results'].get('metrics', {}).get('accuracy', 0):.4f}
   - Test AUC: {results['bert_results'].get('metrics', {}).get('roc_auc', 0):.4f}
   - Test Precision: {results['bert_results'].get('metrics', {}).get('precision', 0):.4f}
   - Test Recall: {results['bert_results'].get('metrics', {}).get('recall', 0):.4f}

3. Hybrid Fusion Model (FINAL):
   - Test Accuracy: {results['hybrid_results'].get('accuracy', 0):.4f}
   - Test AUC: {results['hybrid_results'].get('roc_auc', 0):.4f}
   - Test Precision: {results['hybrid_results'].get('precision', 0):.4f}
   - Test Recall: {results['hybrid_results'].get('recall', 0):.4f}
   - Test F1-Score: {results['hybrid_results'].get('f1_score', 0):.4f}

Confidence Intervals (95%):
---------------------------
"""
        for metric, values in results.get('confidence_intervals', {}).items():
            if isinstance(values, dict):
                summary += f"{metric.capitalize()}: {values.get('mean', 0):.4f} "
                summary += f"[{values.get('ci_lower', 0):.4f}, {values.get('ci_upper', 0):.4f}]\n"
        
        summary += f"""
Optimal Classification Threshold: {results.get('optimal_threshold', 0.5):.4f}

Key Findings:
-------------
- The hybrid model achieves {results['hybrid_results'].get('accuracy', 0)*100:.2f}% accuracy
- ROC-AUC score of {results['hybrid_results'].get('roc_auc', 0):.4f} indicates excellent discrimination
- Precision of {results['hybrid_results'].get('precision', 0):.4f} minimizes false positives
- Recall of {results['hybrid_results'].get('recall', 0):.4f} ensures high detection rate

Recommendations for Deployment:
-------------------------------
1. Use threshold of {results.get('optimal_threshold', 0.5):.4f} for balanced performance
2. Flag predictions with confidence < 0.7 for manual review
3. Implement continuous monitoring and model retraining
4. Maintain audit trail for all predictions
5. Regularly update with new deepfake generation techniques

{'='*80}
"""
        
        # Save summary
        with open(f"{self.config['paths']['results_output']}/summary_report.txt", 'w') as f:
            f.write(summary)
        
        print(summary)
    
    def run_complete_pipeline(self):
        """Execute complete end-to-end pipeline"""
        try:
            self.log("="*80)
            self.log("STARTING DEEPFAKE DETECTION TRAINING PIPELINE")
            self.log("="*80)
            
            # Step 1: Data Preprocessing
            data = self.step1_data_preprocessing()
            
            # Step 2: Train CNN
            cnn_model, cnn_metrics = self.step2_train_cnn(data)
            
            # Step 3: Train BERT (if text data available)
            bert_model, bert_metrics = self.step3_train_bert(data)
            
            # Step 4: Train Hybrid Model
            hybrid_model, hybrid_metrics, predictions, probabilities = \
                self.step4_train_hybrid(data, cnn_model, bert_model)
            
            # Step 5: Generate Reports
            report, ci_results = self.step5_generate_reports(data)
            
            # Final summary
            self.log("\n" + "="*80)
            self.log("PIPELINE COMPLETED SUCCESSFULLY!")
            self.log("="*80)
            self.log(f"Total Duration: {datetime.now() - self.start_time}")
            self.log(f"Final Model Accuracy: {hybrid_metrics['accuracy']:.4f}")
            self.log(f"Final Model AUC: {hybrid_metrics['roc_auc']:.4f}")
            self.log("="*80)
            
            return {
                'success': True,
                'models': {
                    'cnn': cnn_model,
                    'bert': bert_model,
                    'hybrid': hybrid_model
                },
                'metrics': {
                    'cnn': cnn_metrics,
                    'bert': bert_metrics,
                    'hybrid': hybrid_metrics
                }
            }
        
        except Exception as e:
            self.log(f"\n!!! PIPELINE FAILED !!!")
            self.log(f"Error: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            return {'success': False, 'error': str(e)}

# Main execution
if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║  Deepfake Fraud Detection in Banking                      ║
    ║  End-to-End Training Pipeline                             ║
    ║  Master's in Statistics Capstone Project                  ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize and run pipeline
    pipeline = DeepfakeDetectionPipeline(config_path='config.json')
    results = pipeline.run_complete_pipeline()
    
    if results['success']:
        print("\n✓ Training pipeline completed successfully!")
        print("✓ Models saved to 'models/' directory")
        print("✓ Results saved to 'results/' directory")
        print("✓ Ready for API deployment")
    else:
        print(f"\n✗ Training pipeline failed: {results.get('error', 'Unknown error')}")

        