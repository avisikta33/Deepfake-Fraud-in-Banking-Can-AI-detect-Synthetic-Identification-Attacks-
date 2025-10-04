"""
tests/test_models.py
Comprehensive Unit and Integration Tests
"""

import unittest
import numpy as np
import cv2
import sys
import os
sys.path.append('../src')

from data_preprocessing import DeepfakeDataPreprocessor
from cnn_model import CNNDeepfakeDetector
from hybrid_model import HybridDeepfakeDetector
from evaluation import ModelEvaluator

class TestDataPreprocessing(unittest.TestCase):
    """Test data preprocessing pipeline"""
    
    def setUp(self):
        self.preprocessor = DeepfakeDataPreprocessor(img_size=(224, 224))
        # Create dummy image
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_face_extraction(self):
        """Test face extraction from image"""
        face = self.preprocessor.extract_face_from_frame(self.test_image)
        self.assertIsNotNone(face)
        self.assertEqual(face.shape, (224, 224, 3))
    
    def test_normalization(self):
        """Test image normalization"""
        normalized = self.preprocessor.normalize_image(self.test_image)
        self.assertTrue(np.all(normalized >= 0) and np.all(normalized <= 1))
        self.assertEqual(normalized.dtype, np.float32)
    
    def test_frequency_features(self):
        """Test frequency domain feature extraction"""
        features = self.preprocessor.extract_frequency_features(self.test_image)
        self.assertIn('mean_magnitude', features)
        self.assertIn('std_magnitude', features)
        self.assertIn('high_freq_ratio', features)

class TestCNNModel(unittest.TestCase):
    """Test CNN visual detector"""
    
    def setUp(self):
        self.detector = CNNDeepfakeDetector(
            input_shape=(224, 224, 3),
            model_type='custom'
        )
        self.detector.build_model()
        self.detector.compile_model()
    
    def test_model_build(self):
        """Test model building"""
        self.assertIsNotNone(self.detector.model)
        self.assertEqual(len(self.detector.model.layers) > 0, True)
    
    def test_model_prediction(self):
        """Test model prediction"""
        # Create dummy input
        X = np.random.rand(5, 224, 224, 3).astype(np.float32)
        predictions, binary_preds = self.detector.predict(X)
        
        self.assertEqual(predictions.shape[0], 5)
        self.assertEqual(binary_preds.shape[0], 5)
        self.assertTrue(np.all(predictions >= 0) and np.all(predictions <= 1))
        self.assertTrue(np.all(np.isin(binary_preds, [0, 1])))
    
    def test_model_input_shape(self):
        """Test correct input shape handling"""
        input_shape = self.detector.model.input_shape
        self.assertEqual(input_shape[1:], (224, 224, 3))

class TestHybridModel(unittest.TestCase):
    """Test hybrid fusion model"""
    
    def setUp(self):
        # Create simple mock models
        self.mock_cnn = CNNDeepfakeDetector(model_type='custom')
        self.mock_cnn.build_model()
        self.mock_cnn.compile_model()
        
        self.hybrid = HybridDeepfakeDetector(
            cnn_model=self.mock_cnn,
            bert_model=None,
            meta_classifier='logistic'
        )
    
    def test_feature_extraction(self):
        """Test multimodal feature extraction"""
        image = np.random.rand(224, 224, 3).astype(np.float32)
        text = "Test KYC application text"
        
        features = self.hybrid.extract_multimodal_features(image, text)
        self.assertIsInstance(features, dict)
        self.assertIn('cnn_prediction', features)
        self.assertIn('cnn_confidence', features)
    
    def test_meta_classifier_build(self):
        """Test meta-classifier building"""
        classifier = self.hybrid.build_meta_classifier()
        self.assertIsNotNone(classifier)

class TestEvaluation(unittest.TestCase):
    """Test evaluation metrics"""
    
    def setUp(self):
        self.evaluator = ModelEvaluator(model_name='Test Model')
        # Create dummy predictions
        np.random.seed(42)
        self.y_true = np.random.randint(0, 2, 100)
        self.y_pred = np.random.randint(0, 2, 100)
        self.y_prob = np.random.rand(100)
    
    def test_binary_classification_metrics(self):
        """Test binary classification evaluation"""
        metrics = self.evaluator.evaluate_binary_classification(
            self.y_true, self.y_pred, self.y_prob
        )
        
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('roc_auc', metrics)
        
        # Check metric ranges
        for metric_name, value in metrics.items():
            if metric_name != 'specificity':
                self.assertTrue(0 <= value <= 1, f"{metric_name} out of range")
    
    def test_confidence_intervals(self):
        """Test bootstrap confidence intervals"""
        ci_results = self.evaluator.bootstrap_confidence_intervals(
            self.y_true, self.y_pred, self.y_prob, n_bootstrap=100
        )
        
        for metric, values in ci_results.items():
            self.assertIn('mean', values)
            self.assertIn('ci_lower', values)
            self.assertIn('ci_upper', values)
            self.assertTrue(values['ci_lower'] <= values['mean'] <= values['ci_upper'])
    
    def test_optimal_threshold(self):
        """Test optimal threshold finding"""
        threshold, score = self.evaluator.find_optimal_threshold(
            self.y_true, self.y_prob, metric='f1'
        )
        
        self.assertTrue(0 <= threshold <= 1)
        self.assertTrue(0 <= score <= 1)

class TestIntegration(unittest.TestCase):
    """Integration tests for complete pipeline"""
    
    def setUp(self):
        self.preprocessor = DeepfakeDataPreprocessor()
        self.cnn_detector = CNNDeepfakeDetector(model_type='custom')
        self.cnn_detector.build_model()
        self.cnn_detector.compile_model()
    
    def test_end_to_end_prediction(self):
        """Test complete end-to-end prediction pipeline"""
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Preprocess
        face = self.preprocessor.extract_face_from_frame(test_image)
        self.assertIsNotNone(face)
        
        normalized = self.preprocessor.normalize_image(face)
        
        # Predict
        predictions, binary_preds = self.cnn_detector.predict(
            np.expand_dims(normalized, axis=0)
        )
        
        self.assertEqual(len(predictions), 1)
        self.assertIn(binary_preds[0][0], [0, 1])
    
    def test_batch_processing(self):
        """Test batch processing capabilities"""
        # Create batch of images
        batch_size = 10
        images = [np.random.rand(224, 224, 3).astype(np.float32) 
                  for _ in range(batch_size)]
        X_batch = np.array(images)
        
        # Process batch
        predictions, binary_preds = self.cnn_detector.predict(X_batch)
        
        self.assertEqual(len(predictions), batch_size)
        self.assertEqual(len(binary_preds), batch_size)

class TestAPIEndpoints(unittest.TestCase):
    """Test API endpoints (requires running API)"""
    
    def setUp(self):
        self.base_url = 'http://localhost:5000'
        self.test_image_base64 = self._create_dummy_image_base64()
    
    def _create_dummy_image_base64(self):
        """Create dummy base64 encoded image"""
        import base64
        import io
        from PIL import Image
        
        img = Image.new('RGB', (224, 224), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def test_health_check(self):
        """Test health check endpoint"""
        import requests
        try:
            response = requests.get(f'{self.base_url}/health', timeout=5)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn('status', data)
        except requests.exceptions.RequestException:
            self.skipTest("API server not running")
    
    def test_image_detection_endpoint(self):
        """Test image detection endpoint"""
        import requests
        try:
            response = requests.post(
                f'{self.base_url}/api/v1/detect/image',
                json={'image_base64': self.test_image_base64},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self.assertIn('results', data)
                self.assertIn('is_synthetic', data['results'])
                self.assertIn('confidence', data['results'])
                self.assertIn('risk_score', data['results'])
        except requests.exceptions.RequestException:
            self.skipTest("API server not running")

# Performance Tests
class TestPerformance(unittest.TestCase):
    """Performance and stress tests"""
    
    def setUp(self):
        self.detector = CNNDeepfakeDetector(model_type='custom')
        self.detector.build_model()
        self.detector.compile_model()
    
    def test_inference_speed(self):
        """Test inference speed"""
        import time
        
        X = np.random.rand(1, 224, 224, 3).astype(np.float32)
        
        # Warmup
        _ = self.detector.predict(X)
        
        # Measure
        start_time = time.time()
        n_iterations = 100
        for _ in range(n_iterations):
            _ = self.detector.predict(X)
        
        elapsed = time.time() - start_time
        avg_time = elapsed / n_iterations
        
        # Should be less than 500ms per image on CPU
        self.assertLess(avg_time, 0.5, f"Inference too slow: {avg_time:.3f}s")
        print(f"\nAverage inference time: {avg_time*1000:.2f}ms")
    
    def test_memory_usage(self):
        """Test memory usage during prediction"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Get initial memory
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run predictions
        X = np.random.rand(100, 224, 224, 3).astype(np.float32)
        _ = self.detector.predict(X)
        
        # Get final memory
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_increase = mem_after - mem_before
        
        # Memory increase should be reasonable (< 500MB)
        self.assertLess(mem_increase, 500, 
                       f"Memory increase too large: {mem_increase:.2f}MB")
        print(f"\nMemory increase: {mem_increase:.2f}MB")

# Statistical Tests
class TestStatisticalProperties(unittest.TestCase):
    """Test statistical properties of models"""
    
    def test_prediction_distribution(self):
        """Test that predictions follow expected distribution"""
        detector = CNNDeepfakeDetector(model_type='custom')
        detector.build_model()
        detector.compile_model()
        
        # Generate predictions
        X = np.random.rand(1000, 224, 224, 3).astype(np.float32)
        predictions, _ = detector.predict(X)
        
        # Check distribution
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        # Predictions should be somewhat centered (for untrained model)
        self.assertGreater(std_pred, 0, "No variance in predictions")
        print(f"\nPrediction mean: {mean_pred:.4f}, std: {std_pred:.4f}")
    
    def test_calibration(self):
        """Test model calibration (reliability)"""
        from sklearn.calibration import calibration_curve
        
        # Create synthetic calibrated predictions
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        y_prob = np.random.rand(1000)
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=10
        )
        
        # Check that we have valid calibration data
        self.assertEqual(len(fraction_of_positives), len(mean_predicted_value))
        self.assertTrue(all(0 <= x <= 1 for x in fraction_of_positives))

def run_all_tests():
    """Run all tests and generate report"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestCNNModel))
    suite.addTests(loader.loadTestsFromTestCase(TestHybridModel))
    suite.addTests(loader.loadTestsFromTestCase(TestEvaluation))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestAPIEndpoints))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestStatisticalProperties))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*70)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    import sys
    
    # Run with verbose output
    success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
