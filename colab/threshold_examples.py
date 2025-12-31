"""
Example script demonstrating dynamic thresholding for security-focused vulnerability detection.
This script shows how to use the new auto-thresholding features.
"""

import torch
import sys
import os

# Add parent directory to path if running from colab folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference_example import (
    load_trained_model,
    find_optimal_threshold,
    evaluate_model,
    predict_sequences
)

def example_1_find_optimal_threshold():
    """Example 1: Find and use optimal threshold for a single model."""
    print("="*80)
    print("EXAMPLE 1: Finding Optimal Threshold")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths (adjust these for your setup)
    model_path = '/content/drive/MyDrive/romeo/10k/model/c_lstm.pt'
    test_sequences_path = '/content/drive/MyDrive/romeo/10k/evaluation/c/test/sequences.pt'
    test_labels_path = '/content/drive/MyDrive/romeo/10k/evaluation/c/test/labels.pt'
    
    # Load model and data
    print("\nLoading model and test data...")
    model, checkpoint = load_trained_model(model_path, device)
    test_sequences = torch.load(test_sequences_path)
    test_labels = torch.load(test_labels_path)
    
    print(f"Test samples: {len(test_sequences)}")
    print(f"Positive samples: {test_labels.sum().item()}")
    
    # Find optimal threshold
    print("\nFinding optimal threshold (optimizing F2 with min_recall=0.90)...")
    optimal_thresh, optimal_metrics, threshold_analysis = find_optimal_threshold(
        model, test_sequences, test_labels,
        device=device,
        metric='f2',
        min_recall=0.90
    )
    
    print(f"\nOptimal Threshold: {optimal_thresh:.4f}")
    print(f"  Precision: {optimal_metrics['precision']:.4f}")
    print(f"  Recall: {optimal_metrics['recall']:.4f}")
    print(f"  F1: {optimal_metrics['f1']:.4f}")
    print(f"  F2: {optimal_metrics['f2']:.4f}")
    
    # Show different threshold options
    print("\nThreshold Strategy Comparison:")
    print("-" * 80)
    for name, info in threshold_analysis.items():
        print(f"\n{name.upper()}: threshold={info['threshold']:.4f}")
        print(f"  {info['description']}")
        print(f"  Precision: {info['precision']:.4f}")
        print(f"  Recall: {info['recall']:.4f}")
        print(f"  F2: {info['f2']:.4f}")
    
    # Evaluate with optimal threshold
    print("\n" + "="*80)
    print("Evaluating with optimal threshold...")
    metrics = evaluate_model(model, test_sequences, test_labels, device, threshold=optimal_thresh)
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['true_positives']:4d}")
    print(f"  True Negatives:  {metrics['true_negatives']:4d}")
    print(f"  False Positives: {metrics['false_positives']:4d} (safe code flagged)")
    print(f"  False Negatives: {metrics['false_negatives']:4d} (missed vulnerabilities) ⚠️")
    
    print(f"\nSecurity Metrics:")
    print(f"  Recall: {metrics['recall']:.4f} (% of vulnerabilities detected)")
    print(f"  F2 Score: {metrics['f2']:.4f} (security-focused metric)")
    
    print("\n✓ Example 1 Complete")


def example_2_compare_thresholds():
    """Example 2: Compare performance at different thresholds."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Comparing Different Thresholds")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths (adjust these for your setup)
    model_path = '/content/drive/MyDrive/romeo/10k/model/c_lstm.pt'
    test_sequences_path = '/content/drive/MyDrive/romeo/10k/evaluation/c/test/sequences.pt'
    test_labels_path = '/content/drive/MyDrive/romeo/10k/evaluation/c/test/labels.pt'
    
    # Load model and data
    print("\nLoading model and test data...")
    model, checkpoint = load_trained_model(model_path, device)
    test_sequences = torch.load(test_sequences_path)
    test_labels = torch.load(test_labels_path)
    
    # Test different thresholds
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    print(f"\n{'Threshold':<12} {'Recall':<10} {'Precision':<12} {'F1':<10} {'F2':<10} {'FN':<8} {'FP':<8}")
    print("-" * 80)
    
    for threshold in thresholds:
        metrics = evaluate_model(model, test_sequences, test_labels, device, threshold=threshold)
        
        print(f"{threshold:<12.2f} "
              f"{metrics['recall']:<10.4f} "
              f"{metrics['precision']:<12.4f} "
              f"{metrics['f1']:<10.4f} "
              f"{metrics['f2']:<10.4f} "
              f"{metrics['false_negatives']:<8d} "
              f"{metrics['false_positives']:<8d}")
    
    print("\nObservations:")
    print("  - Lower threshold → Higher recall (catch more vulnerabilities)")
    print("  - Lower threshold → More false positives (more code to review)")
    print("  - For security: Prioritize low false negatives (FN)")
    
    print("\n✓ Example 2 Complete")


def example_3_security_focused_evaluation():
    """Example 3: Evaluate with different security levels."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Security-Level Focused Evaluation")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths (adjust these for your setup)
    model_path = '/content/drive/MyDrive/romeo/10k/model/c_lstm.pt'
    test_sequences_path = '/content/drive/MyDrive/romeo/10k/evaluation/c/test/sequences.pt'
    test_labels_path = '/content/drive/MyDrive/romeo/10k/evaluation/c/test/labels.pt'
    
    # Load model and data
    print("\nLoading model and test data...")
    model, checkpoint = load_trained_model(model_path, device)
    test_sequences = torch.load(test_sequences_path)
    test_labels = torch.load(test_labels_path)
    
    # Different security levels
    security_levels = {
        'MAXIMUM': 0.95,
        'HIGH': 0.90,
        'MODERATE': 0.85,
        'STANDARD': 0.80
    }
    
    print("\nSecurity Level Analysis:")
    print("=" * 80)
    
    for level_name, min_recall in security_levels.items():
        print(f"\n{level_name} Security (min_recall={min_recall}):")
        print("-" * 80)
        
        # Find optimal threshold for this security level
        optimal_thresh, optimal_metrics, _ = find_optimal_threshold(
            model, test_sequences, test_labels,
            device=device,
            metric='f2',
            min_recall=min_recall
        )
        
        # Evaluate
        metrics = evaluate_model(model, test_sequences, test_labels, device, threshold=optimal_thresh)
        
        print(f"  Threshold: {optimal_thresh:.4f}")
        print(f"  Recall: {metrics['recall']:.4f} (detecting {metrics['recall']*100:.1f}% of vulnerabilities)")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  F2 Score: {metrics['f2']:.4f}")
        print(f"  False Negatives: {metrics['false_negatives']} (missed vulnerabilities)")
        print(f"  False Positives: {metrics['false_positives']} (safe code to review)")
        
        # Recommendation
        if level_name == 'MAXIMUM':
            print("  💡 Use for: Critical infrastructure, compliance requirements")
        elif level_name == 'HIGH':
            print("  💡 Use for: Security-focused applications (RECOMMENDED)")
        elif level_name == 'MODERATE':
            print("  💡 Use for: Balanced security with limited review resources")
        else:
            print("  💡 Use for: Lower-risk applications")
    
    print("\n✓ Example 3 Complete")


def example_4_batch_prediction_with_threshold():
    """Example 4: Make predictions on new code samples."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Batch Prediction with Custom Threshold")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths (adjust these for your setup)
    model_path = '/content/drive/MyDrive/romeo/10k/model/c_lstm.pt'
    test_sequences_path = '/content/drive/MyDrive/romeo/10k/evaluation/c/test/sequences.pt'
    
    # Load model and sequences
    print("\nLoading model...")
    model, checkpoint = load_trained_model(model_path, device)
    sequences = torch.load(test_sequences_path)
    
    # Use a security-focused threshold (lower than 0.5)
    security_threshold = 0.35
    
    print(f"\nMaking predictions with security threshold: {security_threshold}")
    print(f"Number of samples: {len(sequences)}")
    
    # Get predictions
    probabilities, predictions = predict_sequences(
        model, sequences,
        device=device,
        threshold=security_threshold
    )
    
    # Analyze results
    num_flagged = predictions.sum().item()
    avg_prob = probabilities.mean().item()
    max_prob = probabilities.max().item()
    min_prob = probabilities.min().item()
    
    print(f"\nPrediction Results:")
    print(f"  Flagged as vulnerable: {num_flagged}/{len(sequences)} ({num_flagged/len(sequences)*100:.1f}%)")
    print(f"  Average probability: {avg_prob:.4f}")
    print(f"  Max probability: {max_prob:.4f}")
    print(f"  Min probability: {min_prob:.4f}")
    
    # Show high-confidence predictions
    high_conf_vulnerable = (probabilities > 0.8).sum().item()
    high_conf_safe = (probabilities < 0.2).sum().item()
    uncertain = ((probabilities >= 0.2) & (probabilities <= 0.8)).sum().item()
    
    print(f"\nConfidence Distribution:")
    print(f"  High confidence vulnerable (p > 0.8): {high_conf_vulnerable}")
    print(f"  Uncertain (0.2 ≤ p ≤ 0.8): {uncertain}")
    print(f"  High confidence safe (p < 0.2): {high_conf_safe}")
    
    print("\n💡 Tip: Review high-confidence vulnerable samples first")
    
    print("\n✓ Example 4 Complete")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("DYNAMIC THRESHOLDING EXAMPLES")
    print("Security-Focused Vulnerability Detection")
    print("="*80)
    
    # Check if paths need to be updated
    print("\n⚠️  Before running: Update file paths in each example function")
    print("   to match your environment (Colab or local)")
    
    # Uncomment the examples you want to run:
    
    # Example 1: Basic usage - find and use optimal threshold
    # example_1_find_optimal_threshold()
    
    # Example 2: Compare different thresholds
    # example_2_compare_thresholds()
    
    # Example 3: Compare security levels
    # example_3_security_focused_evaluation()
    
    # Example 4: Make predictions with custom threshold
    # example_4_batch_prediction_with_threshold()
    
    print("\n" + "="*80)
    print("To run examples, uncomment the desired function calls above")
    print("="*80)
