from cwe_db import CVE_DB
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import os



def count_lines_in_functions(db_path):
    """
    Count the number of lines in each function and return the results as lists
    separated by vulnerable (positive) and non-vulnerable (negative) functions.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Count lines in vulnerable functions
    cursor.execute("""
        SELECT code, vuln FROM funcs
        WHERE vuln IS NOT NULL AND vuln != ''
    """)
    positive_results = [(len(code.split('\n')), vuln) for code, vuln in cursor.fetchall()]
    
    # Count lines in non-vulnerable functions
    cursor.execute("""
        SELECT code FROM funcs
        WHERE vuln IS NULL OR vuln = ''
    """)
    negative_results = [len(code.split('\n')) for code, in cursor.fetchall()]
    
    conn.close()
    
    positive_line_counts = [count for count, _ in positive_results]
    
    return positive_line_counts, negative_results

def create_line_count_plots(db_path, language, output_dir="plots"):
    """
    Create plots showing line count distribution similar to the provided examples.
    """
    positive_counts, negative_counts = count_lines_in_functions(db_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the figure
    plt.figure(figsize=(16, 12))
    
    # Plot 1: All Functions Line Count Distribution
    total_functions = len(positive_counts) + len(negative_counts)
    plt.subplot(2, 2, 1)
    all_counts = positive_counts + negative_counts
    plt.hist(all_counts, bins=50, color='blue')
    plt.title(f'All Functions Line Count Distribution\n({total_functions} functions)')
    plt.xlabel('Lines of Code')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Plot 2: Positive vs Negative Functions
    plt.subplot(2, 2, 2)
    plt.hist([positive_counts, negative_counts], bins=50, label=['Positive (Vulnerable)', 'Negative (Secure)'],
             color=['red', 'green'])
    plt.title('Positive vs Negative Functions')
    plt.xlabel('Lines of Code')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Line Count Distribution (Box Plot)
    plt.subplot(2, 2, 3)
    box_data = [positive_counts, negative_counts]
    plt.boxplot(box_data, labels=['Positive', 'Negative'])
    plt.title('Line Count Distribution (Box Plot)')
    plt.ylabel('Lines of Code')
    plt.grid(True)
    
    # Plot 4: Cumulative Distribution
    plt.subplot(2, 2, 4)
    sorted_counts = np.sort(all_counts)
    cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    plt.plot(sorted_counts, cumulative, 'b-')
    plt.title('Cumulative Distribution')
    plt.xlabel('Lines of Code')
    plt.ylabel('Cumulative Probability')
    plt.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'juliet_{language}_stats.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"Plots for {language} saved to {output_path}")
    
    return {
        'total': total_functions,
        'vulnerable': len(positive_counts),
        'non_vulnerable': len(negative_counts),
        'avg_vulnerable_lines': np.mean(positive_counts),
        'avg_non_vulnerable_lines': np.mean(negative_counts),
        'max_lines': np.max(all_counts)
    }

# Generate plots for both databases
print("Generating plots for C codebase...")
c_stats = create_line_count_plots('c_10+.db', 'c')
print(f"C database statistics:")
print(f"- Total functions: {c_stats['total']}")
print(f"- Vulnerable functions: {c_stats['vulnerable']} ({c_stats['vulnerable']/c_stats['total']:.2%})")
print(f"- Non-vulnerable functions: {c_stats['non_vulnerable']} ({c_stats['non_vulnerable']/c_stats['total']:.2%})")
print(f"- Average lines in vulnerable functions: {c_stats['avg_vulnerable_lines']:.2f}")
print(f"- Average lines in non-vulnerable functions: {c_stats['avg_non_vulnerable_lines']:.2f}")
print(f"- Maximum lines: {c_stats['max_lines']}")

print("\nGenerating plots for Java codebase...")
java_stats = create_line_count_plots('java_10+.db', 'java')
print(f"Java database statistics:")
print(f"- Total functions: {java_stats['total']}")
print(f"- Vulnerable functions: {java_stats['vulnerable']} ({java_stats['vulnerable']/java_stats['total']:.2%})")
print(f"- Non-vulnerable functions: {java_stats['non_vulnerable']} ({java_stats['non_vulnerable']/java_stats['total']:.2%})")
print(f"- Average lines in vulnerable functions: {java_stats['avg_vulnerable_lines']:.2f}")
print(f"- Average lines in non-vulnerable functions: {java_stats['avg_non_vulnerable_lines']:.2f}")
print(f"- Maximum lines: {java_stats['max_lines']}")