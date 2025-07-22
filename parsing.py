import os
import re
import ast
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JulietParser:
    """
    Parser for Juliet Test Suite following the strategy outlined in juliet_structure.txt
    """
    
    def __init__(self):
        self.test_cases = defaultdict(list)
        self.function_labels = {}
        self.positive_functions = set()
        self.negative_functions = set()
        
    def group_files_by_testcase(self, source_directory: str) -> Dict[str, List[str]]:
        """
        Group files by test case ID based on CWE and test case pattern.
        
        Args:
            source_directory: Path to the Juliet source directory
            
        Returns:
            Dictionary mapping test case IDs to lists of file paths
        """
        test_cases = defaultdict(list)
        
        # Find all source files
        extensions = ['*.c', '*.cpp', '*.java', '*.cs']
        all_files = []
        
        for ext in extensions:
            pattern = os.path.join(source_directory, '**', ext)
            all_files.extend(glob.glob(pattern, recursive=True))
        
        logger.info(f"Found {len(all_files)} source files in {source_directory}")
        
        for file_path in all_files:
            filename = os.path.basename(file_path)
            
            # Extract test case ID using regex
            # Pattern: CWE[number]_Description__variant_details_suffix
            match = re.match(r'(CWE\d+_[^_]+(?:_[^_]+)*?)(?:_\d+[a-z]?)?(?:_bad|_good)?\.(?:c|cpp|java|cs)$', filename)
            
            if match:
                test_case_id = match.group(1)
                test_cases[test_case_id].append(file_path)
            else:
                # Fallback: use filename without extension and numeric suffixes
                base_name = re.sub(r'_\d+[a-z]?(?:_bad|_good)?$', '', os.path.splitext(filename)[0])
                test_cases[base_name].append(file_path)
        
        logger.info(f"Grouped files into {len(test_cases)} test cases")
        return dict(test_cases)
    
    def label_function(self, func_name: str, file_suffix: str, all_functions: Set[str]) -> str:
        """
        Label function as POSITIVE (vulnerable) or NEGATIVE (non-vulnerable).
        
        Args:
            func_name: Name of the function
            file_suffix: Suffix of the file (_bad, _good, etc.)
            all_functions: Set of all function names in the test case
            
        Returns:
            "POSITIVE" for vulnerable functions, "NEGATIVE" for non-vulnerable
        """
        func_lower = func_name.lower()
        
        # Primary flaw indicators - these are POSITIVE (vulnerable)
        if ("bad" in func_lower and not "good" in func_lower) or "badsink" in func_lower:
            return "POSITIVE"
        
        # Non-vulnerable variants - these are NEGATIVE (secure)
        if "good" in func_lower or file_suffix == "_good":
            return "NEGATIVE"
            
        # Check if this is a main function (typically NEGATIVE)
        if func_name == "main":
            return "NEGATIVE"
        
        # For other functions, check the file suffix
        if file_suffix == "_bad":
            return "POSITIVE"
        
        # Default to NEGATIVE per Juliet conventions
        return "NEGATIVE"
    
    def extract_functions_from_c_cpp(self, file_path: str) -> List[Tuple[str, int, int]]:
        """
        Extract function definitions from C/C++ files.
        
        Args:
            file_path: Path to the C/C++ file
            
        Returns:
            List of tuples (function_name, start_line, end_line)
        """
        functions = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Improved regex patterns for C/C++ function extraction
            # Pattern 1: Standard function definitions
            func_patterns = [
                r'^(?:static\s+)?(?:void|int|char\s*\*|double|float|long|short|unsigned\s+\w+|\w+\s*\*?)\s+(\w+)\s*\([^)]*\)\s*$',
                r'^(?:static\s+)?(?:void|int|char\s*\*|double|float|long|short|unsigned\s+\w+|\w+\s*\*?)\s+(\w+)\s*\([^)]*\)\s*\{',
                # Pattern for Juliet-style long function names
                r'^(?:void|int|char\s*\*|double|float|long|short|\w+)\s+([A-Za-z]\w*(?:_\w+)*)\s*\([^)]*\)\s*$',
                r'^(?:void|int|char\s*\*|double|float|long|short|\w+)\s+([A-Za-z]\w*(?:_\w+)*)\s*\([^)]*\)\s*\{'
            ]
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('//') or line.startswith('/*'):
                    i += 1
                    continue
                
                func_name = None
                
                # Try each pattern
                for pattern in func_patterns:
                    match = re.match(pattern, line)
                    if match:
                        func_name = match.group(1)
                        break
                
                if func_name:
                    # Check if opening brace is on the same line or next line
                    start_line = i + 1
                    brace_line = i
                    
                    if '{' not in line:
                        # Look for opening brace on next few lines
                        for j in range(i + 1, min(i + 3, len(lines))):
                            if '{' in lines[j].strip():
                                brace_line = j
                                break
                        else:
                            # No opening brace found, skip this match
                            i += 1
                            continue
                    
                    # Find the end of the function by counting braces
                    brace_count = lines[brace_line].count('{') - lines[brace_line].count('}')
                    end_line = brace_line + 1
                    
                    for j in range(brace_line + 1, len(lines)):
                        line_content = lines[j]
                        brace_count += line_content.count('{') - line_content.count('}')
                        if brace_count == 0:
                            end_line = j + 1
                            break
                    
                    functions.append((func_name, start_line, end_line))
                    
                    # Skip to end of function
                    i = end_line
                else:
                    i += 1
            
        except Exception as e:
            logger.warning(f"Error parsing C/C++ file {file_path}: {e}")
        
        return functions
    
    def extract_functions_from_java(self, file_path: str) -> List[Tuple[str, int, int]]:
        """
        Extract function definitions from Java files.
        
        Args:
            file_path: Path to the Java file
            
        Returns:
            List of tuples (function_name, start_line, end_line)
        """
        functions = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Improved Java method patterns
            method_patterns = [
                r'^\s*(?:public|private|protected)?\s*(?:static)?\s*(?:void|int|String|boolean|double|float|long|short|\w+(?:\[\])?)\s+(\w+)\s*\([^)]*\)\s*$',
                r'^\s*(?:public|private|protected)?\s*(?:static)?\s*(?:void|int|String|boolean|double|float|long|short|\w+(?:\[\])?)\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+\w+(?:\s*,\s*\w+)*)?\s*\{',
                r'^\s*(?:public|private|protected)?\s*(?:static)?\s*(?:void|int|String|boolean|double|float|long|short|\w+(?:\[\])?)\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+\w+(?:\s*,\s*\w+)*)?\s*$'
            ]
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('//') or line.startswith('/*'):
                    i += 1
                    continue
                
                method_name = None
                
                # Try each pattern
                for pattern in method_patterns:
                    match = re.match(pattern, line)
                    if match:
                        method_name = match.group(1)
                        break
                
                if method_name:
                    # Check if opening brace is on the same line or next line
                    start_line = i + 1
                    brace_line = i
                    
                    if '{' not in line:
                        # Look for opening brace on next few lines
                        for j in range(i + 1, min(i + 3, len(lines))):
                            if '{' in lines[j].strip():
                                brace_line = j
                                break
                        else:
                            # No opening brace found, skip this match
                            i += 1
                            continue
                    
                    # Find the end of the method by counting braces
                    brace_count = lines[brace_line].count('{') - lines[brace_line].count('}')
                    end_line = brace_line + 1
                    
                    for j in range(brace_line + 1, len(lines)):
                        line_content = lines[j]
                        brace_count += line_content.count('{') - line_content.count('}')
                        if brace_count == 0:
                            end_line = j + 1
                            break
                    
                    functions.append((method_name, start_line, end_line))
                    
                    # Skip to end of method
                    i = end_line
                else:
                    i += 1
            
        except Exception as e:
            logger.warning(f"Error parsing Java file {file_path}: {e}")
        
        return functions
    
    def extract_functions_from_cs(self, file_path: str) -> List[Tuple[str, int, int]]:
        """
        Extract function definitions from C# files.
        
        Args:
            file_path: Path to the C# file
            
        Returns:
            List of tuples (function_name, start_line, end_line)
        """
        functions = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
            
            # C# method pattern: access_modifier return_type method_name(parameters) {
            method_pattern = r'^\s*(?:public|private|protected|internal)?\s*(?:static)?\s*(?:void|int|string|bool|double|float|long|short|\w+(?:\[\])?)\s+(\w+)\s*\([^)]*\)\s*\{'
            
            for i, line in enumerate(lines):
                match = re.match(method_pattern, line.strip())
                if match:
                    method_name = match.group(1)
                    
                    # Find the end of the method by counting braces
                    brace_count = 1
                    end_line = i + 1
                    
                    for j in range(i + 1, len(lines)):
                        line_content = lines[j]
                        brace_count += line_content.count('{') - line_content.count('}')
                        if brace_count == 0:
                            end_line = j + 1
                            break
                    
                    functions.append((method_name, i + 1, end_line))
            
        except Exception as e:
            logger.warning(f"Error parsing C# file {file_path}: {e}")
        
        return functions
    
    def is_trivial_wrapper(self, file_path: str, func_name: str, start_line: int, end_line: int) -> bool:
        """
        Check if a function is a trivial wrapper (only calls other functions).
        
        Args:
            file_path: Path to the source file
            func_name: Name of the function
            start_line: Starting line number of the function
            end_line: Ending line number of the function
            
        Returns:
            True if the function is a trivial wrapper, False otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            func_body = ''.join(lines[start_line:end_line-1])
            
            # Remove comments and whitespace
            func_body = re.sub(r'//.*', '', func_body)
            func_body = re.sub(r'/\*.*?\*/', '', func_body, flags=re.DOTALL)
            func_body = re.sub(r'\s+', ' ', func_body).strip()
            
            # Count meaningful statements (excluding braces and simple assignments)
            statements = re.findall(r';', func_body)
            
            # If function has only 1-2 statements and contains a function call, it's likely trivial
            if len(statements) <= 2 and re.search(r'\w+\s*\([^)]*\)\s*;', func_body):
                return True
                
        except Exception as e:
            logger.warning(f"Error checking trivial wrapper for {func_name} in {file_path}: {e}")
        
        return False
    
    def parse_test_case(self, test_case_id: str, file_paths: List[str]) -> Dict:
        """
        Parse a complete test case and label functions.
        
        Args:
            test_case_id: ID of the test case
            file_paths: List of file paths in this test case
            
        Returns:
            Dictionary with parsing results
        """
        results = {
            'test_case_id': test_case_id,
            'files': [],
            'positive_functions': [],
            'negative_functions': [],
            'total_functions': 0
        }
        
        all_functions = set()
        
        # First pass: collect all function names
        for file_path in file_paths:
            ext = Path(file_path).suffix.lower()
            
            if ext in ['.c', '.cpp']:
                functions = self.extract_functions_from_c_cpp(file_path)
            elif ext == '.java':
                functions = self.extract_functions_from_java(file_path)
            elif ext == '.cs':
                functions = self.extract_functions_from_cs(file_path)
            else:
                continue
            
            for func_name, _, _ in functions:
                all_functions.add(func_name)
        
        # Second pass: label functions and extract details
        for file_path in file_paths:
            file_result = {
                'path': file_path,
                'functions': []
            }
            
            ext = Path(file_path).suffix.lower()
            filename = os.path.basename(file_path)
            file_suffix = ""
            
            if "_bad" in filename:
                file_suffix = "_bad"
            elif "_good" in filename:
                file_suffix = "_good"
            
            if ext in ['.c', '.cpp']:
                functions = self.extract_functions_from_c_cpp(file_path)
            elif ext == '.java':
                functions = self.extract_functions_from_java(file_path)
            elif ext == '.cs':
                functions = self.extract_functions_from_cs(file_path)
            else:
                continue
            
            for func_name, start_line, end_line in functions:
                # Skip trivial wrappers
                if self.is_trivial_wrapper(file_path, func_name, start_line, end_line):
                    continue
                
                label = self.label_function(func_name, file_suffix, all_functions)
                
                func_result = {
                    'name': func_name,
                    'start_line': start_line,
                    'end_line': end_line,
                    'label': label
                }
                
                file_result['functions'].append(func_result)
                
                if label == "POSITIVE":
                    results['positive_functions'].append(func_result)
                else:
                    results['negative_functions'].append(func_result)
                
                results['total_functions'] += 1
            
            results['files'].append(file_result)
        
        return results
    
    def parse_juliet_dataset(self, source_directory: str) -> Dict:
        """
        Parse the entire Juliet dataset.
        
        Args:
            source_directory: Path to the Juliet source directory
            
        Returns:
            Dictionary with complete parsing results
        """
        logger.info(f"Starting to parse Juliet dataset at {source_directory}")
        
        # Group files by test case
        test_cases = self.group_files_by_testcase(source_directory)
        
        results = {
            'source_directory': source_directory,
            'total_test_cases': len(test_cases),
            'test_cases': {},
            'summary': {
                'total_functions': 0,
                'positive_functions': 0,
                'negative_functions': 0
            }
        }
        
        # Parse each test case
        for test_case_id, file_paths in test_cases.items():
            logger.info(f"Parsing test case: {test_case_id} ({len(file_paths)} files)")
            
            test_case_result = self.parse_test_case(test_case_id, file_paths)
            results['test_cases'][test_case_id] = test_case_result
            
            # Update summary
            results['summary']['total_functions'] += test_case_result['total_functions']
            results['summary']['positive_functions'] += len(test_case_result['positive_functions'])
            results['summary']['negative_functions'] += len(test_case_result['negative_functions'])
        
        logger.info(f"Parsing complete. Found {results['summary']['total_functions']} total functions")
        logger.info(f"Positive functions: {results['summary']['positive_functions']}")
        logger.info(f"Negative functions: {results['summary']['negative_functions']}")
        
        return results


def quick_test_parser(source_directory: str, max_test_cases: int = 5) -> Dict:
    """
    Quick test of the parser on a limited number of test cases.
    
    Args:
        source_directory: Path to the Juliet source directory
        max_test_cases: Maximum number of test cases to process
        
    Returns:
        Dictionary with parsing results
    """
    parser = JulietParser()
    
    # Group files by test case
    test_cases = parser.group_files_by_testcase(source_directory)
    
    # Take only the first few test cases for testing
    limited_test_cases = dict(list(test_cases.items())[:max_test_cases])
    
    results = {
        'source_directory': source_directory,
        'total_test_cases': len(limited_test_cases),
        'test_cases': {},
        'summary': {
            'total_functions': 0,
            'positive_functions': 0,
            'negative_functions': 0
        }
    }
    
    # Parse each test case
    for test_case_id, file_paths in limited_test_cases.items():
        print(f"Testing parser on: {test_case_id} ({len(file_paths)} files)")
        
        test_case_result = parser.parse_test_case(test_case_id, file_paths)
        results['test_cases'][test_case_id] = test_case_result
        
        # Update summary
        results['summary']['total_functions'] += test_case_result['total_functions']
        results['summary']['positive_functions'] += len(test_case_result['positive_functions'])
        results['summary']['negative_functions'] += len(test_case_result['negative_functions'])
    
    return results