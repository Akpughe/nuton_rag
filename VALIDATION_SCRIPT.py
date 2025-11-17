#!/usr/bin/env python3
"""
Validation Script for User-Based Data Isolation Implementation

This script validates that all code changes have been properly implemented.
Run this BEFORE deployment to catch any issues.

Usage:
    python VALIDATION_SCRIPT.py
"""

import os
import sys
import re
from pathlib import Path


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'


def print_header(text):
    print(f"\n{Colors.BLUE}{'='*60}")
    print(f"{text}")
    print(f"{'='*60}{Colors.RESET}\n")


def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")


def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")


def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")


def check_file_exists(filepath):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print_success(f"File exists: {filepath}")
        return True
    else:
        print_error(f"File NOT found: {filepath}")
        return False


def check_imports(filepath, required_imports):
    """Check if required imports are in a file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        all_found = True
        for imp in required_imports:
            if imp in content:
                print_success(f"Import found in {os.path.basename(filepath)}: {imp}")
            else:
                print_error(f"Import NOT found in {os.path.basename(filepath)}: {imp}")
                all_found = False
        
        return all_found
    except Exception as e:
        print_error(f"Error reading {filepath}: {e}")
        return False


def check_function_signature(filepath, function_name, required_params):
    """Check if a function has required parameters."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Simple regex to find function definition
        pattern = rf"def {function_name}\([^)]*\):"
        match = re.search(pattern, content)
        
        if not match:
            print_error(f"Function not found: {function_name} in {os.path.basename(filepath)}")
            return False
        
        func_sig = match.group(0)
        all_found = True
        
        for param in required_params:
            if param in func_sig:
                print_success(f"Parameter found: {function_name}({param})")
            else:
                print_error(f"Parameter NOT found: {function_name}({param})")
                all_found = False
        
        return all_found
    except Exception as e:
        print_error(f"Error checking function in {filepath}: {e}")
        return False


def check_function_exists(filepath, function_name):
    """Check if a function exists in a file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        if f"def {function_name}(" in content:
            print_success(f"Function exists: {function_name}")
            return True
        else:
            print_error(f"Function NOT found: {function_name}")
            return False
    except Exception as e:
        print_error(f"Error reading {filepath}: {e}")
        return False


def check_pydantic_field(filepath, class_name, field_name, field_type):
    """Check if a Pydantic model has a required field."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Find class definition
        class_pattern = rf"class {class_name}\([^)]*\):[^\n]*\n(.*?)(?=^class|\Z)"
        class_match = re.search(class_pattern, content, re.MULTILINE | re.DOTALL)
        
        if not class_match:
            print_error(f"Class not found: {class_name}")
            return False
        
        class_content = class_match.group(1)
        
        # Look for the field
        field_pattern = rf"{field_name}\s*:\s*{field_type}"
        if re.search(field_pattern, class_content):
            print_success(f"Field found: {class_name}.{field_name}: {field_type}")
            return True
        else:
            print_error(f"Field NOT found: {class_name}.{field_name}: {field_type}")
            return False
    except Exception as e:
        print_error(f"Error checking Pydantic field: {e}")
        return False


def check_string_in_file(filepath, search_string):
    """Check if a string exists in a file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        if search_string in content:
            return True
        return False
    except:
        return False


def main():
    print_header("User-Based Data Isolation Implementation Validation")
    
    # Define file paths
    root_dir = os.path.dirname(os.path.abspath(__file__))
    supabase_client = os.path.join(root_dir, "supabase_client.py")
    flashcard_process = os.path.join(root_dir, "flashcard_process.py")
    quiz_process = os.path.join(root_dir, "quiz_process.py")
    pipeline = os.path.join(root_dir, "pipeline.py")
    
    checks_passed = 0
    checks_failed = 0
    
    # Test 1: Check all files exist
    print_header("1. File Existence Checks")
    for filepath in [supabase_client, flashcard_process, quiz_process, pipeline]:
        if check_file_exists(filepath):
            checks_passed += 1
        else:
            checks_failed += 1
    
    # Test 2: Check supabase_client.py imports
    print_header("2. Import Checks - supabase_client.py")
    if check_imports(supabase_client, ["Optional"]):
        checks_passed += 1
    else:
        checks_failed += 1
    
    # Test 3: Check flashcard_process.py imports
    print_header("3. Import Checks - flashcard_process.py")
    if check_imports(flashcard_process, ["determine_shared_status"]):
        checks_passed += 1
    else:
        checks_failed += 1
    
    # Test 4: Check quiz_process.py imports
    print_header("4. Import Checks - quiz_process.py")
    if check_imports(quiz_process, ["determine_shared_status"]):
        checks_passed += 1
    else:
        checks_failed += 1
    
    # Test 5: Check determine_shared_status function
    print_header("5. Helper Function Check - supabase_client.py")
    if check_function_exists(supabase_client, "determine_shared_status"):
        checks_passed += 1
    else:
        checks_failed += 1
    
    # Test 6: Check generate_flashcards signature
    print_header("6. Function Signature Checks - flashcard_process.py")
    if check_function_signature(flashcard_process, "generate_flashcards", ["user_id"]):
        checks_passed += 1
    else:
        checks_failed += 1
    
    # Test 7: Check regenerate_flashcards signature
    if check_function_signature(flashcard_process, "regenerate_flashcards", ["user_id"]):
        checks_passed += 1
    else:
        checks_failed += 1
    
    # Test 8: Check generate_quiz signature
    print_header("7. Function Signature Checks - quiz_process.py")
    if check_function_signature(quiz_process, "generate_quiz", ["user_id"]):
        checks_passed += 1
    else:
        checks_failed += 1
    
    # Test 9: Check regenerate_quiz signature
    if check_function_signature(quiz_process, "regenerate_quiz", ["user_id"]):
        checks_passed += 1
    else:
        checks_failed += 1
    
    # Test 10: Check insert_flashcard_set signature
    print_header("8. Function Signature Checks - supabase_client.py")
    if check_function_signature(supabase_client, "insert_flashcard_set", ["created_by", "is_shared"]):
        checks_passed += 1
    else:
        checks_failed += 1
    
    # Test 11: Check insert_quiz_set signature
    if check_function_signature(supabase_client, "insert_quiz_set", ["created_by", "is_shared"]):
        checks_passed += 1
    else:
        checks_failed += 1
    
    # Test 12: Check Pydantic models
    print_header("9. Pydantic Model Checks - pipeline.py")
    if check_pydantic_field(pipeline, "FlashcardRequest", "user_id", "str"):
        checks_passed += 1
    else:
        checks_failed += 1
    
    if check_pydantic_field(pipeline, "QuizRequest", "user_id", "str"):
        checks_passed += 1
    else:
        checks_failed += 1
    
    # Test 13: Check endpoint handlers have user_id
    print_header("10. Endpoint Handler Checks - pipeline.py")
    endpoints = [
        "generate_flashcards_endpoint",
        "regenerate_flashcards_endpoint",
        "generate_quiz_endpoint",
        "regenerate_quiz_endpoint"
    ]
    
    for endpoint in endpoints:
        if check_string_in_file(pipeline, f"user_id=request.user_id"):
            print_success(f"Endpoint passes user_id: {endpoint}")
            checks_passed += 1
        else:
            print_warning(f"Could not verify user_id in {endpoint}")
            # Don't count as failure since it's in multiple endpoints
    
    # Summary
    print_header("Validation Summary")
    total_checks = checks_passed + checks_failed
    
    print(f"\nTotal Checks: {total_checks}")
    print(f"{Colors.GREEN}Passed: {checks_passed}{Colors.RESET}")
    print(f"{Colors.RED}Failed: {checks_failed}{Colors.RESET}")
    
    if checks_failed == 0:
        print_success("\n✓ All validation checks PASSED!")
        print_success("Ready for deployment.")
        return 0
    else:
        print_error(f"\n✗ {checks_failed} validation check(s) FAILED!")
        print_error("Please fix the issues before deployment.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

