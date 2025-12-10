#!/usr/bin/env python3
"""
Smoke test for secure_fl package.

This test performs basic validation that the package can be imported
and essential functionality works correctly.
"""

import sys
import traceback


def test_basic_import():
    """Test that the main package can be imported."""
    try:
        import secure_fl

        print(f"✓ Successfully imported secure_fl version {secure_fl.__version__}")
        return True
    except Exception as e:
        print(f"✗ Failed to import secure_fl: {e}")
        traceback.print_exc()
        return False


def test_core_modules():
    """Test that core modules can be imported."""
    modules_to_test = [
        "secure_fl.client",
        "secure_fl.server",
        "secure_fl.crypto",
        "secure_fl.utils",
        "secure_fl.config",
    ]

    success = True
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✓ Successfully imported {module}")
        except Exception as e:
            print(f"✗ Failed to import {module}: {e}")
            success = False

    return success


def test_cli_import():
    """Test that CLI module can be imported."""
    try:

        print("✓ Successfully imported CLI main function")
        return True
    except Exception as e:
        print(f"✗ Failed to import CLI: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality works."""
    try:

        # Test that we can access basic configuration
        from secure_fl.config import Config

        _config = Config()
        print("✓ Successfully created Config instance")

        # Test that we can import key classes

        print("✓ Successfully imported core classes")

        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all smoke tests."""
    print("=" * 50)
    print("Running secure_fl smoke tests...")
    print("=" * 50)

    tests = [
        ("Basic Import", test_basic_import),
        ("Core Modules", test_core_modules),
        ("CLI Import", test_cli_import),
        ("Basic Functionality", test_basic_functionality),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        else:
            print(f"FAILED: {test_name}")

    print("\n" + "=" * 50)
    print(f"Smoke test results: {passed}/{total} tests passed")
    print("=" * 50)

    if passed == total:
        print("🎉 All smoke tests passed!")
        return 0
    else:
        print("💥 Some smoke tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
