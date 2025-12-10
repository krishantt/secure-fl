#!/usr/bin/env python3
"""
Code Cleanup Script for Secure FL Project

This script identifies and removes redundant code, consolidates model definitions,
and updates imports throughout the project to use the centralized models module.

Usage:
    python scripts/cleanup_redundancy.py [--dry-run] [--verbose]

Options:
    --dry-run: Show what would be changed without making changes
    --verbose: Show detailed output
"""

import argparse
import ast
import re
from pathlib import Path


class CodeCleanup:
    """Main class for cleaning up redundant code"""

    def __init__(
        self, project_root: Path, dry_run: bool = False, verbose: bool = False
    ):
        self.project_root = project_root
        self.dry_run = dry_run
        self.verbose = verbose
        self.changes_made = 0
        self.files_processed = 0

    def log(self, message: str, level: str = "info"):
        """Log messages based on verbosity level"""
        if self.verbose or level == "info":
            prefix = "DRY RUN: " if self.dry_run else ""
            print(f"[{level.upper()}] {prefix}{message}")

    def find_duplicate_model_definitions(self) -> dict[str, list[Path]]:
        """Find duplicate model class definitions across files"""
        model_definitions = {}

        # Common model patterns to look for
        model_patterns = [
            r"class\s+(\w*Model)\s*\(",
            r"class\s+(Simple\w+)\s*\(",
            r"class\s+(Test\w+)\s*\(",
        ]

        for py_file in self.project_root.rglob("*.py"):
            if (
                "models" in str(py_file)
                or "__pycache__" in str(py_file)
                or ".venv" in str(py_file)
                or "venv" in str(py_file)
                or ".pytest_cache" in str(py_file)
                or "node_modules" in str(py_file)
                or ".git" in str(py_file)
                or "site-packages" in str(py_file)
            ):
                continue

            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                for pattern in model_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        if match not in model_definitions:
                            model_definitions[match] = []
                        model_definitions[match].append(py_file)

            except Exception as e:
                self.log(f"Error reading {py_file}: {e}", "error")

        # Filter to only duplicates
        duplicates = {k: v for k, v in model_definitions.items() if len(v) > 1}
        return duplicates

    def extract_inline_model_definitions(
        self, file_path: Path
    ) -> list[tuple[str, str, int, int]]:
        """Extract inline model definitions from a file"""
        models = []

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")

            # Look for class definitions inside functions/methods
            in_function = False
            indent_level = 0
            current_model = None
            model_lines = []
            model_start_line = 0

            for i, line in enumerate(lines):
                stripped = line.strip()

                # Check for function definitions
                if re.match(r"def\s+\w+", stripped):
                    in_function = True
                    indent_level = len(line) - len(line.lstrip())

                # Check for class definitions inside functions
                if (
                    in_function
                    and stripped.startswith("class ")
                    and "nn.Module" in stripped
                ):
                    match = re.match(r"class\s+(\w+)", stripped)
                    if match:
                        current_model = match.group(1)
                        model_lines = [line]
                        model_start_line = i

                # Collect model lines
                elif current_model and (
                    line.startswith(" " * (indent_level + 4)) or stripped == ""
                ):
                    model_lines.append(line)

                # End of model definition
                elif (
                    current_model
                    and len(line) - len(line.lstrip()) <= indent_level
                    and stripped
                ):
                    models.append(
                        (current_model, "\n".join(model_lines), model_start_line, i)
                    )
                    current_model = None
                    model_lines = []

                # End of function
                elif (
                    in_function
                    and len(line) - len(line.lstrip()) <= indent_level
                    and stripped
                    and not stripped.startswith("#")
                ):
                    in_function = False
                    if current_model:
                        models.append(
                            (current_model, "\n".join(model_lines), model_start_line, i)
                        )
                        current_model = None
                        model_lines = []

        except Exception as e:
            self.log(f"Error extracting models from {file_path}: {e}", "error")

        return models

    def update_imports_in_file(self, file_path: Path) -> bool:
        """Update imports in a file to use centralized models"""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # Patterns to replace
            replacements = [
                # Remove inline model definitions and replace with imports
                (
                    r"(\s+)(class\s+(MNISTModel|CIFAR10Model|SimpleModel|SimpleTestModel|TestModel)\s*\([^:]+:[^}]+})",
                    r"\1# Model moved to secure_fl.models\n\1from secure_fl.models import \3",
                ),
                # Add import if models are used but not imported
                (
                    r"^((?:(?!from secure_fl\.models import).)*)(MNISTModel|CIFAR10Model|SimpleModel)",
                    r"from secure_fl.models import \2\n\1\2",
                ),
            ]

            for pattern, replacement in replacements:
                content = re.sub(
                    pattern, replacement, content, flags=re.MULTILINE | re.DOTALL
                )

            # Clean up duplicate imports
            lines = content.split("\n")
            seen_imports = set()
            cleaned_lines = []

            for line in lines:
                if line.strip().startswith("from secure_fl.models import"):
                    if line.strip() not in seen_imports:
                        seen_imports.add(line.strip())
                        cleaned_lines.append(line)
                else:
                    cleaned_lines.append(line)

            content = "\n".join(cleaned_lines)

            if content != original_content:
                if not self.dry_run:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                self.log(f"Updated imports in {file_path}")
                return True

        except Exception as e:
            self.log(f"Error updating imports in {file_path}: {e}", "error")

        return False

    def remove_duplicate_functions(self, file_path: Path) -> bool:
        """Remove duplicate utility functions"""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # Common duplicate function patterns
            duplicate_patterns = [
                # Remove duplicate create_synthetic_dataset functions
                r"def\s+create_synthetic_dataset\s*\([^:]+:[\s\S]*?return\s+.*",
                # Remove duplicate model creation functions that are now in models module
                r"def\s+create_test_model\s*\([^:]+:[\s\S]*?return\s+.*",
                # Remove duplicate parameter conversion functions (already in utils)
                r"def\s+parameters_to_ndarrays\s*\([^:]+:[\s\S]*?return\s+.*",
                r"def\s+ndarrays_to_parameters\s*\([^:]+:[\s\S]*?return\s+.*",
            ]

            for pattern in duplicate_patterns:
                matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
                if matches:
                    self.log(
                        f"Found duplicate function in {file_path}: {pattern[:50]}..."
                    )
                    if not self.dry_run:
                        content = re.sub(
                            pattern, "", content, flags=re.MULTILINE | re.DOTALL
                        )

            if content != original_content:
                if not self.dry_run:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                self.log(f"Removed duplicate functions from {file_path}")
                return True

        except Exception as e:
            self.log(f"Error removing duplicates from {file_path}: {e}", "error")

        return False

    def cleanup_unused_imports(self, file_path: Path) -> bool:
        """Remove unused imports from a file"""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Parse AST to find actual usage
            try:
                tree = ast.parse(content)
            except SyntaxError:
                return False

            # Find all imports
            imports = []
            used_names = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append((alias.name, alias.asname))
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        imports.append((alias.name, alias.asname))
                elif isinstance(node, ast.Name):
                    used_names.add(node.id)
                elif isinstance(node, ast.Attribute):
                    used_names.add(node.attr)

            # Remove unused imports (simplified version)
            lines = content.split("\n")
            cleaned_lines = []

            for line in lines:
                if line.strip().startswith("import ") or line.strip().startswith(
                    "from "
                ):
                    # Keep import if it's likely used (simplified heuristic)
                    import_match = re.search(r"import\s+(\w+)", line)
                    if import_match:
                        imported_name = import_match.group(1)
                        if (
                            imported_name in used_names
                            or imported_name.lower() in content.lower()
                        ):
                            cleaned_lines.append(line)
                        else:
                            self.log(f"Removing unused import: {line.strip()}")
                    else:
                        cleaned_lines.append(line)
                else:
                    cleaned_lines.append(line)

            new_content = "\n".join(cleaned_lines)

            if new_content != content:
                if not self.dry_run:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                return True

        except Exception as e:
            self.log(f"Error cleaning imports in {file_path}: {e}", "error")

        return False

    def consolidate_constants(self) -> bool:
        """Consolidate scattered constants into a central config"""
        constants_found = {}

        # Look for common constants that are duplicated
        constant_patterns = [
            r"DEFAULT_\w+\s*=\s*.*",
            r"CONFIG_\w+\s*=\s*.*",
            r"\w+_TIMEOUT\s*=\s*\d+",
            r"\w+_BATCH_SIZE\s*=\s*\d+",
        ]

        for py_file in self.project_root.rglob("*.py"):
            if (
                "__pycache__" in str(py_file)
                or ".venv" in str(py_file)
                or "venv" in str(py_file)
                or ".pytest_cache" in str(py_file)
                or "node_modules" in str(py_file)
                or ".git" in str(py_file)
                or "site-packages" in str(py_file)
            ):
                continue

            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                for pattern in constant_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        const_name = match.split("=")[0].strip()
                        if const_name not in constants_found:
                            constants_found[const_name] = []
                        constants_found[const_name].append((py_file, match))

            except Exception as e:
                self.log(f"Error reading {py_file}: {e}", "error")

        # Report duplicates
        duplicates = {k: v for k, v in constants_found.items() if len(v) > 1}
        if duplicates:
            self.log("Found duplicate constants:")
            for const, locations in duplicates.items():
                self.log(f"  {const}: {len(locations)} occurrences")

        return len(duplicates) > 0

    def fix_inconsistent_formatting(self, file_path: Path) -> bool:
        """Fix inconsistent code formatting"""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # Fix common formatting issues
            fixes = [
                # Fix inconsistent indentation (convert tabs to spaces)
                (r"\t", "    "),
                # Fix trailing whitespace
                (r"[ \t]+$", "", re.MULTILINE),
                # Fix multiple blank lines
                (r"\n\s*\n\s*\n", "\n\n"),
                # Fix import spacing
                (r"(import\s+\w+)\s*,\s*", r"\1, "),
                # Fix function spacing
                (r"\n(def\s+)", r"\n\n\1"),
                (r"\n(class\s+)", r"\n\n\1"),
            ]

            for pattern, replacement, *flags in fixes:
                flag = flags[0] if flags else 0
                content = re.sub(pattern, replacement, content, flags=flag)

            # Remove excessive blank lines at end
            content = content.rstrip() + "\n"

            if content != original_content:
                if not self.dry_run:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                self.log(f"Fixed formatting in {file_path}")
                return True

        except Exception as e:
            self.log(f"Error fixing formatting in {file_path}: {e}", "error")

        return False

    def run_cleanup(self):
        """Run the complete cleanup process"""
        self.log("Starting code cleanup...")

        # 1. Find and report duplicate model definitions
        self.log("Step 1: Finding duplicate model definitions...")
        duplicates = self.find_duplicate_model_definitions()
        if duplicates:
            self.log("Found duplicate model definitions:")
            for model, files in duplicates.items():
                self.log(
                    f"  {model}: {[str(f.relative_to(self.project_root)) for f in files]}"
                )
        else:
            self.log("No duplicate model definitions found")

        # 2. Process all Python files
        self.log("Step 2: Processing Python files...")
        python_files = [
            f
            for f in self.project_root.rglob("*.py")
            if not any(
                exclude in str(f)
                for exclude in [
                    ".venv",
                    "venv",
                    "__pycache__",
                    ".pytest_cache",
                    "node_modules",
                    ".git",
                    "site-packages",
                ]
            )
        ]

        for py_file in python_files:
            if (
                "__pycache__" in str(py_file)
                or "models/__init__.py" in str(py_file)
                or ".venv" in str(py_file)
                or "venv" in str(py_file)
                or ".pytest_cache" in str(py_file)
                or "node_modules" in str(py_file)
                or ".git" in str(py_file)
                or "site-packages" in str(py_file)
            ):
                continue

            self.files_processed += 1

            # Update imports
            if self.update_imports_in_file(py_file):
                self.changes_made += 1

            # Remove duplicate functions
            if self.remove_duplicate_functions(py_file):
                self.changes_made += 1

            # Clean up unused imports
            if self.cleanup_unused_imports(py_file):
                self.changes_made += 1

            # Fix formatting
            if self.fix_inconsistent_formatting(py_file):
                self.changes_made += 1

        # 3. Consolidate constants
        self.log("Step 3: Checking for duplicate constants...")
        if self.consolidate_constants():
            self.log("Consider consolidating duplicate constants into a central config")

        # 4. Summary
        self.log("Cleanup completed!")
        self.log(f"Files processed: {self.files_processed}")
        self.log(f"Changes made: {self.changes_made}")

        if self.dry_run:
            self.log("This was a dry run. No files were actually modified.")
            self.log("Run without --dry-run to apply changes.")


def main():
    parser = argparse.ArgumentParser(
        description="Clean up redundant code in secure-fl project"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show changes without applying them"
    )
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument(
        "--project-root", type=Path, default=Path.cwd(), help="Project root directory"
    )

    args = parser.parse_args()

    # Validate project structure
    if not (args.project_root / "secure_fl").exists():
        print("Error: This doesn't appear to be the secure-fl project root")
        print(f"Looking for secure_fl directory in: {args.project_root}")
        return 1

    # Run cleanup
    cleanup = CodeCleanup(args.project_root, args.dry_run, args.verbose)
    cleanup.run_cleanup()

    return 0


if __name__ == "__main__":
    exit(main())
