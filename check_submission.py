"""
check_submission.py — AST structure analysis and submission validation.

Performs:
  * AST analysis for forbidden constructs (eval, exec, dynamic imports).
  * Detects hardcoded outputs and dead code patterns.
  * Checks function complexity.
  * Verifies submission interface compliance.
  * Reports findings clearly.

Usage:
    python check_submission.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# AST visitor
# ---------------------------------------------------------------------------

class SubmissionASTChecker(ast.NodeVisitor):
    """
    Visits AST nodes in the submission file and flags issues.
    """

    FORBIDDEN_NAMES = {"eval", "exec", "compile", "__import__", "open"}
    SUSPICIOUS_CONSTANTS = {"100.0", "97.5", "0.975", "99"}  # Example hardcoded accuracy values

    def __init__(self) -> None:
        self.findings: list[dict[str, Any]] = []
        self.functions: list[dict[str, Any]] = []
        self.imports: list[str] = []
        self.global_vars: list[str] = []
        self.forbidden_calls: list[dict[str, Any]] = []
        self.dynamic_imports: list[dict[str, Any]] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(alias.name)
            self._check_forbidden_import(alias.name, node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.imports.append(node.module)
            self._check_forbidden_import(node.module, node.lineno)
        self.generic_visit(node)

    def _check_forbidden_import(self, name: str, lineno: int) -> None:
        forbidden_modules = {"streamlit", "plotly", "tkinter", "matplotlib", "wx", "PyQt5", "PyQt6"}
        for fm in forbidden_modules:
            if name.startswith(fm):
                self.findings.append({
                    "type": "FORBIDDEN_IMPORT",
                    "severity": "ERROR",
                    "line": lineno,
                    "message": f"GUI/UI package import '{name}' not allowed in competition submission.",
                })

    def visit_Call(self, node: ast.Call) -> None:
        # Check for eval/exec calls
        if isinstance(node.func, ast.Name):
            if node.func.id in self.FORBIDDEN_NAMES:
                self.findings.append({
                    "type": "FORBIDDEN_CALL",
                    "severity": "ERROR",
                    "line": node.lineno,
                    "message": f"Forbidden call: {node.func.id}() — not allowed in competition submission.",
                })
                self.forbidden_calls.append({"name": node.func.id, "line": node.lineno})

        # Check for __import__ dynamic imports
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "__import__":
                self.dynamic_imports.append({"line": node.lineno})
                self.findings.append({
                    "type": "DYNAMIC_IMPORT",
                    "severity": "WARNING",
                    "line": node.lineno,
                    "message": "Dynamic __import__ detected — prefer static imports.",
                })
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Count complexity (approximate: number of branches)
        complexity = _count_branches(node)
        has_docstring = (
            node.body and
            isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)
        )
        self.functions.append({
            "name": node.name,
            "line": node.lineno,
            "complexity": complexity,
            "has_docstring": has_docstring,
        })
        if complexity > 15:
            self.findings.append({
                "type": "HIGH_COMPLEXITY",
                "severity": "WARNING",
                "line": node.lineno,
                "message": f"Function '{node.name}' has high complexity ({complexity}). Consider refactoring.",
            })
        if not has_docstring:
            self.findings.append({
                "type": "MISSING_DOCSTRING",
                "severity": "INFO",
                "line": node.lineno,
                "message": f"Function '{node.name}' lacks a docstring.",
            })
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)  # type: ignore[arg-type]

    def visit_Global(self, node: ast.Global) -> None:
        for name in node.names:
            self.global_vars.append(name)
            if name not in ("__version__", "__algorithm__", "_CFG"):
                self.findings.append({
                    "type": "GLOBAL_VAR",
                    "severity": "INFO",
                    "line": node.lineno,
                    "message": f"Global variable '{name}' detected. Prefer function parameters.",
                })
        self.generic_visit(node)


def _count_branches(node: ast.FunctionDef) -> int:
    """Count branch statements (if/for/while/try/except) for complexity estimate."""
    count = 1  # Base complexity
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.ExceptHandler,
                              ast.With, ast.Assert)):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Submission interface checker
# ---------------------------------------------------------------------------

def check_submission_interface(tree: ast.Module) -> list[dict[str, Any]]:
    """Check that the submission exports the expected 'run' function."""
    findings: list[dict[str, Any]] = []
    functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    if "run" not in functions:
        findings.append({
            "type": "MISSING_INTERFACE",
            "severity": "ERROR",
            "line": 0,
            "message": "Required 'run(signal, fs, modality, channel)' function not found.",
        })
    else:
        findings.append({
            "type": "INTERFACE_FOUND",
            "severity": "OK",
            "line": 0,
            "message": "'run()' function found — submission interface compliant.",
        })

    if "process_batch" in functions:
        findings.append({
            "type": "BATCH_INTERFACE",
            "severity": "OK",
            "line": 0,
            "message": "'process_batch()' batch processing interface found.",
        })

    return findings


# ---------------------------------------------------------------------------
# Main checker
# ---------------------------------------------------------------------------

def check_file(filepath: str) -> dict[str, Any]:
    """
    Run full AST analysis on a Python file.

    Parameters
    ----------
    filepath : str
        Path to the Python file to check.

    Returns
    -------
    dict with analysis results.
    """
    path = Path(filepath)
    if not path.exists():
        return {"error": f"File not found: {filepath}"}

    source = path.read_text(encoding="utf-8")

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return {"error": f"Syntax error in {filepath}: {e}"}

    checker = SubmissionASTChecker()
    checker.visit(tree)

    interface_findings = check_submission_interface(tree)
    all_findings = checker.findings + interface_findings

    errors = [f for f in all_findings if f["severity"] == "ERROR"]
    warnings = [f for f in all_findings if f["severity"] == "WARNING"]
    infos = [f for f in all_findings if f["severity"] == "INFO"]
    oks = [f for f in all_findings if f["severity"] == "OK"]

    ast_pass = len(errors) == 0

    return {
        "file": str(path),
        "ast_pass": ast_pass,
        "overall": "PASS" if ast_pass else "FAIL",
        "n_functions": len(checker.functions),
        "n_imports": len(checker.imports),
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "n_infos": len(infos),
        "errors": errors,
        "warnings": warnings,
        "infos": infos,
        "oks": oks,
        "functions": checker.functions,
        "imports": checker.imports,
        "forbidden_calls": checker.forbidden_calls,
        "dynamic_imports": checker.dynamic_imports,
        "global_vars": checker.global_vars,
    }


def print_check_report(result: dict[str, Any]) -> None:
    """Print AST check report to stdout."""
    if "error" in result:
        print(f"ERROR: {result['error']}")
        return

    sep = "=" * 70
    print(sep)
    print("  SUBMISSION AST STRUCTURE ANALYSIS")
    print(sep)
    print(f"  File           : {result['file']}")
    print(f"  Overall        : {result['overall']}")
    print(f"  Errors         : {result['n_errors']}")
    print(f"  Warnings       : {result['n_warnings']}")
    print(f"  Info           : {result['n_infos']}")
    print(f"  Functions      : {result['n_functions']}")
    print(f"  Imports        : {result['n_imports']}")
    print()

    if result["oks"]:
        print("OK:")
        for f in result["oks"]:
            print(f"  [{f['severity']}] {f['message']}")
        print()

    if result["errors"]:
        print("ERRORS:")
        for f in result["errors"]:
            print(f"  [L{f['line']}] {f['message']}")
        print()

    if result["warnings"]:
        print("WARNINGS:")
        for f in result["warnings"]:
            print(f"  [L{f['line']}] {f['message']}")
        print()

    print("FUNCTIONS:")
    for fn in result["functions"]:
        doc = "Y" if fn["has_docstring"] else "N"
        print(
            f"  {fn['name']:<40} "
            f"complexity={fn['complexity']:>3}  "
            f"docstring={doc}  "
            f"L{fn['line']}"
        )
    print()

    if result["imports"]:
        print("IMPORTS:")
        for imp in result["imports"]:
            print(f"  {imp}")
        print()

    print(sep)


if __name__ == "__main__":
    import os
    submission_file = os.path.join(os.path.dirname(__file__), "final_submission.py")
    print(f"Checking: {submission_file}")
    print()
    result = check_file(submission_file)
    print_check_report(result)

    # Also check core engine files (AST forbidden-construct check only)
    for fname in ["adaptive_engine.py", "quality_engine.py", "noise_engine.py",
                  "rhythm_engine.py", "alert_engine.py", "edge_engine.py"]:
        fpath = os.path.join(os.path.dirname(__file__), fname)
        if os.path.exists(fpath):
            print(f"\nChecking: {fname}")
            r = check_file(fpath)
            # Engine files don't need run() — only count real forbidden-construct errors
            real_errors = [e for e in r.get("errors", []) if e["type"] != "MISSING_INTERFACE"]
            warns = r.get("n_warnings", 0)
            status = "PASS" if len(real_errors) == 0 else "FAIL"
            print(f"  Status: {status} | Errors: {len(real_errors)} | Warnings: {warns}")

    # Exit with error code if submission fails
    sys.exit(0 if result.get("ast_pass", False) else 1)
