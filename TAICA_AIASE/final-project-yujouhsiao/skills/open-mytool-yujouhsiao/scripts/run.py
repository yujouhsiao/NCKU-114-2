#!/usr/bin/env python3
"""Open Track: Code Complexity Analyzer — file-based output contract."""
import sys
import json
import os
import ast
import argparse


class ComplexityAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.cyclomatic_complexity = 1
        self.function_count = 0
        self.max_nesting = 0
        self.current_nesting = 0

    def visit_FunctionDef(self, node):
        self.function_count += 1
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.function_count += 1
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.generic_visit(node)

    def visit_If(self, node):
        self.cyclomatic_complexity += 1
        self.current_nesting += 1
        self.max_nesting = max(self.max_nesting, self.current_nesting)
        self.generic_visit(node)
        self.current_nesting -= 1

    def visit_For(self, node):
        self.cyclomatic_complexity += 1
        self.current_nesting += 1
        self.max_nesting = max(self.max_nesting, self.current_nesting)
        self.generic_visit(node)
        self.current_nesting -= 1

    def visit_While(self, node):
        self.cyclomatic_complexity += 1
        self.current_nesting += 1
        self.max_nesting = max(self.max_nesting, self.current_nesting)
        self.generic_visit(node)
        self.current_nesting -= 1

    def visit_ExceptHandler(self, node):
        self.cyclomatic_complexity += 1
        self.generic_visit(node)


def classify_complexity(score: float) -> str:
    if score >= 85:
        return "highly_maintainable"
    elif score >= 70:
        return "maintainable"
    elif score >= 50:
        return "moderate_complexity"
    elif score >= 25:
        return "complex"
    else:
        return "highly_complex"


def analyze_code(code: str) -> dict:
    try:
        tree = ast.parse(code)
        analyzer = ComplexityAnalyzer()
        analyzer.visit(tree)
        loc = len([l for l in code.split('\n') if l.strip() and not l.strip().startswith('#')])
        maintainability = 100
        if analyzer.cyclomatic_complexity > 10:
            maintainability -= (analyzer.cyclomatic_complexity - 10) * 5
        if analyzer.max_nesting > 5:
            maintainability -= (analyzer.max_nesting - 5) * 10
        if loc > 500:
            maintainability -= (loc - 500) // 100
        maintainability = max(0, min(100, maintainability))
        return {
            "status": "success",
            "metrics": {
                "cyclomatic_complexity": analyzer.cyclomatic_complexity,
                "lines_of_code": loc,
                "function_count": analyzer.function_count,
                "max_nesting_level": analyzer.max_nesting,
                "maintainability_score": maintainability,
            },
            "assessment": classify_complexity(maintainability),
        }
    except SyntaxError as e:
        return {"status": "syntax_error", "error": str(e), "metrics": {}}
    except Exception as e:
        return {"status": "error", "error": str(e), "metrics": {}}


def resolve_result_path() -> str:
    return os.environ.get("AIASE_RESULT_PATH") or os.path.join(os.getcwd(), "aiase_result.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--confidence", type=float, default=None)
    a = ap.parse_args()

    code = a.code.replace('\\n', '\n')
    analysis = analyze_code(code)
    confidence = a.confidence if a.confidence is not None else (0.9 if analysis["status"] == "success" else 0.3)

    result = {"task_id": a.task_id, "result": analysis, "confidence": confidence}
    path = resolve_result_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    os.replace(tmp, path)
    print(f"written ok -> {path}")


if __name__ == "__main__":
    main()
