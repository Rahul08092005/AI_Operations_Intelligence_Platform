"""Verify that the Python environment is correctly set up.

Run from the project root:  python scripts/check_env.py
"""
import importlib
import sys

MODULES = [
    "fastapi", "uvicorn", "pydantic", "pydantic_settings", "sqlalchemy", "psycopg",
    "pandas", "numpy", "pyarrow", "openpyxl", "sklearn", "xgboost", "shap", "joblib",
]


def main() -> int:
    print(f"Python {sys.version.split()[0]}")
    failed = []
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
            print(f"  OK    {name:<18} {getattr(mod, '__version__', '?')}")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL  {name:<18} {exc}")

    # Smoke test: train a tiny XGBoost model and explain it with SHAP
    try:
        import numpy as np
        import shap
        import xgboost as xgb

        rng = np.random.default_rng(42)
        X = rng.random((200, 4))
        y = (X[:, 0] > 0.5).astype(int)
        model = xgb.XGBClassifier(n_estimators=10, max_depth=2).fit(X, y)
        values = shap.TreeExplainer(model).shap_values(X[:5])
        print(f"  OK    XGBoost + SHAP smoke test (shap output shape: {np.array(values).shape})")
    except Exception as exc:  # noqa: BLE001
        failed.append("xgboost+shap smoke test")
        print(f"  FAIL  XGBoost + SHAP smoke test: {exc}")

    if failed:
        print(f"\nProblems found: {failed}")
        return 1
    print("\nEnvironment looks good.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
