"""
Root conftest.py — ensures the project root is on sys.path so that
`from manga_tracker.models import ...` resolves correctly regardless of
where pytest is invoked from.
"""
import sys
import pathlib

# Add <repo-root>/  (i.e. the folder that CONTAINS manga_tracker/) to sys.path
sys.path.insert(0, str(pathlib.Path(__file__).parent))
