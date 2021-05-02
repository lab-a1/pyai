rm -rf dist/ build/
bump2version --current-version $(python setup.py --version) patch setup.py
python -m build
python -m twine upload --skip-existing --repository pypi dist/*
