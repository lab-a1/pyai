rm -rf dist/ build/
python -m build
python -m twine upload --skip-existing --repository pypi dist/*
