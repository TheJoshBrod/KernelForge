

# AutoLinter
autopep8 --in-place --recursive src/
autopep8 --in-place --recursive benchmarks/


# Reorders import order
find . \
  -type d \( -name venv -o -name .venv -o -name build -o -name __pycache__ \) -prune -o \
  -name '*.py' -print0 | xargs -0 reorder-python-imports


# Linter
pylint src/