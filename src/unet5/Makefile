.PHONY: lint

lint:
	autopep8 --select=W291,W292,W293,W391,E302 --in-place *.py
	flake8 --exit-zero *.py  # see also .flake8

#flake8 --exit-zero --max-line-length=120 --ignore=E128,E402,W504 *.py
