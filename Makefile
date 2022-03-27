DIR := src
VENV := .venv
RUN := poetry run

FILES?=$(DIR)


.PHONY: clean


all: help

help:
	@echo "Commands:"
	@echo "  \033[00;32minstall\033[0m   - setup virtual environment with extra configuration."
	@echo "  \033[00;32mformat\033[0m - format code with Black. Override \033[00;33mFILES\033[0m variable to format certain file or files."
	@echo "  \033[00;32mlint\033[0m   - run linting in the code base. Override \033[00;33mFILES\033[0m variable to lint certain file or files."
	@echo "  \033[00;32mclean\033[0m  - remove all Python artifacts."

app:
	@echo "[ \033[00;32mStarting application\033[0m ]"
	@. $(VENV)/bin/activate && gunicorn --config src/app/gunicorn_config.py "src.app.main:get_app()"

install:
	@poetry config virtualenvs.in-project true
	@poetry install
	@echo "[ \033[00;32mPoetry setup completed. You are good to go!\033[0m ]"

format:
	$(RUN) black $(FILES)

lint:
	@echo "[ \033[00;33mFlake8 linter\033[0m ]"
	$(RUN) flake8 $(FILES)
	@echo "[ \033[00;33mBlack linter\033[0m ]"
	$(RUN) black --check $(FILES)

clean:
	find . -name "*.pyc" -exec rm -f {} +
	find . -name "*.pyo" -exec rm -f {} +
	find . -name "*~" -exec rm -f {} +
	find . -name "__pycache__" -exec rm -fr {} +
