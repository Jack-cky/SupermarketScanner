.PHONY: create install setup run clean

PYTHON_VERSION=3.10.12
ENV_NAME=supermarketscanner
CONDA_PATH=$(shell which conda)
ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh \
	&& conda activate $(ENV_NAME)

create:
	conda create -n $(ENV_NAME) python=$(PYTHON_VERSION) -y
	$(CONDA_PATH) init bash || true

install:
	$(ACTIVATE) && pip install -r requirements.txt

setup: create install

run:
	$(ACTIVATE) && streamlit run demo/streamlit_app.py

clean:
	conda env remove -n $(ENV_NAME) -y
