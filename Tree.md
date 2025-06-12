nicu-kangaroo-care-ml/
│   .gitignore
│   docker-compose.yml
│   Dockerfile
│   Makefile
│   pyproject.toml
│   README.md
│   requirements-dev.txt
│   requirements.txt
│
├── .devcontainer/
│   │   devcontainer.json
│   └── Dockerfile
│
├── configs/
│   └── training_config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── scripts/
│   │   generate_data.py
│   │   start_api.py
│   └── train_model.py
│
├── src/
│   │   __init__.py
│   │
│   ├── data/
│   │   │   __init__.py
│   │   │   generator.py
│   │   │   preprocessing.py
│   │   └── schemas.py
│   │
│   ├── models/
│   │   │   __init__.py
│   │   │   architecture.py
│   │   │   evaluation.py
│   │   └── training.py
│   │
│   ├── serving/
│   │   │   __init__.py
│   │   │   api.py
│   │   │   inference.py
│   │   └── schemas.py
│   │
│   └── utils/
│       │   __init__.py
│       │   config.py
│       └── logger.py
│
└── tests/
   │   __init__.py
   │   conftest.py
   │
   ├── test_data/
   │   │   __init__.py
   │   │   test_generator.py
   │   └── test_schemas.py
   │
   ├── test_models/
   │   │   __init__.py
   │   │   test_architecture.py
   │   │   test_evaluation.py
   │   └── test_training.py
   │
   └── test_serving/
       │   __init__.py
       │   test_api.py
       └── test_inference.py