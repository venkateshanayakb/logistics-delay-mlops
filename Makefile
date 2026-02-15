# ── Logistics Delay MLOps — Makefile ────────────────────────────
# Developer convenience commands for the project.
#
# Usage:
#   make help           Show all available commands
#   make run            Start the API server
#   make test           Run tests with coverage
#   make docker-up      Start the full monitoring stack
# ────────────────────────────────────────────────────────────────

.PHONY: help run frontend test lint docker-build docker-up docker-down clean

# Default
help: ## Show this help message
	@echo.
	@echo  Logistics Delay MLOps - Available Commands
	@echo  ==========================================
	@echo.
	@echo  run            Start FastAPI server (port 8000)
	@echo  frontend       Start Streamlit frontend (port 8501)
	@echo  test           Run pytest with coverage summary
	@echo  lint           Run flake8 linter
	@echo  docker-build   Build the API Docker image
	@echo  docker-up      Start monitoring stack (API + Prometheus + Grafana)
	@echo  docker-down    Stop monitoring stack
	@echo  clean          Remove caches and temp files
	@echo.

# ── Local Development ───────────────────────────────────────────
run: ## Start FastAPI server
	.venv\Scripts\python.exe -m uvicorn api.main:app --reload --port 8000

frontend: ## Start Streamlit frontend
	.venv\Scripts\python.exe -m streamlit run frontend/app.py

# ── Testing & Quality ──────────────────────────────────────────
test: ## Run pytest with coverage
	.venv\Scripts\python.exe -m pytest tests/ -v --tb=short --cov=api --cov=src --cov-report=term-missing

test-quick: ## Run pytest without coverage (faster)
	.venv\Scripts\python.exe -m pytest tests/ -v --tb=short

lint: ## Run flake8 linter
	.venv\Scripts\python.exe -m flake8 api/ src/ tests/ --statistics --count

# ── Docker ──────────────────────────────────────────────────────
docker-build: ## Build the API Docker image
	docker build -t logistics-api:latest .

docker-up: ## Start full monitoring stack
	docker-compose -f monitoring/docker-compose.yml up -d --build

docker-down: ## Stop monitoring stack
	docker-compose -f monitoring/docker-compose.yml down

docker-logs: ## Follow API container logs
	docker-compose -f monitoring/docker-compose.yml logs -f api

# ── Cleanup ─────────────────────────────────────────────────────
clean: ## Remove caches, temp files, and __pycache__
	@echo Cleaning up...
	@if exist __pycache__ rd /s /q __pycache__
	@if exist .pytest_cache rd /s /q .pytest_cache
	@if exist api\__pycache__ rd /s /q api\__pycache__
	@if exist src\__pycache__ rd /s /q src\__pycache__
	@if exist tests\__pycache__ rd /s /q tests\__pycache__
	@if exist _temp_doc.txt del _temp_doc.txt
	@if exist _col_info.txt del _col_info.txt
	@echo Done!
