# route-audit — developer entry points
#
# The suite is standard-library only (no pytest, no network), so it stays
# runnable in an offline sandbox.  Every target shells out to `python -m ...`.

PYTHON ?= python
N_TASKS ?= 200
THRESHOLDS ?= 0.5,0.7,0.3
SEED ?= 20260101
AUDIT_OUT ?= runs/audit.json
REPORT_DIR ?= out
TASK_SET ?= builtin-mixed

.PHONY: help test test-quick audit report analyse compare clean clean-tests models tasksets

help:
	@echo "route-audit make targets"
	@echo "  make test         run the full unittest suite (standard library only)"
	@echo "  make test-quick   run the suite with a short traceback"
	@echo "  make audit        run the audit and save raw JSON to $(AUDIT_OUT)"
	@echo "  make report       render the Markdown report + figures into $(REPORT_DIR)"
	@echo "  make analyse      re-analyse a saved audit without spending tokens"
	@echo "  make compare      print the configuration table for a saved audit"
	@echo "  make models       list the model catalogue and prices"
	@echo "  make tasksets     list the builtin task sets"
	@echo "  make clean        remove build/report artefacts"
	@echo ""
	@echo "Overridable: N_TASKS=$(N_TASKS) THRESHOLDS=$(THRESHOLDS) SEED=$(SEED)"

# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
test:
	$(PYTHON) -m unittest discover -s tests -v

test-quick:
	$(PYTHON) -m unittest discover -s tests

# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #
audit:
	$(PYTHON) -m routeaudit audit \
		--task-set $(TASK_SET) \
		--n-tasks $(N_TASKS) \
		--thresholds $(THRESHOLDS) \
		--seed $(SEED) \
		--out $(AUDIT_OUT)

# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
report:
	$(PYTHON) -m routeaudit report \
		--task-set $(TASK_SET) \
		--n-tasks $(N_TASKS) \
		--thresholds $(THRESHOLDS) \
		--seed $(SEED) \
		--save-audit $(AUDIT_OUT) \
		--out-dir $(REPORT_DIR)

# Re-render from a previously saved audit: costs nothing, spends no tokens.
analyse:
	$(PYTHON) -m routeaudit analyse $(AUDIT_OUT) --out $(REPORT_DIR)/analysis.json

compare:
	$(PYTHON) -m routeaudit compare $(AUDIT_OUT)

# --------------------------------------------------------------------------- #
# Introspection
# --------------------------------------------------------------------------- #
models:
	$(PYTHON) -m routeaudit models

tasksets:
	$(PYTHON) -m routeaudit tasksets

# --------------------------------------------------------------------------- #
# Housekeeping
# --------------------------------------------------------------------------- #
clean-tests:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('tests').glob('_tmp')]"

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	$(PYTHON) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('runs', 'out', 'build', 'dist', '.pytest_cache', 'tests/_tmp')]"
