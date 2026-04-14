.PHONY: quality format checkstyle spotless test verify install-hooks

## Install git pre-commit hooks
install-hooks:
	git config core.hooksPath .githooks

## Auto-fix formatting via Spotless
format:
	./mvnw spotless:apply

## Run Spotless check
spotless:
	./mvnw spotless:check

## Run Checkstyle
checkstyle:
	./mvnw checkstyle:check

## Run unit tests with coverage enforcement (80% line coverage)
test:
	./mvnw test

## Full quality gate: spotless + checkstyle + tests + coverage
quality: spotless checkstyle test

## Full Maven verify (includes all quality checks)
verify:
	./mvnw verify
