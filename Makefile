.PHONY: quality format checkstyle spotless test verify install-hooks build parity reference clean

install-hooks:
	git config core.hooksPath .githooks

format:
	./mvnw spotless:apply

spotless:
	./mvnw spotless:check

checkstyle:
	./mvnw checkstyle:check

build:
	./mvnw compile -q

test:
	./mvnw test

quality: spotless checkstyle test

verify:
	./mvnw verify

parity: reference
	./mvnw test -Dtest=ValidateParityTest

reference:
	python3 tools/validate_against_reference.py > tools/reference_values.json
	@python3 -c "import json; d=json.load(open('tools/reference_values.json')); print(f'Generated {d[\"_num_functions\"]} reference values (numpy {d[\"_numpy_version\"]}, commit {d[\"_numpy_commit\"][:12]})')"

clean:
	./mvnw clean -q
