# Testing Guidelines

## Core Principle
- **TEST DRIVEN DEVELOPMENT**: Write the unit tests first. Then write code that passes those tests. This reduces wasteful design.
- **ALWAYS RUN TESTS AFTER GENERATION**: After generating any test code, IMMEDIATELY execute the tests to verify they compile and pass. This is EXTREMELY IMPORTANT and non-negotiable.
- **COMPLETION CRITERIA**: No suggestions in the agentic loop are complete until:
  1. **Code compiles successfully**: `./mvnw compile` must exit with status 0
  2. **All unit tests pass**: `./mvnw test` must exit with status 0

## Unit Testing
- **Mocking**: Use `@MockitoBean` for mocking Spring beans in tests (`@MockBean` is deprecated)
- **Focus**: Test core functionality and main business logic
- **Function Names**: use camel casing instead of snake_casing for test functions. Correct Example: homePageShouldReturnHomeTemplate.
- **Avoid Over-Testing**: Don't test implementation details to maintain flexibility for future changes
- **Single Concept**: Test one concept per test method
- **Descriptive Names**: Use meaningful test names that describe the behavior being tested
- **Repository Tests**: Use `@DataJdbcTest` for repository tests
- **Security Configuration**: For @WebMvcTest, use `@Import(SecurityConfiguration.class)` the current security setup. DO NOT Create a TestSecurityConfiguration.
- **Isolation**: Mock dependencies to isolate the unit under test
  - Controllers should depend on mocked Services
  - Services should mock Repository dependencies
  - Repositories should be integration tested with `@DataJdbcTest` and flyway migrations validated
- **IMMEDIATE EXECUTION**: After writing any test, immediately run it to ensure it compiles and passes

## Integration Testing
- **Web Page Testing**: Focus on:
  - Page loading correctly (status code 200)
  - Basic content rendering properly
  - Core functionality working as expected
- **Avoid Brittle Tests**: Don't check specific UI element positioning or styling
- **Business Outcomes**: Prefer testing business outcomes over implementation details
- **Test Annotations**:
  - Use `@WebMvcTest` for controller tests
  - Use `@SpringBootTest` for full integration tests
  - Use test containers for database integration tests

## API Documentation
- **Spring RestDocs**: Generate API documentation with Spring RestDocs
- **Test-Driven Docs**: Write tests that document API endpoints

## Test Data Management
- **Test Isolation**: Each test should be independent and repeatable
- **Clean State**: Use `@Transactional` or `@DirtiesContext` appropriately
- **Realistic Data**: Use realistic test data that reflects production scenarios

## Repository Testing — JSONB Column Validation
- **JsonString for JSONB**: Any entity field mapped to a PostgreSQL `jsonb` column MUST use `JsonString` type (from `com.sparrowlogic.cheerleader.config.JsonString`), never plain `String`. Spring Data JDBC sends `String` as VARCHAR, which PostgreSQL rejects for JSONB columns.
- **Verification skill**: When adding or modifying entities with JSONB columns, cross-reference the Flyway migration (`src/main/resources/db/migration/`) to identify all `jsonb` columns, then verify the corresponding entity field uses `JsonString`. Run the `@DataJdbcTest` repository test with a save/findById round-trip to confirm the type mapping works against real PostgreSQL.
- **Raw SQL exemption**: JSONB columns read via raw SQL queries (e.g., `JdbcTemplate`) don't need `JsonString` — the `PgObjectToStringReadConverter` handles the read side automatically.
