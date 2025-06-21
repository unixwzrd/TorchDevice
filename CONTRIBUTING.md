# Contributing to TorchDevice

We love your input! We want to make contributing to TorchDevice as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code lints
6. Issue that pull request!

## Development Setup

1. Clone your fork of the repository
```bash
git clone https://github.com/YOUR_USERNAME/TorchDevice.git
cd TorchDevice
```

2. Create a virtual environment and install dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

3. Set up pre-commit hooks
```bash
pre-commit install
```

## Project Structure

See [Project Structure](docs/project_structure.md) for detailed information about the codebase organization.

## Testing

### Running Tests
```bash
# Run all tests
python run_tests_and_install.py

# Run specific test module
python run_tests_and_install.py --test-only tests/[test-name.py]

# Update expected test outputs
python run_tests_and_install.py --test-only --update-expected tests/[test-name.py]
```

### Writing Tests

1. Create test files in the appropriate directory under `tests/`
2. Follow the existing test structure and naming conventions
3. Use the `PrefixedTestCase` base class for consistent logging
4. Include both positive and negative test cases
5. Test edge cases and error conditions

## Documentation

### API Documentation
- Document all public APIs using docstrings
- Include type hints for all function parameters and return values
- Provide examples in docstrings where appropriate

### README and Guides
- Keep README.md up to date with new features and changes
- Update relevant documentation in the `docs/` directory
- Create new guides for significant features

## Code Style

- Follow PEP 8 guidelines
- Use type hints
- Keep functions focused and single-purpose
- Write clear, descriptive variable and function names
- Comment complex logic
- Use docstrings for all public APIs

## Pull Request Process

1. Update the README.md and documentation with details of changes
2. Update the CHANGELOG.md with a note describing your changes
3. Update the version numbers in relevant files
4. The PR will be merged once you have the sign-off of at least one maintainer

## Issue Reporting

### Bug Reports
When reporting bugs, include:
- A clear description of the issue
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, PyTorch version)
- Relevant log output or error messages

### Feature Requests
When requesting features:
- Clearly describe the feature
- Explain the use case
- Provide examples of how it would be used
- Discuss potential implementation approaches

## External Project Tests

Feel free to try other projects using TorchDevice. If you find issues, please report them as GitHub issues, or submit a patch with a fix in a PR.  I cannot guarantee that all projects will work, and I may not be able to get to every test project, but I appreciate the assistance in debugging and making this work better for the community.

### PyTorch Transformers

Checkout the [test_automation/README.md](test_automation/README.md) for instructions on running the PyTorch Transformers test suite. For how to manage external projects, see [Managing Test Target Projects](#managing-test-target-projects).

to run the transformers tests, use:

```bash
time ( for projectgroup in $( (cd test_projects/transformers/tests; /usr/local/bin/ls -l | grep -v .py  | grep -v __ | awk '{print $NF}' ) ); do python test_automation/run_transformer_tests.py --project_root test_projects/transformers $projectgroup; done ) 2>1 | tee output.log
```

To run the all the tests it takes about 1 hour and 30 minutes.

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.

## References

* [Project Structure](docs/project_structure.md)
* [TorchDevice Behavior](docs/TorchDevice_Behavior.md)
* [API Reference](docs/TorchDevice_Functions.md)
* [CUDA Operations](docs/CUDA-Operations.md) 