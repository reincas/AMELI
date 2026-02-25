import logging
import pytest

# Maximum number of electrons (or holes) for the test functions
# Set 'DEFAULT = False' to run the full test suite
DEBUG = False


class TestContext:
    """ Context information class used to store test function and arguments as a string. """

    info = "Idle"


# Global test context object
context = TestContext()


class TestInfoFilter(logging.Filter):
    """ Filter class used to make the test context available for logging. """

    def filter(self, record):
        """ Add new element 'test_context'to the log record, which is connected to the context info string. """

        record.test_context = context.info
        return True


def pytest_configure(config):
    """ Pytest startup hook. Attach the test context filter to the root logger. """

    root_logger = logging.getLogger()
    root_logger.addFilter(TestInfoFilter())


@pytest.fixture(autouse=True)
def update_context_string(request):
    """ Pytest logging hook. Update function and arguments string in the test context object. """

    # Name of the current test function
    func_name = request.node.originalname or request.node.name

    # Function arguments
    if hasattr(request.node, 'callspec'):
        kwargs = request.node.callspec.params
    else:
        kwargs = request.node.funcargs
    arg_str = ", ".join(repr(value) for value in kwargs.values())

    # Update test context
    context.info = f"{func_name}({arg_str})"

    # Done
    yield

    # Mark remaining log messages when the test run has actually finished
    context.info = "Teardown"


def pytest_html_report_title(report):
    """ Pytest HTML report title. """

    report.title = "AMELI Test Report"


def pytest_exception_interact(node, call, report):
    """ Log exception message of failed tests. """

    logger = logging.getLogger("pytest.error")
    if report.failed:
        exc_name = call.excinfo.typename
        try:
            error_msg = report.longrepr.reprcrash.message
            lineno = report.longrepr.reprcrash.lineno
        except:
            error_msg = str(report.longrepr)
            lineno = "(none)"

        extra_data = {'test_context': context.info}
        logger.error(f"Test FAILED on line {lineno} -> {exc_name}: {error_msg}", extra=extra_data)
