"""Pytest configuration and shared fixtures for enterprise-benchmark tests."""

import pytest


# Configure pytest-asyncio
def pytest_configure(config):
    """Configure pytest with asyncio support."""
    config.addinivalue_line("markers", "asyncio: mark test as async")
