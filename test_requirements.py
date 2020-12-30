import pytest
import os
import subprocess


req = [line.strip() for line in open('requirements.txt') if not line.strip().startswith('#') and len(line.strip()) > 0]


@pytest.fixture
def recorded():
    recorded = [line.strip() for line in open('requirements.txt')
                    if not line.strip().startswith('#') and len(line.strip()) > 0]
    return sorted(recorded, key=str.lower)

@pytest.fixture
def installed():
    pip_freeze = subprocess.check_output('pip freeze', shell=True)
    installed = pip_freeze.decode().split('\n')[:-1]
    installed = [s.lower() for s in installed if not s.startswith('pip-tools')]
    return sorted(installed, key=str.lower)


def test_requirements_up_to_date(recorded, installed):
    for r, i in zip(recorded, installed):
        assert r == i, f'Requirement: {r} should match installed: {i}'





