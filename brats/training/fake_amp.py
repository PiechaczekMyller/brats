"""
Fake module used to mock amp from apex package:
https://nvidia.github.io/apex/

Rationale:

The apex.amp module is used to speed up pytorch computations and optimize memory usage by using mixed precision learning.
The package is available only for linux (and with experimental version for windows).
To be able to test and program efficiently, this mock is added so the package doesn't raise the ImportError.
Whenever the parts of this mock are used, RuntimeError is raised.
"""


def scale_loss(*args, **kwargs):
    raise RuntimeError("Apex not imported! Check installed packages or disable it.")


def initialize(*args, **kwargs):
    raise RuntimeError("Apex not imported! Check installed packages or disable it.")
