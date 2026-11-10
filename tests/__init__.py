"""Test package for route-audit.

The suite uses only the standard library ``unittest`` runner, because the audit
sandbox must be verifiable offline with no third-party test dependency::

    python -m unittest discover -s tests -v

Having this ``__init__.py`` makes ``tests`` a package, so ``discover`` works
with the documented ``-s tests`` invocation without needing ``-p`` or ``-t``.
"""

from __future__ import annotations
