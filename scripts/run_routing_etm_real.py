"""Compatibility entry point for archived Routing ETM experiment commands.

The maintained model and runner are ``ContextualSparseETM`` and
``scripts.run_contextual_sparse_etm``.  This module remains only so commands
recorded in frozen experiment manifests continue to execute.
"""

from scripts.run_contextual_sparse_etm import main

if __name__ == "__main__":
    raise SystemExit(main())
