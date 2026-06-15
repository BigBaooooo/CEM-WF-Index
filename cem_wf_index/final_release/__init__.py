"""Final CEM-WF-Index paper-review implementation package.

The modules in this package contain the final calibrated analogue retrieval
source code used by the paper-review release. Command-line scripts are thin
wrappers around these package modules.
"""

from cem_wf_index.final_release.online_inference import METHOD_DEFINITION, METHOD_NAME, SELECTED_PROFILE

__all__ = ["METHOD_DEFINITION", "METHOD_NAME", "SELECTED_PROFILE"]
