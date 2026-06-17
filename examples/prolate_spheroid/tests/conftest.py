import os
import sys
import jax

# make the example root importable (lamb, geometry, losses, constants, models)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Tests compare strict finite-difference and analytic/autodiff quantities.
# Enabling x64 reduces truncation/roundoff artifacts on CPU.
jax.config.update("jax_enable_x64", True)
