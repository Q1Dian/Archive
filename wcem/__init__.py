import os
from pathlib import Path

# Triton needs a C compiler; the env ships one even when it is not activated.
_cc = Path(__file__).resolve().parent.parent / ".conda" / "bin" / "gcc"
if "CC" not in os.environ and _cc.exists():
    os.environ["CC"] = str(_cc)
