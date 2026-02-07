"""PHI domain package.

Contains vault models, service interfaces, and demo adapters for HIPAA-ready workflows.
"""

from app.phi.db import Base
from app.phi.service import PHIService

__all__ = ["Base", "PHIService"]
