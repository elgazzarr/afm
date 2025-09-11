"""Legacy configuration classes - DEPRECATED. Use config_loader.py instead."""

# This file is kept for backward compatibility only.
# New code should use the unified Config class from config_loader.py

from .config_loader import Config

# Alias for backward compatibility
ModelConfig = Config
        
