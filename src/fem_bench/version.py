"""Version constants for production FEM-Bench artifacts."""

PACKAGE_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0"
LEGACY_SCHEMA_VERSIONS = frozenset({"pilot-0.1"})
SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION, *LEGACY_SCHEMA_VERSIONS})
