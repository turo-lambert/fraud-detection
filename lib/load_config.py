"""Module to define and load config file."""
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class ConfigBaseModel(BaseModel):
    """Base Config."""

    class Config:
        """Pydantic config class to define validation rules for all parameters in ConfigFile."""

        validate_assignment = True
        use_enum_values = True

    @classmethod
    def load_config(cls, config_filepath: Path) -> dict:
        """Load config file using yaml."""
        try:
            with config_filepath.open("rb") as yaml_file:
                config = yaml.safe_load(yaml_file)
        except yaml.YAMLError as e:
            raise Exception(f"Failed loading {config_filepath}: {str(e)}") from e
        return cls(**config)


class PreprocessingConfig(ConfigBaseModel):
    """Preprocessing Config."""

    cols_to_drop: list[str]
    categorical_cols: list[str]
    claim_occured_col: str
    claim_reported_col: str
    types_mapping: Optional[dict[str, str]] = Field(
        default=None, description="Mapping of column names to their data types."
    )


class BaseConfig(ConfigBaseModel):
    """Base Config."""

    preprocessing: PreprocessingConfig
