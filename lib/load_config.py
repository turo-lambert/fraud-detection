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

    cols_to_drop: list[str] = Field(..., description="List of columns to drop if they are not useful for modeling.")
    categorical_cols: list[str] = Field(..., description="List of categorical columns to be encoded.")
    claim_occured_col: str = Field(..., description="Column name for the claim occurrence date.")
    claim_reported_col: str = Field(..., description="Column name for the claim reported date.")
    types_mapping: Optional[dict[str, str]] = Field(
        default=None, description="Mapping of column names to their data types."
    )


class HPTuningConfig(ConfigBaseModel):
    """Hyperparameters Config."""

    float_hp: dict[str, list[float]] = Field(
        ...,
        description="Dictionary containing float hyperparameters and their search space.",
    )
    int_hp: dict[str, list[float]] = Field(
        ...,
        description="Dictionary containing integer hyperparameters and their search space.",
    )
    log_hp: dict[str, list[float]] = Field(
        ...,
        description="Dictionary containing log hyperparameters and their search space.",
    )
    n_trials: int = Field(default=50, description="Number of trials to run for hyperparameter tuning.")


class ModellingConfig(ConfigBaseModel):
    """Modelling Config."""

    params: dict[str, str | int | float]
    hp_tuning: HPTuningConfig


class BaseConfig(ConfigBaseModel):
    """Base Config."""

    preprocessing: PreprocessingConfig
    model: ModellingConfig
