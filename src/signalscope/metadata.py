"""
Metadata definitions, schema validation models, and signal metadata tracking.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class IQDType(str, Enum):
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    FLOAT32 = "float32"
    FLOAT64 = "float64"


class IQOrdering(str, Enum):
    IQ = "IQ"  # Interleaved I, Q, I, Q...
    QI = "QI"  # Interleaved Q, I, Q, I...


class Endianness(str, Enum):
    LITTLE = "little"
    BIG = "big"


class IQFormatConfig(BaseModel):
    """Configuration for parsing raw interleaved binary IQ files."""
    dtype: IQDType = Field(default=IQDType.FLOAT32, description="Data type of binary samples")
    ordering: IQOrdering = Field(default=IQOrdering.IQ, description="I/Q sample interleaving order")
    endianness: Endianness = Field(default=Endianness.LITTLE, description="Byte order for multi-byte dtypes")
    sample_rate: float = Field(default=1000000.0, gt=0, description="Sample rate in Hz")
    center_freq_hz: Optional[float] = Field(default=None, description="Absolute RF Center Frequency in Hz if known")


class MetadataOrigin(str, Enum):
    FILE_HEADER = "File Header"
    USER_INPUT = "User Specified"
    ASSUMED_DEFAULT = "Assumed Default"


class SignalMetadata(BaseModel):
    """Metadata describing an ingested signal recording."""
    filename: str
    file_format: str  # "WAV" or "RAW_IQ"
    file_size_bytes: int
    num_samples: int
    duration_sec: float
    sample_rate_hz: float
    center_freq_hz: Optional[float] = None
    frequency_is_absolute: bool = False
    metadata_origin: MetadataOrigin = MetadataOrigin.USER_INPUT
    iq_config: Optional[IQFormatConfig] = None
    num_channels: int = 1
    preprocessing_applied: List[str] = Field(default_factory=list)

    @property
    def has_center_frequency(self) -> bool:
        return self.center_freq_hz is not None and self.center_freq_hz > 0


class ParameterStatus(str, Enum):
    CONFIDENT = "CONFIDENT"
    ESTIMATED = "ESTIMATED"
    EXPERIMENTAL = "EXPERIMENTAL"
    UNAVAILABLE = "UNAVAILABLE"


class ParameterResult(BaseModel):
    """Container for an extracted signal parameter with explicit units, method, and quality."""
    name: str
    display_name: str
    value: Any
    unit: str
    method: str
    status: ParameterStatus = ParameterStatus.CONFIDENT
    confidence: float = 1.0  # 0.0 to 1.0
    assumptions: str = ""


class QualitySeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class QualityFlag(BaseModel):
    """Warning or diagnostic flag generated during signal processing."""
    code: str
    severity: QualitySeverity
    message: str
    recommendation: str = ""
