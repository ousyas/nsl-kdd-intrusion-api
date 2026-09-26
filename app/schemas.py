from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ConnectionFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration: int = Field(ge=0)
    protocol_type: str = Field(min_length=1)
    service: str = Field(min_length=1)
    flag: str = Field(min_length=1)
    src_bytes: int = Field(ge=0)
    dst_bytes: int = Field(ge=0)
    land: int = Field(ge=0, le=1)
    wrong_fragment: int = Field(ge=0)
    urgent: int = Field(ge=0)
    hot: int = Field(ge=0)
    num_failed_logins: int = Field(ge=0)
    logged_in: int = Field(ge=0, le=1)
    num_compromised: int = Field(ge=0)
    root_shell: int = Field(ge=0, le=1)
    su_attempted: int = Field(ge=0)
    num_root: int = Field(ge=0)
    num_file_creations: int = Field(ge=0)
    num_shells: int = Field(ge=0)
    num_access_files: int = Field(ge=0)
    num_outbound_cmds: int = Field(ge=0)
    is_host_login: int = Field(ge=0, le=1)
    is_guest_login: int = Field(ge=0, le=1)
    count: int = Field(ge=0)
    srv_count: int = Field(ge=0)
    serror_rate: float = Field(ge=0, le=1)
    srv_serror_rate: float = Field(ge=0, le=1)
    rerror_rate: float = Field(ge=0, le=1)
    srv_rerror_rate: float = Field(ge=0, le=1)
    same_srv_rate: float = Field(ge=0, le=1)
    diff_srv_rate: float = Field(ge=0, le=1)
    srv_diff_host_rate: float = Field(ge=0, le=1)
    dst_host_count: int = Field(ge=0)
    dst_host_srv_count: int = Field(ge=0)
    dst_host_same_srv_rate: float = Field(ge=0, le=1)
    dst_host_diff_srv_rate: float = Field(ge=0, le=1)
    dst_host_same_src_port_rate: float = Field(ge=0, le=1)
    dst_host_srv_diff_host_rate: float = Field(ge=0, le=1)
    dst_host_serror_rate: float = Field(ge=0, le=1)
    dst_host_srv_serror_rate: float = Field(ge=0, le=1)
    dst_host_rerror_rate: float = Field(ge=0, le=1)
    dst_host_srv_rerror_rate: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    prediction: str
    attack_probability: float
    threshold: float
    model_version: str


class BatchPredictionRequest(BaseModel):
    records: list[ConnectionFeatures] = Field(min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class ModelInfoResponse(BaseModel):
    model_version: str
    trained_at: str
    threshold: float
    validation_metrics: dict
    test_metrics: dict

