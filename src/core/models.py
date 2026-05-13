from pydantic import BaseModel, Field


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class IntegrationReport(BaseModel):
    model_name: str
    model_version: str
    run_id: str
    all_passed: bool
    checks: list[CheckResult] = Field(default_factory=list)
