from pydantic import BaseModel


class ScenarioPayload(BaseModel):
    scenarioName: str
    temperatureDelta: float = 0.0
    windMultiplier: float = 1.0
    radiationMultiplier: float = 1.0
    cloudDelta: float = 0.0
