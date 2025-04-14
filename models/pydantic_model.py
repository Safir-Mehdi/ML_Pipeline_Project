from pydantic import BaseModel, Field
from typing import Optional

class CensusData(BaseModel):
    age: int
    fnlwgt: int
    education_num: int
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    workclass: Optional[str] = Field(None, description="Employment type")
    education: str
    marital_status: str
    occupation: Optional[str] = Field(None, description="Job role")
    relationship: str
    race: str
    sex: str
    native_country: Optional[str] = Field(None, description="Country of origin")