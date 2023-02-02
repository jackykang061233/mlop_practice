from typing import Any, List, Optional
import numpy as np

from pydantic import BaseModel
import sys
sys.path.insert(0, '/Users/kangchieh/Documents/GitHub/mlops_practice/')
from classification_model.processing.validation import TitanicDataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]


class MultipleTitanicDataInputs(BaseModel):
    inputs: List[TitanicDataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "PassengerId": 892,
                        "Pclass": 3,
                        "Name": 'Kelly, Mr. James',
                        "Sex": 'male',
                        "Age": 34.5,
                        "SibSp": 0,
                        "Parch": 0,
                        "Ticket": 330911,
                        "Fare": 7.8292,
                        "Cabin": 's',
                        "Embarked": 'Q',                     
                    }
                ]
            }
        }
