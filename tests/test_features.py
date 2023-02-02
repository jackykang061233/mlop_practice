from classification_model.config.core import config
from classification_model.processing.features import GroupTransformImputer

import math

def test_group_transformer(sample_input_data):
    # Given
    transformer = GroupTransformImputer(
        variables=config.model_config.categorical_na_mean_group, group=config.model_config.categorical_na_mean_group_param[0], 
        trans=config.model_config.categorical_na_mean_group_param[1]
    )
    assert sample_input_data["PassengerId"].iat[0] == 892

    # When
    subject = transformer.fit_transform(sample_input_data)

    # Then
    assert math.isclose(subject["Age"].iat[-1], 24.027945, abs_tol=1)