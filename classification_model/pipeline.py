from feature_engine.encoding import OrdinalEncoder
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,
)
from feature_engine.transformation import LogTransformer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from classification_model.config.core import config
from classification_model.processing import features as pp

pipe = Pipeline([
    ('mean_imputation', MeanMedianImputer(
        imputation_method='mean', variables=config.model_config.numerical_na_mean
    )),
    ('group_imputation', pp.GroupTransformImputer(
        variables=config.model_config.categorical_na_mean_group, group=config.model_config.categorical_na_mean_group_param[0], 
        trans=config.model_config.categorical_na_mean_group_param[1]
    )),
    ('log', LogTransformer(variables=config.model_config.numericals_log_vars)),
    ('categorical_encoder', OrdinalEncoder(
        encoding_method='ordered', variables=config.model_config.categorical_vars)),
    ('scaler', MinMaxScaler()),
    ('svm', SVC()),
    
])

