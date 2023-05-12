import mlrun
import lightgbm as lgbm
import pandas as pd
from sklearn.model_selection import train_test_split
def train(
    train_set: pd.DataFrame,
    label_column: str = "Admit",
    model_name: str = "lgbm_UCLAAdmit",
    boosting_type: str = "gbdt",
    subsample: float = 0.8,
    min_split_gain: float = 0.5,
    min_child_samples: int = 10,
):
    y = train_set[label_column]
    train_df = train_set.drop(columns=[label_column])
    x_train, x_test, y_train, y_test = train_test_split(
        train_df, y, random_state=84, test_size=0.20
    )
    model = lgbm.LGBMRegressor(
        boosting_type=boosting_type,
        subsample=subsample,
        min_split_gain=min_split_gain,
        min_child_samples=min_child_samples,
    )

    # -------------- The only line you need to add for MLOps --------------------
    # Wraps the model with MLOps (test set is provided for analysis)
    apply_mlrun(model=model, model_name=model_name, x_test=x_test, y_test=y_test)
    model.fit(X=x_train, y=y_train)
    return model