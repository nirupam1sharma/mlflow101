import mlrun
import pandas as pd
# get/create a project and register the data prep and trainer function in it
project = mlrun.get_or_create_project(
    name="Ucla admit", user_project=True, context="./"
)
project.set_function('data_prep.py', 'data-prep', image='mlrun/ml-models',
                     handler='data_preparation', kind="job")
project.set_function('trainer.py', 'trainer', image='mlrun/ml-models',
                     handler='train', kind="job")
project.set_function("hub://auto_trainer", 'evaluate', image='mlrun/ml-models',
                     handler='evaluate', kind="job")
project.save()
# execute the function through MLRun SDK (run locally instead of over k8s)
data_prep = project.run_function(
    "data-prep", inputs={"dataset": "Admission_Predict.csv"}, local=True
)
# get the returned data artifact
train_dataset = data_prep.artifact("train_dataset").as_df()
print(train_dataset.head())