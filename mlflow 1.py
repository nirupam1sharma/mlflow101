import mlrun
import pandas as pd
import numpy as np
# run the hub function "describe" with the prepared dataset
run = mlrun.run_function('hub://describe',
                         params={'label_column': 'Admit'},
                         inputs={"table": 'Admission_Predict.csv'})

# view generated artifact names (charts)
run.outputs.keys()

['describe-csv', 'plots/hist.html', 'histograms', 'scatter-2d',
 'violin', 'correlation-matrix-csv', 'correlation', 'dataset']

# View the generated violin plot:
run.artifact("violin").show()