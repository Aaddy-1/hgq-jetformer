# MSc-transformer4physics

## How to create the datasets
```python
python -m src.build_dataset --num_particles [NUM_PARTICLES] --num_feats [NUM_FEATS]
```


## How to run training
From the root folder, run the following commands:
For non-quantized training, run:
```python
python -m src.train --num_particles [NUM_PARTICLES] --num_feats [NUM_FEATS] --num_epochs [NUM_EPOCHS]
```
For performing QAT, add a --quantize flag at the end, so the command would become 
```python
python -m src.train --num_particles [NUM_PARTICLES] --num_feats [NUM_FEATS] --num_epochs [NUM_EPOCHS] --quantize
```
The model will be saved in the `models` folder with the name `[num_particles]_[num_feats].keras`. If using --quantize, then the model will be saved in the `models/quantized` folder with the same name.
The outputs, which contain the training loss in the `.npz` file, along with the plots and metrics will be saved in the `outputs` folder with the same naming convention.

If you want to train a model and not save it in those folders so that they may not overwrite the previous results, you can add a --experiment flag to the end of the training command, all model files and outputs will be saved in `experiment/[EXPERIMENT_NAME]`.
```python
python -m src.train --num_particles [NUM_PARTICLES] --num_feats [NUM_FEATS] --num_epochs [NUM_EPOCHS] --quantize --experiment [EXPERIMENT_NAME]
```

Note: Training will fail unless the corresponding dataset has been created.
