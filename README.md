# Implementation of data distirbution inference attack in FL

## Requirements
- torch 1.7.1
- tensorflow-privacy 0.5.1
- numpy 1.16.2

## Files
> FLModel.py: definition of the FL party and FL server class

> MLModel.py: ML Models

> DPMechanisms.py: generate gaussian noise

> train-$dataset-name$.ipynb: code for DP-based FL model training

> attack-$dataset-name$.ipynb: attack method

## Usag
1. Install [tensorflow-privacy]( https://github.com/tensorflow/privacy)
2. Train FL models
3. Run attack_$dataset$.ipynb to launch attacks

### Parameters for FL training
```python
# code segment in test-$dataset$.ipynb
lr = 0.15				# initial learning rate
fl_param = {
    'output_size': 5,             # number of units in output layer
    'client_num': client_num,     # number of parties
    'model': LogisticRegression,  # ML model
    'data': d,          # dataset
    'lr': lr,           # learning rate
    'E': 100,           # number of local iterations
    'eps': 4.0,         # privacy budget
    'delta': 1e-5,      # approximate DP: (epsilon, delta)-DP
    'q': 0.05,          # sampling rate
    'clip': 2,          # pre-example gradient norm clipping threshold
    'tot_T': 5,         # number of global iterations
    ...
}
```


