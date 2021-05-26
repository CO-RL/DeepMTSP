# Graph Convolutional Neural Network with Attention Model to Solve Multiple Traveling Salesman Problems

Haojian Liang, Shaohua Wang, Huilai Li

## Installation

See installation instructions [here](INSTALL.md).

## Running the experiments

### MTSP_2w
```
# Generate MILP instances
python 01_generate_instances.py MTSP_ori
# Generate supervised learning datasets
python 02_generate_datasets.py MTSP_ori -j 4  # number of available CPUs
# Training
for i in {0..4}
do
    python 03_train_gcnn.py MTSP_ori -m baseline -s $i
    python 03_train_gcnn.py MTSP_ori -m attention -s $i
    python 03_train_competitor.py MTSP_ori -m extratrees -s $i
    python 03_train_competitor.py MTSP_ori -m svmrank -s $i
    python 03_train_competitor.py MTSP_ori -m lambdamart -s $i
done
# Test
python 04_test.py MTSP_ori
```

### MTSP_10w
```
# Generate supervised learning datasets
python 02_generate_datasets.py MTSP_ori_10w -j 4  # number of available CPUs
# Training
for i in {0..4}
do
    python 03_train_gcnn.py MTSP_ori -m baseline -s $i
    python 03_train_gcnn.py MTSP_ori -m attention -s $i
    python 03_train_competitor.py MTSP_ori -m extratrees -s $i
    python 03_train_competitor.py MTSP_ori -m svmrank -s $i
    python 03_train_competitor.py MTSP_ori -m lambdamart -s $i
done
# Test
python 04_test.py MTSP_ori
```
### minmax-mtsp_2w
```
# Generate MILP instances
python 01_generate_instances.py minmax-mtsp
# Generate supervised learning datasets
python 02_generate_datasets.py minmax-mtsp -j 4  # number of available CPUs
# Training
for i in {0..4}
do
    python 03_train_gcnn.py minmax-mtsp -m baseline -s $i
    python 03_train_gcnn.py minmax-mtsp -m attention -s $i
    python 03_train_competitor.py minmax-mtsp -m extratrees -s $i
    python 03_train_competitor.py minmax-mtsp -m svmrank -s $i
    python 03_train_competitor.py minmax-mtsp -m lambdamart -s $i
done
# Test
python 04_test.py MTSP_ori
```

### minmax-mtsp_10w
```
# Generate supervised learning datasets
python 02_generate_datasets.py minmax-mtsp_10w -j 4  # number of available CPUs
# Training
for i in {0..4}
do
    python 03_train_gcnn.py minmax-mtsp_10w -m baseline -s $i
    python 03_train_gcnn.py minmax-mtsp_10w -m attention -s $i
    python 03_train_competitor.py minmax-mtsp_10w -m extratrees -s $i
    python 03_train_competitor.py minmax-mtsp_10w -m svmrank -s $i
    python 03_train_competitor.py minmax-mtsp_10w -m lambdamart -s $i
done
# Test
python 04_test.py MTSP_ori
```

## Citation
Please cite our paper if you use this code in your work.


