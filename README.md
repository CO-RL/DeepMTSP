# Graph Convolutional Neural Network with Attention Model to Solve Multiple Traveling Salesman Problems

Haojian Liang, Shaohua Wang, Huilai Li

## Installation

See installation instructions [here](INSTALL.md).

## Running the experiments
```
# Generate MILP instances
python 01_generate_instances.py Standard_MTSP
python 01_generate_instances.py MinMax_MTSP
python 01_generate_instances.py Bounded_MTSP
```

### Standard_MTSP(2W Training set)
```
# Generate supervised learning datasets
python 02_generate_datasets.py Standard_MTSP -j 4  # number of available CPUs
# Training
for i in {0..4}
do
    python 03_train_gcnn.py Standard_MTSP -m baseline -s $i
    python 03_train_gcnn.py Standard_MTSP -m attention -s $i
    python 03_train_competitor.py Standard_MTSP -m extratrees -s $i
    python 03_train_competitor.py Standard_MTSP -m svmrank -s $i
    python 03_train_competitor.py Standard_MTSP -m lambdamart -s $i
done
# Test
python 04_test.py MTSP_ori
```

### Standard_MTSP(10W Training set)
```
# Generate supervised learning datasets
python 02_generate_datasets.py Standard_MTSP_10w -j 4  # number of available CPUs
# Training
for i in {0..4}
do
    python 03_train_gcnn.py Standard_MTSP_10w -m baseline -s $i
    python 03_train_gcnn.py Standard_MTSP_10w -m attention -s $i
    python 03_train_competitor.py Standard_MTSP_10w -m extratrees -s $i
    python 03_train_competitor.py Standard_MTSP_10w -m svmrank -s $i
    python 03_train_competitor.py Standard_MTSP_10w -m lambdamart -s $i
done
# Test
python 04_test.py Standard_MTSP_10w
```
### MinMax_MTSP（2W Training set）
```
# Generate supervised learning datasets
python 02_generate_datasets.py MinMax_MTSP -j 4  # number of available CPUs
# Training
for i in {0..4}
do
    python 03_train_gcnn.py MinMax_MTSP -m baseline -s $i
    python 03_train_gcnn.py MinMax_MTSP -m attention -s $i
    python 03_train_competitor.py MinMax_MTSP -m extratrees -s $i
    python 03_train_competitor.py MinMax_MTSP -m svmrank -s $i
    python 03_train_competitor.py MinMax_MTSP -m lambdamart -s $i
done
# Test
python 04_test.py MinMax_MTSP
```

### MinMax_MTSP（10W Training set）
```
# Generate supervised learning datasets
python 02_generate_datasets.py MinMax_MTSP_10w -j 4  # number of available CPUs
# Training
for i in {0..4}
do
    python 03_train_gcnn.py MinMax_MTSP_10w -m baseline -s $i
    python 03_train_gcnn.py MinMax_MTSP_10w -m attention -s $i
    python 03_train_competitor.py MinMax_MTSP_10w -m extratrees -s $i
    python 03_train_competitor.py MinMax_MTSP_10w -m svmrank -s $i
    python 03_train_competitor.py MinMax_MTSP_10w -m lambdamart -s $i
done
# Test
python 04_test.py MinMax_MTSP
```

### Bounded_MTSP（2W Training set）
```
# Generate supervised learning datasets
python 02_generate_datasets.py Bounded_MTSP -j 4  # number of available CPUs
# Training
for i in {0..4}
do
    python 03_train_gcnn.py Bounded_MTSP -m baseline -s $i
    python 03_train_gcnn.py Bounded_MTSP -m attention -s $i
    python 03_train_competitor.py Bounded_MTSP -m extratrees -s $i
    python 03_train_competitor.py Bounded_MTSP -m svmrank -s $i
    python 03_train_competitor.py Bounded_MTSP -m lambdamart -s $i
done
# Test
python 04_test.py Bounded_MTSP
```

### Bounded_MTSP（10W Training set）
```
# Generate supervised learning datasets
python 02_generate_datasets.py Bounded_MTSP_10w -j 4  # number of available CPUs
# Training
for i in {0..4}
do
    python 03_train_gcnn.py Bounded_MTSP_10w -m baseline -s $i
    python 03_train_gcnn.py Bounded_MTSP_10w -m attention -s $i
    python 03_train_competitor.py Bounded_MTSP_10w -m extratrees -s $i
    python 03_train_competitor.py Bounded_MTSP_10w -m svmrank -s $i
    python 03_train_competitor.py Bounded_MTSP_10w -m lambdamart -s $i
done
# Test
python 04_test.py Bounded_MTSP_10w
```

## Citation
Please cite our paper if you use this code in your work.


