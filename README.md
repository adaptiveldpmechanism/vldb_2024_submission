The full technical report is [here](https://arxiv.org/ftp/arxiv/papers/2404/2404.01625.pdf)

# Implementation of A3M Mechanism

## Dependencies 
1. A3M needs the optimization module of [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html).
2. To run real-world data experiments (including multidimensional data), additional packages of [pandas](https://pandas.pydata.org/) and [folktable](https://github.com/socialfoundations/folktables) are needed.
3. Code for preprocessing the real-world data is in **real_dataset.ipynb** (you can also try **multidimension.ipynb** to play with the ACSData).

## Synthetic data experiments

> python synthetic_exp_dis.py

This script shows the performance of A3M, in comparison with several other baselines (run the script *_sw.py for SW) over a range of choices of epsilon. The data type and various hyperparameters (e.g., data size, bin size, split ratio) can be specified in the input arguments. For example, for 10000 Gaussian data points, run

> python synthetic_exp_dis.py --data_type=GAUSSIAN --n=10000 --bin_size=0.5 --axRatio=4 --s=0.1

with bin_size (namely, N) set to 0.5; axRatio set to 4; and split ratio set to 0.1.

## Real-world data experiments 

> python real_income.py

> python real_retirement.py

> python real_green_taxi.py

> python multidimension.py

These scripts show the performance of A3M on multi-dimensional data for multiple real-world datasets, in comparison with several other baselines (run the script *_sw.py for SW). 

## Hyperparameter studies of A3M 

Run AAA with different values of bin size (or equivalently, number of bins).
> python vary_N_synthetic.py

Run AAA with different values of aX ratio.
> python vary_ax_synthetic.py

Run AAA with different values of split portion.
> python vary_split_synthetic.py
