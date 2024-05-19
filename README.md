Wasserstein Distributionally Robust Control and State Estimation for Partially Observable Linear Systems
====================================================

This repository includes the source code for implementing 
Linear-Quadratic-Gaussian(LQG), Wasserstein Distributionally Robust Controller(WDRC), Distributionally Robust Linear Quadratic Control (DRLQC),
and Distributionally Robust Control and Estimation(WDRCE)

## Requirements
- Python (>= 3.5)
- numpy (>= 1.17.4)
- scipy (>= 1.6.2)
- matplotlib (>= 3.1.2)
- control (>= 0.9.4)
- **[CVXPY](https://www.cvxpy.org/)**
- **[MOSEK (>= 9.3)](https://www.mosek.com/)**
- (pickle5) if relevant error occurs
- joblib (>=1.4.2, Optional : Used for parallel computation of out-of-sample experiment)
## Additional Requirements to run DRLQC paper
- Pytorch 2.0
- [Pymanopt] https://pymanopt.org/

## Code explanation

To get Figure 1 (a)
First, generate the Total Cost data using
```
python main_param_nonzeromean.py --dist quadratic --noise_dist quadratic
```
After data generation, plot the results using
```
python plot_params4_drlqc_nonzeromean.py --dist quadratic --noise_dist quadratic
```

To get Figure 1 (b)
First, generate the Computation_time data using
```
python main_time.py
```
Note that main_time.py is a time-consuming process.
After Data ge1neration, plot the results using
```
python plot_time.py
```

To get Figure 2 (a)
First, generate the Total Cost data using
```
python main_param_zeromean.py
```
After data generation, plot the results using
```
python plot_params4_drlqc_zeromean.py
```

To get Figure 2 (b)
First, generate the Total Cost data using
```
python main_param_zeromean.py --dist quadratic --noise_dist quadratic
```
After data generation, plot the results using
```
python plot_params4_drlqc_zeromean.py --dist quadratic --noise_dist quadratic
```

To get Figure 3 (a)
First, generate the Total Cost data using
```
python main_param_filter.py
```
After data generation, plot the results using
```
python plot_params4_F.py
```

To get Figure 3 (b)
First, generate the Total Cost data using
```
python main_param_filter.py --dist quadratic --noise_dist quadratic
```
After data generation, plot the results using
```
python plot_params4_F.py --dist quadratic --noise_dist quadratic
```

To get Figure 4 (a)
First, generate the Total Cost data using
```
python main_param_longT.py
```
After data generation, plot the results using
```
python plot_params_long.py
```

To get Figure 4 (b)
First, generate the Total Cost data using
```
python main_param_longT.py --dist quadratic --noise_dist quadratic
```
After data generation, plot the results using
```
python plot_params_long.py --dist quadratic --noise_dist quadratic
```

To get Figure 5 (a)
First, generate the Total Cost data using
```
python main_3.py
```
After data generation, plot the results using
```
python plot_J.py
```

To get Figure 5 (b)
First, generate the Total Cost data using
```
python main_3.py --dist quadratic --noise_dist quadratic
```
After data generation, plot the results using
```
python plot_J.py --dist quadratic --noise_dist quadratic
```

To get Figure 6 (a)
First, generate the Total Cost data using
```
python main_param_3_s21.py
```
After data generation, plot the results using
```
python plot_params_21.py
```

To get Figure 6 (b)
First, generate the Total Cost data using
```
python main_param_3_s21.py --dist quadratic --noise_dist quadratic
```
After data generation, plot the results using
```
python plot_params_21.py --dist quadratic --noise_dist quadratic
```
