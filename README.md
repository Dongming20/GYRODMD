GYRODMD is a python code for Dynamic Mode Decomposition analysis for Gyrokinetics.

Please put the gyrodmd.py file in the same directory of your python code and import it using following:
```bash
from gyrodmd import GYRODMD
```
To use it, define the GYRODMD class and fit it as follow:
```bash
gyro_dmd = GYRODMD(rank=0)
gyro_dmd.fit(data=your_data[:,:], dt=your_dt)
```
The data should be 2D np.array and its shape is (N_space, N_time), dt is the time step.
After fitting it, the ith omega and mode are:
```bash
growth_rate = gyro_dmd.omega[i].real
frequency = gyro_dmd.omega[i].imag
mode[:,i] = gyro_dmd.mode[:,i]
```

To use Total Least Square (TLS) DMD, re-define the GYRODMD with tls_rank argument:
```bash
gyro_dmd = GYRODMD(rank=0, tls_rank=number)
```
where if tls_rank=0, this will automatically compute the optimal tls_rank for you. Default is "None". 

To use Higher order + TLS_DMD, define the GYRODMD class as follow:
```bash
gyro_dmd = GYRODMD(rank=0, tls_rank=number, DMD_order=number)
```
