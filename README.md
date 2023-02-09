# bayesian-attractor-model-python-cpp-

"bayesian-attractor-model-python-cpp-" is a program implementing core functions of Yuragi learning [1] with Python.
To speed up the program, this core function is written in C++. And I made a simple interface program written in Python using pybind11. 


Book [1] says
```
Yuragi, which is the Japanese term describing the noise or fluctuations
```

Yuragi learning is a new artificial intelligence paradigm that models the top-down learning mechanism of the human brain. The above mentioned core functions of Yuragi learning is "Perception of Observation" that is based on the Bayesian attractor model [2]. 


# Requirement
bayesian-attractor-model-python-cpp- used [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) and [pybind11](https://github.com/pybind/pybind11).

## Eigen

[Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) is "a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms." To install it, please see: https://eigen.tuxfamily.org/dox/GettingStarted.html

### note
Our program assumed that Eigen is placed in the same directory. see global.h.

## pybind11

[pybind11](https://github.com/pybind/pybind11) is "a lightweight header-only library that exposes C++ types in Python and vice versa, mainly to create Python bindings of existing C++ code." To install it, 
```pip
pip install pybind11
```

# Usage

(Environments under Ubuntu 20.04 is tested)

At first, compile bam-module.cpp and create a library file (bam-module.so)
```
g++ -O3 -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` bam-module.cpp -o bam_module.so
```

# Description
- bam-module.cpp
  - Main program implements the Bayesian attractor model [2].
  - This program describes the Bayesian attractor model, takes multidimensional series data as input and outputs a confidence level indicating which of the pre-stored alternatives it is similar to. 

- global.h
  - global variables are written in this file.

- const.h
  - const variables are written in this file.

- Yuragi_sample.py
  - This Python program is a sample program that uses the functions provided in the Bayesian attractor model program written in C++.
  - First, it create an instance of the bayesian attractor model class written in C++. 
  ```
  BAM = bam_module.bam(3, 3, 2, 1)
  ```
  - Then, call methods of bam-module.cpp
    - BAM.set_k_dim: Set the number of attractors
    - BAM.set_f_dim: Set the number of dimensions of a feature
    - BAM.set_norm_prm: Set the parameters for data normalization
    - BAM.set_q: Reset the dynamics uncertainty of BAM
    - BAM.set_r: Reset the sensory uncertainty of BAM
    - BAM.upd_f: Features are set to attractors
    - BAM.ukf_z: Inputs observations (retrieved from the argument) into the Bayesian attractor model and calculates the decision variables
    - BAM.get_z: Outputs the decision variables in the Bayesian attractor model
    - BAM.get_p: Outputs the variance-covariance matrix of the current decision variables
    - BAM.get_c: Outputs the confidence values in the Bayesian attractor model
    - BAM.msg_on: Turns on/off display of debug messages

# Example
```
python Yuragi_sample.py sample_feature.txt sample_input.txt sample_norm.txt
```
It generates res.csv that contains confidence level of how similar the input series is to the features listed in each line of the sample_feature.txt.

The input series are 3-dimensional data X = (x_1, x_2, x_3) where x_i = (1+r_1, 1+r_2, -1+r_3) and r_1, r_2, r_3 ~ Uniform(-1,1). The three attractors store (1, -1, -1), (-1, 1, 1), and (1, 1, -1) respectively. In time step 1--125, (x_1, x_2, x_3) = (1, 1, -1) and after that (x_1, x_2, x_3) = (-1, 1, 1).

![fig_input](https://user-images.githubusercontent.com/47323363/216059213-b15dc4c2-1c50-43f5-899c-5c22d0dc02aa.png)

Yuragi learning outputs a confidence level indicating how close the input series is to the information stored in 3 attractors.

![fig_conf](https://user-images.githubusercontent.com/47323363/216059204-7c22a371-1fb9-455a-9180-300e12ffa886.png)

# Tips
- Normalization
Yuragi learning can improve accuracy by normalizing the input series. The set_norm_prm() function is prepared for this purpose. In the file of the third argument, the mean value is written on the first line for each dimension of the features, separated by tab characters, and the standard deviation is written on the second line for each dimension of the features, separated by tab characters. Normalization (subtracting the mean and dividing by the standard deviation) is automatically performed when storing the features to attractors and providing input series.

# Author

* Daichi Kominami
* Assistant Professor of Osaka University

# License

"bayesian-attractor-model-python-cpp-" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).

Enjoy Yuragi learning!


[1]  M. Murata and K. Leibnitz, "Fluctuation-Induced Network Control and Learning: Applying the Yuragi Principle of Brain and Biological Systems," Springer, March 2021.

[2]  S. Bitzer, J. Bruineberg, and S. J. Kiebel, "A bayesian attractor model for perceptual decision making," PLOS Computational Biology, August 2015.

