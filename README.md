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

[Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) is "a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms." To install it, please see: https://eigen.tuxfamily.org/dox/GettingStarted.html


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
    - BAM.set_k_dim: The number of attractors can be changed
    - BAM.set_f_dim: Changes the number of dimensions of a feature
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
python Yuragi_sample.py sample_feature.txt sample_input.txt
```
It generates res.csv that contains confidence level of how similar the input series is to the features listed in each line of the sample_feature.txt.


# Author

* Daichi Kominami
* Assistant Professor of Osaka University

# License

"bayesian-attractor-model-python-cpp-" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).

Enjoy Yuragi learning!


[1]  M. Murata and K. Leibnitz, "Fluctuation-Induced Network Control and Learning: Applying the Yuragi Principle of Brain and Biological Systems," Springer, March 2021.

[2]  S. Bitzer, J. Bruineberg, and S. J. Kiebel, "A bayesian attractor model for perceptual decision making," PLOS Computational Biology, August 2015.

