# Directionality-Aware Mixture Model Parallel Sampling for Efficient Dynamical System Learning

_Last updated: September 2, 2025_
> [!WARNING]  
> This **C++ implementation** (`main_cpp` branch) is **legacy** and no longer maintained.  
> Please use the **Python version** available in the `main` branch, which is the actively supported.


Barebone implementation of Directionality-Aware Mixture Model(DAMM) tailored to the learning of Linear Parameter Varying Dynamical System (LPV-DS). This module cannot be used standalone, and has been integrated as a part of the [LPV-DS](https://github.com/SunannnSun/lpvds) framework. Please refer to the [LPV-DS](https://github.com/SunannnSun/lpvds) repository for the specific use.


--- 

### Dependencies
- **[Required]** [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page): Eigen 3.4 is required.
- **[Required]** [Boost](https://www.boost.org/): Boost 1.74 is recommended.
- **[Required]** [OpenMP](https://www.openmp.org/): OpenMP 5.0 is recommended.
- **[Required]** [OpenCV](https://opencv.org/) : OpenCV 4.8 is recommended.

---

### Compilation

Mac users please download LLVM compiler separately. The built-in clang compiler in Xcode does not support OpenMP. Please refer to ``CMakeLists.txt`` in ``src`` folder to ensure correct compiler.


```
mkdir build
cd build
cmake ../src
make
```
