# Cython-Filter-Files


Need to install Openmp, Gsl and Cython, I suggest you do it using conda. 
After that you need to go to setup.py and change the direction of the include and lib folders to your anaconda folders or where you have installed the packages.
Then install with:

python setup.py build_ext --inplace

(the package will be installed in this folder, so you need to add the path to it when importing if everything works I will modify it to install globally.)

the functions to be used are in filter_f, you will see that there are two versions of smooth_bao, version 1 is faster than version 2, but version two agrees better with the old code, while version 1 agrees with 0.02%, the difference lies how the second derivative is calculated and consequently the position of the minimums and maximums
