# Linear models demystified

A notebook to explain the power and interpretability of linear models. Hopefully this will make complex modelling packages like `prophet` seem less like magic.

## Installation

The best way to install this notebook is with [miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) (conda's little cousin).

1. Download, and unzip, the folder from github (drop down from the "code" button) 
2. Download and install [miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)
3. Open the anaconda prompt (or terminal if you know how) in the project folder
4. create a new environment with the following command

```shell
conda env create -f environment.yaml 
```

Activate your newly created environment with the following:
```shell
conda activate linear_models
```

These steps should install all the necessary packages. If you get an error that says package not installed then run the following 
```
python -m pip install [package-name]
```

## VSCode

I use [VSCode](https://code.visualstudio.com/) as my code editor since it is free and open-source. You can also use things like PyCharm (which I think is included at Trainline). The benefit of these editors is that you can open and view Jupyter notebooks without launching a full jupyter server. This makes it easier to edit and move files, you can change jupyter kernels easier and all round better coding experience.

You just need to install the Jupyter extension on [VSCode](https://code.visualstudio.com/) and you should be able to the notebooks included here.

