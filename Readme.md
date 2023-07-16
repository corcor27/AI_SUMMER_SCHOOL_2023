# AI_SUMMER_SCHOOL_2023

Section1: Introduction to Python

- Introduction anaconda: https://swcarpentry.github.io/python-novice-inflammation/
- anaconda cheat sheet: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment
- Installing Anaconda
- Create Enviroment using .yml file (back up option just incase)
- Install Juypter & load up juypter

Section2: Data types & handling data Pandas

- introduction to data types: https://swcarpentry.github.io/python-novice-inflammation/01-intro.html
- Examples of possible data types: Features-wise, Images and Time series
- introduction to pandas: https://ucsbcarpentry.github.io/2019-10-10-Python-UCSB/08-data-frames/
- Creating a data structure
- Accessing & manipulating your data
- Plotting with Matplotlib

Section3: Machine Learning & Visualiation

- What is AI / Machine Learning
- Examples of machine learning models for the following types of data: Features-wise, Images and Time series
- Charts and Feature Extraction




# Introduction to Python: Installing Anaconda

1.1 (Everyone) Firstly, for you to start spriting or testing any python code your going to need an enviroment. For this we employ to use of anaconda, which can be downloaded from the following link: https://www.anaconda.com/ . Make sure to download your correct os version

1.2 (Windows) Click install and complete installer windows. Make sure when asked to "conda init" say yes

1.2 (Linux) In console, locate your downloaded file and run the bash script e.g. bash ~/Downloads/Anaconda3-2020.05-Linux-x86_64.sh

1.2 (Apple) click the following link: https://docs.anaconda.com/free/anaconda/install/mac-os/ and download the MacOS installer.

1.3 (Everyone) go to https://github.com/corcor27/AI_SUMMER_SCHOOL_2023/tree/main and download the repositry and unzip the downloaded file.

1.4 (Linux & apple) open terminal and navigate to inside the just downloaded git repositry and enter "conda env create -f linux_enviroment.yml". This with install all the packages. In terminal if you dont have (base) shown in the bottom corner, please signal for help.

1.4 (Windows) open your downloaded anaconda-navigator. on the home screen (shown in top left) click the install button for "CMD.exe Prompt". Once installed, click the now "run" button shown under the "CMD.exe Prompt" and navigate to the just downloaded git repositry. Then run command "conda env create -f windows_enviroment.yml"

1.5 (Everyone) now returning to anaconda navigator (if linux or apple enter "anaconda-navigator" in terminal") and click the button to install "Jupyter Notebook".

1.6 (Everyone) Once installed open jupyter Notebook, then click "New" (top right) and python 3 and this will open you a new notebook to code some python.



# Data handling with Pandas
Time Series Mosquito Dataset: https://github.com/takluyver/ucb-ipython-intro/blob/master/python_intermediate/A1_mosquito_data.csv

Image Classification Datset: https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification

Feature-wise Dataset: https://www.kaggle.com/datasets/uciml/iris




# Machine Learning & Visualiation

Time series: https://swcarpentry.github.io/python-intermediate-mosquitoes/02-modularization-documentation.html

