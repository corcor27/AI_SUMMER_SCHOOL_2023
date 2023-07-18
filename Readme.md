# AI_SUMMER_SCHOOL_2023

Section1: Introduction to Python

- Introduction anaconda: https://swcarpentry.github.io/python-novice-inflammation/
- anaconda cheat sheet: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment
- Installing Anaconda
- Create Enviroment using .yml file (back up option just incase)
- Install Juypter & load up juypter

Section2: Data types & handling data Pandas

- introduction to data types: https://swcarpentry.github.io/python-novice-inflammation/01-intro.html

- introduction to pandas: https://ucsbcarpentry.github.io/2019-10-10-Python-UCSB/08-data-frames/
- Creating a data structure
- Accessing & manipulating your data
- Examples of possible data types: Features-wise, Images and Time series
- Plotting with Matplotlib

Section3: Machine Learning & Visualiation

- What is AI / Machine Learning
- Examples of machine learning models for the following types of data: Features-wise, Images and Time series
- Charts and Feature Extraction




# Introduction to Python: Installing Anaconda

- Introduction anaconda: https://swcarpentry.github.io/python-novice-inflammation/
- anaconda cheat sheet: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment

1.1 (Everyone) Firstly, for you to start spriting or testing any python code your going to need an enviroment. For this we employ to use of anaconda, which can be downloaded from the following link: https://www.anaconda.com/ . Make sure to download your correct os version

1.2 (Windows) Click install and complete installer windows. Make sure when asked to "conda init" say yes

1.2 (Linux) In console, locate your downloaded file and run the bash script e.g. bash ~/Downloads/Anaconda3-2020.05-Linux-x86_64.sh

1.2 (Apple) click the following link: https://docs.anaconda.com/free/anaconda/install/mac-os/ and download the MacOS installer.

1.3 (Everyone) go to https://github.com/corcor27/AI_SUMMER_SCHOOL_2023/tree/main and download the repositry and unzip the downloaded file.

1.4 (Linux & apple) open terminal and navigate to inside the just downloaded git repositry and enter "conda env create -f linux_enviroment.yml". This with install all the packages. In terminal if you dont have (base) shown in the bottom corner, please signal for help.

1.4 (Windows) open your downloaded anaconda-navigator. on the home screen (shown in top left) click the install button for "CMD.exe Prompt". Once installed, click the "run" button shown under the "CMD.exe Prompt" and navigate to the just downloaded git repositry. Then run command "conda env create -f windows_enviroment.yml"

1.5 (Everyone) now returning to anaconda navigator (if linux or apple enter "anaconda-navigator" in terminal") and click the button to install "Jupyter Notebook".

1.6 (Everyone) Once installed open jupyter Notebook, then click "New" (top right) and python 3 and this will open you a new notebook to code some python.

1.7 (Everyone) Find where your conda enviroment is sorted and zip up enviroment such that you have a backup. You will thank me later

# Datasets



Image Classification Datset: https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification (medium)

Feature-wise Dataset: https://www.kaggle.com/datasets/uciml/iris (easy)

FordA time series Dataset: http://www.timeseriesclassification.com/description.php?Dataset=FordA (hard)


# Data types & handling data Pandas

2.1 introduction to data types: https://swcarpentry.github.io/python-novice-inflammation/01-intro.html (2 pages)

2.2 introduction to pandas: https://ucsbcarpentry.github.io/2019-10-10-Python-UCSB/08-data-frames/ (1 pages)

2.3 Visualizing Tabular Data: https://swcarpentry.github.io/python-novice-inflammation/03-matplotlib.html

2.4 Futher work, In the git there are 3 different folders/datasets (shown above). Pick one that is the most similar to your data and have ago at reading in the data and producing a graph.




# Machine Learning & Visualiation

Beginners: Start on the iris dataset

