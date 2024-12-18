# A modification of atomic centered symmetry functions (ACSF) for predicting birefringence of nonlinear optical crystals

## 1 Browse all models
Details of all trained models mentioned in our paper are gathered in several directories, which has a prefix **bm**. The following list shows the meaning of the suffix.
+ jvs: models predicting 5 classification tasks based on JARVIS-DFT database, using p-wACSF crystal features.
+ jvs0: models predicting 5 classification tasks based on JARVIS-DFT database, using wACSF crystal features.
+ 348: models predicting two tasks 348-004 and 348-008, using p-wACSF features.

In **bm_jvs** and **jvs0** directory, the name of each csv file involves inner sorting criterion and its target name. In **bm_348** directory, all models with different target and random seed are all listed in the csv file.

## 2 Generating features and training models from zero
The directory **mfo** contains necessary data and code for building stacking models from zero. Here, we take 348-004 or 348-008 for example, to show the generation of p-wACSF features.
### 2.1 Collecting crystal data 
The directory **348** contains crystal cif files and NLO properties of 348 database.
### 2.2 p-wACSF calculation stage
Run **ndw1p.py** to generate job files for calculation. Note that prepare two empty directories for job files and calcualtion results. 
Then copy **nd_wacsf1p.py** to the job directory and arrange function calculation stage.
### 2.3 Histogram stage 
After p-wACSF calculation, run **ndw2g_1.py** and **ndw2g_2.py** to generate crystal features 
### 2.4 Data splitting and preprocessing
Run **ndfsc.py** to split data and resample training set.
### 2.5 Train classification models
Run **ndml2_gen.py** to generate job files for model training.

Note that datasets directly for model training canbe found in `https://figshare.com/account/articles/28053213`

Then copy **nd_ml2.py** to the job directory.
