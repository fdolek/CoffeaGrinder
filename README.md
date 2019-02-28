# CoffeeGrinder
### Prerequisites

* Python 2.7
* numpy version 1.13.1 
* jupyter 4.3+   
* ipython version 5.7+  
* histbook     
* vega version 1.1 
* uproot 

```
pip install numpy --user
pip install jupyter --user
pip install ipython # you may need to install it through conda: conda install ipython
pip install histbook --user
pip install vega==1.1 --user
pip install vegascope --user
pip install uproot --user
pip install awkward --user
pip install fnal_column_analysis_tools -U
```
Note: you might need to upgrade pip to version 18 to get Jupyter 4.3 or above  (`pip install --upgrade pip`). 

To upgrade the packages after they have been installed, use the following command.
```
pip install package_name --user --upgrade
```

### Code versions
There are two ways to run the notebooks. Those with *_uproot.ipynb can be run on your laptop using local nanoAOD files copied to your laptop from the LPC. The notebooks with *_striped.ipynb require access to the striped cluster. The uproot notebooks are useful for doing development and making sure your setup is correct. The actual analysis code will be run over all datasets using the striped cluster. 

### Striped environment new Installation¶

Once you have installed the prerequisites, set up the striped client
```
  git clone http://cdcvs.fnal.gov/projects/nosql-ldrd striped     
  cd striped
  python setup.py install --user 
```

Now in a different directory, clone the Coffea repository
```
  git clone https://github.com/CoffeaTeam/CoffeaGrinder.git
  
 ```
 
 Ask Allie if you do not have a password to access Striped. Your password XX and your username YY need to be placed in a striped.yaml file with the following format. 
 ```
 JobServer: {
 host: ifdb02.fnal.gov,
 port: 8766
 }
 Username: YY
 Password: XX
 ```
 
 Change the passwords to protect your password from reading by group and others:
 ```
 chmod 0700 striped.yaml
 ```
 
 The file location needs to be passed in using the "session" parameter in the jupyter notebook before launching the jobs:
 ```
 session = Session("/Users/ahall/striped.yaml")

 ```
### Setting up Fermilab VPN¶

* Go to [here!](https://vpn.fnal.gov) and log in with your Services account 
    * This will automatically install CISCO AnyConnect VPN 
* Open CISCO AnyConnect VPN
* Type "vpn.fnal.gov" and click connect
* Log in with your Services account username and password


### Run
Launch Jupyter Notebook
```
   cd CoffeaGrinder/
   jupyter notebook
```
It should open a new page in your default browser. If not, you can follow the link displayed in the terminal.  At the page you will find the directories and files from the location you ran jupyter notebook. select one of the samples to run it.

