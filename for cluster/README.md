# Python Job Template

This branch contains an example that shows how to run a Python script on the cluster using SLURM. For a more general guide see the [master](https://oscm.maximilian-schiffer.com/it/it-wiki/) branch. The script uses Gurobi to solve the classic diet model. This example was adapted from Gurobi's [`diet.py`](https://www.gurobi.com/documentation/9.1/examples/diet_py.html). The diet model is described by a set of food items and a set of nutrients. The goal is to select a combination of food items with minimal cost, while satisfying a set of constraints that ensure the consumption of a given amount of nutrients in the diet.

## Instances

There are two instances in the `data` folder: `c1.txt` and `c2.txt`. Each of these files contains the cost vector, the constraint matrix and vectors that contain the lower and upper bounds on the nutrients. Files `p1.txt` and `p2.txt` contain, for each corresponding instance, the number of nutrients and the number of available food items separated by a comma. 

## Getting started

### During/before code development (on your local machine)

During code development, the following steps are necessary so that the code can be run on the cluster. These set up a virtual environment which can be replicated on the cluster. You can skip these steps and start with [preparing the cluster](https://oscm.maximilian-schiffer.com/it/it-wiki/-/examples/python/README.md#preparing-the-cluster) if you already have a virtual environment (via `venv`/`pip`) configured.

- Create an environment:  
`python3 -m venv venv`  
This will create a subfolder named `venv` in the root directory. For more information about virtual environments, visit the [documentation](https://docs.python.org/3/tutorial/venv.html). Please note that `(ana)conda` environments are not supported and a migration to `venv/pip` environments is nessesary in such cases. Here, we recommend to set up a new virtual environment using `venv` and `pip` directly on the cluster if using windows.

- Activate the environment:  `source venv/bin/activate`

- To check whether the environment is correctly activated, you may run: `which python3`  
This command should print to the terminal the absolute path of the python interpreter of the newly created environment. It should be something like `/path/to/directory/venv/bin/python`.

- With the environment activated, install gurobipy, pandas and other necessary packages: `pip install gurobipy && pip install pandas`

- (After code is ready) write requirements: `pip list --format=freeze > requirements.txt`  
This file will contain all necessary packages for running your code.

- Upload code to your git repository with `git push`  

- Once the code is ready and working on your local machine, follow the general steps (as described [here](https://oscm.maximilian-schiffer.com/it/it-wiki#getting-started)).

### Preparing the cluster

0. Replicate the virtual environment you have created on the cluster. For this purpose, first copy `requirements.txt` to the directory on the cluster where your code should run. Then, run the following commands, which will load the appropiate python module, create a new virtual environment, activate the virtual environment, and install the required packages from `requirements.txt`. 
```
module load python/3.8.7
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
deactivate
module unload python/3.8.7
```
1. Replace `executable` with your script/program. This is for example a bash script which calls your python script. Make sure that this script is marked as executable: `chmod +x ./executable`.
2. Edit `configuration.sh`: here you need to load the necessary modules (`python/3.8.7` and `gurobi/9.1.2` in this example) and activate the environment.
3. Generate `run_list.csv`. Note that this example contains three runs, where the third run tries to load an instance which does not exist and thus results an error.
4. Add folders containing your instances/data/settings, e.g. via sftp. In this example, all data is located in the folder `data`.


### Running the code on the cluster:
1. Start a screen session `screen -S <your_session_name>`, run `./run.sh`. You can detach your session by pressing `Ctrl + A + D` to run other commands. To reattach your session type `screen -r <your_session_name>`. See [screen](https://linuxize.com/post/how-to-use-linux-screen/) for a quickstart guide on `screen`.
2. `results`, `logs` and `error-logs` will contain your program's output as soon as your program is done.

## How your program is executed

The main problem with concurrent runs is that result files and logs need to be written in a fashion such that no conflicts occur, i.e., each run is isolated in a dedicated environment. The job template script manages this environment for you. This is done as follows: Upon starting a run, the script creates a temporary directory on the local filesystem that mirrors the base directory. It also sets up your shell environment (loaded modules, variables, etc.) such that each run mimics the configuration when running `./run.sh`. Then your program is run via `./EXECUTABLE <line from run_list argument 1> <line from run_list argument 2> ...`. Finally, all files written to `results, logs, error-logs` will be transferred back to the base directory. As such, any files you write to `results/` `logs/` and `error-logs/` will be in the respective directories (e.g., `results/run-name/`) inside the shared base directory after your program has finished running. All other files will be discarded. This guarantees that no conflicts can occur.


## Considerations

* Please be as accurate as possible with the resources allocated for your script. Especially the time limit is important as it is used by SLURM's internal scheduler. 
