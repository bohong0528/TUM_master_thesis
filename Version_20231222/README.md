# Objective
Given a traffic simulation (MATSim) which outputs for a given scenario the number of vehicles driven on a street link (street link count), we predict the street link count using a **surrogate model**. 
Imagine the traffic simulation $f$, and a scenario $x$ and the traffic link count vector $y = f(x)$, then we learn $\hat y = f'(x)$ with $f'$ being the surrogate with the objective of $\min |y - \hat y|$.

# Environments
In general we distinguish three different environments all originating from the MATSim Berlin environment.
1. **Cutout world**: I run the Berlin MATSim simulation. The Berlin MATSim output data is huge. To have a smaller output data, I cut out the trips within a certain district of Berlin. We only consider this district for the prediction. Here the different scenarios represent different districts.
2. **Sparse world**: I sparsed the Berlin street network, and also sampled the population to generate a "smaller" Berlin environment. I run the Berlin MATSim simulation on these "smaller" environments. Here the different scenarios represent differnt sample seeds.
3. **Small world**: I generated artificial environments with an artificial street network and an artificial population. I run the MATSim simulation on these artificial environments. The different scenarios represent different seeds to generate the artificial environments.

In this directory you find two folders:
1. **Scenario** folder: In this folder are all the input files and output files of the MATSim simulation. (You do not have to run the MATSim simulation.) E.g. the environment *small_worlds* contains different *scenarios* and in each *scenario* folder you find the *config.xml*, *network.xml*, and *plans.xml* files which are input data for the MATSim simulation. You also find an *output* directory which is the output of the MATSim simulation. **You can use the input data for you surrogate model. Please do not use the output data for your surrogate model - This would be cheating :)**
2. **Data** folder: Here you find the training data -> Postprocessed data from the *Scenario* folder. You find the exact same scenarios as in the *Scenario* folder, each scenario assigned to *Train*, *Validate*, or *Test*. The variable *link_count* refers to the target variable that we want to predict. The dictionary *link_count* is a mapping from the link ID to the link count.
 - Variables:
    - nodes_x: x-position of node
    - nodes_y: y-position of node
    - nodes_id: new node ID starting at 0 and ending at len(nodes)
    - link_from: node ID from node where link starts
    - link_to: node ID from node where link ends
    - links_id: new link ID starting at 0 and ending at len(links)
    - link_length: length of link
    - link_freespeed: freespeed without congestion of link
    - link_capacity: capacity parameter for calculating congestion on link
    - link_permlanes: numer of lanes of link
    - link_counts: number of vehicles using this link
    - o_d_pairs: (start node ID, end node ID) of planned trips
    - work_x: x-position of work place of person
    - work_y: y-position of work place of person
    - home_x: x-position of home place of person
    - home_y: y-position of home place of person
    - go_to_work: time when person starts trip to go to home
    - go_to_home: time when person starts trip to go to work

# Summary
- Which data can you use as input for you model: All data in the *Scenario* folders except the *output* directories. And all data in the *Data* folder except the *link_count* variable. Please use the different data sets for training, validation, and testing. Best start: ignore the *Scenario* folder and just start with the data you find in the *Data* folder.
- Where do I find the target variable: The target variable is the *link_count* variable in the files in the *Data* folder.

If there are any further questions, please do not hesitate to contact me!

You are not allowed to use this data for anything else except this master thesis / IDP.