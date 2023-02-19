# AIML_426_project2
 
### How to run:
Each part is stored in the separate directory. 
All parts expect for part 4 and 5 use Python 3.10. Part 4 uses Python 3.9 and Part 5 uses Python 3.7.
The libraries needed for Part 1 to 3 are:
    NumPy
    Seaborn
    MatPlotLib
    Pandas
    DEAP
    PyGraphviz
    Scikit
    
Part 2 uses arguments which are the filepaths to the knapsack data files. For part 4.1, flgp_q4.py takes the best FLGP functions generated from IDGP_main.py and converts the npy data into csv files which are written into the directory called 'data. Part 4.2 is done in classifier_q4.py where it reads and classifies the the csv data and outputs the performance and training time of each classifier. Part 5 uses run_rl.py to output training models and training performance to the 'logs' folder, eval_rl.py is also used to evaluate the test performance which is also outputted to the 'logs' folder. The output csv files is then read to chart_performance.py chart the performance of the policy neural networks. chart_performance.py arguments are the folder paths to the log files with a separate argument for each seed.