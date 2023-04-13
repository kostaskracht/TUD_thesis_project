import time

from OLS import execute_OLS
from RA import execute_RA
from Benchmarks import Benchmarks
from parallel_execution_new import MindmapPPOMultithread
import os
import yaml
import numpy as np
from datetime import datetime

if __name__ == "__main__":

    os.chdir("../")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + str(np.random.choice(1000)).zfill(3)

    model_file = "src/model_params_mt.yaml"
    env_file = "environments/env_params.yaml"
    reuse_mode = "no"
    epsilon = 0.05
    continue_execution = False
    file_to_load = "outputs/ols/20230327125850_642/iters/iter_1.json"
    message = ""
    ra = False # TODO - SOS Change to False afterwards!
    execution = "ppo" # ols, ra, ppo
    run_benchmark = True

    ## Full environment parameters set
    # new_env_params = {
    #     "seed": 1234,
    #     "w_rewards": [1, 0, 0],
    #     "is_objective_active": [True, False, False],
    #     "components_to_keep": None,
    #     "max_reward": [9.53120000e+01, 5.88369327e+08, 3.05471214e+08],
    #     "std_reward": [48.8203, 1.0, 1.0],
    #     "norm_method": "scaled"
    #     # "quiet": False,
    # }
    #
    # # Update model file parameters
    # new_model_params = {
    #     "seed": 1234,
    #     "n_epochs": 150000,
    #     "actor_arch": [450, "relu", 450, "relu"],
    #     "critic_arch": [450, "relu", 450, "relu"],
    #     "quiet": False
    # }


# Update env file parameters
    new_env_params = {
        "seed": 1234,
        "w_rewards": [0.3, 0.2, 0.5]
    }

    # Update model file parameters
    new_model_params = {
        "seed": 1234,
        "quiet": False,
        # "normalize_advantage": False,
        # "n_epochs": 15000,
        # "multirunner": False,
    }

    files_dict = {env_file: new_env_params, model_file: new_model_params}

    # Update the yaml files
    new_filenames = []
    for filename, new_params in files_dict.items():
        new_filename = filename.replace(".yaml", f"_{timestamp}.yaml")
        with open(filename, 'r') as outfile:
            file_dict = yaml.safe_load(outfile)
            for key, value in new_params.items():
                file_dict[key] = value
        with open(new_filename, 'w+') as outfile:
            yaml.dump(file_dict, outfile, default_flow_style=False)

        new_filenames.append(new_filename)
        print(f"{new_filename}")

    start_time = time.time()

    if execution == "ols":
        # Execute OLS
        ols = execute_OLS(model_file=new_filenames[1], env_file=new_filenames[0], reuse_mode=reuse_mode, epsilon=epsilon,
                          continue_execution=continue_execution)
        weights = ols.env.w_rewards
        print(f"Weights are {ols.env.w_rewards}")
        f = open(f"{ols.output_dir}/logs/README", "w")

        if run_benchmark:
            benchmarks = Benchmarks(env_file=new_filenames[0])
            print(f"Saving benchmark execution under {benchmarks.timestamp}")

            print(f"Executing CBM:")
            cbm_value, cbm_values = benchmarks.execute_benchmarks(output_dir=ols.output_dir)
            print(f"CBM return for weights: {benchmarks.env.w_rewards} is {cbm_value}")


    elif execution == "ppo":
        ppo = MindmapPPOMultithread(param_file=new_filenames[1], env_file=new_filenames[0], ra=ra)
        print(f"Weights are {ppo.env.w_rewards}")
        ppo.run(exec_mode="train")
        f = open(f"{ppo.output_dir}/README", "w")

        if run_benchmark:
            benchmarks = Benchmarks(env_file=new_filenames[0])
            print(f"Saving benchmark execution under {benchmarks.timestamp}")

            print(f"Executing CBM:")
            cbm_value, cbm_values = benchmarks.execute_benchmarks(output_dir=ppo.output_dir)
            print(f"CBM return for weights: {benchmarks.env.w_rewards} is {cbm_value}")

    elif execution == "ra":
        ra = execute_RA(model_file=new_filenames[1], env_file=new_filenames[0], reuse_mode=reuse_mode, epsilon=epsilon,
                        continue_execution=continue_execution)
        print(f"Weights are {ra.env.w_rewards}")
        f = open(f"{ra.output_dir}/logs/README", "w")

    f.write(message)

    # Remove parameter files
    os.remove(new_filenames[0])
    os.remove(new_filenames[1])

    print(f"Total time {time.time() - start_time} seconds.")
    f.close()
