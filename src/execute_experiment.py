import time

from OLS import execute_OLS
from RA import execute_RA
from PF import execute_PF
from Benchmarks import Benchmarks
from MO_benchmark import execute_CBM
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
    seed = 1244
    file_to_load = "outputs/pf/20230418120208_220/iters/iter_12.json"
    # file_to_load = "outputs/ols/20230410150924_687/iters/iter_48.json" # iter2
    #file_to_load = "outputs/ols/20230401165818_954/iters/iter_25.json" # iter1
    execution = "cbm"  # ols, ra, ppo, cbm
    message = f"{execution} Execution - {reuse_mode} reuse - seed {seed}"
    run_benchmark = False

    # Update env file parameters
    new_env_params = {
        "seed": seed,
        # "w_rewards": [0.33, 0.33, 0.34]
    }

    # Update model file parameters
    new_model_params = {
        "seed": seed,
        # "quiet": False,
        # "n_epochs": 50
    }

    files_dict = {env_file: new_env_params, model_file: new_model_params}

    print(message)

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
                          continue_execution=continue_execution, file_to_load=file_to_load)
        weights = ols.env.w_rewards
        print(f"Weights are {ols.env.w_rewards}")
        f = open(f"{ols.output_dir}/logs/README", "w")


    elif execution == "ppo":
        ppo = MindmapPPOMultithread(param_file=new_filenames[1], env_file=new_filenames[0], ra=False)
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
                        continue_execution=continue_execution, file_to_load=file_to_load)
        print(f"Weights are {ra.env.w_rewards}")
        f = open(f"{ra.output_dir}/logs/README", "w")

    elif execution == "pf":
        reuse_mode = "full"
        pf = execute_PF(model_file=new_filenames[1], env_file=new_filenames[0], reuse_mode=reuse_mode, epsilon=epsilon,
                        continue_execution=continue_execution, file_to_load=file_to_load)
        # print(f"Weights are {ra.env.w_rewards}")
        f = open(f"{pf.output_dir}/logs/README", "w")

    elif execution == "cbm":
        reuse_mode = "no"
        cbm = execute_CBM(model_file=new_filenames[1], env_file=new_filenames[0], reuse_mode=reuse_mode, epsilon=epsilon,
                        continue_execution=continue_execution, file_to_load=file_to_load)
        # print(f"Weights are {ra.env.w_rewards}")
        f = open(f"{cbm.output_dir}/logs/README", "w")

    f.write(message)

    # Remove parameter files
    os.remove(new_filenames[0])
    os.remove(new_filenames[1])

    print(f"Total time {time.time() - start_time} seconds.")
    f.close()
