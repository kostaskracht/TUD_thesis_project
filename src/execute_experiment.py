from OLS import execute_OLS
from RA import execute_RA
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
    message = ""

    # Update env file parameters
    new_env_params = {
        "seed": 1234,
        "w_rewards": [0.1, 0.1, 0.8]
    }

    # Update model file parameters
    new_model_params = {
        "seed": 1234,
        "quiet": False
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

    # Execute OLS
    # ols = execute_OLS(model_file=new_filenames[1], env_file=new_filenames[0], reuse_mode=reuse_mode, epsilon=epsilon,
    #                   continue_execution=continue_execution)

    ppo = MindmapPPOMultithread(param_file=new_filenames[1], env_file=new_filenames[0], ra=False)
    print(f"Weights are {ppo.env.w_rewards}")
    ppo.run(exec_mode="train")
    f = open(f"{ppo.output_dir}/logs/README", "w")

    # f = open(f"{ols.output_dir}/logs/README", "w")
    f.write(message)
    f.close()
