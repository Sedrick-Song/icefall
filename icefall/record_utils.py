from torch.utils.tensorboard import SummaryWriter
import os
import json
import torch
import sys
from pathlib import Path, PosixPath

def record_time():
    from datetime import datetime
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    return date_time

def backup(backpath, runfile_path, params):
    os.makedirs(backpath, exist_ok=True)
    if type(runfile_path)!=list:
        runfile_path = [runfile_path]
    else:
        runfile_path = runfile_path
    runfile_path = [os.path.abspath(item) for item in runfile_path]
    for item in runfile_path:
        os.system(f"cp {item} {backpath}")
    if "run_script" in params and params.run_script is not None:
        if "run_script_dir" in params and params.run_script_dir is not None:
            run_script_dir = params.run_script_dir
        else:
            run_script_dir = ""

        os.system(f"cp {os.path.join(run_script_dir, params.run_script)} {backpath}")
    with open(os.path.join(backpath, "params.json"), 'w') as f:
        params_record = {item: params[item] for item in params}
        for item in params_record:
            if type(params_record[item]) == PosixPath:
                params_record[item] = os.path.abspath(params_record[item])

        f.write(json.dumps(params_record, indent=4))

    with open(os.path.join(backpath, "args.sh"), 'w') as f:
        print(f"# possible run path: {os.getcwd()}", file=f)
        print(" ".join(sys.argv), file=f)

    torch.save(params, os.path.join(backpath, "params.pth"))


class RecordWriter:
    def __init__(self, params, runfile_path, backup_path) -> None:
        if params.tensorboard:
            self.tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
        else:
            self.tb_writer = None
        if "debug" in params and params.debug:
            self.wandb_flag = False
        elif "wandb_project" in params and params.wandb_project is not None:
            import wandb
            self.wandb_flag = True
            self.wandb = wandb
            if "wandb_name" in params and params.wandb_name is not None:
                self.wandb_name = params.wandb_name
            else:
                exp_dir = os.path.join(os.getcwd(), params.exp_dir).split("/")
                exp_dir = [item for item in exp_dir if len(item)>0]
                self.wandb_name="_".join(exp_dir[-2:])

            wandb.init(
                project=params.wandb_project,
                name=self.wandb_name,
            )
        else:
            self.wandb_flag = False
                # config=params)
        self.params = params
        self.runfile_path=runfile_path
        self.backup_path = backup_path
        backup(self.backup_path, self.runfile_path, self.params)

    def add_scalar(self, name, value, step):
        if self.tb_writer:
            self.tb_writer.add_scalar(name, value, step)
        if self.wandb_flag:
            self.wandb.log({name:value}, step=step)