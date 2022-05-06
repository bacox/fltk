import copy
from pathlib import Path
import os
import yaml
from fltk.util.generate_docker_compose_2 import generate_compose_file, generate_compose_file_from_dict


def rm_tree(pth: Path):
    for child in pth.iterdir():
        if child.is_file():
            child.unlink()
        # else:
        #     rm_tree(child)
    # pth.rmdir()


def check_num_clients_consistency(cfg_data: dict):
    if type(cfg_data) is str:
        cfg_data = yaml.safe_load(copy.deepcopy(cfg_data))

    # if 'deploy' in cfg_data and 'docker' in cfg_data['deploy']:
    #     num_docker_clients = sum([x['amount'] for x in cfg_data['deploy']['docker']['clients'].values()])
    #     if cfg_data['num_clients'] != num_docker_clients:
    #         print('[Warning]\t Number of docker clients is not equal to the num_clients property!')


def generate(base_path: Path):
    descr_path = base_path / 'descr.yaml'

    exp_cfg_list = [x for x in base_path.iterdir() if '.cfg' in x.suffixes]
    descr_data = ''
    with open(descr_path) as descr_f:
        descr_data = descr_f.read()
    exps_path = base_path / 'exps'
    exps_path.mkdir(parents=True, exist_ok=True)
    rm_tree(exps_path)

    check_num_clients_consistency(descr_data)
    for exp_cfg in exp_cfg_list:
        exp_cfg_data = ''
        with open(exp_cfg) as exp_f:
            exp_cfg_data = exp_f.read()

        exp_data = descr_data + exp_cfg_data
        exp_data += f'\nexperiment_prefix: \'{base_path.name}_{exp_cfg.name.split(".")[0]}\'\n'
        filename = '.'.join([exp_cfg.name.split('.')[0], exp_cfg.name.split('.')[2]])
        with open(exps_path / filename, mode='w') as f:
            f.write(exp_data)
    print('Done')


# def run():
#     base_path = Path(__file__).parent
#     descr_path = base_path / 'descr.yaml'
#
#     exp_cfg_list = [x for x in base_path.iterdir() if '.cfg' in x.suffixes]
#     descr_data = ''
#     with open(descr_path) as descr_f:
#         descr_data = descr_f.read()
#
#     exps_path = base_path / 'exps'
#     exps_path.mkdir(parents=True, exist_ok=True)
#     for exp_cfg in exp_cfg_list:
#         exp_cfg_data = ''
#         replications = 1
#         with open(exp_cfg) as exp_f:
#             exp_cfg_data = exp_f.read()
#         for replication_id in range(replications):
#             exp_data = descr_data + exp_cfg_data
#             exp_data += f'\nexperiment_prefix: \'{Path(__file__).parent.name}_{exp_cfg.name.split(".")[0]}\'\n'
#             filename = '.'.join([exp_cfg.name.split('.')[0], exp_cfg.name.split('.')[2]])
#             with open(exps_path / filename, mode='w') as f:
#                 f.write(exp_data)
#     print('Done')


def run(base_path: Path):
    print(f'Run {base_path}')
    print(list(base_path.iterdir()))
    descr_path = base_path / 'descr.yaml'
    exp_cfg_list = [x for x in base_path.iterdir() if '.cfg' in x.suffixes]
    descr_data = ''
    with open(descr_path) as descr_f:
        descr_data = yaml.safe_load(descr_f.read())

    replications = 1
    if 'replications' in descr_data:
        replications = descr_data['replications']
    run_docker = False
    if 'deploy' in descr_data and 'docker' in descr_data['deploy']:
    # if 'docker_system' in descr_data:
        # Run in docker
        # Generate Docker
        print(descr_data)
        # docker_deploy_path = Path(descr_data['deploy']['docker']['base_path'])
        #
        # print(docker_deploy_path)
        run_docker = True
        generate_compose_file_from_dict(descr_data['deploy']['docker'])
        # generate_compose_file(docker_deploy_path)

    exp_files = [x for x in (base_path / 'exps').iterdir() if x.suffix in ['.yaml', '.yml']]

    cmd_list = []
    print(exp_files)
    if run_docker:
        first_prefix = '--build'
        for exp_cfg_file in exp_files:
            for replication_id in range(replications):
                cmd = f'export OPTIONAL_PARAMS="--prefix={replication_id}";export EXP_CONFIG_FILE="{exp_cfg_file}"; docker-compose --compatibility up {first_prefix};'
                cmd_list.append(cmd)
                first_prefix = ''
    else:
        print('Switching to direct mode')
        for exp_cfg_file in exp_files:
            for replication_id in range(replications):
                # cmd = f'export OPTIONAL_PARAMS="--prefix={replication_id}";export EXP_CONFIG_FILE="{exp_cfg_file}"; docker-compose --compatibility up {first_prefix};'
                cmd = f'python3 -m fltk single {exp_cfg_file} --prefix={replication_id}'
                cmd_list.append(cmd)

    [print(x) for x in cmd_list]
    for cmd in cmd_list:
        print(f'Running cmd: "{cmd}"')
        os.system(cmd)
    print('Done')
    # docker_system


    # name = 'dev'
    # generate_docker(name)
    # base_path = f'{Path(__file__).parent}'
    # exp_list = [
    #     'fedavg.yaml',
    #     ]
    # exp_list = [f'{base_path}/exps/{x}' for x in exp_list]
    # first_prefix = '--build'
    # for exp_cfg_file in exp_list:
    #     cmd = f'export EXP_CONFIG_FILE="{exp_cfg_file}"; docker-compose --compatibility up {first_prefix};'
    #     print(f'Running cmd: "{cmd}"')
    #     os.system(cmd)
    #     first_prefix = ''

    # print('Done')

def generate_tifl_23_uniform():
    return generate_linear(23)

def generate_tifl_2x6_uniform():
    import numpy as np
    num = 6
    step = 1.0 / num
    speeds = np.arange(1+step, 2+step, step)
    cpus = np.ones(num) * 2

    num = 11
    step = 1.0 / num
    # speeds2 = np.arange(1 + step, 2 + step, step)
    # cpus2 = np.ones(num)
    speeds = np.concatenate((speeds, np.arange(step, 1 + step, step)), axis=0)
    cpus = np.concatenate((cpus, np.ones(num)), axis=0)
    print(speeds)
    print(cpus)
    build_client_specs(speeds.tolist(), cpus.tolist())

def generate_tifl_23_random():
    import numpy as np
    num = 23
    speeds = np.random.uniform(0.1, 1, num)
    cpus = np.ones(num)
    print(speeds)
    print(cpus)
    build_client_specs(speeds.tolist(), cpus.tolist())

def generate_linear(num: int):
    import numpy as np
    print(1.0 / num)
    step = 1.0 / num
    speeds = np.arange(step, 1+step, step)
    cpus = np.ones(num)
    print(speeds)
    print(cpus)
    build_client_specs(speeds.tolist(), cpus.tolist())

def generate_equal(num: int):
    import numpy as np
    print(1.0 / num)
    step = 1.0 / num
    speeds = np.ones(num)
    cpus = np.ones(num)
    print(speeds)
    print(cpus)
    build_client_specs(speeds.tolist(), cpus.tolist())

def build_client_specs(speeds, cpus):
    clients = []
    for idx, (speed, num_cpu) in enumerate(zip(speeds, cpus)):
        print(f'Item {idx}, s: {speed}, c: {num_cpu}')
        obj = {
            'rank': idx + 1,
            'num-cores': num_cpu,
            'cpu-speed': speed,
            'stub-name': 'stub_default.yml'
        }
        clients.append(obj)

    print(yaml.dump(clients))
    with open('client_dump.yaml', 'w+') as file:
        yaml.dump(clients, file)

def generate_client_specs():
    import numpy as np
    speeds = np.arange(0.1, 1.1, 0.1)

    print(speeds)
    pass

if __name__ == '__main__':
    print('Generate client set')
    # generate_client_specs()
    # generate_equal(6)
    generate_tifl_23_uniform()
    generate_tifl_23_random()
    # generate_tifl_2x6_uniform()
    # generate_linear(6)

# if __name__ == '__main__':
#     base_path = Path(__file__).parent
#     descr_path = base_path / 'descr.yaml'
#
#     exp_cfg_list = [x for x in base_path.iterdir() if '.cfg' in x.suffixes]
#     descr_data = ''
#     with open(descr_path) as descr_f:
#         descr_data = descr_f.read()
#     exps_path = base_path / 'exps'
#     exps_path.mkdir(parents=True, exist_ok=True)
#     for exp_cfg in exp_cfg_list:
#         exp_cfg_data = ''
#         with open(exp_cfg) as exp_f:
#             exp_cfg_data = exp_f.read()
#
#         exp_data = descr_data + exp_cfg_data
#         exp_data += f'\nexperiment_prefix: \'{Path(__file__).parent.name}_{exp_cfg.name.split(".")[0]}\'\n'
#         filename = '.'.join([exp_cfg.name.split('.')[0], exp_cfg.name.split('.')[2]])
#         with open(exps_path / filename, mode='w') as f:
#             f.write(exp_data)
#     print('Done')
#
#
