import numpy as np

def client_matching(client_data, performance_data):
    mct = np.mean(list(performance_data.values()))
    return [(client_data[0].name, client_data[1].name)]

def offloading_callable(self_obj, state: dict, deadline, training_start_time) -> bool:
    client_ids = [x.name for x in self_obj.selected_clients]
    if 'first_cycle' not in state:
        for client_id in client_ids:
            self_obj.message_async(client_id, 'lock')
        state['first_cycle'] = False

    if 'inactive' not in state and all(item in self_obj.performance_data for item in client_ids):
        # We got all performance data
        # Make offloading decision
        # Make offloading calls
        self_obj.logger.info(f'Client ids are {client_ids}')
        self_obj.logger.info(f'Performance data is {self_obj.performance_data}')
        self_obj.logger.info(f'list {list(self_obj.performance_data.values())}')
        self_obj.logger.info(f'Mean compute time is {np.mean(list(self_obj.performance_data.values()))}')
        self_obj.logger.info(f'Starting offload')
        offloading_decision = client_matching(self_obj.selected_clients, self_obj.performance_data)
        self_obj.logger.info(f'Offloading decision {offloading_decision}')
        for c1, c2 in offloading_decision:
            self_obj.logger.info(f'Client {c1} will offload to client {c2}')
            self_obj.message_async(c1, 'receive_offloading_decision', c2, 0)
            self_obj.message_async(c1, 'unlock')

        state['inactive'] = True

    return False