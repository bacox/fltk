import time
from fltk.core.client import Client


def deadline_callable(self_obj, state: dict, deadline, training_start_time) -> bool:
    if time.time() > training_start_time + deadline:
        self_obj.logger.warning('Deadline has passed!')
        # Notify clients to stop
        for client in self_obj.selected_clients:
            client.valid_response = False
            self_obj.message_async(client.ref, Client.stop_training)
        # Break out waiting loop
        return True