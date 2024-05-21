from typing import Any, Callable, Dict, Union
import inspect
import json


def load_configuration(config_path: Union[str, Dict], argument_name: str = "opts") -> Any:

    def decorator(func: Callable) -> Any:

        param_names: tuple = list(inspect.signature(func).parameters.keys())
        argument_index: int = list(param_names).index(argument_name)


        def inner(*args, **kwargs) -> Any:

            def update_kwargs():
                if isinstance(config_path, str):    
                    with open(config_path, "r") as file:
                        kwargs[argument_name] = json.load(file)
                elif isinstance(config_path, dict):
                    kwargs[argument_name] = config_path

                return kwargs

            try:
                if len(args) > argument_index or kwargs[argument_name] is not None:
                    pass
                else:
                    kwargs = update_kwargs()

            except KeyError:
                kwargs = update_kwargs()

            return func(*args, **kwargs)

        return inner
    
    return decorator
