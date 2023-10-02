import importlib, inspect
from pathlib import Path

def make_class_dict(custom_module=None, custom_use_class=True, custom_use_function=False, existing_modules=[]):
    class_dict = {}
    for module, package in existing_modules:
        module = importlib.import_module(module, package)
        for name, cls in inspect.getmembers(module, inspect.isclass):
            class_dict[name] = cls

    if custom_module is None:
        return class_dict
    
    for file in Path(custom_module).glob(r'*.py'):
        if file == '__init__.py':
            continue

        module = importlib.import_module(f'.{file.stem}', package=custom_module)
        if custom_use_class:
            for name, cls in inspect.getmembers(module, inspect.isclass):
                if cls.__module__ == module.__name__:
                    class_dict[name] = cls

        if custom_use_function:
            for name, cls in inspect.getmembers(module, inspect.isfunction):
                if cls.__module__ == module.__name__:
                    class_dict[name] = cls 

    return class_dict
