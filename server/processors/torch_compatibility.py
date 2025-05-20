"""
Direct PyTorch Function Interceptor

This script provides an alternative approach to patching PyTorch functions by
using Python's module import hooks to intercept and modify the original PyTorch
functions at import time. This ensures that all functions are properly patched
before any code tries to use them.
"""

import sys
import types
import importlib.abc
import importlib.machinery
import functools

# Store original importer
original_import = __import__

# Store patched functions to avoid double patching
patched_functions = set()

def patch_function(func, func_name):
    """Create a patched version of a PyTorch function that handles dim/axis conversion"""
    if func_name in patched_functions:
        return func
        
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # First try with original arguments
            return func(*args, **kwargs)
        except TypeError as e:
            error_msg = str(e)
            if 'dim' in kwargs and ('unexpected keyword' in error_msg or 'axis' in error_msg):
                # Convert 'dim' to 'axis'
                new_kwargs = kwargs.copy()
                dim_value = new_kwargs.pop('dim')
                new_kwargs['axis'] = dim_value
                print(f"AUTO-PATCHED: Converting 'dim' to 'axis' in {func_name}")
                return func(*args, **new_kwargs)
            elif 'axis' in kwargs and ('unexpected keyword' in error_msg or 'dim' in error_msg):
                # Convert 'axis' to 'dim'
                new_kwargs = kwargs.copy()
                axis_value = new_kwargs.pop('axis')
                new_kwargs['dim'] = axis_value
                print(f"AUTO-PATCHED: Converting 'axis' to 'dim' in {func_name}")
                return func(*args, **new_kwargs)
            raise
    
    patched_functions.add(func_name)
    return wrapper

def patch_torch_module(module):
    """Recursively patch all functions in a module that might use dim/axis"""
    
    # List of function names to patch in torch modules
    functions_to_patch = [
        'min', 'max', 'mean', 'sum', 'prod', 'std', 'var',
        'argmin', 'argmax', 'all', 'any', 'logsumexp',
        'norm', 'median', 'mode', 'topk', 'sort', 'argsort',
        'cumsum', 'cumprod', 'cummax', 'cummin',
        'amin', 'amax', '_amin', '_amax'
    ]
    
    # Check if this is a type we should recurse into
    if isinstance(module, (types.ModuleType, type)):
        for attr_name in dir(module):
            # Skip private attributes except _amin and _amax
            if attr_name.startswith('_') and attr_name not in ['_amin', '_amax']:
                continue
                
            try:
                attr = getattr(module, attr_name)
                
                # Patch the function if it's in our list
                if callable(attr) and attr_name in functions_to_patch:
                    full_name = f"{module.__name__}.{attr_name}" if hasattr(module, "__name__") else attr_name
                    patched_attr = patch_function(attr, full_name)
                    setattr(module, attr_name, patched_attr)
                
                # Recursively patch submodules and classes
                if isinstance(attr, (types.ModuleType, type)) and attr != module:
                    patch_torch_module(attr)
                    
            except (AttributeError, TypeError):
                # Skip attributes that can't be accessed
                pass

class TorchImportInterceptor(importlib.abc.MetaPathFinder):
    """Import hook that intercepts torch imports to patch functions"""
    
    def find_spec(self, fullname, path, target=None):
        # Let the normal import system find the module
        return None
        
    def find_module(self, fullname, path=None):
        # This is only called in Python < 3.4
        return None
        
    def exec_module(self, module):
        # This is never called, as we don't create the module ourselves
        pass
        
    def create_module(self, spec):
        # Let the normal import system create the module
        return None

def torch_import_hook(name, globals=None, locals=None, fromlist=(), level=0):
    """Custom import hook for torch modules"""
    module = original_import(name, globals, locals, fromlist, level)
    
    # Check if this is a torch module
    if name == 'torch' or name.startswith('torch.'):
        print(f"AUTO-PATCHING: Intercepted import of {name}")
        patch_torch_module(module)
        
    return module

def install_hooks():
    """Install the import hooks"""
    sys.meta_path.insert(0, TorchImportInterceptor())
    __builtins__['__import__'] = torch_import_hook
    print("PyTorch import hooks installed - functions will be auto-patched for dim/axis compatibility")

# Install hooks when this module is imported
install_hooks()

if __name__ == "__main__":
    print("PyTorch compatibility layer loaded")
    print("Any torch imports will now be automatically patched for dim/axis compatibility")
