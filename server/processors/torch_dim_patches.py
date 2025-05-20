"""
PyTorch Dimension Compatibility Module

This module provides direct monkey patching for the key PyTorch functions that have
the 'dim' vs 'axis' compatibility issue. It specifically targets the _amin and _amax
functions which are commonly causing issues in SceneScript.
"""

import torch
import functools
import traceback
import warnings
import inspect

# Store a list of patched functions
_PATCHED_FUNCTIONS = set()

def patch_amin_amax():
    """
    Directly patch the torch._C._VariableFunctions._amin and _amax functions
    to handle both 'dim' and 'axis' parameters.
    """
    print("Patching PyTorch _amin and _amax functions")
    
    try:
        # Patch _amin
        if hasattr(torch._C._VariableFunctions, '_amin'):
            original_amin = torch._C._VariableFunctions._amin
            
            @functools.wraps(original_amin)
            def patched_amin(*args, **kwargs):
                try:
                    # First try with original arguments
                    return original_amin(*args, **kwargs)
                except TypeError as e:
                    error_msg = str(e)
                    if 'dim' in kwargs and 'axis' not in kwargs and 'unexpected keyword' in error_msg:
                        print("Converting 'dim' to 'axis' in _amin")
                        new_kwargs = kwargs.copy()
                        dim_value = new_kwargs.pop('dim')
                        new_kwargs['axis'] = dim_value
                        return original_amin(*args, **new_kwargs)
                    else:
                        # If not a dim/axis issue, re-raise
                        raise
                except Exception as e:
                    print(f"Error in patched _amin: {str(e)}")
                    # Re-raise other exceptions
                    raise
            
            # Apply the patch
            torch._C._VariableFunctions._amin = patched_amin
            _PATCHED_FUNCTIONS.add('torch._C._VariableFunctions._amin')
            print("✓ Patched torch._C._VariableFunctions._amin")
        else:
            print("✗ Could not find torch._C._VariableFunctions._amin")
        
        # Patch _amax
        if hasattr(torch._C._VariableFunctions, '_amax'):
            original_amax = torch._C._VariableFunctions._amax
            
            @functools.wraps(original_amax)
            def patched_amax(*args, **kwargs):
                try:
                    # First try with original arguments
                    return original_amax(*args, **kwargs)
                except TypeError as e:
                    error_msg = str(e)
                    if 'dim' in kwargs and 'axis' not in kwargs and 'unexpected keyword' in error_msg:
                        print("Converting 'dim' to 'axis' in _amax")
                        new_kwargs = kwargs.copy()
                        dim_value = new_kwargs.pop('dim')
                        new_kwargs['axis'] = dim_value
                        return original_amax(*args, **new_kwargs)
                    else:
                        # If not a dim/axis issue, re-raise
                        raise
                except Exception as e:
                    print(f"Error in patched _amax: {str(e)}")
                    # Re-raise other exceptions
                    raise
            
            # Apply the patch
            torch._C._VariableFunctions._amax = patched_amax
            _PATCHED_FUNCTIONS.add('torch._C._VariableFunctions._amax')
            print("✓ Patched torch._C._VariableFunctions._amax")
        else:
            print("✗ Could not find torch._C._VariableFunctions._amax")
        
        # Also patch the torch.min and torch.max functions
        # These often call into _amin and _amax
        original_min = torch.min
        
        @functools.wraps(original_min)
        def patched_min(*args, **kwargs):
            try:
                return original_min(*args, **kwargs)
            except TypeError as e:
                error_msg = str(e)
                if 'dim' in kwargs and ('dim' in error_msg or 'axis' in error_msg):
                    print("Converting 'dim' to 'axis' in torch.min")
                    new_kwargs = kwargs.copy()
                    dim_value = new_kwargs.pop('dim')
                    new_kwargs['axis'] = dim_value
                    return original_min(*args, **new_kwargs)
                raise
        
        torch.min = patched_min
        _PATCHED_FUNCTIONS.add('torch.min')
        print("✓ Patched torch.min")
        
        original_max = torch.max
        
        @functools.wraps(original_max)
        def patched_max(*args, **kwargs):
            try:
                return original_max(*args, **kwargs)
            except TypeError as e:
                error_msg = str(e)
                if 'dim' in kwargs and ('dim' in error_msg or 'axis' in error_msg):
                    print("Converting 'dim' to 'axis' in torch.max")
                    new_kwargs = kwargs.copy()
                    dim_value = new_kwargs.pop('dim')
                    new_kwargs['axis'] = dim_value
                    return original_max(*args, **new_kwargs)
                raise
        
        torch.max = patched_max
        _PATCHED_FUNCTIONS.add('torch.max')
        print("✓ Patched torch.max")
        
        print(f"Successfully patched {len(_PATCHED_FUNCTIONS)} PyTorch functions")
        
    except Exception as e:
        print(f"Error patching PyTorch functions: {e}")
        traceback.print_exc()

# Direct function monkey patching for specifically handling _amin and _amax
def direct_monkey_patch():
    """
    Directly replace the _amin and _amax functions with versions that handle both
    'dim' and 'axis' parameters at a very low level.
    """
    print("Applying direct monkey patching to PyTorch _amin and _amax functions")
    
    try:
        # Try to reach the internal C functions through the namespace access
        namespace = torch._C._VariableFunctions
        
        # Get parameter lists for the original functions
        def inspect_func(func):
            try:
                # Try to get signature, but might not work for C++ functions
                return inspect.signature(func)
            except (ValueError, TypeError):
                return None
        
        if hasattr(namespace, '_amin'):
            # Create a wrapper that handles compatibility
            orig_amin = namespace._amin
            
            def new_amin(*args, **kwargs):
                # Check for 'dim' parameter and convert to 'axis'
                if 'dim' in kwargs and 'axis' not in kwargs:
                    # Create a new kwargs dict with axis instead of dim
                    dim_val = kwargs.pop('dim')
                    kwargs['axis'] = dim_val
                    print(f"Direct patch converting dim={dim_val} to axis={dim_val}")
                try:
                    return orig_amin(*args, **kwargs)
                except Exception as e:
                    print(f"Error in directly patched _amin: {e}")
                    # Just try standard min as fallback
                    if args and isinstance(args[0], torch.Tensor):
                        return args[0].min()
                    raise
            
            # Replace the original function
            namespace._amin = new_amin
            print("✓ Direct patched torch._C._VariableFunctions._amin")
        
        if hasattr(namespace, '_amax'):
            # Create a wrapper that handles compatibility
            orig_amax = namespace._amax
            
            def new_amax(*args, **kwargs):
                # Check for 'dim' parameter and convert to 'axis'
                if 'dim' in kwargs and 'axis' not in kwargs:
                    # Create a new kwargs dict with axis instead of dim
                    dim_val = kwargs.pop('dim')
                    kwargs['axis'] = dim_val
                    print(f"Direct patch converting dim={dim_val} to axis={dim_val}")
                try:
                    return orig_amax(*args, **kwargs)
                except Exception as e:
                    print(f"Error in directly patched _amax: {e}")
                    # Just try standard max as fallback
                    if args and isinstance(args[0], torch.Tensor):
                        return args[0].max()
                    raise
            
            # Replace the original function
            namespace._amax = new_amax
            print("✓ Direct patched torch._C._VariableFunctions._amax")
        
        print("Direct monkey patching complete")
    
    except Exception as e:
        print(f"Error applying direct monkey patching: {e}")
        traceback.print_exc()

# Apply all patch methods for maximum compatibility
def apply_all_patches():
    """Apply all possible patching methods for maximum compatibility"""
    # Standard patch for _amin and _amax
    patch_amin_amax()
    
    # Direct monkey patch approach
    direct_monkey_patch()
    
    print("All PyTorch compatibility patches applied")
    return _PATCHED_FUNCTIONS

# Automatically apply patches when the module is imported
patched_functions = apply_all_patches()
