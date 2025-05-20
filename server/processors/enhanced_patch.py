import torch
import inspect
import sys
import functools
import types

def apply_comprehensive_torch_patches():
    """
    Apply comprehensive patches to PyTorch functions to handle version compatibility issues.
    This version goes beyond just patching min/max functions and handles any function
    that might have dim/axis parameter confusion.
    """
    print("Applying comprehensive PyTorch compatibility patches")
    
    # Track patched functions
    patched_functions = []
    
    # Create a monkey patch for any function that might use 'dim' or 'axis'
    def create_dim_axis_adapter(func, func_name):
        @functools.wraps(func)
        def patched_func(*args, **kwargs):
            try:
                # First try with original arguments
                return func(*args, **kwargs)
            except TypeError as e:
                error_msg = str(e)
                
                # Check if the error is related to 'dim' vs 'axis'
                if ('dim' in kwargs and ('dim' in error_msg or 'axis' in error_msg)):
                    print(f"Converting 'dim' to 'axis' in {func_name}")
                    # Make a copy of kwargs to avoid modifying the original
                    new_kwargs = kwargs.copy()
                    # Move 'dim' to 'axis'
                    dim_value = new_kwargs.pop('dim')
                    new_kwargs['axis'] = dim_value
                    return func(*args, **new_kwargs)
                elif ('axis' in kwargs and ('dim' in error_msg or 'axis' in error_msg)):
                    print(f"Converting 'axis' to 'dim' in {func_name}")
                    # Make a copy of kwargs to avoid modifying the original
                    new_kwargs = kwargs.copy()
                    # Move 'axis' to 'dim' 
                    axis_value = new_kwargs.pop('axis')
                    new_kwargs['dim'] = axis_value
                    return func(*args, **new_kwargs)
                
                # If we get here, it's some other TypeError, so re-raise
                raise
            except Exception as e:
                print(f"Warning: Error in patched {func_name}: {str(e)}")
                # Just re-raise the exception
                raise
                
        return patched_func
    
    # Patch torch._C._VariableFunctions - this is where most of the low-level operations are
    if hasattr(torch, '_C') and hasattr(torch._C, '_VariableFunctions'):
        for attr_name in dir(torch._C._VariableFunctions):
            # Skip private/special methods and non-callable attributes
            if attr_name.startswith('_') and not attr_name.startswith('_a'):
                continue
                
            attr = getattr(torch._C._VariableFunctions, attr_name)
            if callable(attr):
                # Patch and replace the function
                patched_attr = create_dim_axis_adapter(attr, f"torch._C._VariableFunctions.{attr_name}")
                setattr(torch._C._VariableFunctions, attr_name, patched_attr)
                patched_functions.append(f"torch._C._VariableFunctions.{attr_name}")
    
    # Patch common functions that might use dim/axis
    for func_name in ['min', 'max', 'mean', 'sum', 'argmin', 'argmax', 'all', 'any']:
        if hasattr(torch, func_name):
            orig_func = getattr(torch, func_name)
            patched_func = create_dim_axis_adapter(orig_func, f"torch.{func_name}")
            setattr(torch, func_name, patched_func)
            patched_functions.append(f"torch.{func_name}")
    
    # Patch tensor methods too - this is important for methods called on tensor objects
    tensor_methods = [
        'min', 'max', 'mean', 'sum', 'argmin', 'argmax', 
        'all', 'any', 'norm', 'prod', 'var', 'std', 'logsumexp'
    ]
    
    # Create a sample tensor to get its methods
    sample_tensor = torch.zeros(1)
    
    for method_name in tensor_methods:
        if hasattr(sample_tensor, method_name):
            # Get the original method
            orig_method = getattr(sample_tensor.__class__, method_name)
            
            # Create a patched version
            patched_method = create_dim_axis_adapter(orig_method, f"torch.Tensor.{method_name}")
            
            # Replace the method on the Tensor class
            setattr(sample_tensor.__class__, method_name, patched_method)
            patched_functions.append(f"torch.Tensor.{method_name}")
    
    print(f"Applied {len(patched_functions)} PyTorch compatibility patches")
    return patched_functions

# This allows the file to be both imported and run directly
if __name__ == "__main__":
    apply_comprehensive_torch_patches()
