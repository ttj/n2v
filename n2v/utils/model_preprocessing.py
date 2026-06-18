"""
Model preprocessing utilities for neural network verification.

Provides BatchNorm fusion to simplify models before reachability analysis.
Fusing BatchNorm into preceding Conv2d/Linear layers eliminates BatchNorm
from the computation graph, which avoids the need for a dedicated BatchNorm
reachability implementation.

Also provides final-softmax stripping. Star reachability has a sound Softmax
rule (softmax_star, a box relaxation), but stripping is exact and
order-preserving: it leaves argmax/argmin unchanged and avoids the box
relaxation's correlation-dropping over-approximation. Valid only for specs over
logits/argmax, not softmax probabilities -- see strip_final_softmax.

Fusion formula (Conv2d + BatchNorm2d):
    scale = gamma / sqrt(running_var + eps)
    W_fused = scale.reshape(-1, 1, 1, 1) * W_conv
    b_fused = scale * (b_conv - running_mean) + beta

For Linear + BatchNorm1d, the scale reshape is (-1, 1).
"""

import copy

import torch
import torch.nn as nn

# onnx2torch converts an ONNX Softmax at opset <= 11 to OnnxSoftmaxV1V11, which
# is NOT an nn.Softmax subclass (opset >= 13 maps to nn.Softmax). Import it so
# detection/stripping recognizes that wrapper. Optional: without onnx2torch such
# a module can't appear, so the None fallback is sound (no new dep -- n2v.utils
# already imports onnx2torch via model_loader).
try:
    from onnx2torch.node_converters.activations import (
        OnnxSoftmaxV1V11 as _OnnxSoftmaxV1V11,
    )
except Exception:  # pragma: no cover - onnx2torch is optional here
    _OnnxSoftmaxV1V11 = None


def has_batchnorm(model: nn.Module) -> bool:
    """
    Check whether a model contains any BatchNorm layers.

    Args:
        model: PyTorch model to inspect.

    Returns:
        True if any module is an instance of BatchNorm1d, BatchNorm2d,
        or BatchNorm3d; False otherwise.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            return True
    return False


def _is_softmax_module(module: nn.Module) -> bool:
    """
    Check whether a module is a plain (non-log) softmax activation.

    Recognizes nn.Softmax and onnx2torch's OnnxSoftmaxV1V11 wrapper (the
    non-nn.Softmax module an ONNX Softmax at opset <= 11 converts to). The
    log-softmax variants are excluded: nn.LogSoftmax is not an nn.Softmax
    subclass, and the wrapper sets is_log=True.
    """
    if isinstance(module, nn.Softmax):
        return True
    if _OnnxSoftmaxV1V11 is not None and isinstance(module, _OnnxSoftmaxV1V11):
        return not getattr(module, 'is_log', False)
    return False


def _is_softmax_call_node(node) -> bool:
    """
    Check whether an fx node is a non-module softmax call (functional or method).

    Matches a call_function whose target is named 'softmax' (e.g.
    torch.softmax, F.softmax) or a call_method named 'softmax' (e.g.
    ``x.softmax(-1)``); log_softmax is excluded by the exact name. Some ONNX
    importers emit softmax this way rather than as a module. In every matched
    form the softmax input is the node's first positional arg.
    """
    op = getattr(node, 'op', None)
    target = getattr(node, 'target', None)
    if op == 'call_function':
        return getattr(target, '__name__', None) == 'softmax'
    if op == 'call_method':
        return target == 'softmax'
    return False


def has_softmax(model: nn.Module) -> bool:
    """
    Check whether a model contains any (non-log) Softmax layers.

    Recognizes nn.Softmax and onnx2torch's OnnxSoftmaxV1V11 wrapper (emitted
    for ONNX Softmax at opset <= 11), plus a non-module softmax (call_function
    or call_method node) in a torch.fx.GraphModule. LogSoftmax variants are not
    counted.

    Args:
        model: PyTorch model to inspect.

    Returns:
        True if the model contains a softmax activation; False otherwise.
    """
    if any(_is_softmax_module(module) for module in model.modules()):
        return True
    # Functional / method softmax shows up as a call node, not a module.
    import torch.fx
    if isinstance(model, torch.fx.GraphModule):
        return any(_is_softmax_call_node(n) for n in model.graph.nodes)
    return False


def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    Fuse a Conv2d layer with a following BatchNorm2d layer.

    Args:
        conv: The convolutional layer.
        bn: The batch normalization layer (must be in eval mode).

    Returns:
        A new Conv2d layer with fused weights and bias.
    """
    # Extract BN parameters
    gamma = bn.weight  # scale
    beta = bn.bias  # shift
    running_mean = bn.running_mean
    running_var = bn.running_var
    eps = bn.eps

    # Compute scale factor: gamma / sqrt(var + eps)
    scale = gamma / torch.sqrt(running_var + eps)

    # Fused weight: scale reshaped for broadcasting over conv weight dims
    # Conv weight shape: (out_channels, in_channels/groups, kH, kW)
    fused_weight = scale.reshape(-1, 1, 1, 1) * conv.weight

    # Fused bias: scale * (b_conv - mean) + beta
    if conv.bias is not None:
        fused_bias = scale * (conv.bias - running_mean) + beta
    else:
        fused_bias = scale * (-running_mean) + beta

    # Create new Conv2d with same configuration but with bias
    fused_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
    )

    fused_conv.weight = nn.Parameter(fused_weight)
    fused_conv.bias = nn.Parameter(fused_bias)

    return fused_conv


def _fuse_linear_bn(linear: nn.Linear, bn: nn.BatchNorm1d) -> nn.Linear:
    """
    Fuse a Linear layer with a following BatchNorm1d layer.

    Args:
        linear: The linear layer.
        bn: The batch normalization layer (must be in eval mode).

    Returns:
        A new Linear layer with fused weights and bias.
    """
    # Extract BN parameters
    gamma = bn.weight
    beta = bn.bias
    running_mean = bn.running_mean
    running_var = bn.running_var
    eps = bn.eps

    # Compute scale factor
    scale = gamma / torch.sqrt(running_var + eps)

    # Linear weight shape: (out_features, in_features)
    fused_weight = scale.reshape(-1, 1) * linear.weight

    # Fused bias
    if linear.bias is not None:
        fused_bias = scale * (linear.bias - running_mean) + beta
    else:
        fused_bias = scale * (-running_mean) + beta

    # Create new Linear with same configuration but with bias
    fused_linear = nn.Linear(
        in_features=linear.in_features,
        out_features=linear.out_features,
        bias=True,
    )

    fused_linear.weight = nn.Parameter(fused_weight)
    fused_linear.bias = nn.Parameter(fused_bias)

    return fused_linear


def _is_fusable_pair(layer_a: nn.Module, layer_b: nn.Module) -> bool:
    """
    Check if two adjacent layers form a fusable pair.

    Returns True for Conv2d+BatchNorm2d or Linear+BatchNorm1d pairs.
    """
    if isinstance(layer_a, nn.Conv2d) and isinstance(layer_b, nn.BatchNorm2d):
        return True
    if isinstance(layer_a, nn.Linear) and isinstance(layer_b, nn.BatchNorm1d):
        return True
    return False


def _fuse_pair(layer_a: nn.Module, layer_b: nn.Module) -> nn.Module:
    """
    Fuse a fusable pair of layers into a single layer.

    Args:
        layer_a: Conv2d or Linear layer.
        layer_b: Corresponding BatchNorm layer.

    Returns:
        Fused layer.
    """
    if isinstance(layer_a, nn.Conv2d) and isinstance(layer_b, nn.BatchNorm2d):
        return _fuse_conv_bn(layer_a, layer_b)
    if isinstance(layer_a, nn.Linear) and isinstance(layer_b, nn.BatchNorm1d):
        return _fuse_linear_bn(layer_a, layer_b)
    raise ValueError(f"Cannot fuse {type(layer_a)} with {type(layer_b)}")


def _fuse_sequential(seq: nn.Sequential) -> nn.Sequential:
    """
    Fuse BatchNorm layers in a Sequential module.

    Walks through child modules pairwise, fusing Conv+BN and Linear+BN pairs.
    Replaces fused BN layers with nn.Identity().

    Args:
        seq: A Sequential module (already deep-copied).

    Returns:
        The same Sequential with fused layers (modified in-place on the copy).
    """
    children = list(seq.children())
    num_children = len(children)

    # First, recurse into any child Sequential modules
    for i, child in enumerate(children):
        if isinstance(child, nn.Sequential):
            children[i] = _fuse_sequential(child)

    # Now fuse adjacent pairs at this level
    i = 0
    while i < num_children - 1:
        layer_a = children[i]
        layer_b = children[i + 1]

        if _is_fusable_pair(layer_a, layer_b):
            children[i] = _fuse_pair(layer_a, layer_b)
            children[i + 1] = nn.Identity()
            i += 2  # skip the Identity we just placed
        else:
            i += 1

    # Rebuild the Sequential with the updated children
    # Preserve the original keys (integer indices for plain Sequential)
    new_seq = nn.Sequential()
    for idx, child in enumerate(children):
        new_seq.add_module(str(idx), child)

    return new_seq


def _fuse_generic_module(module: nn.Module) -> nn.Module:
    """
    Fuse BatchNorm layers in an arbitrary module (not just Sequential).

    For Sequential submodules, uses _fuse_sequential.
    For other modules with named_children, recurses and replaces children.
    Also handles torch.fx.GraphModule via graph-based fusion.

    Args:
        module: A module (already deep-copied).

    Returns:
        The module with fused layers (modified in-place on the copy).
    """
    # Handle Sequential directly
    if isinstance(module, nn.Sequential):
        return _fuse_sequential(module)

    # Handle torch.fx.GraphModule
    import torch.fx
    if isinstance(module, torch.fx.GraphModule):
        return _fuse_graph_module(module)

    # For generic modules, recurse into children
    for name, child in module.named_children():
        fused_child = _fuse_generic_module(child)
        if fused_child is not child:
            setattr(module, name, fused_child)

    # Now attempt to fuse adjacent children at this level
    # Get ordered children and look for fusable pairs
    child_names = []
    child_modules = []
    for name, child in module.named_children():
        child_names.append(name)
        child_modules.append(child)

    i = 0
    while i < len(child_modules) - 1:
        layer_a = child_modules[i]
        layer_b = child_modules[i + 1]

        if _is_fusable_pair(layer_a, layer_b):
            fused = _fuse_pair(layer_a, layer_b)
            setattr(module, child_names[i], fused)
            setattr(module, child_names[i + 1], nn.Identity())
            child_modules[i] = fused
            child_modules[i + 1] = nn.Identity()
            i += 2
        else:
            i += 1

    return module


def _fuse_graph_module(gm: 'torch.fx.GraphModule') -> 'torch.fx.GraphModule':
    """
    Fuse BatchNorm layers in a torch.fx.GraphModule by iterating over graph nodes.

    Finds sequential call_module pairs where the first is Conv2d/Linear and
    the second is the corresponding BatchNorm, fuses them, and recompiles.

    Args:
        gm: A torch.fx.GraphModule (already deep-copied).

    Returns:
        The GraphModule with fused layers and recompiled graph.
    """

    graph = gm.graph
    nodes = list(graph.nodes)

    # Build a map from node name to the module it references
    modules = dict(gm.named_modules())

    for node in nodes:
        if node.op != 'call_module':
            continue

        # Check if this node's module is a BatchNorm
        mod = modules.get(node.target)
        if not isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            continue

        # Find the preceding node (the input to this BN)
        if len(node.args) == 0:
            continue
        prev_node = node.args[0]
        if not hasattr(prev_node, 'op') or prev_node.op != 'call_module':
            continue

        prev_mod = modules.get(prev_node.target)
        if prev_mod is None:
            continue

        if _is_fusable_pair(prev_mod, mod):
            # Fuse the pair
            fused = _fuse_pair(prev_mod, mod)

            # Replace the Conv/Linear module in the GraphModule
            # Navigate the module hierarchy to set the attribute
            _set_module_by_name(gm, prev_node.target, fused)

            # Replace BN with Identity
            _set_module_by_name(gm, node.target, nn.Identity())

            # Redirect BN's output users to point to the Identity
            # (the Identity will just pass through the conv/linear output)
            node.replace_all_uses_with(prev_node)

            # Remove the BN node from the graph
            graph.erase_node(node)

    gm.recompile()
    return gm


def _set_module_by_name(model: nn.Module, target: str, new_module: nn.Module) -> None:
    """
    Set a module in a model by its dotted name path.

    For example, target="layer1.conv" sets model.layer1.conv = new_module.
    """
    parts = target.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def fuse_batchnorm(model: nn.Module) -> nn.Module:
    """
    Fuse BatchNorm layers into preceding Conv2d/Linear layers.

    Creates a deep copy of the model and fuses all Conv2d+BatchNorm2d and
    Linear+BatchNorm1d pairs. The BatchNorm layers are replaced with
    nn.Identity(). The original model is not modified.

    The fused model produces numerically identical outputs (within floating
    point tolerance) to the original model in eval mode.

    Args:
        model: PyTorch model to preprocess. Must be in eval mode or will
               be set to eval mode on the copy.

    Returns:
        A new model with BatchNorm layers fused into their preceding
        Conv2d/Linear layers. The model is in eval mode.

    Example:
        >>> model = nn.Sequential(
        ...     nn.Conv2d(3, 16, 3, padding=1),
        ...     nn.BatchNorm2d(16),
        ...     nn.ReLU(),
        ... )
        >>> model.eval()
        >>> fused = fuse_batchnorm(model)
        >>> # fused has no BatchNorm layers
        >>> assert not has_batchnorm(fused)
    """
    # Deep copy to avoid mutating original
    fused_model = copy.deepcopy(model)
    fused_model.eval()

    # Fuse BatchNorm layers
    fused_model = _fuse_generic_module(fused_model)
    fused_model.eval()

    return fused_model


def _strip_graph_module_softmax(gm: 'torch.fx.GraphModule') -> None:
    """Remove a trailing Softmax node from an fx GraphModule, in place.

    Rewires the output to the softmax's input, erases the node, then re-lints
    and recompiles. Handles module nodes (nn.Softmax or OnnxSoftmaxV1V11) and
    call_function / call_method softmax. No-op if the output isn't a softmax.
    """
    graph = gm.graph
    out_node = next(n for n in graph.nodes if n.op == 'output')
    last = out_node.args[0]
    if (getattr(last, 'op', None) == 'call_module'
            and _is_softmax_module(gm.get_submodule(last.target))):
        out_node.args = (last.args[0],)  # return the softmax's input instead
        graph.erase_node(last)
        # Replace the now-orphaned Softmax submodule with Identity so it no
        # longer shows up in model.modules().
        _set_module_by_name(gm, last.target, nn.Identity())
        graph.lint()
        gm.recompile()
    elif _is_softmax_call_node(last):
        # Functional / method softmax: its input is the first positional arg
        # (or the 'input' kwarg). Rewire the output to it and erase the node.
        softmax_input = last.args[0] if last.args else last.kwargs.get('input')
        if softmax_input is not None:
            out_node.args = (softmax_input,)
            graph.erase_node(last)
            graph.lint()
            gm.recompile()


def strip_final_softmax(model: nn.Module) -> nn.Module:
    """
    Remove a trailing Softmax layer from a model.

    Returns a deep copy with the final softmax dropped; the original is
    unchanged. Softmax is order-preserving, so this leaves argmax/argmin intact
    and exactly preserves the stripped model's logits.

    Precondition: the output spec must be over logits/argmax. Stripping is
    unsound for specs over softmax probabilities (e.g. "P(class) >= 0.9");
    verify those in-graph via softmax_star instead.

    Handles torch.fx.GraphModule (onnx2torch output) and nn.Sequential. The
    softmax is recognized as an nn.Softmax module, an OnnxSoftmaxV1V11 wrapper
    (ONNX opset <= 11), or a call_function / call_method node; LogSoftmax is
    left in place. Returns the copy unchanged if no trailing softmax is found.

    Args:
        model: PyTorch model to preprocess.

    Returns:
        A new model in eval mode with the trailing softmax removed.
    """
    import torch.fx

    stripped = copy.deepcopy(model)
    stripped.eval()

    if isinstance(stripped, torch.fx.GraphModule):
        _strip_graph_module_softmax(stripped)
    elif isinstance(stripped, nn.Sequential) and len(stripped) > 0 \
            and _is_softmax_module(stripped[-1]):
        stripped = nn.Sequential(*list(stripped)[:-1])

    stripped.eval()
    return stripped
