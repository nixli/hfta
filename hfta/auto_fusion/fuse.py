import torch
from torch import fx
from torch.fx.node import map_arg
from ..ops import get_hfta_op_for, expand_to_hfta_op



def fuse(entities):

    if not isinstance(entities, (list, tuple)) or len(entities) == 0:
        raise TypeError("Entities must be a list or a tuple with >0 elements, not {}"
            .format(type(entities)))

    types = [type(e) for e in entities]
    if len(set(types)) > 1:
        raise ValueError("All members of the entities must be the same")

    e0 = entities[0]
    for t in FusionTable:
        if isinstance(e0, t):
            return FusionTable[t](entities)

    raise TypeError("Fusion of type {} is not supported".format(type(e0)))


def fuse_modules(modules):
    B = len(modules)
    if B == 1:
        return modules[0]


    graphs = []

    for m in modules:
        graph_trace = fx.symbolic_trace(m)
        graphs.append(graph_trace)

    # Currently we assume all modules are the same and simply create the
    # ops with the identical B version
    fx_module = graphs[0]
    modules = dict(fx_module.named_modules())

    fused_graph = fx.Graph()
    val_map = {}
    hfta_modules = {}

    for node in fx_module.graph.nodes:

        hfta_target = "{}_hfta_B_{}".format(node.target, B)

        if node.op == "call_module":
            # change to HFTA equivalent
            module = modules[node.target]
            args = map_arg(node.args, lambda n: val_map[n])
            kwargs = map_arg(node.kwargs, lambda n: val_map[n])
            val_map[node] = fused_graph.create_node(
                node.op, hfta_target, args, kwargs, node.name + "_hfta"
            )
            fused_op = expand_to_hfta_op(module, B=B)
            hfta_modules[hfta_target] = fused_op

        elif node.op == "call_function":
            cur_node = fused_graph.node_copy(node, lambda n: val_map[n])
            if node.target == torch.flatten:
                args = cur_node.args
                if args[1] == 1:
                    args = (args[0], 2)
                    cur_node.args = args
                # add a transpose to the args
                users = cur_node.users

                transpose_node = fused_graph.create_node(
                    "call_function", torch.transpose,
                    (cur_node, 0, 1), {}, "transpose_hfta"
                    )

                # This is the "sort of" equivalent of "replaces_all_uses_with"
                # since the users of the fused graph has not been created yet
                val_map[node] = transpose_node
            else:
                val_map[node] = cur_node

        elif node.op == "placeholder":
            # add new dimemsion of the palceholder
            val_map[node] = fused_graph.node_copy(node, lambda n: val_map[n])

        elif node.op == "output":
            # not sure what needs to be done here, presumably same as placeholer?
            val_map[node] = fused_graph.node_copy(node, lambda n: val_map[n])
        else:
            raise NotImplementedError

    fused = fx.GraphModule(hfta_modules, fused_graph)
    fused.recompile()
    fused.graph.lint()
    print(fused)
    return fused

def fuse_optimizers(optimizers):
    raise NotImplementedError


def split(module):
    pass


FusionTable = {
    torch.nn.Module : fuse_modules,
    torch.optim.Optimizer : fuse_optimizers,
}