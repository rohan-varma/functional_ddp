import torch
import torch.distributed as dist
from functools import partial



def ddp_forward_pre():
    pass

def ddp_forward_post():

def make_ddp(module: nn.Module, process_group: dist.ProcessGroup, gradient_as_bucket_view: bool = False, static_graph: bool = False, find_unused_parameters: bool = False) -> None:
    # Assumes single device module already placed on CUDA device
    # TODO - module ignore params, param verification, sync, etc
    params, expected_sparse_grad = _build_params_for_reducer(module)
    bucket_bytes_cap = int(25 * 1024 * 1024) # bucket_cap_mb needs to be configurable
    # Computing bucket indices and sizes
    bucket_size_limits = [sys.maxsize] if (static_graph or not find_unused_parameters) else [dist._DEFAULT_FIRST_BUCKET_BYTES, bucket_bytes_cap]
    bucket_indices, per_bucket_size_limits = dist._compute_bucket_assignment_by_size(parameters, bucket_size_limits, expect_sparse_gradients)
    reducer = dist.Reducer(parameters, list(reversed(bucket_indices)), list(reversed(per_bucket_size_limits)), process_group, expect_sparse_gradient, bucket_bytes_cap, find_unused_parameters,
            gradient_as_bucket_view, {}, dist._DEFAULT_FIRST_BUCKET_BYTES)
    # TODO logger, syncBN

    def pre_forward(reducer, mod, inp):
        if torch.is_grad_enabled():
            # TODO - no_sync()
            reducer.prepare_for_forward()

        # TODO - join()
        if torch.is_grad_enabled() and reducer._rebuild_buckets():
            print("Buckets are rebuilt!")

        # TODO - buffer sync

    def post_forward(mod, inp, output):
        if torch.is_grad_enabled():
            # TODO - no_sync()
            require_forward_param_sync = True
            if find_unused_parameters and not static_graph:
                reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                reducer.prepare_for_backward([])
        else:
            pass
        if (find_unused_parameters and not static_graph) or (static_graph and _is_first_iter):
            _is_first_iter = False
            state = {'static_graph': static_graph, '


    pass
