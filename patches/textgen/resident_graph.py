"""Per-block CUDA graphs for resident text-encoder weights.

Stock ComfyUI captures a graph only for the aimdo/vbar streaming path, where
weights are paged in per block. With --highvram the weights stay resident, that
path never engages, and decode runs eagerly (thousands of tiny launches per
step). This module supplies the resident counterpart: capture one graph per
transformer block, then replay it on every later decode step.

It replaces the three-file core overlay this pack used to ship. Every core
function is wrapped, never copied: wrappers call the original, so a ComfyUI
upgrade degrades to eager decode instead of silently diverging.

A captured graph bakes the addresses of the tensors it ran on. generate()
allocates a fresh KV cache per call, so graphs are dropped when a generate
starts and when a prompt ends. Decode buffers are allocated once per generate
so their addresses stay fixed for the whole capture.
"""

import logging
import threading
import weakref

import torch

import comfy.model_management
import comfy.model_prefetch as mp
from comfy.cli_args import args as cli_args

RESIDENT_GRAPH_MODULES = weakref.WeakSet()
GRAPH_CAPTURE_STREAMS = {}

_gate = None


def _enabled():
    return _gate is None or _gate()


def drop_resident_graphs(module):
    """Invalidate every graph under a module before its buffers are replaced."""
    for child in module.modules():
        mp._drop_graph(child)
    RESIDENT_GRAPH_MODULES.clear()


def cleanup():
    for module in RESIDENT_GRAPH_MODULES:
        mp._drop_graph(module)
    RESIDENT_GRAPH_MODULES.clear()
    mp.cleanup_prefetch_queues()


def _capture_stream(device):
    stream = GRAPH_CAPTURE_STREAMS.get(device)
    if stream is not None:
        return stream
    stream = torch.cuda.Stream(device=device)
    # keep PyTorch's persistent BLAS workspaces outside the allocation graph
    with mp.pause_malloc_graph():
        with torch.cuda.stream(stream):
            torch.cuda.current_blas_handle()
            one = torch.empty((2, 2), device=device)
            torch.addmm(one[0], one, one)
    GRAPH_CAPTURE_STREAMS[device] = stream
    return stream


def _resident_graph_step(module, core, device, generator):
    graph = getattr(module, "_comfy_graph", None)
    if graph is not None:
        graph["graph"].replay()
        return
    stream = _capture_stream(device)
    if module not in mp.GRAPH_WARMED_MODULES:
        stream.wait_stream(comfy.model_management.current_stream(device))
        with torch.cuda.stream(stream):
            core()
        comfy.model_management.current_stream(device).wait_stream(stream)
        mp.GRAPH_WARMED_MODULES.add(module)
        return
    graph = torch.cuda.CUDAGraph()
    if generator is not None:
        graph.register_generator_state(generator)
    comfy.model_management.synchronize()
    stream.wait_stream(comfy.model_management.current_stream(device))
    with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
        core()
    comfy.model_management.current_stream(device).wait_stream(stream)
    graph.replay()
    module._comfy_graph = {"graph": graph, "signature": None}
    RESIDENT_GRAPH_MODULES.add(module)


def _patch_prefetch_queue_pop():
    original = mp.prefetch_queue_pop
    if getattr(original, "_gfx906_resident", False):
        return True

    def prefetch_queue_pop(queue, device, module, dtype=None, core=None,
                           enable_graph=False, generator=None, malloc_scope=None):
        if not _enabled() or core is None:
            return original(queue, device, module, dtype, core, enable_graph,
                            generator, malloc_scope)
        malloc_graph = mp.MALLOC_GRAPHS.get(threading.get_ident())
        if malloc_graph is not None and not malloc_graph._comfy_active:
            malloc_graph = None
        resident = (enable_graph and not cli_args.disable_cuda_graphs
                    and comfy.model_management.is_device_cuda(device)
                    and malloc_graph is None and module is not None
                    and getattr(module, "_v_block", None) is None)
        if not resident:
            return original(queue, device, module, dtype, core, enable_graph,
                            generator, malloc_scope)
        real_core = core

        def graphed_core():
            _resident_graph_step(module, real_core, device, generator)

        # force stock down its plain core() path; our closure owns the graph
        return original(queue, device, module, dtype, graphed_core, False,
                        generator, malloc_scope)

    prefetch_queue_pop._gfx906_resident = True
    mp.prefetch_queue_pop = prefetch_queue_pop
    return True


def _patch_forward():
    from comfy.text_encoders import llama

    original = llama.Llama2_.forward
    if getattr(original, "_gfx906_resident", False):
        return True

    def forward(self, x, *args, **kwargs):
        # decode_buffers is the 13th positional after x. generate passes it by
        # keyword, so a present keyword means the caller already supplied one.
        if (not _enabled() or not getattr(self, "graph_dynamic_vbar_blocks", False)
                or cli_args.disable_cuda_graphs or len(args) >= 13
                or "decode_buffers" in kwargs):
            return original(self, x, *args, **kwargs)
        src = kwargs.get("embeds")
        if src is None:
            src = x
        buffers = getattr(self, "_gfx906_decode_buffers", None)
        if isinstance(src, torch.Tensor) and (buffers is None
                or buffers[0].shape[0] != src.shape[0]
                or buffers[0].device != src.device
                or buffers[0].dtype != src.dtype):
            buffers = self.init_decode_buffers(src.shape[0], src.device, src.dtype)
            self._gfx906_decode_buffers = buffers
        if buffers is not None:
            kwargs["decode_buffers"] = buffers
        return original(self, x, *args, **kwargs)

    forward._gfx906_resident = True
    llama.Llama2_.forward = forward
    return True


def _wrap_generate(cls):
    original = vars(cls).get("generate")
    if original is None:
        return False
    if getattr(original, "_gfx906_resident", False):
        return True

    def generate(self, *args, **kwargs):
        if not _enabled():
            return original(self, *args, **kwargs)
        drop_resident_graphs(self.model)
        self.model._gfx906_decode_buffers = None
        return original(self, *args, **kwargs)

    generate._gfx906_resident = True
    cls.generate = generate
    return True


def _patch_generate():
    from comfy.text_encoders import llama

    patched = _wrap_generate(llama.BaseGenerate)
    # Qwen35 overrides generate to route MTP through its own loop; wrap it too
    # so graphs from an earlier generate are dropped on that path as well.
    try:
        from comfy.text_encoders import qwen35
    except ImportError:
        qwen35 = None
    if qwen35 is not None and getattr(qwen35, "Qwen35", None) is not None:
        _wrap_generate(qwen35.Qwen35)
    # The vendored fork owns its own BaseGenerate/Qwen35 class copies (core's
    # volatile qwen35/llama classes are not used). Wrap those too, otherwise
    # their generate() never drops stale graphs and the next decode replays a
    # graph into recycled buffers -> HSA fault. Only wrap when the fork is
    # installed, to avoid importing vendor on cores where it is disabled.
    try:
        from vendor import qwen35 as vendor_qwen35
    except ImportError:
        vendor_qwen35 = None
    if vendor_qwen35 is not None and getattr(vendor_qwen35, "_installed", False):
        from vendor.qwen35 import llama_base as vendor_llama_base
        from vendor.qwen35 import model as vendor_model
        _wrap_generate(vendor_llama_base.BaseGenerate)
        _wrap_generate(vendor_model.Qwen35)
    return patched


def _patch_prompt_end():
    import execution

    original = execution.PromptExecutor.execute_async
    if getattr(original, "_gfx906_resident_cleanup", False):
        return True

    async def execute_async(self, *args, **kwargs):
        try:
            return await original(self, *args, **kwargs)
        finally:
            # always drop: turning the textgen gate off mid-session must not
            # leave graphs captured against buffers the next generate recycles
            cleanup()

    execute_async._gfx906_resident_cleanup = True
    execution.PromptExecutor.execute_async = execute_async
    return True


def install(gate=None):
    global _gate
    _gate = gate
    # Every hook calls the stock function it wraps. Check the touch points up
    # front so a ComfyUI version that moved them degrades to eager decode
    # instead of failing to start; real logic errors still raise below.
    try:
        from comfy.text_encoders import llama
        import execution
    except ImportError:
        logging.warning("gpu-tune: resident graph hooks unavailable on this build, decode runs eagerly")
        return False
    core_symbols = ("prefetch_queue_pop", "_drop_graph", "pause_malloc_graph",
                    "GRAPH_WARMED_MODULES", "MALLOC_GRAPHS", "cleanup_prefetch_queues")
    if (not all(hasattr(mp, name) for name in core_symbols)
            or not hasattr(llama, "Llama2_")
            or not hasattr(llama.BaseGenerate, "generate")
            or not hasattr(execution.PromptExecutor, "execute_async")):
        logging.warning("gpu-tune: resident graph hooks unavailable on this build, decode runs eagerly")
        return False
    return (_patch_prefetch_queue_pop() and _patch_forward()
            and _patch_generate() and _patch_prompt_end())
