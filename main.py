# isort: off
import torch
import fa

# isort: on
import math
import time


# Our minimal flash attention aims to be faster than this by avoiding HBM read/
# writes of N^2 matrices.
def manual_attn(q, k, v):
    att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
    att = torch.nn.functional.softmax(att, dim=-1)
    y = att @ v
    return y

length = 8192
dim = 64

q = torch.rand((length, dim), dtype=torch.float, device="cuda")
k = torch.rand((length, dim), dtype=torch.float, device="cuda")
v = torch.rand((length, dim), dtype=torch.float, device="cuda")

stream = torch.cuda.Stream()

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

print("==========================ManualAttention==========================")
with torch.cuda.stream(stream):
    start_event.record()
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=98, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs"),
    ) as p:
        for i in range(100):
            manual_result = manual_attn(q, k, v)
            p.step()
    end_event.record()
    print(p.key_averages().table(sort_by="cuda_time_total"))

end_event.synchronize()
print(f"Manual attention elapsed: {start_event.elapsed_time(end_event)}ms")


print("==========================FlashAttention==========================")
with torch.cuda.stream(stream):
    start_event.record()
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=98, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs2"),
    ) as p:
        for i in range(100):
            fa_result = fa.attention(q, k, v)
            p.step()
    end_event.record()
    print(p.key_averages().table(sort_by="cuda_time_total"))

end_event.synchronize()
print(f"Flash attention elapsed: {start_event.elapsed_time(end_event)}ms")

print("==========================FlashAttentionKVParallel==========================")
with torch.cuda.stream(stream):
    start_event.record()
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=98, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs3"),
    ) as p:
        for i in range(100):
            fa_kv_parallel_result = fa.attention_kv_parallel(q, k, v)
            p.step()
    end_event.record()
    print(p.key_averages().table(sort_by="cuda_time_total"))

end_event.synchronize()
print(f"Flash attention kv parallel elapsed: {start_event.elapsed_time(end_event)}ms")

stream.synchronize()

print("fa sanity checks: ", torch.allclose(manual_result, fa_result))
print("fa_kv_parallel sanity checks: ", torch.allclose(manual_result, fa_kv_parallel_result))
