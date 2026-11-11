# The GPU's kernel and user-space contract

What `kernel/mediatek/common`'s MappleGPU driver offers the Metal driver. The structures are in
`uapi/mapple_gpu_uapi.h`; the calls follow Linux's Panfrost DRM interface, so Mesa's Panfrost driver runs on
them through `libdrm-xnu`.

- **Service.** `MappleGPUAccelerator`, a subclass of `IOAcceleratorES` (which the kernel defines, having no
  IOGPUFamily). Its properties `MetalPluginName = MappleMTLDriver` and `MetalPluginClassName = VMDevice` make
  Metal load `/System/Library/Extensions/MappleMTLDriver.bundle`. With boot-arg `mp_gpu=0` it isn't published
  and Metal sees no device.
- **Connection.** `IOServiceOpen` with type `VGPU_CLIENT_TYPE` gives a `MappleGPUUserClient`: one GPU address
  space, with its own buffers (BOs) and sync objects. Mesa opens one per Metal device.
- **Calls.** `IOConnectCallStructMethod(selector, &args, ...)`: the structure goes in and comes back updated;
  its size is checked against `vgpu_selector_size[]`. Pointers inside it are the caller's addresses. Errors are
  `kIOReturn*` codes, turned into errno by `vgpu`.

| Selector | As Panfrost's | Notes |
|---|---|---|
| `GET_VERSION`, `GET_PARAM` | `GET_PARAM` | GPU ID, shader and job-slot features; page size 16384 |
| `CREATE_BO`, `GET_BO_OFFSET`, `GEM_CLOSE`, `MADVISE`, `QUERY_BO_INFO` | the same | wired memory at GPU addresses 32 MiB to 4 GiB; 4 KiB GPU pages under 16 KiB XNU pages; `HEAP` BOs grow on a fault |
| `MMAP_BO` | `MMAP_BO` | returns a token; `IOConnectMapMemory64(memoryType = token)` maps it, cacheable; the kernel keeps CPU caches in step around each job |
| `SUBMIT` | `SUBMIT` | fragment jobs on job slot 0, the rest on slot 1; waits for `in_syncs`, signals `out_sync`; a hung chain is stopped and its sync object gets the error, never a panic |
| `WAIT_BO`, `SYNC_BO` | the same | absolute `CLOCK_MONOTONIC` timeout |
| `SYNCOBJ_CREATE/DESTROY/WAIT/RESET/SIGNAL` | binary DRM sync objects | |
| `SYNCOBJ_TRANSFER` | (ours) | copies one sync object's fence into another, instead of sync files |
| `IMPORT_IOSURFACE` | (ours) | maps an IOSurface's pages into the GPU without a copy, instead of dma-buf |
| `IMPORT_USERPTR` | (ours) | maps a task's own pages (`newBufferWithBytesNoCopy`) |
| `JM_CTX_CREATE/DESTROY` | the same | one context per connection |

Not used: a DRM node, dma-buf, sync files, KMS (frames go through IOSurface and AppleCLCD), performance
counters. `vgpu/test_vgpu` checks the contract against the kernel (`VGPU_BACKEND=iokit`) or an in-process mock.
