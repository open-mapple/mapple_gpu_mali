// BO mappings are gone once their BOs are: N BOs created, mapped (drmMap, as Panfrost does), released to the mapping
// cache (drmUnmap) and closed, half through drmCloseBufferHandle and half through DRM_IOCTL_GEM_CLOSE; then N BOs
// mapped with vgpu_mmap and closed without an unmap (a client that forgets: the kernel must drop the mapping).
// The live mappings must be back at the baseline: the mock's count on the Mac, the kernel's memory maps and buffer
// descriptors (kern.mapple_gpu_memory) on the phone (VGPU_BACKEND=iokit).
// Mac: cc -I include -I include/libdrm test_maps.c src/xf86drm_xnu.c ../vgpu/vgpu.c ../vgpu/vgpu_mock.c
//      ../vgpu/vgpu_iokit.c -framework IOKit -framework IOSurface -framework CoreFoundation -o out/test_maps
// Usage: test_maps [N]   (default 200)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include "xf86drm.h"
#include "libdrm/drm.h"
#include "libdrm/panfrost_drm.h"
#include "../vgpu/vgpu.h"

// Kernel counts from the objects line, or the mock's mappings (descriptors -1 then).
static void counts(long *maps, long *descs)
{
    size_t len = 0;
    char *buf = NULL, *p;
    *maps = *descs = -1;
    if (sysctlbyname("kern.mapple_gpu_memory", NULL, &len, NULL, 0) == 0 && (buf = malloc(len + 1)) &&
        sysctlbyname("kern.mapple_gpu_memory", buf, &len, NULL, 0) == 0) {
        buf[len] = 0;
        if ((p = strstr(buf, "memory maps "))) *maps = strtol(p + 12, NULL, 10);
        if ((p = strstr(buf, "buffer descriptors "))) *descs = strtol(p + 19, NULL, 10);
    } else {
        *maps = (long)vgpu_mock_mappings();
    }
    free(buf);
}

static uint32_t new_bo(int fd, uint32_t size)
{
    struct drm_panfrost_create_bo c = { .size = size };
    return drmIoctl(fd, DRM_IOCTL_PANFROST_CREATE_BO, &c) == 0 ? c.handle : 0;
}

int main(int argc, char **argv)
{
    int n = argc > 1 ? atoi(argv[1]) : 200, fails = 0;
    int fd = vgpu_open();
    if (fd < 0) { printf("vgpu_open: %d\n", fd); return 1; }
    printf("backend %s, %d BOs per loop\n", vgpu_backend_name(), n);
    long m0, d0, m1, d1;
    counts(&m0, &d0);
    for (int i = 0; i < n; i++) {
        uint32_t h = new_bo(fd, 256 << 10);
        struct drm_panfrost_mmap_bo mb = { .handle = h };
        void *p = NULL;
        if (!h || drmIoctl(fd, DRM_IOCTL_PANFROST_MMAP_BO, &mb) || drmMap(fd, mb.offset, 256 << 10, &p)) { fails++; break; }
        memset(p, i, 4096);
        drmUnmap(p, 256 << 10);
        if (i & 1) {
            struct drm_gem_close gc = { .handle = h };
            drmIoctl(fd, DRM_IOCTL_GEM_CLOSE, &gc);
        } else {
            drmCloseBufferHandle(fd, h);
        }
    }
    counts(&m1, &d1);
    printf("libdrm map/close: memory maps %ld -> %ld, buffer descriptors %ld -> %ld\n", m0, m1, d0, d1);
    if (m1 > m0 || d1 > d0) fails++;

    for (int i = 0; i < n; i++) {
        struct vgpu_create_bo cb = { .size = 256 << 10 };
        if (vgpu_ioctl(fd, VGPU_SEL_CREATE_BO, &cb)) { fails++; break; }
        void *p = vgpu_mmap(fd, cb.handle, cb.size);
        if (p) memset(p, i, 4096);
        struct vgpu_gem_close gc = { .handle = cb.handle };
        vgpu_ioctl(fd, VGPU_SEL_GEM_CLOSE, &gc);
    }
    long m2, d2;
    counts(&m2, &d2);
    // the mock has no kernel to drop them: there the count shows what a forgotten unmap leaves
    printf("close without unmap: memory maps %ld -> %ld, buffer descriptors %ld -> %ld%s\n", m1, m2, d1, d2,
           d2 < 0 ? " (mock: no kernel cleanup)" : "");
    if (d2 >= 0 && (m2 > m1 || d2 > d1)) fails++;
    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
