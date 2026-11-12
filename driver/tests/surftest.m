// surftest: the GPU draws a known colour into an IOSurface the size of the screen (1125x2436 BGRA, pitch 4608, as
// backboardd's), for every combination of the kernel's experiment switches (DEVELOPMENT kernels, as root):
//   kern.mapple_gpu_surf_uncached  new surfaces write-combined
//   kern.mapple_gpu_surf_copy      the driver copies the surface instead of importing it
//   kern.mapple_gpu_linear         render targets linear
//   kern.mapple_gpu_surf_maint     after the job: CPU lines dropped at the next CPU access (0), at once (1), never (2)
// and reads it back through IOSurfaceLock and the kernel's kern.mapple_gpu_surf_debug (DRAM, the CPU's view, after a
// clean). The switches are back at 0 at the end. Usage: surftest [combination index]
// surftest debug ID...: only the kernel's read-back of those surfaces (e.g. backboardd's, from the SURF: log)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <IOSurface/IOSurfaceRef.h>
#include <errno.h>
#include <string.h>
#include <sys/sysctl.h>

static int set(const char *name, int v)
{
    return sysctlbyname(name, NULL, NULL, &v, sizeof(v));
}

static uint32_t at(IOSurfaceRef s, uint32_t x, uint32_t y)
{
    const uint8_t *p = IOSurfaceGetBaseAddress(s);
    return *(const uint32_t *)(p + (size_t)y * IOSurfaceGetBytesPerRow(s) + (size_t)x * 4);
}

int main(int argc, char **argv) { @autoreleasepool {
    if (argc > 2 && strcmp(argv[1], "debug") == 0) {
        for (int i = 2; i < argc; i++) {
            uint32_t id = (uint32_t)strtoul(argv[i], NULL, 0);
            char dbg[1024] = "";
            size_t len = sizeof(dbg);
            if (sysctlbyname("kern.mapple_gpu_surf_debug", dbg, &len, &id, sizeof(id)) == 0) printf("%s", dbg);
            else printf("surface %u: %s\n", id, strerror(errno));
        }
        return 0;
    }
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { printf("no Metal device (link: gpu on, gpu publish)\n"); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    const uint32_t W = 1125, H = 2436, BPR = 4608;
    int only = argc > 1 ? atoi(argv[1]) : -1, fails = 0, n = 0;
    printf("Metal device %s; uncached copy linear maint: lock start / middle (want) -> result\n", dev.name.UTF8String);
    for (int unc = 0; unc < 2; unc++)
    for (int copy = 0; copy < 2; copy++)
    for (int lin = 0; lin < 2; lin++)
    for (int maint = 0; maint < 3; maint++, n++) {
        if (only >= 0 && n != only) continue;
        if (set("kern.mapple_gpu_surf_uncached", unc) || set("kern.mapple_gpu_surf_copy", copy) ||
            set("kern.mapple_gpu_linear", lin) || set("kern.mapple_gpu_surf_maint", maint)) {
            printf("cannot set the kern.mapple_gpu_* switches (root, DEVELOPMENT kernel?)\n");
            return 1;
        }
        NSDictionary *props = @{ (id)kIOSurfaceWidth: @(W), (id)kIOSurfaceHeight: @(H), (id)kIOSurfaceBytesPerElement: @4,
                                 (id)kIOSurfaceBytesPerRow: @(BPR), (id)kIOSurfacePixelFormat: @(0x42475241u) };
        IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)props);
        if (!s) { printf("%2d: IOSurfaceCreate failed\n", n); fails++; continue; }
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                      width:W height:H mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModeShared;
        id<MTLTexture> t = [dev newTextureWithDescriptor:td iosurface:s plane:0];
        // a colour of its own for each combination: a stale earlier one cannot pass
        uint8_t b = (uint8_t)(0x10 + n * 8), g = 0x80, r = (uint8_t)(0xf0 - n * 4);
        uint32_t want = 0xff000000u | (uint32_t)r << 16 | (uint32_t)g << 8 | b;
        MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = t;
        rp.colorAttachments[0].loadAction = MTLLoadActionClear;
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        rp.colorAttachments[0].clearColor = MTLClearColorMake(r / 255.0, g / 255.0, b / 255.0, 1.0);
        id<MTLCommandBuffer> cb = [q commandBuffer];
        [[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
        uint32_t s0 = at(s, 0, 0), s1 = at(s, W / 2, H / 2);
        IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
        bool ok = s0 == want && s1 == want && cb.status == MTLCommandBufferStatusCompleted;
        fails += !ok;
        printf("%2d: %d %d %d %d: %08x / %08x (%08x) -> %s%s\n", n, unc, copy, lin, maint, s0, s1, want,
               ok ? "ok" : "WRONG", cb.error ? [[cb.error description] UTF8String] : "");
        uint32_t id = IOSurfaceGetID(s);
        char dbg[1024] = "";
        size_t len = sizeof(dbg);
        if (sysctlbyname("kern.mapple_gpu_surf_debug", dbg, &len, &id, sizeof(id)) == 0) printf("%s", dbg);
        t = nil;
        CFRelease(s);
    }
    set("kern.mapple_gpu_surf_uncached", 0); set("kern.mapple_gpu_surf_copy", 0);
    set("kern.mapple_gpu_linear", 0); set("kern.mapple_gpu_surf_maint", 0);
    printf("%d wrong\n", fails);
    return fails != 0;
} }
