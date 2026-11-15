// Rotated IOSurface render targets, as CoreAnimation draws landscape into the display's portrait surfaces: a texture
// with MTLTextureDescriptor's private rotation (quarter turns; width and height swapped for odd turns) over a surface.
// A draw writes each pixel's position, a second pass loads and stores it again; then, over a surface the CPU filled,
// a pass loads it and adds each pixel's position. The surface must hold what the Mac's GPU leaves in it, for each turn,
// with 64-byte pitches (turned on the GPU) and one that is not (turned on the CPU).
// Usage: rotation   (llvmpipe; VMG_DRIVER=panfrost VGPU_BACKEND=mock runs Panfrost's import and passes, but the mock
// runs no jobs, so its pixels differ)
#import <Metal/Metal.h>
#import <IOSurface/IOSurfaceRef.h>
#include <stdio.h>
#include <string.h>

id<MTLDevice> TSMtlCreateDevice(void);
void vmg_rotation_counters(uint64_t *gpu, uint64_t *cpu);
@interface MTLTextureDescriptor (VMRotation)
@property (nonatomic) NSUInteger rotation;
@property (nonatomic) BOOL framebufferOnly;
@end

static NSString *const kMSL = @
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "struct V { float4 pos [[position]]; };\n"
    "vertex V tri(uint v [[vertex_id]]) { float2 p = float2((v << 1) & 2, v & 2); V o; o.pos = float4(p * 2 - 1, 0, 1); return o; }\n"
    "fragment float4 where(V i [[stage_in]]) { return float4(floor(i.pos.xy) / 255.0, 0.5, 1); }\n"
    "fragment float4 where0(V i [[stage_in]]) { return float4(floor(i.pos.xy) / 255.0, 0, 0); }\n";

static IOSurfaceRef surface(int w, int h, int bpr, bool filled) {
    IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)@{ (id)kIOSurfaceWidth: @(w), (id)kIOSurfaceHeight: @(h),
        (id)kIOSurfaceBytesPerElement: @4, (id)kIOSurfaceBytesPerRow: @(bpr), (id)kIOSurfacePixelFormat: @((uint32_t)'BGRA') });
    if (!filled) return s;
    IOSurfaceLock(s, 0, NULL);
    uint8_t *p = IOSurfaceGetBaseAddress(s);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            uint8_t *q = p + y * bpr + x * 4;
            q[0] = (uint8_t)x; q[1] = (uint8_t)y; q[2] = (uint8_t)(0x40 + ((x ^ y) & 0x3f)); q[3] = 0xff;
        }
    IOSurfaceUnlock(s, 0, NULL);
    return s;
}

// add: load the surface and add each pixel's position (blending); else clear, draw, then load and store again.
static void draw(id<MTLDevice> dev, IOSurfaceRef s, NSUInteger rotation, bool add) {
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:kMSL options:nil error:&err];
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction = [lib newFunctionWithName:@"tri"];
    pd.fragmentFunction = [lib newFunctionWithName:add ? @"where0" : @"where"];
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    if (add) {
        pd.colorAttachments[0].blendingEnabled = YES;
        pd.colorAttachments[0].sourceRGBBlendFactor = pd.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
        pd.colorAttachments[0].destinationRGBBlendFactor = pd.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOne;
    }
    id<MTLRenderPipelineState> ps = [dev newRenderPipelineStateWithDescriptor:pd error:&err];
    size_t w = IOSurfaceGetWidth(s), h = IOSurfaceGetHeight(s);
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                 width:rotation & 1 ? h : w
                                                                                height:rotation & 1 ? w : h mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget;
    td.rotation = rotation;
    if (rotation) td.framebufferOnly = YES;
    id<MTLTexture> t = [dev newTextureWithDescriptor:td iosurface:s plane:0];
    id<MTLCommandQueue> q = [dev newCommandQueue];
    for (int pass = 0; pass < (add ? 1 : 2); pass++) {
        MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = t;
        rp.colorAttachments[0].loadAction = pass || add ? MTLLoadActionLoad : MTLLoadActionClear;
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
        if (!pass) { [e setRenderPipelineState:ps]; [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3]; }
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> ours = TSMtlCreateDevice(), ref = MTLCreateSystemDefaultDevice();
        static const int shapes[][3] = { { 61, 29, 256 }, { 47, 31, 192 }, { 37, 53, 176 } };   // w, h, bpr (176: not 64-aligned)
        int failed = 0;
        for (int add = 0; add < 2; add++)
            for (unsigned k = 0; k < 3; k++)
                for (NSUInteger r = 0; r < 4; r++) {
                    int w = shapes[k][0], h = shapes[k][1], bpr = shapes[k][2];
                    IOSurfaceRef a = surface(w, h, bpr, add), b = surface(w, h, bpr, add);
                    uint64_t gpu0, cpu0, gpu1, cpu1;
                    vmg_rotation_counters(&gpu0, &cpu0);
                    draw(ours, a, r, add);
                    vmg_rotation_counters(&gpu1, &cpu1);
                    draw(ref, b, r, add);
                    IOSurfaceLock(a, kIOSurfaceLockReadOnly, NULL);
                    IOSurfaceLock(b, kIOSurfaceLockReadOnly, NULL);
                    const uint8_t *pa = IOSurfaceGetBaseAddress(a), *pb = IOSurfaceGetBaseAddress(b);
                    int bad = 0, fx = -1, fy = -1;
                    for (int y = 0; y < h; y++)
                        for (int x = 0; x < w; x++)
                            if (memcmp(pa + y * bpr + x * 4, pb + y * bpr + x * 4, 4) && !bad++) { fx = x; fy = y; }
                    // Rotated textures turn on the GPU where the surface has a 64-byte pitch, else on the CPU.
                    bool gpu = gpu1 > gpu0, cpu = cpu1 > cpu0, path_ok = !r || (bpr % 64 ? cpu && !gpu : gpu && !cpu);
                    printf("%s %s rotation %lu, %dx%d bpr %d, %s: ", bad || !path_ok ? "FAIL" : "ok  ", add ? "load+add" : "draw    ",
                           (unsigned long)r, w, h, bpr, !r ? "not turned" : gpu && !cpu ? "GPU" : cpu && !gpu ? "CPU" : gpu ? "GPU+CPU" : "none");
                    if (bad) {
                        const uint8_t *qa = pa + fy * bpr + fx * 4, *qb = pb + fy * bpr + fx * 4;
                        printf("%d pixels differ, first (%d, %d): ours %02x%02x%02x%02x, Mac %02x%02x%02x%02x\n", bad, fx, fy,
                               qa[3], qa[2], qa[1], qa[0], qb[3], qb[2], qb[1], qb[0]);
                    } else {
                        printf("%s\n", path_ok ? "same as the Mac" : "same as the Mac, wrong path");
                    }
                    failed += bad || !path_ok;
                    IOSurfaceUnlock(a, kIOSurfaceLockReadOnly, NULL);
                    IOSurfaceUnlock(b, kIOSurfaceLockReadOnly, NULL);
                    CFRelease(a);
                    CFRelease(b);
                }
        uint64_t gpu, cpu;
        vmg_rotation_counters(&gpu, &cpu);
        printf("%s (turned %llu times on the GPU, %llu on the CPU)\n", failed ? "rotation: FAILED" : "rotation: all passed",
               (unsigned long long)gpu, (unsigned long long)cpu);
        return failed != 0;
    }
}
