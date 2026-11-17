// BGR10_XR (554) and BGR10_XR_sRGB (555), CoreAnimation's extended-range format: a private texture and one over an
// IOSurface ('w30r'), filled from the CPU with packed texels, read by a kernel, then cleared to extended values by a
// render pass and read back packed; our device and the Mac's GPU must agree (values within 1/510, texels within one code).
// Usage: xr   (llvmpipe)
#import <Metal/Metal.h>
#import <IOSurface/IOSurfaceRef.h>
#include <math.h>
#include <stdio.h>

id<MTLDevice> TSMtlCreateDevice(void);

enum { W = 8, H = 2 };
static NSString *const kMSL = @
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "kernel void k(texture2d<float> t [[texture(0)]], device float4 *o [[buffer(0)]], uint2 p [[thread_position_in_grid]]) {\n"
    "  o[p.y * 8 + p.x] = t.read(p); }\n";

static const uint32_t kTexels[W] = { 384u | 894u << 10 | 639u << 20, 1023u | 512u << 20, 1023u << 20, 3u << 30 | 600u,
                                     0, 0x3fffffffu, 767u | 128u << 10 | 1000u << 20, 384u | 384u << 10 | 384u << 20 };

// The texels read back (float4 each) and, after a clear to (1.2, -0.5, 0.25), the packed texels.
static BOOL run(id<MTLDevice> dev, MTLPixelFormat f, BOOL surface, float *values, uint32_t *packed) {
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f width:W height:H mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    IOSurfaceRef s = NULL;
    id<MTLTexture> t;
    if (surface) {
        s = IOSurfaceCreate((__bridge CFDictionaryRef)@{ (id)kIOSurfaceWidth: @(W), (id)kIOSurfaceHeight: @(H),
            (id)kIOSurfaceBytesPerElement: @4, (id)kIOSurfacePixelFormat: @((uint32_t)'w30r') });
        IOSurfaceLock(s, 0, NULL);
        for (int y = 0; y < H; y++) memcpy((uint8_t *)IOSurfaceGetBaseAddress(s) + y * IOSurfaceGetBytesPerRow(s), kTexels, sizeof kTexels);
        IOSurfaceUnlock(s, 0, NULL);
        td.storageMode = MTLStorageModeShared;
        t = [dev newTextureWithDescriptor:td iosurface:s plane:0];
    } else {
        t = [dev newTextureWithDescriptor:td];
        for (int y = 0; y < H; y++) [t replaceRegion:MTLRegionMake2D(0, y, W, 1) mipmapLevel:0 withBytes:kTexels bytesPerRow:sizeof kTexels];
    }
    if (!t) { if (s) CFRelease(s); return NO; }
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:kMSL options:nil error:&err];
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k"] error:&err];
    id<MTLBuffer> o = [dev newBufferWithLength:W * H * 16 options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:ps];
    [e setTexture:t atIndex:0];
    [e setBuffer:o offset:0 atIndex:0];
    [e dispatchThreads:MTLSizeMake(W, H, 1) threadsPerThreadgroup:MTLSizeMake(W, H, 1)];
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    memcpy(values, o.contents, W * H * 16);
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = t;
    rp.colorAttachments[0].loadAction = MTLLoadActionClear;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    rp.colorAttachments[0].clearColor = MTLClearColorMake(1.2, -0.5, 0.25, 1);
    cb = [q commandBuffer];
    [[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (surface) {
        IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
        memcpy(packed, IOSurfaceGetBaseAddress(s), W * 4);
        IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
        CFRelease(s);
    } else {
        [t getBytes:packed bytesPerRow:W * 4 fromRegion:MTLRegionMake2D(0, 0, W, 1) mipmapLevel:0];
    }
    return YES;
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> ours = TSMtlCreateDevice(), ref = MTLCreateSystemDefaultDevice();
        int failed = 0;
        for (int f = 554; f <= 555; f++)
            for (int surface = 0; surface < 2; surface++) {
                float a[W * H * 4], b[W * H * 4];
                uint32_t pa[W], pb[W];
                BOOL okA = run(ours, (MTLPixelFormat)f, surface, a, pa), okB = run(ref, (MTLPixelFormat)f, surface, b, pb);
                int bad = 0, badp = 0;
                for (int i = 0; okA && okB && i < W * H * 4; i++) bad += fabsf(a[i] - b[i]) > 1.0f / 510;
                for (int i = 0; okA && okB && i < W; i++)
                    for (int c = 0; c < 3; c++) badp += abs((int)((pa[i] >> 10 * c) & 1023) - (int)((pb[i] >> 10 * c) & 1023)) > 1;
                printf("%s format %d %s: ", okA && okB && !bad && !badp ? "ok  " : "FAIL", f, surface ? "IOSurface" : "private  ");
                if (!okA || !okB) printf("no texture on %s\n", !okA ? "ours" : "the Mac");
                else if (bad || badp) printf("%d values and %d clear texels differ (texel 0: ours %08x, Mac %08x)\n", bad, badp, pa[0], pb[0]);
                else printf("same as the Mac\n");
                failed += !okA || !okB || bad || badp;
            }
        printf("%s\n", failed ? "xr: FAILED" : "xr: all passed");
        return failed != 0;
    }
}
