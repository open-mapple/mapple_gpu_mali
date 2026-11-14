// Rotated IOSurface render targets, as CoreAnimation draws landscape into the display's portrait surfaces: a texture
// with MTLTextureDescriptor's private rotation (quarter turns; width and height swapped for odd turns) over a surface.
// A draw writes each pixel's position, a second pass loads and stores it again; the surface must hold what the Mac's
// GPU leaves in it, for each turn, with a 64-byte pitch and with one that is not.
// Usage: rotation   (llvmpipe; VMG_DRIVER=panfrost VGPU_BACKEND=mock runs Panfrost's import and copies, but the mock
// runs no jobs, so its pixels differ)
#import <Metal/Metal.h>
#import <IOSurface/IOSurfaceRef.h>
#include <stdio.h>
#include <string.h>

id<MTLDevice> TSMtlCreateDevice(void);
@interface MTLTextureDescriptor (VMRotation)
@property (nonatomic) NSUInteger rotation;
@property (nonatomic) BOOL framebufferOnly;
@end

static NSString *const kMSL = @
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "struct V { float4 pos [[position]]; };\n"
    "vertex V tri(uint v [[vertex_id]]) { float2 p = float2((v << 1) & 2, v & 2); V o; o.pos = float4(p * 2 - 1, 0, 1); return o; }\n"
    "fragment float4 where(V i [[stage_in]]) { return float4(floor(i.pos.xy) / 255.0, 0.5, 1); }\n";

static IOSurfaceRef surface(int w, int h, int bpr) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{ (id)kIOSurfaceWidth: @(w), (id)kIOSurfaceHeight: @(h),
        (id)kIOSurfaceBytesPerElement: @4, (id)kIOSurfaceBytesPerRow: @(bpr), (id)kIOSurfacePixelFormat: @((uint32_t)'BGRA') });
}

static void draw(id<MTLDevice> dev, IOSurfaceRef s, NSUInteger rotation) {
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:kMSL options:nil error:&err];
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction = [lib newFunctionWithName:@"tri"];
    pd.fragmentFunction = [lib newFunctionWithName:@"where"];
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
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
    for (int pass = 0; pass < 2; pass++) {
        MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = t;
        rp.colorAttachments[0].loadAction = pass ? MTLLoadActionLoad : MTLLoadActionClear;
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
        static const int shapes[][3] = { { 37, 53, 192 }, { 37, 53, 160 } };   // w, h, bpr (160: not 64-aligned)
        int failed = 0;
        for (unsigned k = 0; k < 2; k++)
            for (NSUInteger r = 0; r < 4; r++) {
                int w = shapes[k][0], h = shapes[k][1], bpr = shapes[k][2];
                IOSurfaceRef a = surface(w, h, bpr), b = surface(w, h, bpr);
                draw(ours, a, r);
                draw(ref, b, r);
                IOSurfaceLock(a, kIOSurfaceLockReadOnly, NULL);
                IOSurfaceLock(b, kIOSurfaceLockReadOnly, NULL);
                const uint8_t *pa = IOSurfaceGetBaseAddress(a), *pb = IOSurfaceGetBaseAddress(b);
                int bad = 0, fx = -1, fy = -1;
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                        if (memcmp(pa + y * IOSurfaceGetBytesPerRow(a) + x * 4, pb + y * IOSurfaceGetBytesPerRow(b) + x * 4, 4) &&
                            !bad++) { fx = x; fy = y; }
                printf("%s rotation %lu, %dx%d bpr %zu: ", bad ? "FAIL" : "ok  ", (unsigned long)r, w, h, IOSurfaceGetBytesPerRow(a));
                if (bad) {
                    const uint8_t *qa = pa + fy * IOSurfaceGetBytesPerRow(a) + fx * 4, *qb = pb + fy * IOSurfaceGetBytesPerRow(b) + fx * 4;
                    printf("%d pixels differ, first (%d, %d): ours texel (%d, %d), Mac (%d, %d)\n", bad, fx, fy, qa[2], qa[1], qb[2], qb[1]);
                } else {
                    printf("same as the Mac\n");
                }
                failed += bad != 0;
                IOSurfaceUnlock(a, kIOSurfaceLockReadOnly, NULL);
                IOSurfaceUnlock(b, kIOSurfaceLockReadOnly, NULL);
                CFRelease(a);
                CFRelease(b);
            }
        printf("%s\n", failed ? "rotation: FAILED" : "rotation: all passed");
        return failed != 0;
    }
}
