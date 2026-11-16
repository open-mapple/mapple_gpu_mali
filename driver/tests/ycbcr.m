// Two-plane 4:2:0 surfaces ('420v', '420f') sampled through YCBCR8_420_2P (500), its sRGB twin (520) and RGB8_420_2P
// (540), as Photos' thumbnails are: a compute pass samples every texel centre, nearest and linear, on our device and on
// the Mac's GPU; the results must agree (converted formats within 1.5 of 255, the raw one exactly).
// Usage: ycbcr   (llvmpipe; VMG_DRIVER=panfrost VGPU_BACKEND=mock runs Panfrost's path, but the mock runs no jobs)
#import <Metal/Metal.h>
#import <IOSurface/IOSurface.h>
#include <math.h>
#include <stdio.h>

id<MTLDevice> TSMtlCreateDevice(void);

static NSString *const kMSL = @
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "kernel void kn(texture2d<float> t [[texture(0)]], device float4 *o [[buffer(0)]], constant uint2 &n [[buffer(1)]],\n"
    "               uint2 p [[thread_position_in_grid]]) {\n"
    "  constexpr sampler s(filter::nearest);\n"
    "  if (p.x < n.x && p.y < n.y) o[p.y * n.x + p.x] = t.sample(s, (float2(p) + 0.5) / float2(n)); }\n"
    "kernel void kl(texture2d<float> t [[texture(0)]], device float4 *o [[buffer(0)]], constant uint2 &n [[buffer(1)]],\n"
    "               uint2 p [[thread_position_in_grid]]) {\n"
    "  constexpr sampler s(filter::linear);\n"
    "  if (p.x < n.x && p.y < n.y) o[p.y * n.x + p.x] = t.sample(s, (float2(p) + 0.5) / float2(n)); }\n";

static IOSurfaceRef surface(int w, int h, uint32_t format) {
    int cw = (w + 1) / 2, ch = (h + 1) / 2, ybpr = (w + 63) & ~63, cbpr = (2 * cw + 63) & ~63;
    NSDictionary *p0 = @{ (id)kIOSurfacePlaneWidth: @(w), (id)kIOSurfacePlaneHeight: @(h), (id)kIOSurfacePlaneBytesPerElement: @1,
                          (id)kIOSurfacePlaneBytesPerRow: @(ybpr), (id)kIOSurfacePlaneOffset: @0, (id)kIOSurfacePlaneSize: @(ybpr * h) };
    NSDictionary *p1 = @{ (id)kIOSurfacePlaneWidth: @(cw), (id)kIOSurfacePlaneHeight: @(ch), (id)kIOSurfacePlaneBytesPerElement: @2,
                          (id)kIOSurfacePlaneBytesPerRow: @(cbpr), (id)kIOSurfacePlaneOffset: @(ybpr * h),
                          (id)kIOSurfacePlaneSize: @(cbpr * ch) };
    IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)@{ (id)kIOSurfaceWidth: @(w), (id)kIOSurfaceHeight: @(h),
        (id)kIOSurfacePixelFormat: @(format), (id)kIOSurfaceAllocSize: @(ybpr * h + cbpr * ch), (id)kIOSurfacePlaneInfo: @[ p0, p1 ] });
    IOSurfaceLock(s, 0, NULL);
    uint8_t *y = IOSurfaceGetBaseAddressOfPlane(s, 0), *c = IOSurfaceGetBaseAddressOfPlane(s, 1);
    for (int j = 0; j < h; j++)
        for (int i = 0; i < w; i++) y[j * ybpr + i] = (uint8_t)(16 + (37 * i + 11 * j) % 220);
    for (int j = 0; j < ch; j++)
        for (int i = 0; i < cw; i++) {
            c[j * cbpr + 2 * i] = (uint8_t)(16 + (53 * i + 29 * j) % 225);
            c[j * cbpr + 2 * i + 1] = (uint8_t)(240 - (41 * i + 17 * j) % 225);
        }
    IOSurfaceUnlock(s, 0, NULL);
    return s;
}

static float *sample(id<MTLDevice> dev, IOSurfaceRef s, MTLPixelFormat format, uint32_t lin, int w, int h) {
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:kMSL options:nil error:&err];
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:lin ? @"kl" : @"kn"] error:&err];
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> t = [dev newTextureWithDescriptor:td iosurface:s plane:0];
    if (!t || !ps) return NULL;
    id<MTLBuffer> o = [dev newBufferWithLength:(NSUInteger)w * h * 16 options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:ps];
    [e setTexture:t atIndex:0];
    [e setBuffer:o offset:0 atIndex:0];
    uint32_t n[2] = { (uint32_t)w, (uint32_t)h };
    [e setBytes:n length:8 atIndex:1];
    [e dispatchThreadgroups:MTLSizeMake((w + 7) / 8, (h + 7) / 8, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    float *r = malloc((size_t)w * h * 16);
    memcpy(r, o.contents, (size_t)w * h * 16);
    return r;
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> ours = TSMtlCreateDevice(), ref = MTLCreateSystemDefaultDevice();
        static const int sizes[][2] = { { 8, 4 }, { 37, 21 } };
        static const uint32_t surfaces[] = { '420v', '420f' };
        static const MTLPixelFormat formats[] = { 500, 520, 540 };
        int failed = 0;
        for (unsigned z = 0; z < 2; z++)
            for (unsigned k = 0; k < 2; k++)
                for (unsigned f = 0; f < 3; f++)
                    for (uint32_t lin = 0; lin < 2; lin++) {
                        int w = sizes[z][0], h = sizes[z][1];
                        IOSurfaceRef s = surface(w, h, surfaces[k]);
                        float *a = sample(ours, s, formats[f], lin, w, h), *b = sample(ref, s, formats[f], lin, w, h);
                        float tol = formats[f] == 540 ? 0.01f : 1.5f, worst = 0;
                        int bad = 0, fx = 0, fy = 0, fc = 0;
                        for (int i = 0; a && b && i < w * h * 4; i++) {
                            float d = fabsf(a[i] - b[i]) * 255;
                            if (d > worst) worst = d;
                            if (d > tol && !bad++) { fx = i / 4 % w; fy = i / 4 / w; fc = i % 4; }
                        }
                        uint32_t sf = surfaces[k];
                        printf("%s %c%c%c%c %dx%d format %lu %s: ", a && b && !bad ? "ok  " : "FAIL", sf >> 24, (sf >> 16) & 0xff,
                               (sf >> 8) & 0xff, sf & 0xff, w, h, (unsigned long)formats[f], lin ? "linear " : "nearest");
                        if (!a || !b) printf("no texture on %s\n", !a ? "ours" : "the Mac");
                        else if (bad) printf("%d values off, first (%d, %d) channel %d: ours %.1f, Mac %.1f\n", bad, fx, fy, fc,
                                             a[4 * (fy * w + fx) + fc] * 255, b[4 * (fy * w + fx) + fc] * 255);
                        else printf("same as the Mac (largest difference %.2f of 255)\n", worst);
                        failed += !a || !b || bad;
                        free(a);
                        free(b);
                        CFRelease(s);
                    }
        printf("%s\n", failed ? "ycbcr: FAILED" : "ycbcr: all passed");
        return failed != 0;
    }
}
