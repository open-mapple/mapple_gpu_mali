// Framebuffer fetch as CoreAnimation uses it: a draw writes colour attachment S, a later draw reads it back with
// [[color(S)]] and writes the result (plus 0.1) to the other attachment, which is read back. The matrix: S = 0 or 1;
// S memoryless or private; both draws in one encoder, a memory barrier between them, or two encoders (S stored, then
// loaded; not for memoryless); the read as half4 (CoreAnimation's lph shaders) or float4. Then attachment 1 as A8Unorm
// (a mask: Metal reads it back as (0, 0, 0, a)). Then CoreAnimation's shape on a memoryless attachment 1: the reading
// draw blending into attachment 0 (one, one minus source alpha), the writing draw sampling an ASTC texture made with
// newTextureWithBytesNoCopy (as CoreAnimation's images are), and both.
// Usage: fbfetch [W H]   (default 256 256); fbfetch msl: the shaders' source
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

id<MTLDevice> TSMtlCreateDevice(void);
@interface NSObject (VMNoCopyTexture)
- (id<MTLTexture>)newTextureWithBytesNoCopy:(void *)bytes length:(NSUInteger)length descriptor:(MTLTextureDescriptor *)desc
                                deallocator:(void (^)(void *, NSUInteger))deallocator;
@end

static NSString *const kMSL = @
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "struct V { float4 pos [[position]]; };\n"
    "vertex V fs_quad(uint v [[vertex_id]]) { float2 p = float2((v << 1) & 2, v & 2); V o; o.pos = float4(p * 2 - 1, 0, 1); return o; }\n"
    "struct Two { half4 c0 [[color(0)]]; half4 c1 [[color(1)]]; };\n"
    "struct TwoF { float4 c0 [[color(0)]]; float4 c1 [[color(1)]]; };\n"
    "fragment Two put() { Two o; o.c0 = half4(0.2h, 0.4h, 0.6h, 1.0h); o.c1 = o.c0; return o; }\n"
    "fragment Two get0h(half4 s [[color(0)]]) { Two o; o.c0 = s; o.c1 = s + half4(0.1h, 0.1h, 0.1h, 0); return o; }\n"
    "fragment Two get1h(half4 s [[color(1)]]) { Two o; o.c1 = s; o.c0 = s + half4(0.1h, 0.1h, 0.1h, 0); return o; }\n"
    "fragment TwoF get0f(float4 s [[color(0)]]) { TwoF o; o.c0 = s; o.c1 = s + float4(0.1, 0.1, 0.1, 0); return o; }\n"
    "fragment TwoF get1f(float4 s [[color(1)]]) { TwoF o; o.c1 = s; o.c0 = s + float4(0.1, 0.1, 0.1, 0); return o; }\n"
    "struct VT { float4 pos [[position]]; float2 uv; };\n"
    "vertex VT fs_quad_uv(uint v [[vertex_id]]) { float2 p = float2((v << 1) & 2, v & 2); VT o; o.pos = float4(p * 2 - 1, 0, 1); o.uv = p; return o; }\n"
    "fragment Two putT(VT i [[stage_in]], texture2d<half> t [[texture(3)]]) { constexpr sampler s(filter::linear);\n"
    "  Two o; o.c0 = half4(0); o.c1 = t.sample(s, i.uv); return o; }\n"
    "struct Three { half4 c0 [[color(0)]]; half4 c1 [[color(1)]]; half4 c2 [[color(2)]]; };\n"
    "fragment Three put2() { Three o; o.c0 = half4(0); o.c1 = half4(0.2h, 0.4h, 0.6h, 1.0h); o.c2 = half4(0.6h, 0.2h, 0.4h, 1.0h); return o; }\n"
    "fragment Three get2(half4 a [[color(1)]], half4 b [[color(2)]]) { Three o; o.c0 = (a + b) * 0.5h; o.c1 = a; o.c2 = b; return o; }\n"
    "fragment Two putA() { Two o; o.c0 = half4(0); o.c1 = half4(0.2h, 0.4h, 0.6h, 0.8h); return o; }\n"
    "fragment Two getA(half4 s [[color(1)]]) { Two o; o.c1 = s; o.c0 = half4(s.a, s.a, s.a, 1.0h); return o; }\n";

static id<MTLDevice> dev;
static id<MTLLibrary> lib;

// Writes only attachment `only` (the other's write mask off).
static id<MTLRenderPipelineState> rpipe2(NSString *frag, int only, MTLPixelFormat f1, BOOL blend0) {
    MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = [lib newFunctionWithName:[frag isEqualToString:@"putT"] ? @"fs_quad_uv" : @"fs_quad"];
    if (blend0) {
        d.colorAttachments[0].blendingEnabled = YES;
        d.colorAttachments[0].sourceRGBBlendFactor = d.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
        d.colorAttachments[0].destinationRGBBlendFactor = d.colorAttachments[0].destinationAlphaBlendFactor =
            MTLBlendFactorOneMinusSourceAlpha;
    }
    d.fragmentFunction = [lib newFunctionWithName:frag];
    for (int i = 0; i < 2; i++) {
        d.colorAttachments[i].pixelFormat = i == 1 ? f1 : MTLPixelFormatBGRA8Unorm;
        d.colorAttachments[i].writeMask = i == only ? MTLColorWriteMaskAll : MTLColorWriteMaskNone;
    }
    NSError *e = nil;
    id<MTLRenderPipelineState> p = [dev newRenderPipelineStateWithDescriptor:d error:&e];
    if (!p) printf("pipeline %s: %s\n", frag.UTF8String, e.description.UTF8String);
    return p;
}
static id<MTLRenderPipelineState> rpipe(NSString *frag, int only, MTLPixelFormat f1) {
    MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = [lib newFunctionWithName:@"fs_quad"];
    d.fragmentFunction = [lib newFunctionWithName:frag];
    for (int i = 0; i < 2; i++) {
        d.colorAttachments[i].pixelFormat = i == 1 ? f1 : MTLPixelFormatBGRA8Unorm;
        d.colorAttachments[i].writeMask = i == only ? MTLColorWriteMaskAll : MTLColorWriteMaskNone;
    }
    NSError *e = nil;
    id<MTLRenderPipelineState> p = [dev newRenderPipelineStateWithDescriptor:d error:&e];
    if (!p) printf("pipeline %s: %s\n", frag.UTF8String, e.description.UTF8String);
    return p;
}

enum { ONE, BARRIER, TWO };
static const char *const kMode[] = { "one encoder", "barrier between", "two encoders" };

int main(int argc, char **argv) { @autoreleasepool {
    if (argc > 1 && strcmp(argv[1], "msl") == 0) { printf("%s", kMSL.UTF8String); return 0; }   // the source, for cafetch-replay
    unsigned w = argc > 2 ? (unsigned)atoi(argv[1]) : 256, h = argc > 2 ? (unsigned)atoi(argv[2]) : 256;
    dev = TSMtlCreateDevice();
    NSError *err = nil;
    lib = [dev newLibraryWithSource:kMSL options:nil error:&err];
    if (!lib) { printf("library: %s\n", err.description.UTF8String); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    int fails = 0, runs = 0;
    uint8_t *px = malloc((size_t)w * h * 4);
    for (int s = 0; s < 2; s++)
    for (int memless = 1; memless >= 0; memless--)
    for (int mode = ONE; mode <= TWO; mode++)
    for (int half = 1; half >= 0; half--) { @autoreleasepool {
        if (memless && mode == TWO) continue;
        int o = 1 - s;
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                     width:w height:h mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget;
        td.storageMode = memless ? MTLStorageModeMemoryless : MTLStorageModePrivate;
        id<MTLTexture> src = [dev newTextureWithDescriptor:td];
        td.storageMode = MTLStorageModeShared;
        id<MTLTexture> out = [dev newTextureWithDescriptor:td];
        if (!src || !out) { printf("S=%d %s: no texture\n", s, memless ? "memoryless" : "private"); fails++; continue; }
        id<MTLRenderPipelineState> p1 = rpipe(@"put", s, MTLPixelFormatBGRA8Unorm);
        NSString *g = [NSString stringWithFormat:@"get%d%s", s, half ? "h" : "f"];
        id<MTLRenderPipelineState> p2 = rpipe(g, o, MTLPixelFormatBGRA8Unorm);
        if (!p1 || !p2) { fails++; continue; }

        MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[s].texture = src;
        rp.colorAttachments[o].texture = out;
        for (int i = 0; i < 2; i++) {
            rp.colorAttachments[i].loadAction = MTLLoadActionClear;
            rp.colorAttachments[i].clearColor = MTLClearColorMake(0, 0, 0, 1);
            rp.colorAttachments[i].storeAction = MTLStoreActionStore;
        }
        if (memless) rp.colorAttachments[s].storeAction = MTLStoreActionDontCare;
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
        [e setRenderPipelineState:p1];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        if (mode == BARRIER && [e respondsToSelector:@selector(memoryBarrierWithScope:afterStages:beforeStages:)])
            [e memoryBarrierWithScope:MTLBarrierScopeTextures afterStages:MTLRenderStageFragment beforeStages:MTLRenderStageFragment];
        if (mode == TWO) {
            [e endEncoding];
            for (int i = 0; i < 2; i++) rp.colorAttachments[i].loadAction = MTLLoadActionLoad;
            e = [cb renderCommandEncoderWithDescriptor:rp];
        }
        [e setRenderPipelineState:p2];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        runs++;
        const char *what = memless ? "memoryless" : "private";
        if (cb.error) { printf("S=%d %-10s %-15s %s: %s\n", s, what, kMode[mode], half ? "half " : "float", cb.error.description.UTF8String); fails++; continue; }
        [out getBytes:px bytesPerRow:(size_t)w * 4 fromRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0];
        // expected (BGRA bytes) 0.7, 0.5, 0.3 -> 179, 128, 77
        const uint8_t *c = px + 4 * ((size_t)(h / 2) * w + w / 2);
        int ok = abs(c[0] - 179) <= 3 && abs(c[1] - 128) <= 3 && abs(c[2] - 77) <= 3;
        printf("%s S=%d %-10s %-15s %s: rgb %3u %3u %3u (expected 77 128 179)\n", ok ? "ok  " : "FAIL", s, what,
               kMode[mode], half ? "half " : "float", c[2], c[1], c[0]);
        fails += !ok;
    }}
    // attachment 1 an A8Unorm mask, read back in the next draw: 0.8 -> 204 in every channel of attachment 0
    for (int memless = 1; memless >= 0; memless--)
    for (int mode = ONE; mode <= BARRIER; mode++) { @autoreleasepool {
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatA8Unorm
                                                                                     width:w height:h mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget;
        td.storageMode = memless ? MTLStorageModeMemoryless : MTLStorageModePrivate;
        id<MTLTexture> mask = [dev newTextureWithDescriptor:td];
        td.pixelFormat = MTLPixelFormatBGRA8Unorm;
        td.storageMode = MTLStorageModeShared;
        id<MTLTexture> out = [dev newTextureWithDescriptor:td];
        id<MTLRenderPipelineState> p1 = rpipe(@"putA", 1, MTLPixelFormatA8Unorm), p2 = rpipe(@"getA", 0, MTLPixelFormatA8Unorm);
        if (!mask || !out || !p1 || !p2) { printf("A8: no texture or pipeline\n"); fails++; continue; }
        MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = out;
        rp.colorAttachments[1].texture = mask;
        for (int i = 0; i < 2; i++) {
            rp.colorAttachments[i].loadAction = MTLLoadActionClear;
            rp.colorAttachments[i].clearColor = MTLClearColorMake(0, 0, 0, 0);
            rp.colorAttachments[i].storeAction = i == 1 && memless ? MTLStoreActionDontCare : MTLStoreActionStore;
        }
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
        [e setRenderPipelineState:p1];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        if (mode == BARRIER && [e respondsToSelector:@selector(memoryBarrierWithScope:afterStages:beforeStages:)])
            [e memoryBarrierWithScope:MTLBarrierScopeTextures afterStages:MTLRenderStageFragment beforeStages:MTLRenderStageFragment];
        [e setRenderPipelineState:p2];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        runs++;
        [out getBytes:px bytesPerRow:(size_t)w * 4 fromRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0];
        const uint8_t *c = px + 4 * ((size_t)(h / 2) * w + w / 2);
        int ok = !cb.error && abs(c[0] - 204) <= 3 && abs(c[1] - 204) <= 3 && abs(c[2] - 204) <= 3;
        printf("%s S=1 A8 %-10s %-15s half : rgb %3u %3u %3u (expected 204 204 204)\n", ok ? "ok  " : "FAIL",
               memless ? "memoryless" : "private", kMode[mode], c[2], c[1], c[0]);
        fails += !ok;
    }}
    // CoreAnimation's shape: attachment 1 memoryless; the writer (attachment 1 only) plain or sampling ASTC, the reader
    // (attachment 0 only) plain or blending. ASTC: one void-extent block per 4x4, the colour (0.2, 0.4, 0.6, 1).
    for (int astc = 0; astc < 2; astc++)
    for (int blend = 0; blend < 2; blend++) { @autoreleasepool {
        if (!astc && !blend) continue;
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                     width:w height:h mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget;
        td.storageMode = MTLStorageModeMemoryless;
        id<MTLTexture> c1 = [dev newTextureWithDescriptor:td];
        td.storageMode = MTLStorageModeShared;
        id<MTLTexture> out = [dev newTextureWithDescriptor:td];
        id<MTLTexture> img = nil;
        if (astc) {
            size_t blocks = (size_t)((w + 3) / 4) * ((h + 3) / 4), len = (blocks * 16 + 16383) & ~(size_t)16383;
            uint8_t *bytes = valloc(len);
            static const uint8_t blk[16] = { 0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                             0x33, 0x33, 0x66, 0x66, 0x99, 0x99, 0xff, 0xff };
            for (size_t b = 0; b < blocks; b++) memcpy(bytes + 16 * b, blk, 16);
            MTLTextureDescriptor *ad = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatASTC_4x4_LDR
                                                                                         width:w height:h mipmapped:NO];
            ad.usage = MTLTextureUsageShaderRead;
            ad.storageMode = MTLStorageModeShared;
            if ([(id)dev respondsToSelector:@selector(newTextureWithBytesNoCopy:length:descriptor:deallocator:)])
                img = [(id)dev newTextureWithBytesNoCopy:bytes length:blocks * 16 descriptor:ad
                                             deallocator:^(void *p, NSUInteger l) { (void)l; free(p); }];
            if (!img) { printf("ASTC: newTextureWithBytesNoCopy not there\n"); free(bytes); fails++; continue; }
        }
        id<MTLRenderPipelineState> p1 = rpipe2(astc ? @"putT" : @"put", 1, MTLPixelFormatBGRA8Unorm, NO),
                                   p2 = rpipe2(@"get1h", 0, MTLPixelFormatBGRA8Unorm, blend);
        if (!c1 || !out || !p1 || !p2) { fails++; continue; }
        MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = out;
        rp.colorAttachments[1].texture = c1;
        for (int i = 0; i < 2; i++) {
            rp.colorAttachments[i].loadAction = MTLLoadActionClear;
            rp.colorAttachments[i].clearColor = MTLClearColorMake(0, 0, 0, i ? 0 : 1);
            rp.colorAttachments[i].storeAction = i ? MTLStoreActionDontCare : MTLStoreActionStore;
        }
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
        [e setRenderPipelineState:p1];
        if (img) [e setFragmentTexture:img atIndex:3];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [e setRenderPipelineState:p2];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        runs++;
        [out getBytes:px bytesPerRow:(size_t)w * 4 fromRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0];
        const uint8_t *c = px + 4 * ((size_t)(h / 2) * w + w / 2);
        int ok = !cb.error && abs(c[0] - 179) <= 3 && abs(c[1] - 128) <= 3 && abs(c[2] - 77) <= 3;
        printf("%s S=1 CA memoryless %s writer, %s reader: rgb %3u %3u %3u (expected 77 128 179)\n", ok ? "ok  " : "FAIL",
               astc ? "ASTC-sampling" : "plain", blend ? "blending" : "plain", c[2], c[1], c[0]);
        fails += !ok;
    }}
    // two memoryless attachments of one shape in one pass (the driver shares memoryless backings by shape: the second
    // must get its own): attachment 1 (0.2, 0.4, 0.6), attachment 2 (0.6, 0.2, 0.4), their mean into attachment 0
    { @autoreleasepool {
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                     width:w height:h mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget;
        td.storageMode = MTLStorageModeMemoryless;
        id<MTLTexture> m1 = [dev newTextureWithDescriptor:td], m2 = [dev newTextureWithDescriptor:td];
        td.storageMode = MTLStorageModeShared;
        id<MTLTexture> out = [dev newTextureWithDescriptor:td];
        MTLRenderPipelineDescriptor *pd[2] = { [MTLRenderPipelineDescriptor new], [MTLRenderPipelineDescriptor new] };
        NSString *fn[2] = { @"put2", @"get2" };
        id<MTLRenderPipelineState> ps[2];
        for (int k = 0; k < 2; k++) {
            pd[k].vertexFunction = [lib newFunctionWithName:@"fs_quad"];
            pd[k].fragmentFunction = [lib newFunctionWithName:fn[k]];
            for (int i = 0; i < 3; i++) {
                pd[k].colorAttachments[i].pixelFormat = MTLPixelFormatBGRA8Unorm;
                pd[k].colorAttachments[i].writeMask = (k == 0) == (i > 0) ? MTLColorWriteMaskAll : MTLColorWriteMaskNone;
            }
            ps[k] = [dev newRenderPipelineStateWithDescriptor:pd[k] error:nil];
        }
        if (!m1 || !m2 || !out || !ps[0] || !ps[1]) { printf("two memoryless: no texture or pipeline\n"); fails++; }
        else {
            MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
            id<MTLTexture> at[3] = { out, m1, m2 };
            for (int i = 0; i < 3; i++) {
                rp.colorAttachments[i].texture = at[i];
                rp.colorAttachments[i].loadAction = MTLLoadActionClear;
                rp.colorAttachments[i].clearColor = MTLClearColorMake(0, 0, 0, 1);
                rp.colorAttachments[i].storeAction = i ? MTLStoreActionDontCare : MTLStoreActionStore;
            }
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
            for (int k = 0; k < 2; k++) {
                [e setRenderPipelineState:ps[k]];
                [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
            }
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            runs++;
            [out getBytes:px bytesPerRow:(size_t)w * 4 fromRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0];
            const uint8_t *c = px + 4 * ((size_t)(h / 2) * w + w / 2);
            int ok = !cb.error && abs(c[0] - 128) <= 3 && abs(c[1] - 77) <= 3 && abs(c[2] - 102) <= 3;
            printf("%s two memoryless attachments of one shape: rgb %3u %3u %3u (expected 102 77 128)\n", ok ? "ok  " : "FAIL",
                   c[2], c[1], c[0]);
            fails += !ok;
        }
    }}
    printf("%d/%d passed\n", runs - fails, runs);
    free(px);
    return fails != 0;
}}
