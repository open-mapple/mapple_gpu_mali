// astctest: an ASTC 4x4 texture through the Metal driver, as PosterBoard's snapshots are stored (Apple's AAPL texture
// files, *.atx: chunks of u32 length, 4-byte tag, data; the 4x4 blocks row by row in the "astc" chunk after its size). The blocks go in
// by replaceRegion, by a blit from a buffer and by newTextureWithBytesNoCopy (CoreAnimation's way for these images);
// each texture is drawn (sampled) into a BGRA target the same size and read back.
// The no-copy texture must hand its memory to the deallocator when it is released, not before.
// Usage: astctest FILE.atx [OUT.ppm [usage=N | view | half]]   (the first drawing as a PPM; the no-copy texture with another
// usage, or sampled through a view, as CoreAnimation may)
// On the Mac against this driver (llvmpipe): built with -DVMG_TEST_DYLIB and linked to libMappleMetal.dylib.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// SPI CoreAnimation calls for compressed images (CA::OGL::MetalContext::update_texture)
@interface NSObject (VMNoCopyTexture)
- (id<MTLTexture>)newTextureWithBytesNoCopy:(void *)bytes length:(NSUInteger)length descriptor:(MTLTextureDescriptor *)desc
                                deallocator:(void (^)(void *, NSUInteger))deallocator;
@end

static NSString *const kMSL =
    @"#include <metal_stdlib>\nusing namespace metal;\n"
     "struct VOut { float4 pos [[position]]; };\n"
     "vertex VOut fs_quad(uint v [[vertex_id]]) { float2 p = float2((v << 1) & 2, v & 2); VOut o; o.pos = float4(p * 2 - 1, 0, 1); return o; }\n"
     "fragment float4 show(VOut i [[stage_in]], texture2d<float> t [[texture(0)]]) {\n"
     "  constexpr sampler s(filter::linear);\n"
     "  return t.sample(s, i.pos.xy / float2(t.get_width(), t.get_height()));\n"
     "}\n"
     "fragment half4 show_half(VOut i [[stage_in]], texture2d<half> t [[texture(0)]]) {\n"
     "  constexpr sampler s(filter::linear);\n"
     "  return t.sample(s, float2(i.pos.xy) / float2(t.get_width(), t.get_height()));\n"
     "}\n";

static void summary(const char *what, const uint8_t *px, uint32_t w, uint32_t h)
{
    uint64_t lit = 0;
    for (size_t i = 0; i < (size_t)w * h; i++)
        if (px[4 * i] | px[4 * i + 1] | px[4 * i + 2]) lit++;
    printf("%s: %.1f%% of pixels not black;", what, 100.0 * lit / ((double)w * h));
    const uint32_t xs[] = { 8, w / 2, w - 8, w / 2 }, ys[] = { 8, h / 2, h - 8, h / 4 };
    for (int k = 0; k < 4; k++) {
        const uint8_t *p = px + 4 * ((size_t)ys[k] * w + xs[k]);
        printf(" (%u,%u) rgb %u %u %u", xs[k], ys[k], p[2], p[1], p[0]);
    }
    printf("\n");
}

int main(int argc, char **argv) { @autoreleasepool {
    if (argc < 2) { printf("usage: astctest FILE.atx [OUT.ppm]\n"); return 2; }
    NSData *file = [NSData dataWithContentsOfFile:@(argv[1])];
    const uint8_t *f = file.bytes;
    if (file.length < 64 || memcmp(f, "AAPL\r\n\x1a\n", 8) != 0 || memcmp(f + 12, "HEAD", 4) != 0) {
        printf("%s: not an AAPL texture file\n", argv[1]);
        return 1;
    }
    uint32_t fmt, w, h;
    memcpy(&fmt, f + 32, 4);
    memcpy(&w, f + 40, 4);
    memcpy(&h, f + 44, 4);
    const uint32_t bw = (w + 3) / 4, bh = (h + 3) / 4, bpr = bw * 16;
    const size_t need = (size_t)bpr * bh;
    if (fmt != 0x93b0 || file.length < need) {
        printf("format 0x%x %ux%u, %zu bytes: not ASTC 4x4 or too short (want %zu)\n", fmt, w, h, (size_t)file.length, need);
        return 1;
    }
    const uint8_t *blocks = NULL;
    for (size_t at = 8; at + 8 <= file.length;) {
        uint32_t len;
        memcpy(&len, f + at, 4);
        if (memcmp(f + at + 4, "astc", 4) == 0 && at + 12 + need <= file.length) { blocks = f + at + 12; break; }
        at += 8 + (size_t)len;       // u32 length, tag, then length bytes (the astc chunk's start with its own u32 size)
    }
    if (!blocks) { printf("%s: no astc chunk with %zu bytes\n", argv[1], need); return 1; }
    printf("%s: ASTC 4x4 %ux%u, %zu bytes of blocks at 0x%zx\n", argv[1], w, h, need, (size_t)(blocks - f));

#ifdef VMG_TEST_DYLIB
    extern id<MTLDevice> TSMtlCreateDevice(void);
    id<MTLDevice> dev = TSMtlCreateDevice();
#else
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
#endif
    if (!dev) { printf("no Metal device\n"); return 1; }
    printf("device %s; Apple families:", dev.name.UTF8String);
    for (int i = 1; i <= 9; i++)
        if ([dev supportsFamily:(MTLGPUFamily)(MTLGPUFamilyApple1 + i - 1)]) printf(" %d", i);
    printf("\n");
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:kMSL options:nil error:&err];
    if (!lib) { printf("library: %s\n", err.description.UTF8String); return 1; }
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction = [lib newFunctionWithName:@"fs_quad"];
    BOOL half = argc > 3 && strcmp(argv[3], "half") == 0;    // sampled as texture2d<half> (CoreAnimation's lph shaders)
    pd.fragmentFunction = [lib newFunctionWithName:half ? @"show_half" : @"show"];
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    id<MTLRenderPipelineState> pipe = [dev newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!pipe) { printf("pipeline: %s\n", err.description.UTF8String); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];

    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:w height:h mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget;
    td.storageMode = MTLStorageModeShared;
    id<MTLTexture> target = [dev newTextureWithDescriptor:td];
    uint8_t *px = malloc((size_t)w * h * 4);
    int fails = 0;

    __block int freed = 0;
    for (int way = 0; way < 3; way++) { @autoreleasepool {
        MTLTextureDescriptor *ad = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatASTC_4x4_LDR width:w height:h mipmapped:NO];
        ad.usage = MTLTextureUsageShaderRead;
        ad.storageMode = way == 0 ? MTLStorageModeShared : MTLStorageModePrivate;
        if (way == 2) ad.storageMode = MTLStorageModeShared;
        if (way == 2 && argc > 3 && strncmp(argv[3], "usage=", 6) == 0) ad.usage = (MTLTextureUsage)strtoul(argv[3] + 6, NULL, 0);
        void *nocopy = NULL;
        if (way == 2) {
            size_t page = (size_t)getpagesize(), len = (need + page - 1) / page * page;
            nocopy = valloc(len);
            memcpy(nocopy, blocks, need);
        }
        if (way == 2 && ![(id)dev respondsToSelector:@selector(newTextureWithBytesNoCopy:length:descriptor:deallocator:)]) {
            printf("newTextureWithBytesNoCopy: not implemented by this device\n");
            free(nocopy);
            fails++;
            continue;
        }
        id<MTLTexture> astc = way == 2 ? [(id)dev newTextureWithBytesNoCopy:nocopy length:need descriptor:ad
                                                               deallocator:^(void *p, NSUInteger l) { (void)l; free(p); freed++; }]
                                       : [dev newTextureWithDescriptor:ad];
        const char *what = way == 0 ? "replaceRegion, shared" : way == 1 ? "blit from a buffer, private"
                                                                       : "newTextureWithBytesNoCopy";
        if (!astc) { printf("%s: no ASTC texture from the device\n", what); fails++; continue; }
        id<MTLCommandBuffer> cb = [q commandBuffer];
        if (way == 2) {
        } else if (way == 0) {
            [astc replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0 withBytes:blocks bytesPerRow:bpr];
        } else {
            id<MTLBuffer> buf = [dev newBufferWithBytes:blocks length:need options:MTLResourceStorageModeShared];
            id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
            [b copyFromBuffer:buf sourceOffset:0 sourceBytesPerRow:bpr sourceBytesPerImage:need sourceSize:MTLSizeMake(w, h, 1)
                    toTexture:astc destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
            [b endEncoding];
        }
        MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = target;
        rp.colorAttachments[0].loadAction = MTLLoadActionClear;
        rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
        [e setRenderPipelineState:pipe];
        [e setFragmentTexture:(way == 2 && argc > 3 && strcmp(argv[3], "view") == 0 ? [astc newTextureViewWithPixelFormat:astc.pixelFormat] : astc) atIndex:0];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) { printf("%s: command buffer: %s\n", what, cb.error.description.UTF8String); fails++; continue; }
        [target getBytes:px bytesPerRow:(size_t)w * 4 fromRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0];
        summary(what, px, w, h);
        if (way == 0 && argc > 2 && strcmp(argv[2], "-") != 0) {
            FILE *o = fopen(argv[2], "wb");
            fprintf(o, "P6 %u %u 255\n", w, h);
            for (size_t i = 0; i < (size_t)w * h; i++) fputc(px[4 * i + 2], o), fputc(px[4 * i + 1], o), fputc(px[4 * i], o);
            fclose(o);
            printf("written %s\n", argv[2]);
        }
        if (way == 2) {
            if (freed) { printf("deallocator called while the texture lives\n"); fails++; }
            astc = nil;
        }
    }}
    if ([(id)dev respondsToSelector:@selector(newTextureWithBytesNoCopy:length:descriptor:deallocator:)]) {
        printf("deallocator after the texture's release: %s\n", freed == 1 ? "called once" : freed ? "called more than once" : "NOT called");
        if (freed != 1) fails++;
    }
    free(px);
    return fails != 0;
}}
