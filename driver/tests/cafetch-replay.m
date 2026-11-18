// SPDX-License-Identifier: GPL-3.0-or-later
// cafetch-replay: a render pass captured by the driver (VMCapture.m: VMG_SHM_STATS_DIR/pass-<pid>-<n>/script.txt and
// its blobs) drawn again from the same library, with the same pipelines (formats, write masks, blending, function
// constants), bindings and draws; attachment 0 read back and summed up (and written as a PPM). Overrides change one
// state field at a time, for bisecting:
//   mask0=M mask1=M   every pipeline's write mask on attachment 0 / 1     blend0=0|1 blend1=0|1   blending off/on
//   c1=private        attachment 1 private instead of memoryless          store1=1  attachment 1 stored
//   split             a new encoder (load, store) before every draw       draws=N   only the first N draws
//   tex=grey          every texture without bytes filled with mid grey (default: a gradient)
//   astc=gradient     ASTC textures replaced by a BGRA8 gradient of their size (llvmpipe on the Mac decodes no ASTC)
// Usage: cafetch-replay DIR LIBRARY [OUT.ppm|-] [override...]   (LIBRARY: a .metallib, else a Metal source file)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

id<MTLDevice> TSMtlCreateDevice(void);

static NSDictionary<NSString *, NSString *> *kv(NSArray<NSString *> *w, NSUInteger from) {
    NSMutableDictionary *d = [NSMutableDictionary dictionary];
    for (NSUInteger i = from; i < w.count; i++) {
        NSRange r = [w[i] rangeOfString:@"="];
        if (r.location != NSNotFound) d[[w[i] substringToIndex:r.location]] = [w[i] substringFromIndex:r.location + 1];
    }
    return d;
}

// "name: X; type: MTLDataTypeY; value: (a, b)" (MTLFunctionConstantValues' description) -> constant values
static MTLFunctionConstantValues *constants(NSString *line) {
    static NSDictionary<NSString *, NSArray *> *types;   // name -> @[MTLDataType, element size, count, float?]
    if (!types) types = @{
        @"MTLDataTypeBool": @[@(MTLDataTypeBool), @1, @1, @0], @"MTLDataTypeUChar": @[@(MTLDataTypeUChar), @1, @1, @0],
        @"MTLDataTypeUChar2": @[@(MTLDataTypeUChar2), @1, @2, @0], @"MTLDataTypeUChar3": @[@(MTLDataTypeUChar3), @1, @3, @0],
        @"MTLDataTypeUChar4": @[@(MTLDataTypeUChar4), @1, @4, @0], @"MTLDataTypeChar": @[@(MTLDataTypeChar), @1, @1, @0],
        @"MTLDataTypeUShort": @[@(MTLDataTypeUShort), @2, @1, @0], @"MTLDataTypeShort": @[@(MTLDataTypeShort), @2, @1, @0],
        @"MTLDataTypeUInt": @[@(MTLDataTypeUInt), @4, @1, @0], @"MTLDataTypeInt": @[@(MTLDataTypeInt), @4, @1, @0],
        @"MTLDataTypeUInt2": @[@(MTLDataTypeUInt2), @4, @2, @0], @"MTLDataTypeInt2": @[@(MTLDataTypeInt2), @4, @2, @0],
        @"MTLDataTypeFloat": @[@(MTLDataTypeFloat), @4, @1, @1], @"MTLDataTypeFloat2": @[@(MTLDataTypeFloat2), @4, @2, @1],
        @"MTLDataTypeFloat4": @[@(MTLDataTypeFloat4), @4, @4, @1], @"MTLDataTypeHalf": @[@(MTLDataTypeHalf), @2, @1, @2],
    };
    MTLFunctionConstantValues *v = [MTLFunctionConstantValues new];
    NSRegularExpression *re = [NSRegularExpression regularExpressionWithPattern:
        @"name: (\\w+); type: (\\w+); value: \\(([^)]*)\\)" options:0 error:nil];
    for (NSTextCheckingResult *m in [re matchesInString:line options:0 range:NSMakeRange(0, line.length)]) {
        NSString *name = [line substringWithRange:[m rangeAtIndex:1]], *type = [line substringWithRange:[m rangeAtIndex:2]];
        NSArray *vals = [[line substringWithRange:[m rangeAtIndex:3]] componentsSeparatedByString:@","];
        NSArray *t = types[type];
        if (!t) { printf("constant %s: type %s not handled\n", name.UTF8String, type.UTF8String); continue; }
        uint8_t buf[64] = {0};
        unsigned size = [t[1] unsignedIntValue], count = [t[2] unsignedIntValue], fl = [t[3] unsignedIntValue];
        for (unsigned k = 0; k < count && k < vals.count; k++) {
            NSString *s = [vals[k] stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet];
            if (fl == 1) { float f = s.floatValue; memcpy(buf + 4 * k, &f, 4); }
            else if (fl == 2) { _Float16 h = (_Float16)s.floatValue; memcpy(buf + 2 * k, &h, 2); }
            else { long long x = s.longLongValue; memcpy(buf + size * k, &x, size); }
        }
        [v setConstantValue:buf type:(MTLDataType)[t[0] unsignedIntegerValue] withName:name];
    }
    return v;
}

int main(int argc, char **argv) { @autoreleasepool {
    if (argc < 3) { printf("usage: cafetch-replay DIR LIBRARY [OUT.ppm|-] [override...]\n"); return 2; }
    NSString *dir = @(argv[1]), *libPath = @(argv[2]);
    NSMutableDictionary<NSString *, NSString *> *ov = [NSMutableDictionary dictionary];
    const char *ppm = NULL;
    for (int i = 3; i < argc; i++) {
        NSString *a = @(argv[i]);
        NSRange r = [a rangeOfString:@"="];
        if (r.location != NSNotFound) ov[[a substringToIndex:r.location]] = [a substringFromIndex:r.location + 1];
        else if ([a isEqualToString:@"split"]) ov[@"split"] = @"1";
        else ppm = argv[i];
    }
    NSString *script = [NSString stringWithContentsOfFile:[dir stringByAppendingPathComponent:@"script.txt"]
                                                 encoding:NSUTF8StringEncoding error:nil];
    if (!script) { printf("%s: no script.txt\n", argv[1]); return 1; }
    id<MTLDevice> dev = TSMtlCreateDevice();
    NSError *err = nil;
    id<MTLLibrary> lib = [libPath hasSuffix:@".metallib"] ? [dev newLibraryWithURL:[NSURL fileURLWithPath:libPath] error:&err]
        : [dev newLibraryWithSource:[NSString stringWithContentsOfFile:libPath encoding:NSUTF8StringEncoding error:nil]
                            options:nil error:&err];
    if (!lib) { printf("library: %s\n", err.description.UTF8String); return 1; }
    NSData *(^blob)(NSString *) = ^NSData *(NSString *n) {
        return [NSData dataWithContentsOfFile:[dir stringByAppendingFormat:@"/b%@.bin", n]];
    };

    // pass 1: attachments and pipelines
    NSArray<NSString *> *lines = [script componentsSeparatedByString:@"\n"];
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    id<MTLTexture> att[8] = {nil};
    NSMutableDictionary<NSString *, MTLRenderPipelineDescriptor *> *pd = [NSMutableDictionary dictionary];
    NSMutableDictionary<NSString *, NSString *> *vconst = [NSMutableDictionary dictionary], *fconst = [NSMutableDictionary dictionary];
    for (NSString *l in lines) {
        NSArray<NSString *> *w = [l componentsSeparatedByString:@" "];
        if (w.count < 2) continue;
        NSDictionary *d = kv(w, 1);
        if ([w[0] isEqualToString:@"att"]) {
            unsigned i = (unsigned)w[1].intValue;
            MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:[d[@"fmt"] integerValue]
                width:[d[@"w"] integerValue] height:[d[@"h"] integerValue] mipmapped:NO];
            td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
            td.storageMode = (MTLStorageMode)[d[@"storage"] integerValue];
            if (i == 1 && [ov[@"c1"] isEqualToString:@"private"]) td.storageMode = MTLStorageModePrivate;
            if (i == 0) td.storageMode = MTLStorageModeShared;        // read back
            att[i] = [dev newTextureWithDescriptor:td];
            NSArray *c = [d[@"clear"] componentsSeparatedByString:@","];
            rp.colorAttachments[i].texture = att[i];
            rp.colorAttachments[i].loadAction = (MTLLoadAction)[d[@"load"] integerValue];
            rp.colorAttachments[i].storeAction = (MTLStoreAction)[d[@"store"] integerValue];
            if (i == 0 || (i == 1 && ov[@"store1"])) rp.colorAttachments[i].storeAction = MTLStoreActionStore;
            if (td.storageMode == MTLStorageModeMemoryless) rp.colorAttachments[i].storeAction = MTLStoreActionDontCare;
            rp.colorAttachments[i].clearColor = MTLClearColorMake([c[0] doubleValue], [c[1] doubleValue], [c[2] doubleValue], [c[3] doubleValue]);
            printf("attachment %u: format %lu %lux%lu storage %lu load %lu store %lu\n", i, (unsigned long)td.pixelFormat,
                   (unsigned long)td.width, (unsigned long)td.height, (unsigned long)td.storageMode,
                   (unsigned long)rp.colorAttachments[i].loadAction, (unsigned long)rp.colorAttachments[i].storeAction);
        } else if ([w[0] isEqualToString:@"pipe"]) {
            MTLRenderPipelineDescriptor *p = [MTLRenderPipelineDescriptor new];
            p.rasterSampleCount = MAX([d[@"samples"] integerValue], 1);
            p.depthAttachmentPixelFormat = [d[@"depth"] integerValue];
            p.stencilAttachmentPixelFormat = [d[@"stencil"] integerValue];
            p.alphaToCoverageEnabled = [d[@"a2c"] boolValue];
            pd[w[1]] = p;
            p.label = [NSString stringWithFormat:@"%@|%@", d[@"vfn"], d[@"ffn"]];
        } else if ([w[0] isEqualToString:@"pca"] && pd[w[1]]) {
            MTLRenderPipelineColorAttachmentDescriptor *c = pd[w[1]].colorAttachments[w[2].intValue];
            NSDictionary *e = kv(w, 3);
            int i = w[2].intValue;
            c.pixelFormat = [e[@"fmt"] integerValue];
            c.writeMask = [(ov[[NSString stringWithFormat:@"mask%d", i]] ?: e[@"mask"]) integerValue];
            c.blendingEnabled = [(ov[[NSString stringWithFormat:@"blend%d", i]] ?: e[@"blend"]) boolValue];
            c.rgbBlendOperation = [e[@"rgbop"] integerValue];
            c.alphaBlendOperation = [e[@"aop"] integerValue];
            c.sourceRGBBlendFactor = [e[@"srgb"] integerValue];
            c.destinationRGBBlendFactor = [e[@"drgb"] integerValue];
            c.sourceAlphaBlendFactor = [e[@"sa"] integerValue];
            c.destinationAlphaBlendFactor = [e[@"da"] integerValue];
        } else if ([w[0] isEqualToString:@"pva"] && pd[w[1]]) {
            NSDictionary *e = kv(w, 3);
            MTLVertexAttributeDescriptor *a = pd[w[1]].vertexDescriptor.attributes[w[2].intValue];
            a.format = [e[@"fmt"] integerValue];
            a.offset = [e[@"off"] integerValue];
            a.bufferIndex = [e[@"buf"] integerValue];
        } else if ([w[0] isEqualToString:@"pvl"] && pd[w[1]]) {
            NSDictionary *e = kv(w, 3);
            MTLVertexBufferLayoutDescriptor *v = pd[w[1]].vertexDescriptor.layouts[w[2].intValue];
            v.stride = [e[@"stride"] integerValue];
            v.stepFunction = [e[@"step"] integerValue];
            v.stepRate = MAX([e[@"rate"] integerValue], 1);
        } else if ([w[0] isEqualToString:@"pconst"] && w.count > 3) {
            NSString *rest = [[w subarrayWithRange:NSMakeRange(3, w.count - 3)] componentsJoinedByString:@" "];
            ([w[2] isEqualToString:@"v"] ? vconst : fconst)[w[1]] = rest;
        }
    }
    NSMutableDictionary<NSString *, id<MTLRenderPipelineState>> *pso = [NSMutableDictionary dictionary];
    for (NSString *k in pd) {
        MTLRenderPipelineDescriptor *p = pd[k];
        NSArray *names = [p.label componentsSeparatedByString:@"|"];
        MTLFunctionConstantValues *vc = vconst[k] ? constants(vconst[k]) : nil, *fc = fconst[k] ? constants(fconst[k]) : nil;
        p.vertexFunction = vc ? [lib newFunctionWithName:names[0] constantValues:vc error:&err] : [lib newFunctionWithName:names[0]];
        p.fragmentFunction = fc ? [lib newFunctionWithName:names[1] constantValues:fc error:&err] : [lib newFunctionWithName:names[1]];
        if (!p.vertexFunction || !p.fragmentFunction) { printf("pipeline %s: functions %s\n", k.UTF8String, err.description.UTF8String); continue; }
        id<MTLRenderPipelineState> s = [dev newRenderPipelineStateWithDescriptor:p error:&err];
        if (!s) { printf("pipeline %s: %s\n", k.UTF8String, err.description.UTF8String); continue; }
        pso[k] = s;
        printf("pipeline %s: %s/%s, masks %lu %lu, blending %d %d\n", k.UTF8String, [names[0] UTF8String], [names[1] UTF8String],
               (unsigned long)p.colorAttachments[0].writeMask, (unsigned long)p.colorAttachments[1].writeMask,
               p.colorAttachments[0].blendingEnabled, p.colorAttachments[1].blendingEnabled);
    }

    // pass 2: the encoder
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
    BOOL split = ov[@"split"] != nil;
    int draws = 0, maxDraws = ov[@"draws"] ? ov[@"draws"].intValue : 1 << 30;
    NSMutableArray *keep = [NSMutableArray array];
    for (NSString *l in lines) {
        NSArray<NSString *> *w = [l componentsSeparatedByString:@" "];
        if (w.count < 2) continue;
        NSString *op = w[0];
        if ([op isEqualToString:@"use"]) {
            if (pso[w[1]]) [e setRenderPipelineState:pso[w[1]]];
        } else if ([op isEqualToString:@"viewport"] && w.count >= 7) {
            [e setViewport:(MTLViewport){ w[1].doubleValue, w[2].doubleValue, w[3].doubleValue, w[4].doubleValue, w[5].doubleValue, w[6].doubleValue }];
        } else if ([op isEqualToString:@"scissor"] && w.count >= 5) {
            [e setScissorRect:(MTLScissorRect){ (NSUInteger)w[1].integerValue, (NSUInteger)w[2].integerValue, (NSUInteger)w[3].integerValue, (NSUInteger)w[4].integerValue }];
        } else if ([op isEqualToString:@"raster"] && w.count >= 5) {
            [e setCullMode:w[1].integerValue];
            [e setFrontFacingWinding:w[2].integerValue];
            [e setTriangleFillMode:w[3].integerValue];
        } else if ([op isEqualToString:@"blendcolor"] && w.count >= 5) {
            [e setBlendColorRed:w[1].floatValue green:w[2].floatValue blue:w[3].floatValue alpha:w[4].floatValue];
        } else if ([op isEqualToString:@"buf"] && w.count >= 4) {
            NSData *b = blob(kv(w, 3)[@"blob"]);
            if (!b.length) continue;
            if (w[1].intValue == 0) [e setVertexBytes:b.bytes length:MIN(b.length, 4096) atIndex:w[2].integerValue];
            else [e setFragmentBytes:b.bytes length:MIN(b.length, 4096) atIndex:w[2].integerValue];
            if (b.length > 4096) {   // setBytes holds 4 KiB: larger ones go in a buffer
                id<MTLBuffer> mb = [dev newBufferWithBytes:b.bytes length:b.length options:MTLResourceStorageModeShared];
                [keep addObject:mb];
                if (w[1].intValue == 0) [e setVertexBuffer:mb offset:0 atIndex:w[2].integerValue];
                else [e setFragmentBuffer:mb offset:0 atIndex:w[2].integerValue];
            }
        } else if ([op isEqualToString:@"tex"] && w.count >= 4) {
            int stage = w[1].intValue;
            NSUInteger slot = (NSUInteger)w[2].integerValue;
            id<MTLTexture> t = nil;
            if (![w[3] isEqualToString:@"none"]) {
                NSMutableDictionary *d = [kv(w, 3) mutableCopy];
                NSInteger f = [d[@"fmt"] integerValue];
                if ([ov[@"astc"] isEqualToString:@"gradient"] && f >= MTLPixelFormatASTC_4x4_sRGB && f <= MTLPixelFormatASTC_12x12_LDR) {
                    d[@"fmt"] = @(MTLPixelFormatBGRA8Unorm).stringValue;
                    d[@"blob"] = @"-1";
                }
                MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:[d[@"fmt"] integerValue]
                    width:[d[@"w"] integerValue] height:[d[@"h"] integerValue] mipmapped:NO];
                td.usage = MTLTextureUsageShaderRead;
                td.storageMode = MTLStorageModeShared;
                NSArray *sw = [d[@"swizzle"] componentsSeparatedByString:@","];
                if (sw.count == 4) td.swizzle = MTLTextureSwizzleChannelsMake([sw[0] intValue], [sw[1] intValue], [sw[2] intValue], [sw[3] intValue]);
                t = [dev newTextureWithDescriptor:td];
                NSData *b = [d[@"blob"] intValue] >= 0 ? blob(d[@"blob"]) : nil;
                NSUInteger tw = t.width, th = t.height;
                if (b && td.pixelFormat >= MTLPixelFormatASTC_4x4_sRGB && td.pixelFormat <= MTLPixelFormatASTC_12x12_LDR) {
                    [t replaceRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0 withBytes:b.bytes bytesPerRow:((tw + 3) / 4) * 16];
                } else if (td.pixelFormat == MTLPixelFormatBGRA8Unorm || td.pixelFormat == MTLPixelFormatRGBA8Unorm ||
                           td.pixelFormat == MTLPixelFormatBGRA8Unorm_sRGB || td.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB) {
                    uint8_t *px = malloc(tw * th * 4);
                    for (NSUInteger y = 0; y < th; y++)
                        for (NSUInteger x = 0; x < tw; x++) {
                            uint8_t *p = px + 4 * (y * tw + x);
                            BOOL grey = [ov[@"tex"] isEqualToString:@"grey"];
                            p[0] = grey ? 128 : (uint8_t)(255 * x / MAX(tw, 1)); p[1] = grey ? 128 : (uint8_t)(255 * y / MAX(th, 1));
                            p[2] = 128; p[3] = 255;
                        }
                    [t replaceRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0 withBytes:px bytesPerRow:tw * 4];
                    free(px);
                }
                [keep addObject:t];
            }
            if (stage == 0) [e setVertexTexture:t atIndex:slot]; else [e setFragmentTexture:t atIndex:slot];
        } else if ([op isEqualToString:@"smp"] && w.count >= 4) {
            if ([w[3] isEqualToString:@"none"]) continue;
            // "min M mag M mip M address S T border B lod A..B aniso N compare C"
            MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
            for (NSUInteger k = 3; k + 1 < w.count; k++) {
                if ([w[k] isEqualToString:@"min"]) sd.minFilter = w[k + 1].integerValue;
                else if ([w[k] isEqualToString:@"mag"]) sd.magFilter = w[k + 1].integerValue;
                else if ([w[k] isEqualToString:@"mip"]) sd.mipFilter = w[k + 1].integerValue;
                else if ([w[k] isEqualToString:@"address"] && k + 2 < w.count) { sd.sAddressMode = w[k + 1].integerValue; sd.tAddressMode = w[k + 2].integerValue; }
            }
            id<MTLSamplerState> s = [dev newSamplerStateWithDescriptor:sd];
            if (w[1].intValue == 0) [e setVertexSamplerState:s atIndex:w[2].integerValue];
            else [e setFragmentSamplerState:s atIndex:w[2].integerValue];
        } else if ([op isEqualToString:@"draw"]) {
            if (draws >= maxDraws) continue;
            NSDictionary *d = kv(w, 1);
            if (split && draws > 0) {
                [e endEncoding];
                for (int i = 0; i < 8; i++) if (rp.colorAttachments[i].texture) {
                    rp.colorAttachments[i].loadAction = MTLLoadActionLoad;
                    if (rp.colorAttachments[i].texture.storageMode != MTLStorageModeMemoryless)
                        rp.colorAttachments[i].storeAction = MTLStoreActionStore;
                }
                e = [cb renderCommandEncoderWithDescriptor:rp];
                printf("(split: the state is not carried into the new encoder; use it with draws=2)\n");
            }
            int ib = [d[@"index"] intValue];
            if (ib >= 0) {
                NSData *b = blob(d[@"index"]);
                id<MTLBuffer> mb = [dev newBufferWithBytes:b.bytes length:b.length options:MTLResourceStorageModeShared];
                [keep addObject:mb];
                [e drawIndexedPrimitives:[d[@"prim"] integerValue] indexCount:[d[@"count"] integerValue]
                               indexType:[d[@"index_size"] intValue] == 4 ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16
                             indexBuffer:mb indexBufferOffset:0 instanceCount:MAX([d[@"instances"] integerValue], 1)
                              baseVertex:[d[@"base_vertex"] integerValue] baseInstance:[d[@"base_instance"] integerValue]];
            } else {
                [e drawPrimitives:[d[@"prim"] integerValue] vertexStart:[d[@"start"] integerValue] vertexCount:[d[@"count"] integerValue]
                    instanceCount:MAX([d[@"instances"] integerValue], 1) baseInstance:[d[@"base_instance"] integerValue]];
            }
            draws++;
        }
    }
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) { printf("command buffer: %s\n", cb.error.description.UTF8String); return 1; }
    id<MTLTexture> out = att[0];
    NSUInteger w = out.width, h = out.height;
    uint8_t *px = malloc(w * h * 4);
    [out getBytes:px bytesPerRow:w * 4 fromRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0];
    uint64_t lit = 0, sum[3] = {0};
    for (NSUInteger i = 0; i < w * h; i++) {
        if (px[4 * i] | px[4 * i + 1] | px[4 * i + 2]) lit++;
        sum[0] += px[4 * i + 2]; sum[1] += px[4 * i + 1]; sum[2] += px[4 * i];
    }
    const uint8_t *c = px + 4 * ((h / 2) * w + w / 2);
    printf("%d draws; attachment 0: %.1f%% of pixels not black, mean rgb %.0f %.0f %.0f, centre rgb %u %u %u\n", draws,
           100.0 * lit / (double)(w * h), sum[0] / (double)(w * h), sum[1] / (double)(w * h), sum[2] / (double)(w * h), c[2], c[1], c[0]);
    if (ppm && strcmp(ppm, "-") != 0) {
        FILE *o = fopen(ppm, "wb");
        fprintf(o, "P6 %lu %lu 255\n", (unsigned long)w, (unsigned long)h);
        for (NSUInteger i = 0; i < w * h; i++) fputc(px[4 * i + 2], o), fputc(px[4 * i + 1], o), fputc(px[4 * i], o);
        fclose(o);
    }
    free(px);
    return 0;
}}
