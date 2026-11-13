// Pass capture (with the capability trace on, while /private/var/tmp/vmg-capture exists): a render pass with two or more colour attachments that draws a
// compressed texture (any pass while /private/var/tmp/vmg-capture-all exists too) is written out whole, for tests/cafetch-replay to rebuild: the attachments, each pipeline's
// descriptor and function constants, every binding and draw in order, the bytes of the buffers each draw has bound
// and of the compressed textures. At most six passes per process, to VMG_SHM_STATS_DIR/pass-<pid>-<n>/
// (script.txt, b<n>.bin); writing vmg-capture-all again allows six more.
#import "VMPrivate.h"
#import <objc/runtime.h>
#import <stdatomic.h>
#include "../vmg_shm_stats.h"
#include <sys/stat.h>

static const void *const kConstantsKey = &kConstantsKey;
static _Atomic int written;     // passes this process has written since vmg-capture-all last changed: at most six

BOOL VMPassCaptureRoom(void) {
    static _Atomic long armed;  // vmg-capture-all's modification time when the count started
    struct stat st;
    if (stat("/private/var/tmp/vmg-capture-all", &st) == 0 && st.st_mtime != atomic_load(&armed)) {
        atomic_store(&armed, st.st_mtime);
        atomic_store(&written, 0);
    }
    return atomic_load(&written) < 6;
}

void VMCaptureSetConstants(id function, NSString *line) {
    if (function && line) objc_setAssociatedObject(function, kConstantsKey, line, OBJC_ASSOCIATION_COPY);
}

@implementation VMPassCapture {
    NSMutableString *_script;
    NSMutableArray<NSData *> *_blobs;
    NSMutableDictionary<NSData *, NSNumber *> *_blobIndex;
    NSMutableSet *_pipes;
    struct { __unsafe_unretained id<MTLBuffer> buf; NSUInteger off; } _bufs[2][31];
    BOOL _compressed, _all;
}

- (instancetype)initWithPass:(MTLRenderPassDescriptor *)desc all:(BOOL)all {
    if (!(self = [super init])) return nil;
    _all = all;
    _script = [NSMutableString string];
    _blobs = [NSMutableArray array];
    _blobIndex = [NSMutableDictionary dictionary];
    _pipes = [NSMutableSet set];
    [_script appendFormat:@"pass w=%lu h=%lu\n", (unsigned long)desc.renderTargetWidth, (unsigned long)desc.renderTargetHeight];
    for (unsigned i = 0; i < 8; i++) {
        MTLRenderPassColorAttachmentDescriptor *c = desc.colorAttachments[i];
        if (!c.texture) continue;
        MTLClearColor k = c.clearColor;
        [_script appendFormat:@"att %u fmt=%lu w=%lu h=%lu storage=%lu load=%lu store=%lu clear=%g,%g,%g,%g\n", i,
         (unsigned long)c.texture.pixelFormat, (unsigned long)c.texture.width, (unsigned long)c.texture.height,
         (unsigned long)c.texture.storageMode, (unsigned long)c.loadAction, (unsigned long)c.storeAction, k.red, k.green,
         k.blue, k.alpha];
    }
    return self;
}

- (unsigned)blob:(NSData *)d {
    NSNumber *n = _blobIndex[d];
    if (n) return n.unsignedIntValue;
    _blobIndex[d] = @(_blobs.count);
    [_blobs addObject:d];
    return (unsigned)_blobs.count - 1;
}

- (void)pipeline:(VMRenderPipelineState *)p {
    MTLRenderPipelineDescriptor *d = p.traceDescriptor;
    NSValue *key = [NSValue valueWithNonretainedObject:p];
    [_script appendFormat:@"use %p\n", (__bridge void *)p];
    if (!d || [_pipes containsObject:key]) return;
    [_pipes addObject:key];
    [_script appendFormat:@"pipe %p vfn=%@ ffn=%@ samples=%lu depth=%lu stencil=%lu a2c=%d\n", (__bridge void *)p,
     d.vertexFunction.name, d.fragmentFunction.name, (unsigned long)d.rasterSampleCount,
     (unsigned long)d.depthAttachmentPixelFormat, (unsigned long)d.stencilAttachmentPixelFormat, d.alphaToCoverageEnabled];
    for (unsigned i = 0; i < 8; i++) {
        MTLRenderPipelineColorAttachmentDescriptor *c = d.colorAttachments[i];
        if (c.pixelFormat == MTLPixelFormatInvalid) continue;
        [_script appendFormat:@"pca %p %u fmt=%lu mask=%lu blend=%d rgbop=%lu aop=%lu srgb=%lu drgb=%lu sa=%lu da=%lu\n",
         (__bridge void *)p, i, (unsigned long)c.pixelFormat, (unsigned long)c.writeMask, c.blendingEnabled,
         (unsigned long)c.rgbBlendOperation, (unsigned long)c.alphaBlendOperation, (unsigned long)c.sourceRGBBlendFactor,
         (unsigned long)c.destinationRGBBlendFactor, (unsigned long)c.sourceAlphaBlendFactor,
         (unsigned long)c.destinationAlphaBlendFactor];
    }
    MTLVertexDescriptor *v = d.vertexDescriptor;
    for (unsigned i = 0; i < 31; i++) {
        MTLVertexAttributeDescriptor *a = v.attributes[i];
        if (a.format != MTLVertexFormatInvalid)
            [_script appendFormat:@"pva %p %u fmt=%lu off=%lu buf=%lu\n", (__bridge void *)p, i, (unsigned long)a.format,
             (unsigned long)a.offset, (unsigned long)a.bufferIndex];
        MTLVertexBufferLayoutDescriptor *l = v.layouts[i];
        if (l.stride)
            [_script appendFormat:@"pvl %p %u stride=%lu step=%lu rate=%lu\n", (__bridge void *)p, i, (unsigned long)l.stride,
             (unsigned long)l.stepFunction, (unsigned long)l.stepRate];
    }
    NSString *vc = objc_getAssociatedObject(d.vertexFunction, kConstantsKey), *fc = objc_getAssociatedObject(d.fragmentFunction, kConstantsKey);
    if (vc) [_script appendFormat:@"pconst %p v %@\n", (__bridge void *)p, vc];
    if (fc) [_script appendFormat:@"pconst %p f %@\n", (__bridge void *)p, fc];
}

- (void)buffer:(id<MTLBuffer>)b offset:(NSUInteger)o index:(NSUInteger)i stage:(int)s {
    if (s > 1 || i >= 31) return;
    _bufs[s][i].buf = b;
    _bufs[s][i].off = o;
}

- (void)offset:(NSUInteger)o index:(NSUInteger)i stage:(int)s {
    if (s <= 1 && i < 31) _bufs[s][i].off = o;
}

- (void)texture:(id<MTLTexture>)t index:(NSUInteger)i stage:(int)s {
    if (!t) { [_script appendFormat:@"tex %d %lu none\n", s, (unsigned long)i]; return; }
    NSData *bytes = [t isKindOfClass:[VMTexture class]] ? ((VMTexture *)t).noCopyData : nil;
    if (bytes) _compressed = YES;
    [_script appendFormat:@"tex %d %lu fmt=%lu w=%lu h=%lu levels=%lu storage=%lu usage=%lu swizzle=%u,%u,%u,%u blob=%d\n", s,
     (unsigned long)i, (unsigned long)t.pixelFormat, (unsigned long)t.width, (unsigned long)t.height,
     (unsigned long)t.mipmapLevelCount, (unsigned long)t.storageMode, (unsigned long)t.usage, t.swizzle.red,
     t.swizzle.green, t.swizzle.blue, t.swizzle.alpha, bytes ? (int)[self blob:bytes] : -1];
}

- (void)sampler:(VMSamplerState *)smp index:(NSUInteger)i stage:(int)s {
    [_script appendFormat:@"smp %d %lu %@\n", s, (unsigned long)i, smp ? smp.traceState : @"none"];
}

- (void)line:(NSString *)l {
    [_script appendFormat:@"%@\n", l];
}

- (void)draw:(vmg_draw_info)info index:(id<MTLBuffer>)index {
    for (int s = 0; s < 2; s++)
        for (unsigned i = 0; i < 31; i++) {
            id<MTLBuffer> b = _bufs[s][i].buf;
            if (!b || !b.contents || _bufs[s][i].off >= b.length) continue;
            NSUInteger n = MIN(b.length - _bufs[s][i].off, (NSUInteger)65536);
            [_script appendFormat:@"buf %d %u blob=%u\n", s, i,
             [self blob:[NSData dataWithBytes:(uint8_t *)b.contents + _bufs[s][i].off length:n]]];
        }
    int ib = -1;
    if (index && index.contents) {
        NSUInteger n = MIN(index.length - info.index_offset, (NSUInteger)65536);
        ib = (int)[self blob:[NSData dataWithBytes:(uint8_t *)index.contents + info.index_offset length:n]];
    }
    [_script appendFormat:@"draw prim=%u start=%u count=%u instances=%u base_vertex=%d base_instance=%u index=%d index_size=%u\n",
     (unsigned)info.primitive, info.start, info.count, info.instance_count, info.base_vertex, info.base_instance, ib,
     info.index_size];
}

- (void)finish {
    if (!_compressed && !_all) return;
    int n = atomic_fetch_add(&written, 1);
    if (n >= 6) return;
    NSString *dir = [NSString stringWithFormat:@"%s/pass-%d-%d", VMG_SHM_STATS_DIR, getpid(), n];
    mkdir(VMG_SHM_STATS_DIR, 0777);
    mkdir(dir.UTF8String, 0777);
    [_script writeToFile:[dir stringByAppendingPathComponent:@"script.txt"] atomically:NO encoding:NSUTF8StringEncoding error:nil];
    for (NSUInteger i = 0; i < _blobs.count; i++)
        [_blobs[i] writeToFile:[dir stringByAppendingFormat:@"/b%lu.bin", (unsigned long)i] atomically:NO];
    fprintf(stderr, "vmg: pass captured to %s (%lu blobs)\n", dir.UTF8String, (unsigned long)_blobs.count);
}
@end
