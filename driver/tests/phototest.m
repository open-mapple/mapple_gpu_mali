// phototest: one image through each stage Photos' thumbnails and the photo poster go through, each output checked for
// all-zero pixels: ImageIO decode, CGImageSourceCreateThumbnailAtIndex, IOSurfaceAccelerator (the scaler service,
// mp_scaler), a VideoToolbox pixel transfer, CoreImage on Metal and with the software renderer; each stage's time
// (from its call to its pixels read back).
// Usage: phototest IMAGE [OUTDIR]   (OUTDIR: each stage's output as a PPM)
//        phototest IMAGE --surftrace LIBSURFTRACE: only CoreImage into a BGRA IOSurface (software, then Metal), with every
//        IOSurface call it makes on stderr (the library inserted; tests/surftrace.c)
#import <Foundation/Foundation.h>
#import <CoreImage/CoreImage.h>
#import <CoreVideo/CoreVideo.h>
#import <ImageIO/ImageIO.h>
#import <IOSurface/IOSurfaceRef.h>
#import <Metal/Metal.h>
#import <VideoToolbox/VideoToolbox.h>
#include <dlfcn.h>
#include <stdio.h>

static const char *outdir;
static int zeros;
static double started;      // the stage's start (CFAbsoluteTime)
#define STAGE() (started = CFAbsoluteTimeGetCurrent())

// BGRA or RGBA 8888 pixels: share of pixels with any colour, mean, and a PPM if asked
static void report(const char *stage, const uint8_t *px, size_t w, size_t h, size_t bpr, BOOL bgra) {
    uint64_t lit = 0, sum = 0;
    for (size_t y = 0; y < h; y++)
        for (size_t x = 0; x < w; x++) {
            const uint8_t *p = px + y * bpr + 4 * x;
            if (p[0] | p[1] | p[2]) lit++;
            sum += p[0] + p[1] + p[2];
        }
    double share = w && h ? 100.0 * lit / (double)(w * h) : 0, ms = 1000 * (CFAbsoluteTimeGetCurrent() - started);
    printf("%-34s %4zux%-4zu %5.1f%% of pixels not black, mean %3.0f, %7.1f ms%s\n", stage, w, h, share,
           w && h ? sum / (3.0 * w * h) : 0, ms, lit ? "" : "   ALL ZERO");
    zeros += !lit;
    if (outdir && w && h) {
        char path[1024];
        snprintf(path, sizeof path, "%s/%.20s.ppm", outdir, stage);
        for (char *c = path + strlen(outdir) + 1; *c; c++) if (*c == ' ' || *c == '/') *c = '_';
        FILE *o = fopen(path, "wb");
        if (!o) return;
        fprintf(o, "P6 %zu %zu 255\n", w, h);
        for (size_t y = 0; y < h; y++)
            for (size_t x = 0; x < w; x++) {
                const uint8_t *p = px + y * bpr + 4 * x;
                fputc(bgra ? p[2] : p[0], o), fputc(p[1], o), fputc(bgra ? p[0] : p[2], o);
            }
        fclose(o);
    }
}

static void report_cgimage(const char *stage, CGImageRef img) {
    if (!img) { printf("%-34s no image   ALL ZERO\n", stage); zeros++; return; }
    size_t w = CGImageGetWidth(img), h = CGImageGetHeight(img);
    uint8_t *px = calloc(w * h, 4);
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGContextRef c = CGBitmapContextCreate(px, w, h, 8, w * 4, cs, (CGBitmapInfo)kCGImageAlphaPremultipliedLast);
    CGContextDrawImage(c, CGRectMake(0, 0, w, h), img);
    report(stage, px, w, h, w * 4, NO);
    CGContextRelease(c);
    CGColorSpaceRelease(cs);
    free(px);
}

static void report_surface(const char *stage, IOSurfaceRef s) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    report(stage, IOSurfaceGetBaseAddress(s), IOSurfaceGetWidth(s), IOSurfaceGetHeight(s), IOSurfaceGetBytesPerRow(s), YES);
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
}

static IOSurfaceRef bgra_surface(size_t w, size_t h) {
    NSDictionary *p = @{ (id)kIOSurfaceWidth: @(w), (id)kIOSurfaceHeight: @(h), (id)kIOSurfaceBytesPerElement: @4,
                         (id)kIOSurfacePixelFormat: @((uint32_t)'BGRA') };
    return IOSurfaceCreate((__bridge CFDictionaryRef)p);
}

int main(int argc, char **argv) { @autoreleasepool {
    if (argc < 2) { printf("usage: phototest IMAGE [OUTDIR]\n"); return 2; }
    if (argc > 3 && strcmp(argv[2], "--surftrace") == 0 && !getenv("DYLD_INSERT_LIBRARIES")) {
        setenv("DYLD_INSERT_LIBRARIES", argv[3], 1);
        execv(argv[0], argv);
        perror("execv");
        return 1;
    }
    BOOL surftrace = argc > 3 && strcmp(argv[2], "--surftrace") == 0;
    outdir = argc > 2 && !surftrace ? argv[2] : NULL;
    NSURL *url = [NSURL fileURLWithPath:@(argv[1])];
    CGImageSourceRef src = CGImageSourceCreateWithURL((__bridge CFURLRef)url, NULL);
    if (!src) { printf("%s: ImageIO cannot open it\n", argv[1]); return 1; }
    printf("%s: %s\n", argv[1], [(__bridge NSString *)CGImageSourceGetType(src) UTF8String]);

    // 1. decode
    STAGE();
    CGImageRef img = CGImageSourceCreateImageAtIndex(src, 0, NULL);
    report_cgimage("1 ImageIO decode", img);
    if (surftrace && img) {
        size_t iw = CGImageGetWidth(img), ih = CGImageGetHeight(img);
        uint8_t *pp = calloc(iw * ih, 4);
        CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
        CGContextRef pc = CGBitmapContextCreate(pp, iw, ih, 8, iw * 4, cs, (CGBitmapInfo)kCGImageAlphaPremultipliedLast);
        CGContextDrawImage(pc, CGRectMake(0, 0, iw, ih), img);
        CGImageRef plain = CGBitmapContextCreateImage(pc);
        CIImage *ci = [[CIImage imageWithCGImage:plain] imageByApplyingTransform:CGAffineTransformMakeScale(0.25, 0.25)];
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSArray *ctxs = @[ [CIContext contextWithOptions:@{ kCIContextUseSoftwareRenderer: @YES }], [CIContext contextWithMTLDevice:dev] ];
        for (unsigned m = 0; m < 2; m++) {
            NSDictionary *p = @{ (id)kIOSurfaceWidth: @((int)ci.extent.size.width), (id)kIOSurfaceHeight: @((int)ci.extent.size.height),
                                 (id)kIOSurfaceBytesPerElement: @4, (id)kIOSurfacePixelFormat: @((uint32_t)'BGRA') };
            IOSurfaceRef sf = IOSurfaceCreate((__bridge CFDictionaryRef)p);
            fprintf(stderr, "== CoreImage %s into BGRA surface 0x%x\n", m ? "Metal" : "software", IOSurfaceGetID(sf));
            setenv("SURFTRACE", "1", 1);
            [(CIContext *)ctxs[m] render:ci toIOSurface:sf bounds:ci.extent colorSpace:cs];
            unsetenv("SURFTRACE");
            IOSurfaceLock(sf, kIOSurfaceLockReadOnly, NULL);
            const uint8_t *b = IOSurfaceGetBaseAddress(sf);
            size_t n = IOSurfaceGetBytesPerRow(sf) * IOSurfaceGetHeight(sf), nz = 0;
            for (size_t i = 0; i < n; i++) nz += b[i] != 0;
            IOSurfaceUnlock(sf, kIOSurfaceLockReadOnly, NULL);
            fprintf(stderr, "== %s: %zu of %zu bytes non-zero\n", m ? "Metal" : "software", nz, n);
            CFRelease(sf);
        }
        return 0;
    }

    // 2. ImageIO thumbnail (what assetsd's thumbnails start from)
    NSDictionary *to = @{ (id)kCGImageSourceCreateThumbnailFromImageAlways: @YES, (id)kCGImageSourceThumbnailMaxPixelSize: @256,
                          (id)kCGImageSourceCreateThumbnailWithTransform: @YES };
    STAGE();
    CGImageRef th = CGImageSourceCreateThumbnailAtIndex(src, 0, (__bridge CFDictionaryRef)to);
    report_cgimage("2 ImageIO thumbnail", th);

    // the decoded image in a BGRA IOSurface, for the scaler and VideoToolbox stages
    size_t w = img ? CGImageGetWidth(img) : 0, h = img ? CGImageGetHeight(img) : 0;
    IOSurfaceRef in = w ? bgra_surface(w, h) : NULL, out = w ? bgra_surface(MAX(w / 4, 1), MAX(h / 4, 1)) : NULL;
    if (in) {
        IOSurfaceLock(in, 0, NULL);
        CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
        CGContextRef c = CGBitmapContextCreate(IOSurfaceGetBaseAddress(in), w, h, 8, IOSurfaceGetBytesPerRow(in), cs,
                                               kCGImageAlphaPremultipliedFirst | kCGBitmapByteOrder32Little);
        CGContextDrawImage(c, CGRectMake(0, 0, w, h), img);
        CGContextRelease(c);
        CGColorSpaceRelease(cs);
        IOSurfaceUnlock(in, 0, NULL);
        STAGE();
        report_surface("2b IOSurface input as filled", in);
    }

    // what the surface says about itself (CoreImage asks for plane 0 even on a surface without planes)
    if (in) {
        IOSurfaceLock(in, 0, NULL);
        printf("surface: %zux%zu bpe %zu bpr %zu alloc %zu planes %zu base %p; plane 0: base %p w %zu h %zu bpr %zu bpe %zu "
               "elem %zux%zu comps %zu\n", IOSurfaceGetWidth(in), IOSurfaceGetHeight(in), IOSurfaceGetBytesPerElement(in),
               IOSurfaceGetBytesPerRow(in), IOSurfaceGetAllocSize(in), IOSurfaceGetPlaneCount(in), IOSurfaceGetBaseAddress(in),
               IOSurfaceGetBaseAddressOfPlane(in, 0), IOSurfaceGetWidthOfPlane(in, 0), IOSurfaceGetHeightOfPlane(in, 0),
               IOSurfaceGetBytesPerRowOfPlane(in, 0), IOSurfaceGetBytesPerElementOfPlane(in, 0),
               IOSurfaceGetElementWidthOfPlane(in, 0), IOSurfaceGetElementHeightOfPlane(in, 0),
               IOSurfaceGetNumberOfComponentsOfPlane(in, 0));
        IOSurfaceUnlock(in, 0, NULL);
    }

    // 3. IOSurfaceAccelerator (AppleM2ScalerCSCDriver: our MTKScaler, CPU or MDP backend)
    void *fw = dlopen("/System/Library/PrivateFrameworks/IOSurfaceAccelerator.framework/IOSurfaceAccelerator", RTLD_NOW);
    int (*create)(CFAllocatorRef, uint32_t, void **) = fw ? dlsym(fw, "IOSurfaceAcceleratorCreate") : NULL;
    int (*transfer)(void *, IOSurfaceRef, IOSurfaceRef, CFDictionaryRef, void *, void *, void *, uint32_t *) =
        fw ? dlsym(fw, "IOSurfaceAcceleratorTransferSurface") : NULL;
    void *acc = NULL;
    int r = create && transfer ? create(kCFAllocatorDefault, 0, &acc) : -1;
    if (!acc) {
        printf("%-34s no accelerator (create 0x%x; scaler off: no mp_scaler)\n", "3 IOSurfaceAccelerator", r);
    } else if (in) {
        STAGE();
        r = transfer(acc, in, out, NULL, NULL, NULL, NULL, NULL);
        if (r) printf("%-34s transfer 0x%x   ALL ZERO\n", "3 IOSurfaceAccelerator", r), zeros++;
        else report_surface("3 IOSurfaceAccelerator", out);
    }

    // 4. VideoToolbox pixel transfer (hardware scaler when there is one, else software)
    if (in) {
        CVPixelBufferRef pin = NULL, pout = NULL;
        CVPixelBufferCreateWithIOSurface(NULL, in, NULL, &pin);
        IOSurfaceRef out2 = bgra_surface(MAX(w / 4, 1), MAX(h / 4, 1));
        CVPixelBufferCreateWithIOSurface(NULL, out2, NULL, &pout);
        VTPixelTransferSessionRef vt = NULL;
        STAGE();
        OSStatus st = VTPixelTransferSessionCreate(NULL, &vt);
        if (!st) st = VTPixelTransferSessionTransferImage(vt, pin, pout);
        if (st) printf("%-34s status %d   ALL ZERO\n", "4 VideoToolbox transfer", (int)st), zeros++;
        else report_surface("4 VideoToolbox transfer", out2);
        if (vt) { VTPixelTransferSessionInvalidate(vt); CFRelease(vt); }
        if (pin) CVPixelBufferRelease(pin);
        if (pout) CVPixelBufferRelease(pout);
        CFRelease(out2);
    }

    // 4b. VideoToolbox on plain (malloc) pixel buffers, no IOSurface
    if (img) {
        CVPixelBufferRef pin = NULL, pout = NULL;
        CVPixelBufferCreate(NULL, w, h, kCVPixelFormatType_32BGRA, NULL, &pin);
        CVPixelBufferCreate(NULL, MAX(w / 4, 1), MAX(h / 4, 1), kCVPixelFormatType_32BGRA, NULL, &pout);
        CVPixelBufferLockBaseAddress(pin, 0);
        CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
        CGContextRef c = CGBitmapContextCreate(CVPixelBufferGetBaseAddress(pin), w, h, 8, CVPixelBufferGetBytesPerRow(pin), cs,
                                               kCGImageAlphaPremultipliedFirst | kCGBitmapByteOrder32Little);
        CGContextDrawImage(c, CGRectMake(0, 0, w, h), img);
        CGContextRelease(c);
        CGColorSpaceRelease(cs);
        CVPixelBufferUnlockBaseAddress(pin, 0);
        VTPixelTransferSessionRef vt = NULL;
        STAGE();
        OSStatus st = VTPixelTransferSessionCreate(NULL, &vt);
        if (!st) st = VTPixelTransferSessionTransferImage(vt, pin, pout);
        if (st) printf("%-34s status %d   ALL ZERO\n", "4b VideoToolbox, plain buffers", (int)st), zeros++;
        else {
            CVPixelBufferLockBaseAddress(pout, kCVPixelBufferLock_ReadOnly);
            report("4b VideoToolbox, plain buffers", CVPixelBufferGetBaseAddress(pout), CVPixelBufferGetWidth(pout),
                   CVPixelBufferGetHeight(pout), CVPixelBufferGetBytesPerRow(pout), YES);
            CVPixelBufferUnlockBaseAddress(pout, kCVPixelBufferLock_ReadOnly);
        }
        if (vt) { VTPixelTransferSessionInvalidate(vt); CFRelease(vt); }
        CVPixelBufferRelease(pin);
        CVPixelBufferRelease(pout);
    }

    // 5b. CoreImage in software from a plain bitmap image, rendered into a plain bitmap (no IOSurface)
    if (img) {
        uint8_t *px = calloc(w * h, 4);
        CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
        CGContextRef c = CGBitmapContextCreate(px, w, h, 8, w * 4, cs, (CGBitmapInfo)kCGImageAlphaPremultipliedLast);
        CGContextDrawImage(c, CGRectMake(0, 0, w, h), img);
        CGImageRef plain = CGBitmapContextCreateImage(c);
        CIImage *ci = [CIImage imageWithCGImage:plain];
        CIImage *scaled = [ci imageByApplyingTransform:CGAffineTransformMakeScale(0.25, 0.25)];
        CIContext *soft = [CIContext contextWithOptions:@{ kCIContextUseSoftwareRenderer: @YES }];
        size_t ow = (size_t)scaled.extent.size.width, oh = (size_t)scaled.extent.size.height;
        uint8_t *o = calloc(ow * oh, 4);
        STAGE();
        [soft render:scaled toBitmap:o rowBytes:ow * 4 bounds:scaled.extent format:kCIFormatRGBA8 colorSpace:cs];
        report("5b CoreImage software, bitmaps", o, ow, oh, ow * 4, NO);
        free(o);
        CGImageRelease(plain);
        CGContextRelease(c);
        CGColorSpaceRelease(cs);
        free(px);
    }

    // 5. and 6. CoreImage, a Lanczos scale to a quarter, rendered on Metal and in software
    if (img) {
        CIImage *ci = [CIImage imageWithCGImage:img];
        CIFilter *f = [CIFilter filterWithName:@"CILanczosScaleTransform"];
        [f setValue:ci forKey:kCIInputImageKey];
        [f setValue:@0.25 forKey:kCIInputScaleKey];
        CIImage *scaled = f.outputImage;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        CIContext *metal = dev ? [CIContext contextWithMTLDevice:dev] : nil;
        CIContext *soft = [CIContext contextWithOptions:@{ kCIContextUseSoftwareRenderer: @YES }];
        STAGE();
        CGImageRef a = metal ? [metal createCGImage:scaled fromRect:scaled.extent] : NULL;
        report_cgimage(metal ? "5 CoreImage Metal" : "5 CoreImage Metal (no device)", a);
        STAGE();
        CGImageRef b = [soft createCGImage:scaled fromRect:scaled.extent];
        report_cgimage("6 CoreImage software", b);
        if (a) CGImageRelease(a);
        if (b) CGImageRelease(b);
    }
    // 7. CoreImage rendered into an IOSurface of each format it uses (render:toIOSurface:), software and Metal;
    // the surface's raw bytes read back: none non-zero = the write never landed
    if (img) {
        static const struct { uint32_t fourcc; int bpe; const char *name; } fmts[] = {
            { 'BGRA', 4, "BGRA" }, { 'RGBA', 4, "RGBA" }, { 'RGhA', 8, "RGhA half" }, { 'RGfA', 16, "RGfA float" },
            { 'l10r', 4, "l10r 10-bit" },
        };
        // the source as ImageIO decoded it, and the same pixels as a plain bitmap image (stage 5b's source)
        uint8_t *pp = calloc(w * h, 4);
        CGColorSpaceRef pcs = CGColorSpaceCreateDeviceRGB();
        CGContextRef pc = CGBitmapContextCreate(pp, w, h, 8, w * 4, pcs, (CGBitmapInfo)kCGImageAlphaPremultipliedLast);
        CGContextDrawImage(pc, CGRectMake(0, 0, w, h), img);
        CGImageRef plain = CGBitmapContextCreateImage(pc);
        CIImage *srcs[2] = { [[CIImage imageWithCGImage:img] imageByApplyingTransform:CGAffineTransformMakeScale(0.25, 0.25)],
                             [[CIImage imageWithCGImage:plain] imageByApplyingTransform:CGAffineTransformMakeScale(0.25, 0.25)] };
        CIImage *ci = srcs[0];
        CGRect ext = ci.extent;
        size_t ow = (size_t)ext.size.width, oh = (size_t)ext.size.height;
        CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSArray *ctxs = @[ [CIContext contextWithOptions:@{ kCIContextUseSoftwareRenderer: @YES }],
                           dev ? [CIContext contextWithMTLDevice:dev] : (id)[NSNull null] ];
        for (unsigned from = 0; from < 2; from++)
        for (unsigned k = 0; k < sizeof fmts / sizeof *fmts; k++)
            for (unsigned m = 0; m < 2; m++) {
                if (from && k > 0) continue;   // the plain source: BGRA only
                if (ctxs[m] == (id)[NSNull null]) continue;
                NSDictionary *p = @{ (id)kIOSurfaceWidth: @(ow), (id)kIOSurfaceHeight: @(oh), (id)kIOSurfaceBytesPerElement: @(fmts[k].bpe),
                                     (id)kIOSurfacePixelFormat: @(fmts[k].fourcc) };
                IOSurfaceRef sf = IOSurfaceCreate((__bridge CFDictionaryRef)p);
                char stage[64];
                snprintf(stage, sizeof stage, "7 CI %s into %s%s", m ? "Metal" : "soft", fmts[k].name, from ? ", plain src" : "");
                if (!sf) { printf("%-34s no surface\n", stage); continue; }
                STAGE();
                [(CIContext *)ctxs[m] render:srcs[from] toIOSurface:sf bounds:ext colorSpace:cs];
                IOSurfaceLock(sf, kIOSurfaceLockReadOnly, NULL);
                const uint8_t *b = IOSurfaceGetBaseAddress(sf);
                size_t n = IOSurfaceGetBytesPerRow(sf) * oh, nz = 0;
                for (size_t i = 0; i < n; i++) nz += b[i] != 0;
                IOSurfaceUnlock(sf, kIOSurfaceLockReadOnly, NULL);
                printf("%-34s %4zux%-4zu %5.1f%% of bytes non-zero, %7.1f ms%s\n", stage, ow, oh, 100.0 * nz / (double)n,
                       1000 * (CFAbsoluteTimeGetCurrent() - started), nz ? "" : "   ALL ZERO");
                zeros += !nz;
                CFRelease(sf);
            }
        CGColorSpaceRelease(cs);
        CGImageRelease(plain);
        CGContextRelease(pc);
        CGColorSpaceRelease(pcs);
        free(pp);
    }
    printf("%s\n", zeros ? "a stage gave all zeros" : "no stage gave all zeros");
    if (in) CFRelease(in);
    if (out) CFRelease(out);
    if (th) CGImageRelease(th);
    if (img) CGImageRelease(img);
    CFRelease(src);
    return zeros != 0;
}}
