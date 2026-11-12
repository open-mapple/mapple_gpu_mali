// phototest: one image through each stage Photos' thumbnails and the photo poster go through, each output checked for
// all-zero pixels: ImageIO decode, CGImageSourceCreateThumbnailAtIndex, IOSurfaceAccelerator (the scaler service,
// mp_scaler), a VideoToolbox pixel transfer, CoreImage on Metal and with the software renderer.
// Usage: phototest IMAGE [OUTDIR]   (OUTDIR: each stage's output as a PPM)
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

// BGRA or RGBA 8888 pixels: share of pixels with any colour, mean, and a PPM if asked
static void report(const char *stage, const uint8_t *px, size_t w, size_t h, size_t bpr, BOOL bgra) {
    uint64_t lit = 0, sum = 0;
    for (size_t y = 0; y < h; y++)
        for (size_t x = 0; x < w; x++) {
            const uint8_t *p = px + y * bpr + 4 * x;
            if (p[0] | p[1] | p[2]) lit++;
            sum += p[0] + p[1] + p[2];
        }
    double share = w && h ? 100.0 * lit / (double)(w * h) : 0;
    printf("%-34s %4zux%-4zu %5.1f%% of pixels not black, mean %.0f%s\n", stage, w, h, share,
           w && h ? sum / (3.0 * w * h) : 0, lit ? "" : "   ALL ZERO");
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
    outdir = argc > 2 ? argv[2] : NULL;
    NSURL *url = [NSURL fileURLWithPath:@(argv[1])];
    CGImageSourceRef src = CGImageSourceCreateWithURL((__bridge CFURLRef)url, NULL);
    if (!src) { printf("%s: ImageIO cannot open it\n", argv[1]); return 1; }
    printf("%s: %s\n", argv[1], [(__bridge NSString *)CGImageSourceGetType(src) UTF8String]);

    // 1. decode
    CGImageRef img = CGImageSourceCreateImageAtIndex(src, 0, NULL);
    report_cgimage("1 ImageIO decode", img);

    // 2. ImageIO thumbnail (what assetsd's thumbnails start from)
    NSDictionary *to = @{ (id)kCGImageSourceCreateThumbnailFromImageAlways: @YES, (id)kCGImageSourceThumbnailMaxPixelSize: @256,
                          (id)kCGImageSourceCreateThumbnailWithTransform: @YES };
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
        OSStatus st = VTPixelTransferSessionCreate(NULL, &vt);
        if (!st) st = VTPixelTransferSessionTransferImage(vt, pin, pout);
        if (st) printf("%-34s status %d   ALL ZERO\n", "4 VideoToolbox transfer", (int)st), zeros++;
        else report_surface("4 VideoToolbox transfer", out2);
        if (vt) { VTPixelTransferSessionInvalidate(vt); CFRelease(vt); }
        if (pin) CVPixelBufferRelease(pin);
        if (pout) CVPixelBufferRelease(pout);
        CFRelease(out2);
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
        CGImageRef a = metal ? [metal createCGImage:scaled fromRect:scaled.extent] : NULL;
        report_cgimage(metal ? "5 CoreImage Metal" : "5 CoreImage Metal (no device)", a);
        CGImageRef b = [soft createCGImage:scaled fromRect:scaled.extent];
        report_cgimage("6 CoreImage software", b);
        if (a) CGImageRelease(a);
        if (b) CGImageRelease(b);
    }
    printf("%s\n", zeros ? "a stage gave all zeros" : "no stage gave all zeros");
    if (in) CFRelease(in);
    if (out) CFRelease(out);
    if (th) CGImageRelease(th);
    if (img) CGImageRelease(img);
    CFRelease(src);
    return zeros != 0;
}}
