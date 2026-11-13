// surftrace: every IOSurface call of a process (DYLD_INSERT_LIBRARIES), its first four arguments and its result on
// stderr while SURFTRACE is set in the environment (phototest sets it around one CoreImage render). The functions are
// the public IOSurface API (IOSurfaceRef.h) both this iOS and macOS export: integer and pointer arguments and results
// only, so the generic wrapper fits them; a phone trace and a Mac trace line up call for call.
// Mac: cc -dynamiclib -Wl,-undefined,dynamic_lookup surftrace.c -o libsurftrace.dylib
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef uint64_t u;
#define TRACE(n) \
    extern u n(u, u, u, u, u, u, u, u); \
    static u my_##n(u a, u b, u c, u d, u e, u f, u g, u h) { \
        u r = n(a, b, c, d, e, f, g, h); \
        if (getenv("SURFTRACE")) fprintf(stderr, "surf %s(0x%llx, 0x%llx, 0x%llx, 0x%llx) = 0x%llx\n", #n, a, b, c, d, r); \
        return r; \
    } \
    __attribute__((used)) static const struct { const void *with, *of; } interpose_##n \
        __attribute__((section("__DATA,__interpose"))) = { (const void *)my_##n, (const void *)n };

TRACE(IOSurfaceAlignProperty)
TRACE(IOSurfaceAllowsPixelSizeCasting)
TRACE(IOSurfaceCopyAllValues)
TRACE(IOSurfaceCopyValue)
TRACE(IOSurfaceCreate)
TRACE(IOSurfaceCreateMachPort)
TRACE(IOSurfaceDecrementUseCount)
TRACE(IOSurfaceGetAllocSize)
TRACE(IOSurfaceGetBaseAddress)
TRACE(IOSurfaceGetBaseAddressOfPlane)
TRACE(IOSurfaceGetBitDepthOfComponentOfPlane)
TRACE(IOSurfaceGetBitOffsetOfComponentOfPlane)
TRACE(IOSurfaceGetBytesPerElement)
TRACE(IOSurfaceGetBytesPerElementOfPlane)
TRACE(IOSurfaceGetBytesPerRow)
TRACE(IOSurfaceGetBytesPerRowOfPlane)
TRACE(IOSurfaceGetElementHeight)
TRACE(IOSurfaceGetElementHeightOfPlane)
TRACE(IOSurfaceGetElementWidth)
TRACE(IOSurfaceGetElementWidthOfPlane)
TRACE(IOSurfaceGetHeight)
TRACE(IOSurfaceGetHeightOfPlane)
TRACE(IOSurfaceGetID)
TRACE(IOSurfaceGetNameOfComponentOfPlane)
TRACE(IOSurfaceGetNumberOfComponentsOfPlane)
TRACE(IOSurfaceGetPixelFormat)
TRACE(IOSurfaceGetPlaneCount)
TRACE(IOSurfaceGetPropertyAlignment)
TRACE(IOSurfaceGetPropertyMaximum)
TRACE(IOSurfaceGetRangeOfComponentOfPlane)
TRACE(IOSurfaceGetSeed)
TRACE(IOSurfaceGetSubsampling)
TRACE(IOSurfaceGetTypeID)
TRACE(IOSurfaceGetTypeOfComponentOfPlane)
TRACE(IOSurfaceGetUseCount)
TRACE(IOSurfaceGetWidth)
TRACE(IOSurfaceGetWidthOfPlane)
TRACE(IOSurfaceIncrementUseCount)
TRACE(IOSurfaceIsInUse)
TRACE(IOSurfaceLock)
TRACE(IOSurfaceLookup)
TRACE(IOSurfaceLookupFromMachPort)
TRACE(IOSurfaceRemoveAllValues)
TRACE(IOSurfaceRemoveValue)
TRACE(IOSurfaceSetOwnershipIdentity)
TRACE(IOSurfaceSetPurgeable)
TRACE(IOSurfaceSetValue)
TRACE(IOSurfaceSetValues)
TRACE(IOSurfaceUnlock)
