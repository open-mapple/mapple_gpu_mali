# mapple_gpu_mali

The Metal driver for Mali (Valhall) GPUs: a Metal front end (`driver/metal`) on Mesa's Panfrost driver and
Valhall compiler, a translator from Metal's AIR to NIR (`driver/vmg_air.c`), a direct AIR to Valhall compiler
(`driver/isa`), and libdrm on IOKit (`libdrm-xnu`, `vgpu`). Mesa is fetched and patched when building
(`mesa/build.sh`, `mesa/patches`). The kernel side is MappleGPU in `kernel/mediatek/common`; the contract
between the two is `docs/uapi.md`.

Checked out at `mapple/gpu/mali` by the [Open Mapple manifest](https://github.com/open-mapple/manifest), which says how to get, build and install it.

Licence: GPL-3.0-or-later (`LICENSE`) with an additional term under section 7(b): keep the attribution
"viva-gpu by Nicola Leone Ciardi - https://github.com/Nik300/viva-gpu" and the credits in `NOTICE`, show them in
the documentation and any About screen, and mark modified versions as modified. Mesa and its patches, the libdrm
headers and the tables from Arm's Valhall ISA description keep their MIT licences (`NOTICE`).
