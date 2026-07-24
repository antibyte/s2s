# qwentts.cpp local patch series

The Docker SYCL build is pinned to:

- qwentts.cpp `82cd05b`
- ggml `c044c6f`

The first pair preserves the complete dirty source state that existed before
the upstream update:

1. `0001-qwentts-local.patch` — language/server/B580 changes plus ABI v3,
   256-frame protection and the selectable code sampler.
2. `0002-ggml-local.patch` — existing Vulkan/SYCL device fixes plus the
   device-resident sampling kernel.

The Docker and clean native builds use the conflict-resolved pair:

1. `0101-qwentts-rebased.patch` — applies directly to qwentts `82cd05b`.
2. `0102-ggml-rebased.patch` — applies directly to ggml `c044c6f`.

The rebased series keeps upstream's new batch-slot implementation. Device
sampling state is slot-major, logits and next-token inputs stay on the device,
and the final `[batch, 16]` token matrix is transferred once per audio frame.
The old Vulkan misalignment hunk was dropped from the rebased patch because
the same correction is already present in `c044c6f`.

Patch bases:

- qwentts.cpp `95b4840ad3722b0b67acb945cd57682aae1ac9ca`
- ggml `9e2947f17583acc2f657a77c29b6593ca0fbc6c4`

`scripts/prepare_qwentts_source.ps1` creates a clean source tree at the pinned
revisions and applies the rebased series. It never resets or cleans the dirty
reference checkout under `third_party/qwentts.cpp`.
