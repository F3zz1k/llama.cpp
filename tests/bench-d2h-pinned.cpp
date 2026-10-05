// bench-d2h-pinned: device-to-host copy bandwidth into pageable versus pinned host memory (A2.7 of the disk-cache
// plan). The auto disk cache copies a unit's state off the device with ggml_backend_tensor_get into ordinary
// pageable memory (docs/disk-cache.md, Known limits); this measures what pinned memory (the device's host buffer
// type, where the backend has one) would change, for the chunk sizes the writer uses.
//
//   bench-d2h-pinned [--dev N] [--mib SIZE] [--reps R]
//
// It allocates SIZE MiB on the device, fills it, and times R copies of the whole buffer, and of 8 MiB and 64 MiB
// slices (the deferred-copy trickle and idle slices), into each kind of host memory. Run it on an otherwise idle
// card. On a CPU-only build the "device" is host memory and both rows measure memcpy.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    int    dev_idx = -1;
    size_t mib     = 1024;
    int    reps    = 5;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--dev" && i + 1 < argc) {
            dev_idx = atoi(argv[++i]);
        } else if (a == "--mib" && i + 1 < argc) {
            mib = (size_t) atoll(argv[++i]);
        } else if (a == "--reps" && i + 1 < argc) {
            reps = atoi(argv[++i]);
        } else {
            fprintf(stderr, "usage: %s [--dev N] [--mib SIZE] [--reps R]\n", argv[0]);
            return 1;
        }
    }

    ggml_backend_load_all();
    ggml_backend_dev_t dev = nullptr;
    if (dev_idx >= 0) {
        dev = ggml_backend_dev_get((size_t) dev_idx);
    } else {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!dev) {
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        }
    }
    if (!dev) {
        fprintf(stderr, "no device\n");
        return 2;
    }
    printf("device: %s (%s)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));

    const size_t n_bytes = mib << 20;
    ggml_init_params ip = { ggml_tensor_overhead() * 4, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) n_bytes);
    ggml_backend_buffer_t dbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(dev));
    if (!dbuf) {
        fprintf(stderr, "could not allocate %zu MiB on the device\n", mib);
        return 3;
    }
    {
        std::vector<uint8_t> fill(n_bytes);
        for (size_t i = 0; i < n_bytes; ++i) {
            fill[i] = (uint8_t) (i * 2654435761u >> 24);
        }
        ggml_backend_tensor_set(t, fill.data(), 0, n_bytes);
    }

    // pageable: plain heap memory, touched once so the first copy does not pay for page faults
    std::vector<uint8_t> pageable(n_bytes, 1);

    // pinned: the device's host buffer type, when the backend has one
    ggml_backend_buffer_type_t hbt = ggml_backend_dev_host_buffer_type(dev);
    ggml_backend_buffer_t hbuf = hbt ? ggml_backend_buft_alloc_buffer(hbt, n_bytes) : nullptr;
    uint8_t * pinned = hbuf ? (uint8_t *) ggml_backend_buffer_get_base(hbuf) : nullptr;
    if (pinned) {
        memset(pinned, 1, n_bytes);
    } else {
        printf("this backend has no pinned host buffer type: only the pageable row is measured\n");
    }

    const size_t slices[] = { n_bytes, (size_t) 64 << 20, (size_t) 8 << 20 };
    printf("%-10s %10s %12s %12s\n", "memory", "slice MiB", "GB/s (best)", "GB/s (mean)");
    for (const auto & [label, dst] : { std::make_pair("pageable", pageable.data()), std::make_pair("pinned", pinned) }) {
        if (!dst) {
            continue;
        }
        for (size_t slice : slices) {
            slice = std::min(slice, n_bytes);
            double best = 1e30, sum = 0.0;
            for (int r = 0; r < reps; ++r) {
                const double t0 = now_s();
                for (size_t off = 0; off < n_bytes; off += slice) {
                    ggml_backend_tensor_get(t, dst + off, off, std::min(slice, n_bytes - off));
                }
                const double dt = now_s() - t0;
                best = std::min(best, dt);
                sum += dt;
            }
            printf("%-10s %10zu %12.2f %12.2f\n", label, slice >> 20, n_bytes / best / 1e9, n_bytes / (sum / reps) / 1e9);
        }
    }

    if (hbuf) {
        ggml_backend_buffer_free(hbuf);
    }
    ggml_backend_buffer_free(dbuf);
    ggml_free(ctx);
    return 0;
}
