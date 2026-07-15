<p align="center"><img width="30%" src="https://raw.githubusercontent.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/master/docs/data/OpenVX_logo.png" /></p>

# OpenVX 1.3.2 Conformance Test Suite

[![Conformance](https://github.com/KhronosGroup/OpenVX-cts/actions/workflows/conformance.yml/badge.svg?branch=openvx_1.3.2)](https://github.com/KhronosGroup/OpenVX-cts/actions/workflows/conformance.yml)

The OpenVX Conformance Test Suite (CTS) verifies that an OpenVX implementation conforms to the [OpenVX 1.3.2 specification](https://www.khronos.org/registry/OpenVX/). It covers the core API, immediate-mode utility functions, graph-based node functions, and optional KHR extensions.

## Prerequisites

- **CMake** 3.10 or later
- **C99-compatible compiler** (GCC, Clang, or MSVC)
- **Git LFS** — required to fetch the binary test images in `test_data/` (see [Cloning](#cloning) below)
- An OpenVX implementation — either a pre-built library or the Khronos sample implementation

## Cloning

Since the `openvx_1.3.2` branch, the binary test images in `test_data/` (e.g. `lena.bmp`) are stored using [Git LFS](https://git-lfs.com/). **Install Git LFS before cloning**, otherwise `test_data/` will contain small text pointer files instead of the actual images.

```bash
# Install Git LFS (once per machine), then enable it for your user
git lfs install

# Clone as usual — LFS objects are fetched automatically
git clone <repository-url>
```

If you already cloned without Git LFS, install it and then pull the real files:

```bash
git lfs install
git lfs pull
```

Without Git LFS the files appear present on disk, but the CTS fails at runtime with errors such as:

```
FAILED at test_engine/test_image.c:387
    Can't open image file: lena.bmp
```

> **Note:** This applies to `openvx_1.3.2` and later. The `openvx_1.3` branch stored test images as regular git blobs and did not require Git LFS.

## Building

The CTS supports two modes: linking against a **pre-built OpenVX library** or building the **Khronos sample implementation** from source alongside the CTS.

### Option A: Pre-built OpenVX Library

If you already have a built OpenVX implementation (e.g., from a vendor SDK), point CMake at the headers and libraries using the `OPENVX_LIBRARIES` and `OPENVX_INCLUDES` variables.

```bash
export OPENVX_DIR=<path to prebuilt OpenVX>
mkdir build && cd build

cmake \
    -DOPENVX_INCLUDES=$OPENVX_DIR/include \
    -DOPENVX_LIBRARIES="$OPENVX_DIR/lib/libopenvx.so;$OPENVX_DIR/lib/libvxu.so;pthread;dl;m;rt" \
    <path-to-cts-source>

cmake --build .
```

### Option B: Build with the Khronos Sample Implementation

When `OPENVX_LIBRARIES` is **not** defined, the CTS build system automatically compiles the Khronos OpenVX sample implementation from source via `cmake/openvx.cmake`. This requires the CTS source tree to be located inside the sample implementation repository as a subdirectory (the default layout when using the `sample-impl` repo, which includes `cts/` as a git submodule).

Clone the sample implementation first:

```bash
git clone --recursive --branch openvx_1.3.2 \
    https://github.com/KhronosGroup/OpenVX-sample-impl.git
```

The expected directory structure is:

```
sample-impl/
├── include/            # OpenVX headers
├── sample/             # Sample implementation source
│   ├── framework/      # Core framework (libopenvx)
│   ├── vxu/            # Utility library (libvxu)
│   └── targets/        # Target backends (c_model)
├── kernels/            # Kernel implementations
├── cts/                # <-- This CTS repository (submodule)
│   ├── CMakeLists.txt
│   └── ...
└── ...
```

To build from the sample implementation tree:

```bash
cd sample-impl/cts
mkdir build && cd build
cmake ..
cmake --build .
```

This compiles both the sample implementation (`libopenvx`, `libvxu`, `libopenvx-c_model`) and the CTS test binary in a single build.

### Option C: Build the Sample Implementation Separately, Then Link

You can also build the sample implementation as a standalone step and then point the CTS at the resulting libraries using Option A.

> **Important:** Pass `-DCMAKE_INSTALL_BINDIR=bin -DCMAKE_INSTALL_LIBDIR=bin` when building the sample implementation. Without these flags, the install step does not set install directories correctly and `libopenvx.so` lands at the install prefix root rather than a `bin/` subdirectory, causing the CTS link step to fail.

```bash
# 1. Clone the sample implementation
git clone --recursive --branch openvx_1.3.2 \
    https://github.com/KhronosGroup/OpenVX-sample-impl.git
export SAMPLE_IMPL_DIR=$(pwd)/OpenVX-sample-impl
export OPENVX_DIR=$SAMPLE_IMPL_DIR/install

# 2. Build and install the sample implementation
mkdir -p "$SAMPLE_IMPL_DIR/build" && cd "$SAMPLE_IMPL_DIR/build"
cmake "$SAMPLE_IMPL_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$OPENVX_DIR" \
    -DCMAKE_INSTALL_BINDIR=bin \
    -DCMAKE_INSTALL_LIBDIR=bin \
    -DBUILD_X64=1
make install -j"$(nproc)"

# 3. Build the CTS against the installed sample implementation
cd <path-to-cts>
mkdir build && cd build
cmake \
    -DOPENVX_INCLUDES="$OPENVX_DIR/include" \
    -DOPENVX_LIBRARIES="$OPENVX_DIR/bin/libopenvx.so;$OPENVX_DIR/bin/libvxu.so;pthread;dl;m;rt" \
    ..
cmake --build .
```

### Platform Notes

#### Linux

Link with system libraries as needed:

```bash
-DOPENVX_LIBRARIES="$OPENVX_DIR/lib/libopenvx.so;$OPENVX_DIR/lib/libvxu.so;pthread;dl;m;rt"
```

#### macOS

Use `.dylib` instead of `.so` and link with `pthread`, `dl`, `m`:

```bash
-DOPENVX_LIBRARIES="$OPENVX_DIR/lib/libopenvx.dylib;$OPENVX_DIR/lib/libvxu.dylib;pthread;dl;m"
```

#### Windows (MSVC)

Use the Visual Studio or Ninja generator. Library paths use `.lib`/`.dll`:

```bash
cmake -G "Visual Studio 17 2022" ^
    -DOPENVX_INCLUDES=%OPENVX_DIR%\include ^
    -DOPENVX_LIBRARIES="%OPENVX_DIR%\lib\openvx.lib;%OPENVX_DIR%\lib\vxu.lib" ^
    ..
cmake --build . --config Release
```

### Build Type

For Makefile/Ninja generators, the default build type is `Release`. Override with:

```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

## CMake Variables

### OpenVX Library Configuration

| Variable | Description |
|---|---|
| `OPENVX_LIBRARIES` | Semicolon-separated list of shared/static libraries to link against. Use absolute paths if libraries are not on the system path. If order-dependent, specify in correct link order. When **not set**, the sample implementation is built from source. |
| `OPENVX_INCLUDES` | Absolute path to the OpenVX headers directory (the parent of the `VX/` subfolder). |
| `OPENVX_DEFINITIONS` | Semicolon-separated list of preprocessor definitions for the target platform. |
| `OPENVX_CFLAGS` | Semicolon-separated list of extra compiler flags for the target platform. |

### Conformance Feature Sets

These options select which conformance profiles to compile and test against.

| Variable | Default | Description |
|---|---|---|
| `OPENVX_CONFORMANCE_VISION` | `ON` | Vision conformance feature set. |
| `OPENVX_CONFORMANCE_NEURAL_NETWORKS` | `ON` | Neural Networks conformance feature set. |
| `OPENVX_CONFORMANCE_NNEF_IMPORT` | `ON` | NNEF Import conformance feature set. See note below about extra build requirements. |
| `OPENVX_USE_ENHANCED_VISION` | `ON` | Enhanced Vision feature set. |

### KHR Extension Toggles

These options enable or disable test cases for specific Khronos extensions.

| Variable | Default | Description |
|---|---|---|
| `OPENVX_USE_IX` | `ON` | Import/Export extension (`vx_khr_ix`). |
| `OPENVX_USE_NN` | `ON` | Neural Network extension (`vx_khr_nn`). |
| `OPENVX_USE_NN_16` | `ON` | Neural Network 16-bit extension. |
| `OPENVX_USE_U1` | `ON` | Binary image (1-bit / U1) feature set. |
| `OPENVX_USE_PIPELINING` | `OFF` | Pipelining extension. |
| `OPENVX_USE_STREAMING` | `OFF` | Streaming extension. |
| `OPENVX_USE_USER_DATA_OBJECT` | `OFF` | User Data Object extension. |

### Other Build Options

| Variable | Default | Description |
|---|---|---|
| `BUILD_TEST_DATA_GENERATORS` | `OFF` | Build the test data generator utilities (in `test_data_generator/`). |
| `CT_DISABLE_TIME_SUPPORT` | not set | When defined, disables test duration timing support. |

### NNEF Import: Extra Build Requirements

Enabling `OPENVX_CONFORMANCE_NNEF_IMPORT=ON` requires two extra steps beyond the standard CTS build:

1. **Additional include path** — the NNEF parser headers from the sample implementation's submodule must be on the include path:
   ```bash
   -DCMAKE_C_FLAGS="-I<sample-impl>/kernels/NNEF-Tools/parser/cpp/include"
   ```

2. **Additional link libraries** — the NNEF static library and `stdc++` must be added to `OPENVX_LIBRARIES`:
   ```bash
   -DOPENVX_LIBRARIES="...;$OPENVX_DIR/bin/libnnef-lib.a;stdc++"
   ```

   > **Note:** With `-DCMAKE_INSTALL_LIBDIR=bin` (as recommended in Option C), `libnnef-lib.a` installs to `$OPENVX_DIR/bin/`, not `lib/`.

### Example: Baseline-Only Build

To compile with no extensions enabled:

```bash
cmake \
    -DOPENVX_INCLUDES=$OPENVX_DIR/include \
    -DOPENVX_LIBRARIES="$OPENVX_DIR/lib/libopenvx.so;$OPENVX_DIR/lib/libvxu.so;pthread;dl;m;rt" \
    -DOPENVX_USE_IX=OFF \
    -DOPENVX_USE_NN=OFF \
    -DOPENVX_USE_NN_16=OFF \
    -DOPENVX_USE_U1=OFF \
    -DOPENVX_CONFORMANCE_NEURAL_NETWORKS=OFF \
    -DOPENVX_CONFORMANCE_NNEF_IMPORT=OFF \
    ..
```

### Example: Full Extensions Build

```bash
cmake \
    -DOPENVX_INCLUDES=$OPENVX_DIR/include \
    -DOPENVX_LIBRARIES="$OPENVX_DIR/lib/libopenvx.so;$OPENVX_DIR/lib/libvxu.so;pthread;dl;m;rt" \
    -DOPENVX_USE_IX=ON \
    -DOPENVX_USE_NN=ON \
    -DOPENVX_USE_NN_16=ON \
    -DOPENVX_USE_U1=ON \
    -DOPENVX_USE_PIPELINING=ON \
    -DOPENVX_USE_STREAMING=ON \
    -DOPENVX_USE_USER_DATA_OBJECT=ON \
    ..
```

## Running the Tests

### Environment Setup

1. Set the test data path:

```bash
export VX_TEST_DATA_PATH=<path-to-cts>/test_data/
```

2. Set the library path so the runtime can find the OpenVX shared libraries:

```bash
# Linux
export LD_LIBRARY_PATH=<path-to-openvx-libs>:$LD_LIBRARY_PATH

# macOS
export DYLD_LIBRARY_PATH=<path-to-openvx-libs>:$DYLD_LIBRARY_PATH
```

### Running

```bash
./build/bin/vx_test_conformance [options]
```

### Command-Line Options

| Option | Description |
|---|---|
| `--filter=<filter>` | Run only tests matching `<filter>`. Uses Google Test filter syntax: a colon-separated list of wildcard patterns, optionally followed by `-` and negative patterns. Example: `--filter=SmokeTest*:Array*:-*Disabled*` |
| `--run_disabled` | Include tests that are disabled by default (not part of the conformance suite). |
| `--global_context=0\|1` | `0` (default): create a new `vx_context` for every test. `1`: run all tests within a single `vx_context`. |
| `--check_any_size=0\|1` | `0` (default): use a restricted set of image sizes (typically VGA) for conformance. `1`: include additional image sizes. Conformance only requires the default restricted set. |
| `--show_test_duration=0\|1` | `0` (default): no timing info. `1`: print test execution time in the log. |
| `--list_tests` | List all test names without running them. |
| `--testid=<id>` | Attach a custom identifier to the conformance report. |
| `--verbose` | Enable extra diagnostic output. |
| `--quiet` | Minimize header and status output. |

### Conformance Run

To produce a valid conformance result, run with **all default options** (no flags):

```bash
./build/bin/vx_test_conformance
```

The other options are provided for debugging and development only.

### Filter Examples

Run only smoke tests:

```bash
./build/bin/vx_test_conformance '--filter=SmokeTest*'
```

Run all tests except Neural Network tensor tests:

```bash
./build/bin/vx_test_conformance '--filter=*:-TensorNN*:TensorNetworks*:TensorOp*'
```

Run a specific test by full name:

```bash
./build/bin/vx_test_conformance '--filter=SmokeTest.vxHint'
```

> **Note:** Quote the `--filter` argument to prevent shell glob expansion (especially in zsh).

### Batch Test Runner (`run_tests.py`)

`run_tests.py` is an optional helper (requires **Python 3**) that wraps the
`vx_test_conformance` binary and runs **each test in its own process**. This
isolates crashes to a single test and, unlike a plain run of the binary,
enforces a **per-test timeout** so a hanging test cannot stall the whole suite.
It then prints an aggregate `#REPORT:` summary line with the total, disabled,
started, completed, passed, and failed counts.

```bash
# Usage: run_tests.py <vx_test_conformance executable> [extra options]
python3 run_tests.py ./build/bin/vx_test_conformance

# Restrict to a subset of tests
python3 run_tests.py ./build/bin/vx_test_conformance --filter='*Canny*'
```

Configuration is via environment variables:

| Variable | Description |
|---|---|
| `VX_TEST_DATA_PATH` | Path to the `test_data/` directory (consumed by `vx_test_conformance`). |
| `VX_TEST_TIMEOUT` | Per-test timeout in seconds (default `65`). A test exceeding this is terminated and counted as failed. |

The script exits `0` only if every test started and none failed; otherwise it
exits `1`. It is a convenience/debugging aid and is **not** required to produce
a conformance result — an official run is a plain invocation of
`vx_test_conformance` with no flags (see [Conformance Run](#conformance-run)).

## Test Data

The `test_data/` directory contains:

- Natural images (VGA resolution) used as inputs for tested algorithms
- Reference output data generated by OpenCV 2.4.7

All required test data is included in the CTS package. Regeneration is not necessary in most cases. See `test_data_generator/README` for details on the generator utilities.

## Continuous Integration

The CI runs on every pull request and push to `openvx_1.3.2` via `.github/workflows/conformance.yml`. It builds the [Khronos sample implementation](https://github.com/KhronosGroup/OpenVX-sample-impl) and runs the following conformance modes:

| Mode | Feature flags | Test filter |
|---|---|---|
| 1 · Vision | `OPENVX_CONFORMANCE_VISION` | full suite |
| 2 · Vision + Enhanced Vision | `OPENVX_CONFORMANCE_VISION`, `OPENVX_USE_ENHANCED_VISION` | full suite |
| 3 · Neural Networks | `OPENVX_CONFORMANCE_NEURAL_NETWORKS` | full suite |
| 4 · NNEF Import | `OPENVX_CONFORMANCE_NNEF_IMPORT` | full suite |
| 5 · Combined | `OPENVX_CONFORMANCE_VISION`, `OPENVX_USE_ENHANCED_VISION`, `OPENVX_CONFORMANCE_NEURAL_NETWORKS`, `OPENVX_USE_NN`, `OPENVX_USE_IX`, `OPENVX_USE_U1` | full suite |
| 6 · User Data Object | `OPENVX_USE_USER_DATA_OBJECT` | full suite |
| 7 · Pipelining | `OPENVX_CONFORMANCE_VISION`, `OPENVX_USE_ENHANCED_VISION`, `OPENVX_USE_PIPELINING` | `GraphPipeline.*` |
| 8 · Streaming | `OPENVX_CONFORMANCE_VISION`, `OPENVX_USE_ENHANCED_VISION`, `OPENVX_USE_PIPELINING`, `OPENVX_USE_STREAMING` | `GraphStreaming.*` |
| 9 · Import/Export | `OPENVX_USE_IX` | `ExtensionObject.*` |
| 10 · U1 | `OPENVX_CONFORMANCE_VISION`, `OPENVX_USE_U1` | `vxBinOp1u.*:vxuBinOp1u.*` |

## Directory Structure

```
cts/
├── CMakeLists.txt              # Top-level build script
├── cmake/
│   ├── openvx.cmake            # Sample implementation build integration
│   └── vcs_version.cmake       # VCS version stamp generator
├── test_engine/                # Test framework (assertions, parameterized tests, runner)
├── test_conformance/           # All conformance test source files
│   ├── Networks/               # Neural network graph tests (AlexNet, etc.)
│   ├── test_smoke.c            # Core API smoke tests
│   ├── test_graph.c            # Graph lifecycle and attribute tests
│   ├── test_array.c            # Array object tests
│   ├── test_vxtensor.c         # Tensor object tests
│   ├── test_threshold.c        # Threshold object tests
│   ├── test_canny.c            # Canny edge detector tests
│   ├── test_binop8u.c          # Binary operations (U8)
│   ├── test_binop16s.c         # Binary operations (S16)
│   ├── test_binop1u.c          # Binary operations (U1)
│   └── ...                     # Additional test files
├── test_data/                  # Input images and reference data
├── test_data_generator/        # Utilities to regenerate test data
└── run_tests.py                # Optional batch runner (per-test isolation + timeout)
```

## License

Copyright (c) 2012-2026 The Khronos Group Inc.

Licensed under the Apache License, Version 2.0. See the [LICENSE](http://www.apache.org/licenses/LICENSE-2.0) for details.
