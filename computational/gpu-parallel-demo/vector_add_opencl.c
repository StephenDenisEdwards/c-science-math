/*
 * GPU Parallel Vector Addition -- OpenCL
 *
 * Same demo as the CUDA version but using OpenCL, which works with
 * NVIDIA, AMD, and Intel GPUs. Compiles with a standard C compiler
 * (gcc, cl) -- no special compiler required.
 *
 * The GPU kernel is written as a string and compiled at runtime by
 * the OpenCL driver, which is the key difference from CUDA's
 * ahead-of-time compilation with nvcc.
 *
 * Compile:
 *   gcc -O2 -o vector_add_opencl vector_add_opencl.c -lOpenCL
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define DEFAULT_N (1 << 24)  /* 16 million elements */

/* ---------------------------------------------------------------------------
 * OpenCL kernel source -- compiled at runtime by the GPU driver
 * ------------------------------------------------------------------------ */
static const char *kernel_source =
    "__kernel void vector_add(__global const float *a,\n"
    "                         __global const float *b,\n"
    "                         __global float *c,\n"
    "                         const int n)\n"
    "{\n"
    "    int i = get_global_id(0);\n"
    "    if (i < n)\n"
    "        c[i] = a[i] + b[i];\n"
    "}\n";

/* ---------------------------------------------------------------------------
 * CPU baseline -- plain serial loop
 * ------------------------------------------------------------------------ */
static void vector_add_cpu(const float *a, const float *b, float *c, int n)
{
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

/* ---------------------------------------------------------------------------
 * Helpers
 * ------------------------------------------------------------------------ */
static void fill_random(float *arr, int n)
{
    for (int i = 0; i < n; i++)
        arr[i] = (float)rand() / RAND_MAX;
}

static int verify(const float *cpu, const float *gpu, int n)
{
    for (int i = 0; i < n; i++) {
        float diff = cpu[i] - gpu[i];
        if (diff < 0) diff = -diff;
        if (diff > 1e-5f) {
            printf("  MISMATCH at index %d: cpu=%.7f gpu=%.7f\n",
                   i, cpu[i], gpu[i]);
            return 0;
        }
    }
    return 1;
}

#define CL_CHECK(call)                                                        \
    do {                                                                      \
        cl_int _err = (call);                                                 \
        if (_err != CL_SUCCESS) {                                             \
            fprintf(stderr, "OpenCL error %d at %s:%d\n",                    \
                    _err, __FILE__, __LINE__);                                \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#ifdef _WIN32
#include <windows.h>
static double get_time_ms(void)
{
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / freq.QuadPart * 1000.0;
}
#else
static double get_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#endif

/* ---------------------------------------------------------------------------
 * Main
 * ------------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    int n = DEFAULT_N;
    if (argc > 1)
        n = atoi(argv[1]);

    size_t bytes = (size_t)n * sizeof(float);
    cl_int err;

    printf("Vector Addition -- CPU vs GPU (OpenCL)\n");
    printf("Elements:  %d (%.1f MB per array)\n\n", n, bytes / (1024.0 * 1024.0));

    /* ----- Find a GPU device ----- */
    cl_platform_id platform;
    cl_uint num_platforms;
    CL_CHECK(clGetPlatformIDs(1, &platform, &num_platforms));
    if (num_platforms == 0) {
        fprintf(stderr, "No OpenCL platforms found\n");
        return 1;
    }

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf("No GPU device found, falling back to CPU device\n");
        CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL));
    }

    char device_name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);

    cl_uint compute_units;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units),
                    &compute_units, NULL);

    printf("Device:    %s\n", device_name);
    printf("Compute units: %u\n\n", compute_units);

    /* ----- Create context and command queue ----- */
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CL_CHECK(err);

#ifdef CL_VERSION_2_0
    cl_command_queue queue = clCreateCommandQueueWithProperties(
        context, device, NULL, &err);
#else
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
#endif
    CL_CHECK(err);

    /* ----- Build the kernel ----- */
    size_t src_len = strlen(kernel_source);
    cl_program program = clCreateProgramWithSource(
        context, 1, &kernel_source, &src_len, &err);
    CL_CHECK(err);

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(log), log, NULL);
        fprintf(stderr, "Kernel build failed:\n%s\n", log);
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);
    CL_CHECK(err);

    /* ----- Allocate host memory ----- */
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c_cpu = (float *)malloc(bytes);
    float *h_c_gpu = (float *)malloc(bytes);

    srand((unsigned)time(NULL));
    fill_random(h_a, n);
    fill_random(h_b, n);

    /* ----- CPU ----- */
    double cpu_start = get_time_ms();
    vector_add_cpu(h_a, h_b, h_c_cpu, n);
    double cpu_ms = get_time_ms() - cpu_start;

    /* ----- GPU ----- */
    cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
    CL_CHECK(err);
    cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
    CL_CHECK(err);
    cl_mem d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
    CL_CHECK(err);

    /* Round-trip: transfer + compute + read back */
    double gpu_start = get_time_ms();

    CL_CHECK(clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, bytes, h_a,
                                   0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, bytes, h_b,
                                   0, NULL, NULL));

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &n));

    size_t global_size = (size_t)n;
    size_t local_size = 256;
    /* Round up global_size to a multiple of local_size */
    if (global_size % local_size != 0)
        global_size += local_size - (global_size % local_size);

    /* Kernel-only timing */
    cl_event kernel_event;
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size,
                                     &local_size, 0, NULL, &kernel_event));
    CL_CHECK(clWaitForEvents(1, &kernel_event));

    cl_ulong k_start, k_end;
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START,
                            sizeof(k_start), &k_start, NULL);
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END,
                            sizeof(k_end), &k_end, NULL);

    CL_CHECK(clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes, h_c_gpu,
                                  0, NULL, NULL));

    double roundtrip_ms = get_time_ms() - gpu_start;
    double kernel_ms = (double)(k_end - k_start) / 1e6;

    /* ----- Results ----- */
    printf("%-30s %10.3f ms\n", "CPU (serial loop):", cpu_ms);
    printf("%-30s %10.3f ms\n", "GPU (kernel only):", kernel_ms);
    printf("%-30s %10.3f ms\n", "GPU (with memory transfers):", roundtrip_ms);
    printf("\n");
    printf("Kernel speedup:      %.1fx\n", cpu_ms / kernel_ms);
    printf("Round-trip speedup:  %.1fx\n", cpu_ms / roundtrip_ms);
    printf("\n");

    if (verify(h_c_cpu, h_c_gpu, n))
        printf("Verification: PASSED -- all %d elements match\n", n);
    else
        printf("Verification: FAILED\n");

    /* Cleanup */
    clReleaseEvent(kernel_event);
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);

    return 0;
}
