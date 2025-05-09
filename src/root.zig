const std = @import("std");

const rllvm = @import("rllvm");
const cuda = rllvm.cuda;
const target = rllvm.llvm.target;
const target_machine_mod = rllvm.llvm.target_machine;
const types = rllvm.llvm.types;
const core = rllvm.llvm.core;
const execution = rllvm.llvm.engine;

const nodes = @import("nodes.zig");
pub const buffers = @import("buffers.zig");
const Builder = @import("builder.zig").Builder;

var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);

pub fn execute(program: nodes.RHLOProgram, input_buffer: buffers.RHLOBuffer, output_buffer: buffers.RHLOBuffer) void {
    _ = program;

    _ = target.LLVMInitializeNativeTarget();
    _ = target.LLVMInitializeNativeAsmPrinter();
    _ = target.LLVMInitializeNativeAsmParser();

    _ = rllvm.llvm.support.LLVMLoadLibraryPermanently("/run/opengl-driver/lib/libcuda.so");

    const module = core.LLVMModuleCreateWithName("main");

    var param_types: [2]types.LLVMTypeRef = .{
        core.LLVMPointerType(core.LLVMVoidType(), 0),
        core.LLVMPointerType(core.LLVMVoidType(), 0),
    };
    const fn_type = core.LLVMFunctionType(core.LLVMInt32Type(), &param_types, 2, 0);
    const function = core.LLVMAddFunction(module, "main", fn_type);

    const entry = core.LLVMAppendBasicBlock(function, "entry");

    const builder = core.LLVMCreateBuilder();
    defer core.LLVMDisposeBuilder(builder);
    core.LLVMPositionBuilderAtEnd(builder, entry);

    try cuda.init(module, builder);
    const cuda_device = try cuda.deviceGet(module, builder);
    const cuda_context = try cuda.contextCreate(module, builder, cuda_device);
    _ = cuda_context;

    const h_input = rllvm.types.OpaqueRef{ .ref = core.LLVMGetParam(function, 0) };
    const d_input = rllvm.types.CudaValueRef.create(builder);
    const d_output = rllvm.types.CudaValueRef.create(builder);

    const four: rllvm.types.IntegerRef = .{ .ref = core.LLVMConstInt(core.LLVMInt64Type(), 4, 0) };
    // const size_bytes = rllvm.types.IntegerRef{ .ref = core.LLVMConstInt(core.LLVMInt32Type(), 4, 0) };
    if (input_buffer.device == false) {
        try cuda.memAlloc(module, builder, d_input, four);
        try cuda.copyHToD(module, builder, d_input, h_input, four);
    }
    if (output_buffer.device == false) {
        try cuda.memAlloc(module, builder, d_output, four);
    }

    const ptx =
        \\//
        \\.version 8.4
        \\.target sm_52
        \\.address_size 64
        \\
        \\.visible .entry main(
        \\  .param .u64 input_ptr,
        \\  .param .u64 output_ptr
        \\)
        \\{
        \\  .reg .b32 %r<2>;
        \\  .reg .b64 %rd<3>;
        \\
        \\  ld.param.u64 %rd1, [input_ptr];
        \\  ld.param.u64 %rd2, [output_ptr];
        \\
        \\  cvta.to.global.u64 %rd1, %rd1;
        \\  cvta.to.global.u64 %rd2, %rd2;
        \\
        \\  ld.global.u32 %r1, [%rd1];
        \\
        \\  st.global.u32 [%rd2], %r1;
        \\
        \\  ret;
        \\}
    ;

    const kernel_len = ptx.len;
    const global_ptx_str = core.LLVMAddGlobal(module, core.LLVMPointerType(core.LLVMInt8Type(), 0), "ptx_str");
    const kernel_constant = core.LLVMConstString(@ptrCast(ptx), @intCast(kernel_len), 0);
    core.LLVMSetInitializer(global_ptx_str, kernel_constant);

    const cuda_module = try cuda.moduleLoadData(module, builder, .{ .ref = global_ptx_str });

    const cuda_function = try cuda.moduleGetFunction(module, builder, cuda_module);

    const int_type = core.LLVMInt32Type();
    const grid_dim_x: rllvm.types.IntegerRef = .{ .ref = core.LLVMConstInt(int_type, 1, 0) };
    const grid_dim_y: rllvm.types.IntegerRef = .{ .ref = core.LLVMConstInt(int_type, 1, 0) };
    const grid_dim_z: rllvm.types.IntegerRef = .{ .ref = core.LLVMConstInt(int_type, 1, 0) };
    const block_dim_x: rllvm.types.IntegerRef = .{ .ref = core.LLVMConstInt(int_type, 1, 0) };
    const block_dim_y: rllvm.types.IntegerRef = .{ .ref = core.LLVMConstInt(int_type, 1, 0) };
    const block_dim_z: rllvm.types.IntegerRef = .{ .ref = core.LLVMConstInt(int_type, 1, 0) };
    const shared_mem_bytes: rllvm.types.IntegerRef = .{ .ref = core.LLVMConstInt(int_type, 0, 0) };
    var kernel_params = [_]rllvm.types.CudaValueRef{ d_input, d_output };
    try cuda.launchKernel(module, builder, cuda_function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y, block_dim_z, shared_mem_bytes, &kernel_params);

    const result_ptr = rllvm.types.OpaqueRef{
        .ref = core.LLVMGetParam(function, 1),
    };
    try cuda.copyDToH(module, builder, d_output, result_ptr, four);

    // const float_type = core.LLVMFloatType();
    const zero_idx = core.LLVMConstInt(core.LLVMInt64Type(), 0, 0);
    // var indices = [_]types.LLVMValueRef{zero_idx};
    // const first_element_ptr = core.LLVMBuildGEP2(builder, float_type, result_ptr.ref, &indices[0], indices.len, "first_element_ptr");
    // const first_element = core.LLVMBuildLoad2(builder, float_type, first_element_ptr, "first_element");

    // _ = core.LLVMBuildStore(builder, zero_idx, result_ptr.ref);

    _ = core.LLVMBuildRet(builder, zero_idx);

    // execution
    var error_msg: [*c]u8 = null;
    var engine: types.LLVMExecutionEngineRef = undefined;
    if (execution.LLVMCreateExecutionEngineForModule(&engine, module, &error_msg) != 0) {
        std.debug.print("Execution engine creation failed: {s}\n", .{error_msg});
        core.LLVMDisposeMessage(error_msg);
        @panic("failed to create exec engine");
    }
    defer execution.LLVMDisposeExecutionEngine(engine);

    _ = core.LLVMDumpModule(module);

    const main_addr = execution.LLVMGetFunctionAddress(engine, "main");
    const MainFn = fn (*void, *void) callconv(.C) f32;
    const main_fn: *const MainFn = @ptrFromInt(main_addr);

    _ = main_fn(input_buffer.data_ptr, output_buffer.data_ptr);
}

pub fn createBuilder() Builder {
    return Builder.init(arena.allocator());
}

test "matmul" {
    var builder = createBuilder();
    const dtype = nodes.DataType.F32;
    const shape: nodes.Shape = &[_]usize{ 2, 2 };
    const param0 = try builder.paramemeter(dtype, shape);
    const param1 = try builder.paramemeter(dtype, shape);

    _ = try builder.opAdd(param0, param1);

    const input_buffer = try arena.allocator().alloc(f32, 4);
    input_buffer[0] = 1.0;
    input_buffer[1] = 2.0;
    input_buffer[2] = 3.0;
    input_buffer[3] = 4.0;
    const output_buffer = try arena.allocator().alloc(f32, 4);
    output_buffer[0] = 9;

    std.debug.print("in matmul: {}\n", .{output_buffer[0]});
    std.debug.print("d_output: {any}\n", .{output_buffer.ptr});
    execute(
        builder.program,
        .{
            .data_ptr = @ptrCast(input_buffer.ptr),
            .device = false,
            .size = 4,
        },
        .{
            .data_ptr = @ptrCast(output_buffer.ptr),
            .device = false,
            .size = 4,
        },
    );

    std.debug.print("in matmul: {}\n", .{output_buffer[0]});
}
