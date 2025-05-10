const std = @import("std");

const rllvm = @import("rllvm");
const cuda = rllvm.cuda;
const target = rllvm.llvm.target;
const target_machine_mod = rllvm.llvm.target_machine;
const types = rllvm.llvm.types;
const core = rllvm.llvm.core;
const execution = rllvm.llvm.engine;

const nodes = @import("nodes.zig");
const Builder = @import("builder.zig").Builder;

var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);

pub fn execute(program: nodes.RHLOProgram, inputs: []*void, outputs: []*void) !void {
    // init code
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

    const four: rllvm.types.IntegerRef = .{ .ref = core.LLVMConstInt(core.LLVMInt64Type(), 4, 0) };

    var h_inputs = std.ArrayList(rllvm.types.OpaqueRef).init(arena.allocator());
    var d_inputs = std.ArrayList(rllvm.types.CudaValueRef).init(arena.allocator());
    for (inputs) |input| {
        const h_input = rllvm.types.OpaqueRef{ .ref = core.LLVMConstIntToPtr(core.LLVMConstInt(core.LLVMInt64Type(), @intFromPtr(input), 0), core.LLVMPointerType(core.LLVMInt8Type(), 0)) };
        try h_inputs.append(h_input);

        const d_input = rllvm.types.CudaValueRef.create(builder);
        try d_inputs.append(d_input);
        try cuda.memAlloc(module, builder, d_input, four);
        try cuda.copyHToD(module, builder, d_input, h_input, four);
    }

    var h_outputs = std.ArrayList(rllvm.types.OpaqueRef).init(arena.allocator());
    var d_outputs = std.ArrayList(rllvm.types.CudaValueRef).init(arena.allocator());
    for (outputs) |output| {
        const h_output = rllvm.types.OpaqueRef{ .ref = core.LLVMConstIntToPtr(core.LLVMConstInt(core.LLVMInt64Type(), @intFromPtr(output), 0), core.LLVMPointerType(core.LLVMInt8Type(), 0)) };
        try h_outputs.append(h_output);

        const d_output = rllvm.types.CudaValueRef.create(builder);
        try d_outputs.append(d_output);
        try cuda.memAlloc(module, builder, d_output, four);
    }

    // gen
    const ptx = try @import("backends/cuda/cuda.zig").generatePTX(module, program);

    const cuda_module = try cuda.moduleLoadData(module, builder, .{ .ref = ptx });
    const cuda_function = try cuda.moduleGetFunction(module, builder, cuda_module);

    const int_type = core.LLVMInt32Type();
    const grid_dim_x: rllvm.types.IntegerRef = .{ .ref = core.LLVMConstInt(int_type, 1, 0) };
    const grid_dim_y: rllvm.types.IntegerRef = .{ .ref = core.LLVMConstInt(int_type, 1, 0) };
    const grid_dim_z: rllvm.types.IntegerRef = .{ .ref = core.LLVMConstInt(int_type, 1, 0) };
    const block_dim_x: rllvm.types.IntegerRef = .{ .ref = core.LLVMConstInt(int_type, 1, 0) };
    const block_dim_y: rllvm.types.IntegerRef = .{ .ref = core.LLVMConstInt(int_type, 1, 0) };
    const block_dim_z: rllvm.types.IntegerRef = .{ .ref = core.LLVMConstInt(int_type, 1, 0) };
    const shared_mem_bytes: rllvm.types.IntegerRef = .{ .ref = core.LLVMConstInt(int_type, 0, 0) };
    const kernel_params = try std.mem.concat(arena.allocator(), rllvm.types.CudaValueRef, &[_][]rllvm.types.CudaValueRef{ d_inputs.items, d_outputs.items });
    try cuda.launchKernel(module, builder, cuda_function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y, block_dim_z, shared_mem_bytes, kernel_params);

    for (d_outputs.items, 0..) |d_output, idx| {
        try cuda.copyDToH(module, builder, d_output, h_outputs.items[idx], four);
    }

    const zero = core.LLVMConstInt(core.LLVMInt64Type(), 0, 0);

    _ = core.LLVMBuildRet(builder, zero);

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
    const MainFn = fn () callconv(.C) f32;
    const main_fn: *const MainFn = @ptrFromInt(main_addr);

    _ = main_fn();
}

test "matmul" {
    var builder = Builder.init(arena.allocator());
    const dtype = nodes.DataType.F32;
    const shape: nodes.Shape = &[_]usize{ 2, 2 };
    const param0 = try builder.createParameter(dtype, shape);
    const param1 = try builder.createParameter(dtype, shape);

    const res1 = try builder.opAdd(param0, param1);
    const res2 = try builder.opAdd(param0, res1);
    try builder.createParameterFromRef(res2);

    var input1 = try arena.allocator().alloc(f32, 4);
    input1[0] = 3.0;
    input1[1] = 2.0;
    input1[2] = 3.0;
    input1[3] = 4.0;
    var input2 = try arena.allocator().alloc(f32, 4);
    input2[0] = 9.5;
    input2[1] = 2.0;
    input2[2] = 3.0;
    input2[3] = 4.0;
    const output = try arena.allocator().alloc(f32, 4);

    var inputs = [_]*void{ @ptrCast(input1.ptr), @ptrCast(input2.ptr) };
    var outputs = [_]*void{@ptrCast(output.ptr)};

    try execute(
        builder.program,
        &inputs,
        &outputs,
    );

    std.debug.print("first of output: {d}\n", .{output[0]});
}
