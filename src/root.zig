const std = @import("std");

const rllvm = @import("rllvm");
const cuda = rllvm.cuda;
const target = rllvm.llvm.target;
const target_machine_mod = rllvm.llvm.target_machine;
const types = rllvm.llvm.types;
const core = rllvm.llvm.core;
const execution = rllvm.llvm.engine;

const pretty_printer = @import("pretty-printer.zig");
const nodes = @import("nodes.zig");
const Builder = @import("builder.zig").Builder;

var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);

// TODO: merge inputs and outputs into one params array, and memcpy based on program.params data
// currently all inputs and outputs must be of same shape
pub fn execute(program: nodes.RHLOProgram, params: []*void) !void {
    // quick checks
    for (program.tensor_store.items) |tensor| {
        if (tensor.dimensions[0] != tensor.dimensions[1]) @panic("All tensors must be of symmetric shape");
        if (tensor.dimensions[0] > 25) @panic("Tensors only of any dim smaller than 5 supported");
    }

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

    // allocate d and h memory
    var h_inputs = std.ArrayList(rllvm.types.OpaqueRef).init(arena.allocator());
    var d_inputs = std.ArrayList(rllvm.types.CudaValueRef).init(arena.allocator());
    var h_outputs = std.ArrayList(rllvm.types.OpaqueRef).init(arena.allocator());
    var d_outputs = std.ArrayList(rllvm.types.CudaValueRef).init(arena.allocator());
    var output_sizes = std.ArrayList(rllvm.types.IntegerRef).init(arena.allocator());

    for (program.params.items, 0..) |param, idx| {
        var size_elements: usize = 1;
        for (program.tensor_store.items[param.id].dimensions) |dim| {
            size_elements *= dim;
        }
        const size_bytes = rllvm.types.IntegerRef{ .ref = core.LLVMConstInt(core.LLVMInt64Type(), 4 * size_elements, 0) }; // TODO: only works with 32 bit vars, make it work with any

        if (param.input == true) {
            const input = params[idx];
            const h_input = rllvm.types.OpaqueRef{ .ref = core.LLVMConstIntToPtr(core.LLVMConstInt(core.LLVMInt64Type(), @intFromPtr(input), 0), core.LLVMPointerType(core.LLVMInt8Type(), 0)) };
            try h_inputs.append(h_input);

            const d_input = rllvm.types.CudaValueRef.create(builder);
            try d_inputs.append(d_input);

            try cuda.memAlloc(module, builder, d_input, size_bytes);
            try cuda.copyHToD(module, builder, d_input, h_input, size_bytes);
        }

        if (param.output == true) {
            const output = params[idx];
            const h_output = rllvm.types.OpaqueRef{ .ref = core.LLVMConstIntToPtr(core.LLVMConstInt(core.LLVMInt64Type(), @intFromPtr(output), 0), core.LLVMPointerType(core.LLVMInt8Type(), 0)) };
            try h_outputs.append(h_output);

            const d_output = rllvm.types.CudaValueRef.create(builder);
            try d_outputs.append(d_output);

            try output_sizes.append(size_bytes);

            try cuda.memAlloc(module, builder, d_output, size_bytes);
        }
    }

    var block_dims: usize = 0;
    for (program.tensor_store.items) |tensor| {
        if (tensor.dimensions[0] > block_dims) block_dims = tensor.dimensions[0];
    }

    // gen
    const ptx = try @import("backends/cuda/cuda.zig").generatePTX(module, program);

    const cuda_module = try cuda.moduleLoadData(module, builder, .{ .ref = ptx });
    const cuda_function = try cuda.moduleGetFunction(module, builder, cuda_module);

    const int_type = core.LLVMInt32Type();
    const grid_dim_x: rllvm.types.IntegerRef = .{ .ref = core.LLVMConstInt(int_type, 1, 0) };
    const grid_dim_y: rllvm.types.IntegerRef = .{ .ref = core.LLVMConstInt(int_type, 1, 0) };
    const grid_dim_z: rllvm.types.IntegerRef = .{ .ref = core.LLVMConstInt(int_type, 1, 0) };
    const block_dim_x: rllvm.types.IntegerRef = .{ .ref = core.LLVMConstInt(int_type, block_dims, 0) };
    const block_dim_y: rllvm.types.IntegerRef = .{ .ref = core.LLVMConstInt(int_type, block_dims, 0) };
    const block_dim_z: rllvm.types.IntegerRef = .{ .ref = core.LLVMConstInt(int_type, 1, 0) };
    const shared_mem_bytes: rllvm.types.IntegerRef = .{ .ref = core.LLVMConstInt(int_type, 0, 0) };
    const kernel_params = try std.mem.concat(arena.allocator(), rllvm.types.CudaValueRef, &[_][]rllvm.types.CudaValueRef{ d_inputs.items, d_outputs.items });
    try cuda.launchKernel(module, builder, cuda_function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y, block_dim_z, shared_mem_bytes, kernel_params);

    // copy results to h
    for (d_outputs.items, 0..) |d_output, idx| {
        try cuda.copyDToH(module, builder, d_output, h_outputs.items[idx], output_sizes.items[idx]);
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

    // _ = core.LLVMDumpModule(module);

    const main_addr = execution.LLVMGetFunctionAddress(engine, "main");
    const MainFn = fn () callconv(.C) f32;
    const main_fn: *const MainFn = @ptrFromInt(main_addr);

    _ = main_fn();
}

test "add" {
    var builder = Builder.init(arena.allocator());
    const dtype = nodes.DataType.F32;
    const shape: nodes.Shape = &[_]usize{ 2, 2 };
    const param0 = try builder.createParameter(dtype, shape);
    const param1 = try builder.createParameter(dtype, shape);

    const result = try builder.opAdd(param0, param1);
    try builder.createParameterFromRef(result);

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

    var params = [_]*void{ @ptrCast(input1.ptr), @ptrCast(input2.ptr), @ptrCast(output.ptr) };

    try execute(
        builder.program,
        &params,
    );

    try std.testing.expect(output[0] == 12.5);
    try std.testing.expect(output[1] == 4);
    try std.testing.expect(output[2] == 6);
    try std.testing.expect(output[3] == 8);
}

test "multi step add" {
    var builder = Builder.init(arena.allocator());
    const dtype = nodes.DataType.F32;
    const shape: nodes.Shape = &[_]usize{ 2, 2 };
    const param0 = try builder.createParameter(dtype, shape);
    const param1 = try builder.createParameter(dtype, shape);

    const intermediate_result = try builder.opAdd(param0, param1);
    const result = try builder.opAdd(intermediate_result, param1);
    try builder.createParameterFromRef(result);

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

    var params = [_]*void{ @ptrCast(input1.ptr), @ptrCast(input2.ptr), @ptrCast(output.ptr) };

    try execute(
        builder.program,
        &params,
    );

    try std.testing.expect(output[0] == 22);
    try std.testing.expect(output[1] == 6);
    try std.testing.expect(output[2] == 9);
    try std.testing.expect(output[3] == 12);
}

// TODO: fix numerical inaccuracy in these tests
test "chained 2x2 gemm" {
    var builder = Builder.init(arena.allocator());
    const dtype = nodes.DataType.F32;
    const shape: nodes.Shape = &[_]usize{ 2, 2 };
    const param0 = try builder.createParameter(dtype, shape);
    const param1 = try builder.createParameter(dtype, shape);

    const intermediate_result = try builder.opMatmul(param0, param1);
    const result = try builder.opMatmul(intermediate_result, param1);
    try builder.createParameterFromRef(result);

    var input1 = try arena.allocator().alloc(f32, 4);
    input1[0] = 0.5;
    input1[1] = 2.0;
    input1[2] = 3.0;
    input1[3] = 2.0;
    var input2 = try arena.allocator().alloc(f32, 4);
    input2[0] = 1.5;
    input2[1] = 2.0;
    input2[2] = 0.3;
    input2[3] = 0.2;
    const output = try arena.allocator().alloc(f32, 4);

    var params = [_]*void{ @ptrCast(input1.ptr), @ptrCast(input2.ptr), @ptrCast(output.ptr) };

    try execute(
        builder.program,
        &params,
    );

    const rounded_output = [_]f32{
        roundTo3DecimalPlaces(output[0]),
        roundTo3DecimalPlaces(output[1]),
        roundTo3DecimalPlaces(output[2]),
        roundTo3DecimalPlaces(output[3]),
    };

    try std.testing.expect(rounded_output[0] == 2.445);
    try std.testing.expect(rounded_output[1] == 2.98);
    try std.testing.expect(rounded_output[2] == 9.57);
    try std.testing.expect(rounded_output[3] == 11.48);
}

test "chained 2x2 gemm 2" {
    var builder = Builder.init(arena.allocator());
    const dtype = nodes.DataType.F32;
    const shape: nodes.Shape = &[_]usize{ 2, 2 };
    const param0 = try builder.createParameter(dtype, shape);
    const param1 = try builder.createParameter(dtype, shape);

    const intermediate_result = try builder.opMatmul(param0, param1);
    const result = try builder.opMatmul(param1, intermediate_result);
    try builder.createParameterFromRef(result);

    var input1 = try arena.allocator().alloc(f32, 4);
    input1[0] = 0.5;
    input1[1] = 2.0;
    input1[2] = 3.0;
    input1[3] = 2.0;
    var input2 = try arena.allocator().alloc(f32, 4);
    input2[0] = 1.5;
    input2[1] = 2.0;
    input2[2] = 0.3;
    input2[3] = 0.2;
    const output = try arena.allocator().alloc(f32, 4);

    var params = [_]*void{ @ptrCast(input1.ptr), @ptrCast(input2.ptr), @ptrCast(output.ptr) };

    try execute(
        builder.program,
        &params,
    );

    // try pretty_printer.prettyPrint(builder.program, "hi.rhlo");

    const rounded_output = [_]f32{
        roundTo3DecimalPlaces(output[0]),
        roundTo3DecimalPlaces(output[1]),
        roundTo3DecimalPlaces(output[2]),
        roundTo3DecimalPlaces(output[3]),
    };

    // 1.35
    // 1.4
    // 5.1
    // 6.4

    try std.testing.expect(rounded_output[0] == 12.225);
    try std.testing.expect(rounded_output[1] == 14.9);
    try std.testing.expect(rounded_output[2] == 1.425);
    try std.testing.expect(rounded_output[3] == 1.7);
}

fn roundTo3DecimalPlaces(value: f32) f32 {
    return @floatCast(@round(@as(f32, @floatCast(value)) * 1000.0) / 1000.0);
}
