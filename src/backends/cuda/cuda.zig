const std = @import("std");

const rllvm = @import("rllvm");
const cuda = rllvm.cuda;
const target = rllvm.llvm.target;
const target_machine_mod = rllvm.llvm.target_machine;
const types = rllvm.llvm.types;
const core = rllvm.llvm.core;
const execution = rllvm.llvm.engine;

const ptxast = @import("./ast.zig");
const nodes = @import("../../nodes.zig");

pub fn generatePTX(module: types.LLVMModuleRef, program: nodes.RHLOProgram) !types.LLVMValueRef {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var params = std.ArrayList(ptxast.Parameter).init(allocator);
    var body = std.ArrayList(ptxast.Instruction).init(allocator);
    var directives = std.ArrayList(ptxast.Directive).init(allocator);

    const reg_r_decl = ptxast.Directive{ .reg = .{
        .name = "%r",
        .count = @intCast(program.tensor_store.items.len),
        .type = .f32,
    } };
    const reg_rd_decl = ptxast.Directive{ .reg = .{
        .name = "%rd",
        .count = @intCast(program.params.items.len),
        .type = .u64,
    } };

    try directives.append(reg_r_decl);
    try directives.append(reg_rd_decl);

    for (program.params.items, 0..) |param, idx| {
        _ = param;
        const name = try std.fmt.allocPrint(allocator, "param_{d}", .{idx});
        try params.append(.{ .name = name, .type = .u64 });
    }

    for (params.items, 0..) |param, idx| {
        const reg_string = try std.fmt.allocPrint(allocator, "%rd{d}", .{idx});
        try body.append(.{ .ld = .{ .space = .param, .type = .u64, .src = .{ .parameter = param.name }, .dest = .{ .register = reg_string } } });
        try body.append(.{ .cvta = .{ .to_generic = true, .space = .global, .type = .u64, .src = .{ .register = reg_string }, .dest = .{ .register = reg_string } } });
    }

    try body.append(.{ .ld = .{ .space = .global, .type = .f32, .dest = .{ .register = "%r0" }, .src = .{ .register = "%rd0" } } });
    try body.append(.{ .ld = .{ .space = .global, .type = .f32, .dest = .{ .register = "%r1" }, .src = .{ .register = "%rd1" } } });
    for (program.ops.items) |op| {
        switch (op.kind) {
            .Add => {
                try body.append(.{
                    .add = .{
                        .type = .f32,
                        .src1 = .{ .register = try std.fmt.allocPrint(allocator, "%r{d}", .{op.input_ids[0]}) },
                        .src2 = .{ .register = try std.fmt.allocPrint(allocator, "%r{d}", .{op.input_ids[1]}) },
                        .dest = .{ .register = try std.fmt.allocPrint(allocator, "%r{d}", .{op.output_ids[0]}) },
                    },
                });
            },
        }
    }

    try body.append(.{ .st = .{ .space = .global, .type = .f32, .dest = .{ .register = "%rd2" }, .src = .{ .register = try std.fmt.allocPrint(allocator, "%r{d}", .{program.tensor_store.items.len - 1}) } } });
    try body.append(.ret);

    const kernel = ptxast.Kernel{
        .body = try body.toOwnedSlice(),
        .directives = try directives.toOwnedSlice(),
        .name = "main",
        .params = try params.toOwnedSlice(),
    };

    const globals = &[_]ptxast.GlobalDecl{};
    var kernels = [_]ptxast.Kernel{kernel};
    const ast = ptxast.PTXAst{ .allocator = allocator, .globals = globals, .kernels = &kernels };
    const ptx = try @import("./emission.zig").emit(allocator, ast);

    std.debug.print("{s}\n", .{ptx});

    const kernel_len = ptx.len;
    const global_ptx_str = core.LLVMAddGlobal(module, core.LLVMPointerType(core.LLVMInt8Type(), 0), "ptx_str");
    const kernel_constant = core.LLVMConstString(@ptrCast(ptx), @intCast(kernel_len), 0);
    core.LLVMSetInitializer(global_ptx_str, kernel_constant);
    return global_ptx_str;
}
