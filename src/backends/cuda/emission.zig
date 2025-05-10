const std = @import("std");
const ptxast = @import("./ast.zig");

pub fn emit(allocator: std.mem.Allocator, ast: ptxast.PTXAst) ![]const u8 {
    for (ast.kernels) |kernel| {
        var ptx = std.ArrayList(u8).init(allocator);
        var writer = ptx.writer();

        try writer.writeAll(
            \\.version 8.4
            \\.target sm_52
            \\.address_size 64
        );

        try writer.writeAll("\n.visible .entry main(\n");
        for (kernel.params, 0..) |param, i| {
            try writer.print("  .param .u64 {s}{s}\n", .{ param.name, if (i < kernel.params.len - 1) "," else "" });
        }
        try writer.writeAll(")\n");

        try writer.writeAll(
            \\{
            \\
        );
        for (kernel.directives) |directive| {
            switch (directive) {
                .reg => |reg| {
                    try writer.print("  .reg .{s} {s}<{d}>;\n", .{ reg.type.toString(), reg.name, reg.count });
                },
                else => unreachable,
            }
        }

        try writer.writeAll("\n");

        for (kernel.body) |instruction| {
            switch (instruction) {
                .add => |add| {
                    try writer.print("  add.{s} {s}, {s}, {s};\n", .{ add.type.toString(), emitOperand(add.dest), emitOperand(add.src1), emitOperand(add.src2) });
                },
                .ld => |ld| {
                    try writer.print("  ld.{s}.{s} {s}, [{s}];\n", .{ ld.space.toString(), ld.type.toString(), emitOperand(ld.dest), emitOperand(ld.src) });
                },
                .st => |st| {
                    try writer.print("  st.{s}.{s} [{s}], {s};\n", .{ st.space.toString(), st.type.toString(), emitOperand(st.dest), emitOperand(st.src) });
                },
                .cvta => |cvta| {
                    try writer.print("  cvta{s}.{s}.{s} {s}, {s};\n", .{ if (cvta.to_generic == true) ".to" else "", cvta.space.toString(), cvta.type.toString(), emitOperand(cvta.dest), emitOperand(cvta.src) });
                },
                .ret => try writer.writeAll("  ret;\n"),
                else => unreachable,
            }
        }
        try writer.writeAll(
            \\}
        );

        return try ptx.toOwnedSlice();
    }

    return error.ShouldntHappen;
}

fn emitOperand(operand: ptxast.Operand) []const u8 {
    return switch (operand) {
        .register => |reg| {
            return reg;
        },
        .parameter => |param| {
            return param;
        },
        else => unreachable,
    };
}
