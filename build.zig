const std = @import("std");
const llvm = @import("../wrappers/llvm.zig");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    _ = llvm.setupLLVMInBuild(b, target, optimize);

    const main_module = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    main_module.addImport("llvm", b.modules.get("llvm").?);

    const lib = b.addSharedLibrary(.{
        .name = "rhlo",
        .root_module = main_module,
        .optimize = optimize,
    });
    b.installArtifact(lib);

    const tests = b.addTest(.{
        .root_module = main_module,
        .optimize = optimize,
    });

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&b.addRunArtifact(tests).step);
}
