const rllvm = @import("rllvm");
const cuda = rllvm.cuda;
const target = rllvm.llvm.target;
const target_machine_mod = rllvm.llvm.target_machine;
const types = rllvm.llvm.types;
const core = rllvm.llvm.core;
const execution = rllvm.llvm.engine;

const ptxast = @import("./ast.zig");
const nodes = @import("../../nodes.zig");

pub fn generatePTX(module: types.LLVMModuleRef, program: nodes.RHLOProgram) types.LLVMValueRef {
    for (program.ops.items) |op| {
        switch (op.kind) {
            .Add => {},
        }
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
    return global_ptx_str;
}
