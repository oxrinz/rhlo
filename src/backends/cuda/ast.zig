const std = @import("std");

pub const Register = struct {
    name: []const u8,
    type: DataType,
};

pub const Immediate = union(enum) {
    integer: i64,
    float: f64,
};

pub const Operand = union(enum) {
    register: Register,
    immediate: Immediate,
    memory: MemoryRef,
};

pub const MemoryRef = struct {
    address: []const u8,
    type: DataType,
};

pub const DataType = enum {
    u8,
    s8,
    u16,
    s16,
    u32,
    s32,
    u64,
    s64,
    f16,
    f32,
    f64,
    pred,
};

pub const Instruction = union(enum) {
    add: AddInst,
    ld: LoadInst,
    bra: BranchInst,
    ret: void,
};

pub const AddInst = struct {
    dest: Operand,
    src1: Operand,
    src2: Operand,
    type: DataType,
};

pub const LoadInst = struct {
    dest: Operand,
    src: Operand,
};

pub const BranchInst = struct {
    label: []const u8,
    predicate: ?Operand,
};

pub const Kernel = struct {
    name: []const u8,
    params: []Parameter,
    body: []Instruction,
    registers: []Register,
    directives: []Directive,
};

pub const Parameter = struct {
    name: []const u8,
    type: DataType,
};

pub const Directive = union(enum) {
    reg: RegisterDecl,
    global: GlobalDecl,
};

pub const RegisterDecl = struct {
    type: DataType,
    count: u32,
    name: []const u8,
};

pub const GlobalDecl = struct {
    name: []const u8,
    size: u64,
    type: DataType,
};

pub const PTXAst = struct {
    kernels: []Kernel,
    globals: []GlobalDecl,
    allocator: std.mem.Allocator,

    fn deinit(self: *PTXAst) void {
        for (self.kernels) |kernel| {
            self.allocator.free(kernel.params);
            self.allocator.free(kernel.body);
            self.allocator.free(kernel.registers);
            self.allocator.free(kernel.directives);
        }
        self.allocator.free(self.kernels);
        self.allocator.free(self.globals);
    }
};
