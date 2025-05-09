pub const RHLOBuffer = struct {
    data_ptr: *void,
    size: usize,
    device: bool,
};

pub fn createBuffer(data_ptr: *void, size: usize) RHLOBuffer {
    return .{
        .data_ptr = data_ptr,
        .size = size,
        .device = false,
    };
}
