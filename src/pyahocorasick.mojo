"""Aho-Corasick traversal kernels exposed through a stable C ABI."""

from std.ffi import external_call
from std.sys.info import simd_width_of as simdwidthof

comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime I32Ptr = UnsafePointer[Int32, AnyOrigin[mut=True]]
comptime U32Ptr = UnsafePointer[UInt32, AnyOrigin[mut=True]]


def i64_ptr(address: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=address)


def u32_ptr(address: Int) -> U32Ptr:
    return U32Ptr(unsafe_from_address=address)


def i32_ptr(address: Int) -> I32Ptr:
    return I32Ptr(unsafe_from_address=address)


def find_edge(
    edge_starts: I64Ptr,
    edge_labels: U32Ptr,
    edge_targets: I64Ptr,
    state: Int,
    label: UInt32,
) -> Int:
    var low = Int(edge_starts.load(state))
    var high = Int(edge_starts.load(state + 1))
    while low < high:
        var middle = low + (high - low) // 2
        var candidate = edge_labels.load(middle)
        if candidate < label:
            low = middle + 1
        else:
            high = middle
    if (
        low < Int(edge_starts.load(state + 1))
        and edge_labels.load(low) == label
    ):
        return Int(edge_targets.load(low))
    return -1


def find_label(alphabet: U32Ptr, alphabet_size: Int, label: UInt32) -> Int:
    var low = 0
    var high = alphabet_size
    while low < high:
        var middle = low + (high - low) // 2
        var candidate = alphabet.load(middle)
        if candidate < label:
            low = middle + 1
        else:
            high = middle
    if low < alphabet_size and alphabet.load(low) == label:
        return low
    return -1


@export("mpac_scan")
def mpac_scan(
    text_address: Int,
    text_length: Int,
    edge_starts_address: Int,
    edge_labels_address: Int,
    edge_targets_address: Int,
    failure_address: Int,
    output_starts_address: Int,
    output_ids_address: Int,
    alphabet_address: Int,
    dense_transitions_address: Int,
    alphabet_size: Int,
    alphabet_lookup_size: Int,
    end_indices_address: Int,
    match_ids_address: Int,
    capacity: Int,
    ignore_white_space: Int,
) abi("C") -> Int:
    var text = u32_ptr(text_address)
    var edge_starts = i64_ptr(edge_starts_address)
    var edge_labels = u32_ptr(edge_labels_address)
    var edge_targets = i64_ptr(edge_targets_address)
    var failure = i64_ptr(failure_address)
    var output_starts = i64_ptr(output_starts_address)
    var output_ids = i32_ptr(output_ids_address)
    var alphabet = u32_ptr(alphabet_address)
    var alphabet_lookup = i32_ptr(alphabet_address)
    var dense_transitions = i32_ptr(dense_transitions_address)
    var end_indices = i64_ptr(end_indices_address)
    var match_ids = i32_ptr(match_ids_address)

    var state = 0
    var count = 0
    comptime W = simdwidthof[DType.float64]()
    for index in range(text_length):
        var label = text.load(index)
        if ignore_white_space != 0 and (
            label == UInt32(32) or (label >= UInt32(9) and label <= UInt32(13))
        ):
            continue

        if alphabet_size > 0:
            var column = -1
            if alphabet_lookup_size > 0:
                if Int(label) < alphabet_lookup_size:
                    column = Int(alphabet_lookup.load(Int(label)))
            else:
                column = find_label(alphabet, alphabet_size, label)
            if column >= 0:
                state = Int(
                    dense_transitions.load(state * alphabet_size + column)
                )
            else:
                state = 0
        else:
            var target = find_edge(
                edge_starts, edge_labels, edge_targets, state, label
            )
            while target < 0 and state != 0:
                state = Int(failure.load(state))
                target = find_edge(
                    edge_starts, edge_labels, edge_targets, state, label
                )
            if target >= 0:
                state = target
            else:
                state = 0

        var first = Int(output_starts.load(state))
        var last = Int(output_starts.load(state + 1))
        var output_count = last - first
        var available = capacity - count
        if output_count > 0 and available > 0:
            var writable = min(output_count, available)
            var output_index = first
            var output_limit = first + writable
            var vector_limit = first + (writable // W) * W
            var repeated_index = SIMD[DType.int64, W](Int64(index))
            while output_index < vector_limit:
                var vector_destination = count + output_index - first
                end_indices.store(vector_destination, repeated_index)
                match_ids.store(
                    vector_destination, output_ids.load[width=W](output_index)
                )
                output_index += W
            while output_index < output_limit:
                var scalar_destination = count + output_index - first
                end_indices.store(scalar_destination, Int64(index))
                match_ids.store(
                    scalar_destination, output_ids.load(output_index)
                )
                output_index += 1
        count += output_count
    return count


@export("mpac_materialize")
def mpac_materialize(
    end_indices_address: Int,
    match_ids_address: Int,
    count: Int,
    values: Int,
) abi("C") -> Int:
    if (
        end_indices_address == 0
        or match_ids_address == 0
        or count < 0
        or values == 0
    ):
        return 0
    var end_indices = i64_ptr(end_indices_address)
    var match_ids = i32_ptr(match_ids_address)
    var result = external_call["PyList_New", Int](count)
    if result == 0:
        return 0
    for i in range(count):
        var pair = external_call["PyTuple_New", Int](2)
        if pair == 0:
            external_call["Py_DecRef", NoneType](result)
            return 0
        var end_index = external_call["PyLong_FromLongLong", Int](
            end_indices.load(i)
        )
        if end_index == 0:
            external_call["Py_DecRef", NoneType](pair)
            external_call["Py_DecRef", NoneType](result)
            return 0
        var value = external_call["PyTuple_GetItem", Int](
            values, Int(match_ids.load(i))
        )
        if value == 0:
            external_call["Py_DecRef", NoneType](end_index)
            external_call["Py_DecRef", NoneType](pair)
            external_call["Py_DecRef", NoneType](result)
            return 0
        external_call["Py_IncRef", NoneType](value)
        _ = external_call["PyTuple_SetItem", Int32](pair, 0, end_index)
        _ = external_call["PyTuple_SetItem", Int32](pair, 1, value)
        _ = external_call["PyList_SetItem", Int32](result, i, pair)
    return result
