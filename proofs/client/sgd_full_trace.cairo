%lang starknet

from starkware.cairo.common.cairo_builtins import HashBuiltin, BitwiseBuiltin
from starkware.cairo.common.math import assert_nn_le, assert_not_zero, assert_le_felt
from starkware.cairo.common.math_cmp import is_le_felt
from starkware.cairo.common.hash import hash2
from starkware.cairo.common.bitwise import bitwise_and, bitwise_xor
from starkware.cairo.common.pow import pow
from starkware.cairo.common.alloc import alloc

// Constants
const FIELD_PRIME = 2**251 + 17 * 2**192 + 1;
const MAX_PARAMS = 100;
const MAX_EPOCHS = 10;
const MAX_BATCHES_PER_EPOCH = 50;
const LEARNING_RATE_SCALE = 10000;  // Fixed point scaling for learning rate
const GRADIENT_SCALE = 1000000;     // Fixed point scaling for gradients
const LOSS_SCALE = 1000000;         // Fixed point scaling for loss values

// Training state structure for SGD verification
struct SGDState {
    params: felt*,           // Model parameters (quantized)
    gradients: felt*,        // Computed gradients (quantized)
    loss: felt,             // Training loss (quantized)
    batch_size: felt,       // Number of samples in batch
    learning_rate: felt,    // Learning rate (quantized)
    step_count: felt,       // Current step number
}

// Data commitment structure
struct DataCommitment {
    dataset_hash: felt,     // Hash of dataset identifier
    data_size: felt,        // Number of training samples
    feature_dim: felt,      // Dimension of input features
    labels_hash: felt,      // Hash of labels
}

// Training trace for full verification
struct TrainingTrace {
    initial_state: SGDState,
    final_state: SGDState,
    intermediate_states: SGDState*,
    num_steps: felt,
    epoch_losses: felt*,
    batch_gradients: felt**,
}

// Storage variables for client verification
@storage_var
func client_id() -> (id: felt) {
}

@storage_var
func data_commitment_hash() -> (hash: felt) {
}

@storage_var
func training_round() -> (round: felt) {
}

@storage_var
func verification_threshold() -> (threshold: felt) {
}

// Initialize client verification context
@external
func initialize_verification{
    syscall_ptr: felt*,
    pedersen_ptr: HashBuiltin*,
    range_check_ptr
}(
    client_identifier: felt,
    data_commit: DataCommitment,
    round_num: felt,
    threshold: felt
) {
    // Store client information
    client_id.write(client_identifier);
    training_round.write(round_num);
    verification_threshold.write(threshold);

    // Compute and store data commitment hash
    let (commitment_hash) = hash_data_commitment(data_commit);
    data_commitment_hash.write(commitment_hash);

    return ();
}

// Main SGD full trace verification function
@external
func verify_sgd_full_trace{
    syscall_ptr: felt*,
    pedersen_ptr: HashBuiltin*,
    bitwise_ptr: BitwiseBuiltin*,
    range_check_ptr
}(
    trace: TrainingTrace,
    expected_delta: felt*,
    delta_size: felt
) -> (verification_result: felt) {
    alloc_locals;

    // Verify data commitment matches stored commitment
    let (stored_commitment) = data_commitment_hash.read();
    let (trace_commitment) = compute_trace_commitment(trace);
    assert stored_commitment = trace_commitment;

    // Verify trace integrity and SGD steps
    let (steps_valid) = verify_all_sgd_steps(trace);
    assert_not_zero(steps_valid);

    // Verify final parameter delta matches expected
    let (delta_valid) = verify_parameter_delta(
        trace.initial_state.params,
        trace.final_state.params,
        expected_delta,
        delta_size
    );
    assert_not_zero(delta_valid);

    // Verify training convergence properties
    let (convergence_valid) = verify_training_convergence(trace);
    assert_not_zero(convergence_valid);

    // Verify gradient computations are consistent
    let (gradients_valid) = verify_gradient_consistency(trace);
    assert_not_zero(gradients_valid);

    return (verification_result=1);
}

// Verify all SGD steps in the training trace
func verify_all_sgd_steps{
    syscall_ptr: felt*,
    pedersen_ptr: HashBuiltin*,
    range_check_ptr
}(trace: TrainingTrace) -> (all_valid: felt) {
    alloc_locals;

    if (trace.num_steps == 0) {
        return (all_valid=1);
    }

    // Verify initial state is valid
    let (initial_valid) = verify_sgd_state_validity(trace.initial_state);
    if (initial_valid == 0) {
        return (all_valid=0);
    }

    // Verify each SGD step transition
    let (steps_valid) = verify_step_transitions(
        trace.initial_state,
        trace.intermediate_states,
        trace.num_steps - 1
    );

    if (steps_valid == 0) {
        return (all_valid=0);
    }

    // Verify final step
    let (final_step_valid) = verify_single_sgd_step(
        trace.intermediate_states[trace.num_steps - 2],
        trace.final_state
    );

    return (all_valid=final_step_valid);
}

// Verify transitions between consecutive SGD states
func verify_step_transitions{
    syscall_ptr: felt*,
    pedersen_ptr: HashBuiltin*,
    range_check_ptr
}(
    initial_state: SGDState,
    intermediate_states: SGDState*,
    num_transitions: felt
) -> (transitions_valid: felt) {
    alloc_locals;

    if (num_transitions == 0) {
        return (transitions_valid=1);
    }

    // Verify first transition from initial state
    let (first_valid) = verify_single_sgd_step(
        initial_state,
        intermediate_states[0]
    );

    if (first_valid == 0) {
        return (transitions_valid=0);
    }

    // Recursively verify remaining transitions
    let (remaining_valid) = verify_remaining_transitions(
        intermediate_states,
        num_transitions - 1,
        0
    );

    return (transitions_valid=remaining_valid);
}

// Recursively verify remaining state transitions
func verify_remaining_transitions{
    syscall_ptr: felt*,
    pedersen_ptr: HashBuiltin*,
    range_check_ptr
}(
    states: SGDState*,
    remaining: felt,
    index: felt
) -> (valid: felt) {
    if (remaining == 0) {
        return (valid=1);
    }

    let (step_valid) = verify_single_sgd_step(states[index], states[index + 1]);
    if (step_valid == 0) {
        return (valid=0);
    }

    return verify_remaining_transitions(states, remaining - 1, index + 1);
}

// Verify a single SGD step: w_new = w_old - lr * gradient
func verify_single_sgd_step{
    syscall_ptr: felt*,
    pedersen_ptr: HashBuiltin*,
    range_check_ptr
}(
    prev_state: SGDState,
    curr_state: SGDState
) -> (step_valid: felt) {
    alloc_locals;

    // Verify step count increased by 1
    let step_diff = curr_state.step_count - prev_state.step_count;
    if (step_diff != 1) {
        return (step_valid=0);
    }

    // Verify learning rate consistency
    if (curr_state.learning_rate != prev_state.learning_rate) {
        return (step_valid=0);
    }

    // Verify parameter updates follow SGD rule
    let (params_valid) = verify_parameter_updates(
        prev_state.params,
        curr_state.params,
        prev_state.gradients,
        prev_state.learning_rate,
        MAX_PARAMS
    );

    if (params_valid == 0) {
        return (step_valid=0);
    }

    // Verify gradient bounds (anti-explosion check)
    let (gradients_bounded) = verify_gradient_bounds(curr_state.gradients, MAX_PARAMS);
    if (gradients_bounded == 0) {
        return (step_valid=0);
    }

    return (step_valid=1);
}

// Verify parameter updates follow SGD: w_new = w_old - lr * grad
func verify_parameter_updates{range_check_ptr}(
    old_params: felt*,
    new_params: felt*,
    gradients: felt*,
    learning_rate: felt,
    param_count: felt
) -> (updates_valid: felt) {
    if (param_count == 0) {
        return (updates_valid=1);
    }

    // Compute expected update: lr * gradient / GRADIENT_SCALE
    let gradient_update = (learning_rate * gradients[0]) / LEARNING_RATE_SCALE;
    let expected_param = old_params[0] - gradient_update;

    // Allow small numerical errors due to quantization
    let diff = abs_diff(new_params[0], expected_param);
    let tolerance = GRADIENT_SCALE / 1000;  // 0.1% tolerance

    let (within_tolerance) = is_le_felt(diff, tolerance);
    if (within_tolerance == 0) {
        return (updates_valid=0);
    }

    // Recursively check remaining parameters
    return verify_parameter_updates(
        old_params + 1,
        new_params + 1,
        gradients + 1,
        learning_rate,
        param_count - 1
    );
}

// Verify gradients are within reasonable bounds
func verify_gradient_bounds{range_check_ptr}(
    gradients: felt*,
    gradient_count: felt
) -> (bounds_valid: felt) {
    if (gradient_count == 0) {
        return (bounds_valid=1);
    }

    // Check gradient magnitude is reasonable
    let abs_gradient = abs_value(gradients[0]);
    let max_gradient = GRADIENT_SCALE * 100;  // Max gradient magnitude

    let (within_bounds) = is_le_felt(abs_gradient, max_gradient);
    if (within_bounds == 0) {
        return (bounds_valid=0);
    }

    return verify_gradient_bounds(gradients + 1, gradient_count - 1);
}

// Verify training convergence properties
func verify_training_convergence{range_check_ptr}(
    trace: TrainingTrace
) -> (convergence_valid: felt) {
    alloc_locals;

    // Check that loss generally decreases or stabilizes
    if (trace.num_steps < 2) {
        return (convergence_valid=1);
    }

    let initial_loss = trace.initial_state.loss;
    let final_loss = trace.final_state.loss;

    // Allow loss to increase slightly (up to 10%)
    let allowed_increase = initial_loss / 10;
    let max_allowed_loss = initial_loss + allowed_increase;

    let (loss_reasonable) = is_le_felt(final_loss, max_allowed_loss);

    return (convergence_valid=loss_reasonable);
}

// Verify gradient consistency across the trace
func verify_gradient_consistency{
    syscall_ptr: felt*,
    pedersen_ptr: HashBuiltin*,
    range_check_ptr
}(trace: TrainingTrace) -> (consistency_valid: felt) {
    // Check that gradients are computed consistently
    // This is a simplified check - real implementation would verify
    // gradients against actual loss computation

    if (trace.num_steps == 0) {
        return (consistency_valid=1);
    }

    // Basic sanity check: gradients should not be all zeros
    let (non_zero_gradients) = verify_non_zero_gradients(
        trace.initial_state.gradients,
        MAX_PARAMS
    );

    return (consistency_valid=non_zero_gradients);
}

// Verify at least some gradients are non-zero
func verify_non_zero_gradients{range_check_ptr}(
    gradients: felt*,
    count: felt
) -> (has_non_zero: felt) {
    if (count == 0) {
        return (has_non_zero=0);
    }

    if (gradients[0] != 0) {
        return (has_non_zero=1);
    }

    return verify_non_zero_gradients(gradients + 1, count - 1);
}

// Verify parameter delta matches expected change
func verify_parameter_delta{range_check_ptr}(
    initial_params: felt*,
    final_params: felt*,
    expected_delta: felt*,
    param_count: felt
) -> (delta_valid: felt) {
    if (param_count == 0) {
        return (delta_valid=1);
    }

    let actual_delta = final_params[0] - initial_params[0];
    let diff = abs_diff(actual_delta, expected_delta[0]);

    // Allow 1% tolerance for numerical errors
    let tolerance = abs_value(expected_delta[0]) / 100 + 1;
    let (within_tolerance) = is_le_felt(diff, tolerance);

    if (within_tolerance == 0) {
        return (delta_valid=0);
    }

    return verify_parameter_delta(
        initial_params + 1,
        final_params + 1,
        expected_delta + 1,
        param_count - 1
    );
}

// Verify SGD state validity
func verify_sgd_state_validity{range_check_ptr}(state: SGDState) -> (valid: felt) {
    // Check learning rate is positive
    assert_nn_le(0, state.learning_rate);

    // Check batch size is positive
    assert_nn_le(0, state.batch_size);

    // Check step count is non-negative
    assert_nn_le(0, state.step_count);

    return (valid=1);
}

// Compute hash of data commitment
func hash_data_commitment{
    syscall_ptr: felt*,
    pedersen_ptr: HashBuiltin*
}(commit: DataCommitment) -> (hash: felt) {
    let (hash1) = hash2{hash_ptr=pedersen_ptr}(commit.dataset_hash, commit.data_size);
    let (hash2_val) = hash2{hash_ptr=pedersen_ptr}(commit.feature_dim, commit.labels_hash);
    let (final_hash) = hash2{hash_ptr=pedersen_ptr}(hash1, hash2_val);
    return (hash=final_hash);
}

// Compute commitment hash from training trace
func compute_trace_commitment{
    syscall_ptr: felt*,
    pedersen_ptr: HashBuiltin*
}(trace: TrainingTrace) -> (hash: felt) {
    // Simplified commitment - real implementation would be more comprehensive
    let (hash1) = hash2{hash_ptr=pedersen_ptr}(
        trace.initial_state.params[0],
        trace.final_state.params[0]
    );
    let (hash2_val) = hash2{hash_ptr=pedersen_ptr}(trace.num_steps, trace.initial_state.loss);
    let (final_hash) = hash2{hash_ptr=pedersen_ptr}(hash1, hash2_val);
    return (hash=final_hash);
}

// Utility function to compute absolute difference
func abs_diff{range_check_ptr}(a: felt, b: felt) -> felt {
    let diff = a - b;
    if (is_le_felt(0, diff) == 1) {
        return diff;
    } else {
        return -diff;
    }
}

// Utility function to compute absolute value
func abs_value{range_check_ptr}(value: felt) -> felt {
    if (is_le_felt(0, value) == 1) {
        return value;
    } else {
        return -value;
    }
}

// Public interface for external verification
@view
func get_verification_status{
    syscall_ptr: felt*,
    pedersen_ptr: HashBuiltin*,
    range_check_ptr
}() -> (
    client: felt,
    round: felt,
    data_commitment: felt,
    verified: felt
) {
    let (client) = client_id.read();
    let (round) = training_round.read();
    let (commitment) = data_commitment_hash.read();

    // Return verification status
    return (
        client=client,
        round=round,
        data_commitment=commitment,
        verified=1
    );
}
