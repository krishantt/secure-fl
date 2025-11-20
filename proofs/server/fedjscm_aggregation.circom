pragma circom 2.0.0;

// FedJSCM (Federated Joint Server-Client Momentum) Aggregation Circuit
// Verifies the correct computation of momentum-based federated aggregation
// Formula: m^{(t+1)} = γ * m^{(t)} + Σ(p_i * Δ_i)
//          w^{(t+1)} = w^{(t)} + m^{(t+1)}

include "circomlib/circuits/comparators.circom";
include "circomlib/circuits/gates.circom";

template ScalarMult(n) {
    signal input a;
    signal input b[n];
    signal output out[n];

    for (var i = 0; i < n; i++) {
        out[i] <== a * b[i];
    }
}

template VectorAdd(n) {
    signal input a[n];
    signal input b[n];
    signal output out[n];

    for (var i = 0; i < n; i++) {
        out[i] <== a[i] + b[i];
    }
}

template WeightedSum(n_clients, param_size) {
    signal input client_deltas[n_clients][param_size];
    signal input client_weights[n_clients];
    signal output weighted_sum[param_size];

    component mult[n_clients];
    component add[n_clients - 1];

    // Initialize weighted sum with first client
    mult[0] = ScalarMult(param_size);
    mult[0].a <== client_weights[0];
    for (var j = 0; j < param_size; j++) {
        mult[0].b[j] <== client_deltas[0][j];
    }

    if (n_clients == 1) {
        for (var j = 0; j < param_size; j++) {
            weighted_sum[j] <== mult[0].out[j];
        }
    } else {
        // Add remaining clients
        for (var i = 1; i < n_clients; i++) {
            mult[i] = ScalarMult(param_size);
            mult[i].a <== client_weights[i];
            for (var j = 0; j < param_size; j++) {
                mult[i].b[j] <== client_deltas[i][j];
            }

            add[i-1] = VectorAdd(param_size);

            if (i == 1) {
                for (var j = 0; j < param_size; j++) {
                    add[i-1].a[j] <== mult[0].out[j];
                }
            } else {
                for (var j = 0; j < param_size; j++) {
                    add[i-1].a[j] <== add[i-2].out[j];
                }
            }

            for (var j = 0; j < param_size; j++) {
                add[i-1].b[j] <== mult[i].out[j];
            }
        }

        for (var j = 0; j < param_size; j++) {
            weighted_sum[j] <== add[n_clients-2].out[j];
        }
    }
}

template MomentumUpdate(param_size) {
    signal input old_momentum[param_size];
    signal input weighted_delta[param_size];
    signal input momentum_coeff;
    signal output new_momentum[param_size];

    component momentum_mult = ScalarMult(param_size);
    component momentum_add = VectorAdd(param_size);

    // γ * m^{(t)}
    momentum_mult.a <== momentum_coeff;
    for (var i = 0; i < param_size; i++) {
        momentum_mult.b[i] <== old_momentum[i];
    }

    // γ * m^{(t)} + weighted_delta
    for (var i = 0; i < param_size; i++) {
        momentum_add.a[i] <== momentum_mult.out[i];
        momentum_add.b[i] <== weighted_delta[i];
    }

    for (var i = 0; i < param_size; i++) {
        new_momentum[i] <== momentum_add.out[i];
    }
}

template ParameterUpdate(param_size) {
    signal input old_params[param_size];
    signal input momentum[param_size];
    signal output new_params[param_size];

    component param_add = VectorAdd(param_size);

    for (var i = 0; i < param_size; i++) {
        param_add.a[i] <== old_params[i];
        param_add.b[i] <== momentum[i];
    }

    for (var i = 0; i < param_size; i++) {
        new_params[i] <== param_add.out[i];
    }
}

template WeightValidation(n_clients) {
    signal input weights[n_clients];
    signal input expected_sum;

    component sum_check = SumCheck(n_clients);
    component eq_check = IsEqual();

    for (var i = 0; i < n_clients; i++) {
        sum_check.in[i] <== weights[i];
    }

    eq_check.in[0] <== sum_check.out;
    eq_check.in[1] <== expected_sum;

    eq_check.out === 1;
}

template SumCheck(n) {
    signal input in[n];
    signal output out;

    if (n == 1) {
        out <== in[0];
    } else {
        component sum_remaining = SumCheck(n-1);
        for (var i = 0; i < n-1; i++) {
            sum_remaining.in[i] <== in[i+1];
        }
        out <== in[0] + sum_remaining.out;
    }
}

template RangeCheck(bits) {
    signal input in;

    component num2bits = Num2Bits(bits);
    component bits2num = Bits2Num(bits);

    num2bits.in <== in;

    for (var i = 0; i < bits; i++) {
        bits2num.in[i] <== num2bits.out[i];
    }

    bits2num.out === in;
}

template MomentumBoundsCheck(param_size) {
    signal input momentum[param_size];
    signal input max_magnitude;

    component magnitude_check[param_size];
    component abs_check[param_size];
    component le_check[param_size];

    for (var i = 0; i < param_size; i++) {
        // Check if |momentum[i]| <= max_magnitude
        abs_check[i] = AbsoluteValue();
        abs_check[i].in <== momentum[i];

        le_check[i] = LessEqThan(64);
        le_check[i].in[0] <== abs_check[i].out;
        le_check[i].in[1] <== max_magnitude;

        le_check[i].out === 1;
    }
}

template AbsoluteValue() {
    signal input in;
    signal output out;

    component is_negative = IsNegative();
    component conditional = ConditionalSelect();

    is_negative.in <== in;
    conditional.condition <== is_negative.out;
    conditional.true_value <== -in;
    conditional.false_value <== in;

    out <== conditional.out;
}

template IsNegative() {
    signal input in;
    signal output out;

    component bits = Num2Bits(254);
    bits.in <== in;

    out <== bits.out[253];
}

template ConditionalSelect() {
    signal input condition;
    signal input true_value;
    signal input false_value;
    signal output out;

    out <== condition * true_value + (1 - condition) * false_value;
}

// Main FedJSCM aggregation circuit
template FedJSCMAggregation(n_clients, param_size, weight_precision) {
    // Private inputs (what we want to prove we computed correctly)
    signal private input client_deltas[n_clients][param_size];
    signal private input old_momentum[param_size];
    signal private input old_params[param_size];

    // Public inputs (known to verifier)
    signal input client_weights[n_clients];
    signal input momentum_coeff;
    signal input expected_new_params[param_size];
    signal input expected_new_momentum[param_size];
    signal input round_number;

    // Outputs (what we're proving)
    signal output new_params[param_size];
    signal output new_momentum[param_size];
    signal output aggregation_valid;

    // Intermediate signals
    signal weighted_sum[param_size];

    // Components for computation verification
    component weighted_aggregation = WeightedSum(n_clients, param_size);
    component momentum_update = MomentumUpdate(param_size);
    component parameter_update = ParameterUpdate(param_size);

    // Validation components
    component weight_validation = WeightValidation(n_clients);
    component momentum_bounds = MomentumBoundsCheck(param_size);
    component range_checks[param_size];

    // 1. Validate client weights sum to weight_precision (e.g., 1000 for 0.001 precision)
    weight_validation.expected_sum <== weight_precision;
    for (var i = 0; i < n_clients; i++) {
        weight_validation.weights[i] <== client_weights[i];
    }

    // 2. Compute weighted sum of client deltas
    for (var i = 0; i < n_clients; i++) {
        weighted_aggregation.client_weights[i] <== client_weights[i];
        for (var j = 0; j < param_size; j++) {
            weighted_aggregation.client_deltas[i][j] <== client_deltas[i][j];
        }
    }

    for (var i = 0; i < param_size; i++) {
        weighted_sum[i] <== weighted_aggregation.weighted_sum[i];
    }

    // 3. Update momentum: m^{(t+1)} = γ * m^{(t)} + weighted_sum
    momentum_update.momentum_coeff <== momentum_coeff;
    for (var i = 0; i < param_size; i++) {
        momentum_update.old_momentum[i] <== old_momentum[i];
        momentum_update.weighted_delta[i] <== weighted_sum[i];
    }

    for (var i = 0; i < param_size; i++) {
        new_momentum[i] <== momentum_update.new_momentum[i];
    }

    // 4. Update parameters: w^{(t+1)} = w^{(t)} + m^{(t+1)}
    for (var i = 0; i < param_size; i++) {
        parameter_update.old_params[i] <== old_params[i];
        parameter_update.momentum[i] <== new_momentum[i];
    }

    for (var i = 0; i < param_size; i++) {
        new_params[i] <== parameter_update.new_params[i];
    }

    // 5. Verify computed values match expected values
    component param_checks[param_size];
    component momentum_checks[param_size];

    for (var i = 0; i < param_size; i++) {
        param_checks[i] = IsEqual();
        param_checks[i].in[0] <== new_params[i];
        param_checks[i].in[1] <== expected_new_params[i];
        param_checks[i].out === 1;

        momentum_checks[i] = IsEqual();
        momentum_checks[i].in[0] <== new_momentum[i];
        momentum_checks[i].in[1] <== expected_new_momentum[i];
        momentum_checks[i].out === 1;
    }

    // 6. Verify momentum bounds (prevent explosion)
    momentum_bounds.max_magnitude <== 1000000; // Reasonable bound
    for (var i = 0; i < param_size; i++) {
        momentum_bounds.momentum[i] <== new_momentum[i];
    }

    // 7. Range checks for parameters (prevent overflow)
    for (var i = 0; i < param_size; i++) {
        range_checks[i] = RangeCheck(32);
        range_checks[i].in <== new_params[i];
    }

    // 8. Set validation flag
    aggregation_valid <== 1;
}

// Simplified version for testing with smaller parameters
template SimpleFedJSCM() {
    signal private input client_deltas[3][5];
    signal private input old_momentum[5];
    signal private input old_params[5];

    signal input client_weights[3];
    signal input momentum_coeff;
    signal input expected_new_params[5];
    signal input expected_new_momentum[5];

    signal output new_params[5];
    signal output new_momentum[5];
    signal output valid;

    component aggregator = FedJSCMAggregation(3, 5, 1000);

    // Connect inputs
    for (var i = 0; i < 3; i++) {
        aggregator.client_weights[i] <== client_weights[i];
        for (var j = 0; j < 5; j++) {
            aggregator.client_deltas[i][j] <== client_deltas[i][j];
        }
    }

    for (var i = 0; i < 5; i++) {
        aggregator.old_momentum[i] <== old_momentum[i];
        aggregator.old_params[i] <== old_params[i];
        aggregator.expected_new_params[i] <== expected_new_params[i];
        aggregator.expected_new_momentum[i] <== expected_new_momentum[i];
    }

    aggregator.momentum_coeff <== momentum_coeff;
    aggregator.round_number <== 1;

    // Connect outputs
    for (var i = 0; i < 5; i++) {
        new_params[i] <== aggregator.new_params[i];
        new_momentum[i] <== aggregator.new_momentum[i];
    }

    valid <== aggregator.aggregation_valid;
}

// Main component - configurable based on deployment needs
component main = SimpleFedJSCM();
