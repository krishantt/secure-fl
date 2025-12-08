pragma circom 2.0.0;
// Federated Learning Aggregation Circuit

template FedJSCMAggregation(n_clients, param_size) {
    signal input client_deltas[n_clients][param_size];
    signal input old_momentum[param_size];
    signal input client_weights[n_clients];
    signal input momentum_coeff;
    signal input old_params[param_size];

    signal output new_params[param_size];
    signal output new_momentum[param_size];

    // Intermediate signals for weighted sum
    signal weighted_sum[param_size];
    signal partial_sums[n_clients][param_size];

    var SCALE = 1000000;    // Calculate weighted deltas
    for (var i = 0; i < param_size; i++) {
        for (var j = 0; j < n_clients; j++) {
            partial_sums[j][i] <== client_weights[j] * client_deltas[j][i] / SCALE;
        }
    }

    // Sum the weighted deltas
    for (var i = 0; i < param_size; i++) {
        var sum = 0;
        for (var j = 0; j < n_clients; j++) {
            sum += partial_sums[j][i];
        }
        weighted_sum[i] <== sum;
    }

    // Compute momentum and new params
    signal momentum_term[param_size];
    for (var i = 0; i < param_size; i++) {
        momentum_term[i] <== momentum_coeff * old_momentum[i] / SCALE;
        new_momentum[i] <== momentum_term[i] + weighted_sum[i];
        new_params[i] <== old_params[i] + new_momentum[i];
    }
}

component main = FedJSCMAggregation(2, 5);
