struct GpuResult {
    es: f32,
    peak_idx: u32,
}

@group(0) @binding(0) var<storage, read> abs_scores: array<f32>;
@group(0) @binding(1) var<storage, read> subsets_indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> results: array<GpuResult>;

struct Params {
    k: f32,
    n_total: f32,
    batch_size: f32,
    _pad: f32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    if (f32(batch_idx) >= params.batch_size) {
        return;
    }

    let k = u32(params.k);
    let n_total = params.n_total;
    let n_miss = n_total - params.k;
    
    // Find sum of weights for this subset
    var sum_weights: f32 = 0.0;
    let start_offset = batch_idx * k;
    for (var i: u32 = 0; i < k; i++) {
        let idx = subsets_indices[start_offset + i];
        sum_weights += abs_scores[idx];
    }

    if (sum_weights == 0.0) {
        results[batch_idx] = GpuResult(0.0, 0);
        return;
    }

    var curr_max: f32 = 0.0;
    var curr_min: f32 = 0.0;
    var max_idx: u32 = 0;
    var min_idx: u32 = 0;
    
    var curr_sum_weight: f32 = 0.0;
    for (var j: u32 = 0; j < k; j++) {
        let hit_idx = subsets_indices[start_offset + j];
        let prev_p_hit = curr_sum_weight / sum_weights;
        curr_sum_weight += abs_scores[hit_idx];
        
        let p_hit = curr_sum_weight / sum_weights;
        let p_miss = (f32(hit_idx) - f32(j)) / n_miss;
        
        let es_before = prev_p_hit - p_miss;
        let es_at = p_hit - p_miss;
        
        if (es_before > curr_max) {
            curr_max = es_before;
            max_idx = hit_idx;
        }
        if (es_before < curr_min) {
            curr_min = es_before;
            min_idx = hit_idx;
        }
        
        if (es_at > curr_max) {
            curr_max = es_at;
            max_idx = hit_idx;
        }
        if (es_at < curr_min) {
            curr_min = es_at;
            min_idx = hit_idx;
        }
    }

    if (abs(curr_max) >= abs(curr_min)) {
        results[batch_idx] = GpuResult(curr_max, max_idx);
    } else {
        results[batch_idx] = GpuResult(curr_min, min_idx);
    }
}
