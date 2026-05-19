use std::time::Instant;
use bitcache::TwoStageIndex;
use rand::prelude::*;

fn main() {
    let dim = 384;
    let n = 99_000;
    let nq = 1000;
    let k = 10;

    println!("Building index: {} vectors, dim={}", n, dim);
    let mut rng = StdRng::seed_from_u64(42);
    let vectors: Vec<f32> = (0..n*dim).map(|_| rng.gen_range(-1.0..1.0f32)).collect();
    let queries: Vec<f32> = (0..nq*dim).map(|_| rng.gen_range(-1.0..1.0f32)).collect();

    let mut index = TwoStageIndex::new(dim, 10);
    index.add(&vectors);
    println!("Index built: {} vectors\n", index.len());

    // Warm up
    let _ = index.search(&queries[..dim], k);

    // Sequential benchmark
    let t = Instant::now();
    for i in 0..nq {
        let q = &queries[i*dim..(i+1)*dim];
        index.search(q, k);
    }
    let elapsed_seq = t.elapsed().as_secs_f64();
    let qps_seq = nq as f64 / elapsed_seq;
    let lat_seq = elapsed_seq / nq as f64 * 1_000_000.0;
    println!("Sequential: {} queries in {:.3}s", nq, elapsed_seq);
    println!("  QPS: {:.0}", qps_seq);
    println!("  Latency: {:.0} µs/query\n", lat_seq);

    // Batch (parallel) benchmark
    let t = Instant::now();
    let _ = index.search_batch(&queries, k);
    let elapsed_par = t.elapsed().as_secs_f64();
    let qps_par = nq as f64 / elapsed_par;
    let lat_par = elapsed_par / nq as f64 * 1_000_000.0;
    println!("Parallel batch: {} queries in {:.3}s", nq, elapsed_par);
    println!("  QPS: {:.0}", qps_par);
    println!("  Latency: {:.0} µs/query\n", lat_par);

    println!("Parallel speedup: {:.1}x", elapsed_seq / elapsed_par);
    println!("Effective throughput: {:.0} QPS (was 1,869 before SIMD+Rayon)", qps_par);
}
