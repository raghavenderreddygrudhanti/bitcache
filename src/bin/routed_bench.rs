//! Benchmark: FloatRouted vs TwoStage (exhaustive) throughput comparison.
//!
//! Shows the QPS advantage of scanning only probed partitions.

use std::time::Instant;
use bitcache::{TwoStageIndex, FloatRoutedIndex};
use rand::prelude::*;

fn main() {
    let dim = 384;
    let k = 10;
    let nq = 1000;

    println!("═══════════════════════════════════════════════════════════════");
    println!("  ROUTED vs EXHAUSTIVE: SCALE COMPARISON");
    println!("  dim={}, {} queries, k={}", dim, nq, k);
    println!("═══════════════════════════════════════════════════════════════\n");

    for n in [99_000, 500_000, 1_000_000] {
        println!("─── n = {} ({:.1} MB binary codes) ───", n, n as f64 * 48.0 / 1024.0 / 1024.0);

        let mut rng = StdRng::seed_from_u64(42);
        let vectors: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0..1.0f32)).collect();
        let queries: Vec<f32> = (0..nq * dim).map(|_| rng.gen_range(-1.0..1.0f32)).collect();

        // TwoStage exhaustive
        let mut exhaustive = TwoStageIndex::new(dim, 10);
        exhaustive.add(&vectors);

        let t = Instant::now();
        exhaustive.search_batch(&queries, k);
        let exh_time = t.elapsed().as_secs_f64();
        let exh_qps = nq as f64 / exh_time;

        // FloatRouted P=32, probe=2
        let mut routed = FloatRoutedIndex::new(dim, 32, 2, 100, 5);
        routed.build(&vectors);

        let t = Instant::now();
        routed.search_batch(&queries, k);
        let rt_time = t.elapsed().as_secs_f64();
        let rt_qps = nq as f64 / rt_time;

        let speedup = exh_time / rt_time;
        println!("  Exhaustive:  {:.0} QPS ({:.1}ms/query)", exh_qps, exh_time / nq as f64 * 1000.0);
        println!("  Routed 6.2%: {:.0} QPS ({:.1}ms/query)  [{:.1}x speedup]\n", rt_qps, rt_time / nq as f64 * 1000.0, speedup);
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Routing benefit increases with scale (memory-bandwidth bound)");
    println!("═══════════════════════════════════════════════════════════════");
}
