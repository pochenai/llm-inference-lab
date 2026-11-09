use std::{hint::black_box, time::Instant};
#[inline(never)]
fn early(values: &[f32]) -> bool { values.iter().all(|v| v.is_finite()) }
#[inline(never)]
fn fold(values: &[f32]) -> bool { values.iter().fold(true, |valid, v| valid & v.is_finite()) }
fn main() {
    for len in [0,1,3,4,7,8,16,19,248320] {
        let values: Vec<f32> = (0..len).map(|i| [0.0,-0.0,f32::from_bits(1),f32::MAX,f32::MIN][i%5]).collect();
        assert!(early(&values) && fold(&values));
        for position in [0,len/2,len.saturating_sub(1)] {
            if len == 0 { continue; }
            for invalid in [f32::NAN,f32::INFINITY,f32::NEG_INFINITY] {
                let mut invalids=values.clone(); invalids[position]=invalid;
                assert!(!early(&invalids) && !fold(&invalids));
            }
        }
    }
    let values: Vec<f32> = (0..248320).map(|i| ((i*37%1009) as f32 - 504.0)/128.0).collect();
    for (name, check) in [("all", early as fn(&[f32])->bool),("fold",fold as fn(&[f32])->bool)] {
        let start=Instant::now();
        for _ in 0..2000 { assert!(black_box(check(black_box(&values)))); }
        println!("{name}: {:.6} ms/scan",start.elapsed().as_secs_f64()*1000.0/2000.0);
    }
}
