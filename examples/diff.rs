// Diagnostic: per-step log-prob error vs a llama.cpp fixture.
use kllm::model::{Logits, Model};
use kllm::sampler::log_prob;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let model = Model::load(&a[1]).unwrap();
    let fx: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&a[2]).unwrap()).unwrap();
    println!("types {:?}", model.weight_types());
    for case in fx["cases"].as_array().unwrap().iter().take(2) {
        let ids: Vec<u32> = case["prompt_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        let mut s = model.new_session(512, 64).unwrap();
        let mut logits = s.forward(&ids, Logits::Last).unwrap().to_vec();
        for (n, step) in case["steps"].as_array().unwrap().iter().enumerate().take(6) {
            let top = step["top"].as_array().unwrap();
            let errs: Vec<String> = top
                .iter()
                .map(|t| {
                    format!(
                        "{:+.3}",
                        log_prob(&logits, t[0].as_u64().unwrap() as u32) - t[1].as_f64().unwrap()
                    )
                })
                .collect();
            println!(
                "step {n} ref top lp {:.3} errs {}",
                top[0][1].as_f64().unwrap(),
                errs.join(" ")
            );
            logits = s
                .forward(&[step["id"].as_u64().unwrap() as u32], Logits::Last)
                .unwrap()
                .to_vec();
        }
    }
}
