use std::env;

fn main() {
    let mut rustflags = env::var("RUSTFLAGS").unwrap_or_default();

    let cross_compile = env::var("CROSS_COMPILE").map(|v| v == "1").unwrap_or(false);

    use std::arch::is_x86_feature_detected;

    let mut target_features = Vec::new();

    target_features.push("native");

    if cfg!(feature = "avx2") {
        if !cross_compile && !is_x86_feature_detected!("avx2") {
            panic!(
                "AVX2 is not supported by the CPU. Pass CROSS_COMPILE=1 to cross-compile for AVX2."
            );
        }

        target_features.push("+avx2");
    }

    if cfg!(feature = "avx512") {
        if !cross_compile && !is_x86_feature_detected!("avx512f") {
            panic!(
                "avx512f is not supported by the CPU. Pass CROSS_COMPILE=1 to cross-compile for avx512f."
            );
        }

        if !cross_compile && !is_x86_feature_detected!("avx512vl") {
            panic!(
                "avx512vl is not supported by the CPU. Pass CROSS_COMPILE=1 to cross-compile for avx512vl."
            );
        }

        target_features.push("+avx512f");
        target_features.push("+avx512vl");
        target_features.push("+avx512dq");
        target_features.push("+bmi1");
    }

    rustflags.push_str(" -C target-feature=");
    rustflags.push_str(&target_features.join(","));

    println!("cargo:warning=RUSTFLAGS: {}", rustflags);
    println!("cargo:rustc-env=RUSTFLAGS='{}'", rustflags);
}
