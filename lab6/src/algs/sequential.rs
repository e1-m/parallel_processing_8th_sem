pub fn trapezoidal(a: f64, b: f64, n: usize, f: fn(f64) -> f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let h = (b - a) / (n as f64);
    let mut sum = 0.5 * (f(a) + f(b));

    for i in 1..n {
        sum += f(a + (i as f64) * h);
    }

    sum * h
}

pub fn simpson(a: f64, b: f64, n: usize, f: fn(f64) -> f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let h = (b - a) / (n as f64);
    let mut sum = 0.0;

    for i in 0..n {
        let x_start = a + (i as f64) * h;
        let x_end = x_start + h;
        let mid = (x_start + x_end) / 2.0;

        sum += (h / 6.0) * (f(x_start) + 4.0 * f(mid) + f(x_end));
    }

    sum
}
