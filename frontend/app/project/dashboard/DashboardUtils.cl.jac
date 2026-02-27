"""Dashboard formatting utility functions."""

import from ..shared.MetricsAdapter { to_num }

def:pub format_op(raw: str) -> str {
    raw_text = raw or "";
    prefix = "torch_nn_functional_";
    if raw_text.startsWith(prefix) {
        return "torch.nn.functional." + raw_text.replace(prefix, "");
    }
    return raw_text.replace("_", ".");
}

def:pub format_op_short(raw: str) -> str {
    full = format_op(raw);
    prefix = "torch.nn.functional.";
    if full.startsWith(prefix) {
        return full.replace(prefix, "");
    }
    return full;
}

def:pub format_tick_op(value: any) -> str {
    return format_op_short(str(value));
}

def:pub format_tick_ms(value: any) -> str {
    num = to_num(value);
    if num is None {
        return "0 ms";
    }
    if num <= 0 {
        return "0 ms";
    }
    if num < 0.000001 {
        return f"{num.toExponential(2)} ms";
    }
    if num < 0.001 {
        return f"{(num * 1000).toFixed(3)} µs";
    }
    if num < 0.01 {
        return f"{num.toFixed(5)} ms";
    }
    if num < 1 {
        return f"{num.toFixed(3)} ms";
    }
    return f"{num.toFixed(2)} ms";
}

def:pub format_tick_ratio(value: any) -> str {
    num = to_num(value);
    if num is None {
        return "--";
    }
    return f"{num.toFixed(2)}x";
}
