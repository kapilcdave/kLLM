# AArch64 Q4_0 matvec ceiling

Date: 2026-11-17

kLLM reduces Q4_0 weights against Q8_0 activations during batch-one decode.
The portable path performs 32 scalar integer products per 18-byte weight block.
The AArch64 path unpacks all 32 nibbles with NEON and reduces them as four
eight-lane widening products, removing the scalar inner loop without changing
the integer arithmetic or stored formats.

The traffic-only ceiling is unchanged because both paths read the same Q4_0
weights and prepared activations. The measured gain therefore comes from using
the vector ALUs more efficiently, not from reading fewer bytes. A hardware
counter profile is still needed to separate compute and memory stalls.
