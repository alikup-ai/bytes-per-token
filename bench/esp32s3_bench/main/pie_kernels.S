// ESP32-S3 PIE matrix-vector kernels for the roofline bench.
//
//   void pie_gemv_s8(const int8_t *W, const int8_t *x, int32_t *y, int rows, int n16)
//   void pie_gemv_q4(const uint8_t *W, const int8_t *x, int32_t *y, int rows, int n32)
//
// s8: row-major int8 weights, 16 per EE.VMULAS step.
// q4: row-planar nibbles; each 16-byte block holds 32 weights biased by +8:
//     low nibble of byte j -> weight j, high nibble -> weight 16 + j.
// W and x must be 16-byte aligned; rows >= 1. y[r] is the exact int32 dot
// (ACCX is 40-bit; a 96..512-long int8 dot cannot overflow 32 bits).
// Syntax cross-checked against a published, hardware-tested S3 kernel
// (github.com/iamankushpandit/esp_ai components/story_llm/dot_q4_pie.S);
// the code below is written independently.

    .section .rodata
    .align  4
.Lq4c:
    .byte   0x0F        // nibble mask
    .byte   0x08        // code bias

    .text
    .align  4
    .global pie_gemv_s8
    .type   pie_gemv_s8, @function
pie_gemv_s8:
    entry   a1, 32
.Ls8_row:
    ee.zero.accx
    mov     a8, a3
    loopnez a6, .Ls8_col
    ee.vld.128.ip q0, a2, 16
    ee.vld.128.ip q1, a8, 16
    ee.vmulas.s8.accx q0, q1
.Ls8_col:
    rur.accx_0 a9
    s32i    a9, a4, 0
    addi    a4, a4, 4
    addi    a5, a5, -1
    bnez    a5, .Ls8_row
    retw.n
    .size   pie_gemv_s8, . - pie_gemv_s8

    .align  4
    .global pie_gemv_q4
    .type   pie_gemv_q4, @function
pie_gemv_q4:
    entry   a1, 32
    movi    a8, .Lq4c
    ee.vldbc.8 q7, a8
    addi    a8, a8, 1
    ee.vldbc.8 q6, a8
    ssai    4
.Lq4_row:
    ee.zero.accx
    mov     a8, a3
    loopnez a6, .Lq4_col
    ee.vld.128.ip q0, a2, 16
    ee.vsr.32   q1, q0
    ee.andq     q0, q0, q7
    ee.andq     q1, q1, q7
    ee.vsubs.s8 q0, q0, q6
    ee.vsubs.s8 q1, q1, q6
    ee.vld.128.ip q2, a8, 16
    ee.vld.128.ip q3, a8, 16
    ee.vmulas.s8.accx q0, q2
    ee.vmulas.s8.accx q1, q3
.Lq4_col:
    rur.accx_0 a9
    s32i    a9, a4, 0
    addi    a4, a4, 4
    addi    a5, a5, -1
    bnez    a5, .Lq4_row
    retw.n
    .size   pie_gemv_q4, . - pie_gemv_q4
