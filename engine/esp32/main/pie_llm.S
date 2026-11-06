// ESP32-S3 PIE kernels for the engine. Semantics match llm_ref_q4g /
// llm_ref_s8 in engine/src/llm.c exactly (integer results, no rounding).
//
//   void pie_gemv_q4g(const uint8_t *W, const int8_t *x, int32_t *y, int rows, int groups, int bpg)
//       One int32 per group of bpg 32-weight blocks: y[r*groups+g] = sum (nibble-8) * x.
//       bpg == 2 (our models, 64-weight groups) takes the original unrolled path.
//       A group is two 16-byte blocks; block byte j: low nibble -> weight j,
//       high nibble -> weight 16+j (the layout bench/pie_kernels.S validated).
//   void pie_gemv_s8s(const int8_t *W, const int8_t *x, int32_t *y, int rows, int n16, int stride)
//       y[r] = sum_{j<16*n16} W[r*stride+j] * x[j]   (attention: K q and V^T p)
// W, x and every row start must be 16-byte aligned.

    .section .rodata
    .align  4
.Lq4c:
    .byte   0x0F        // nibble mask
    .byte   0x08        // code bias

    .text
    .align  4
    .global pie_gemv_q4g
    .type   pie_gemv_q4g, @function
pie_gemv_q4g:
    entry   a1, 32
    movi    a8, .Lq4c
    ee.vldbc.8 q7, a8
    addi    a8, a8, 1
    ee.vldbc.8 q6, a8
    ssai    4
    movi    a9, 2
    bne     a7, a9, .Lv_row             // any other group length: general path below
.Lg_row:
    mov     a8, a3                      // x restarts every row
    loopnez a6, .Lg_grp
    ee.zero.accx
    ee.vld.128.ip q0, a2, 16            // block 0: weights 0..31 of the group
    ee.vsr.32   q1, q0
    ee.andq     q0, q0, q7
    ee.andq     q1, q1, q7
    ee.vsubs.s8 q0, q0, q6
    ee.vsubs.s8 q1, q1, q6
    ee.vld.128.ip q2, a8, 16
    ee.vld.128.ip q3, a8, 16
    ee.vmulas.s8.accx q0, q2
    ee.vmulas.s8.accx q1, q3
    ee.vld.128.ip q0, a2, 16            // block 1: weights 32..63
    ee.vsr.32   q1, q0
    ee.andq     q0, q0, q7
    ee.andq     q1, q1, q7
    ee.vsubs.s8 q0, q0, q6
    ee.vsubs.s8 q1, q1, q6
    ee.vld.128.ip q2, a8, 16
    ee.vld.128.ip q3, a8, 16
    ee.vmulas.s8.accx q0, q2
    ee.vmulas.s8.accx q1, q3
    rur.accx_0 a9
    s32i    a9, a4, 0
    addi    a4, a4, 4
.Lg_grp:
    addi    a5, a5, -1
    bnez    a5, .Lg_row
    retw.n
.Lv_row:                                // general path: groups of a7 blocks
    mov     a8, a3
    mov     a10, a6
.Lv_grp:
    ee.zero.accx
    loopnez a7, .Lv_blk
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
.Lv_blk:
    rur.accx_0 a9
    s32i    a9, a4, 0
    addi    a4, a4, 4
    addi    a10, a10, -1
    bnez    a10, .Lv_grp
    addi    a5, a5, -1
    bnez    a5, .Lv_row
    retw.n
    .size   pie_gemv_q4g, . - pie_gemv_q4g

    // void pie_colsum_q4(const uint8_t *const *rows, const int8_t *z, int n, int off, int half, uint8_t *raw)
    //   16 lanes of QACC += z[k] * (nibble - 8) over k < n, taking from each row the
    //   16-byte block at rows[k] + off: its low nibbles (half 0) or high nibbles (half 1).
    //   raw (64 bytes, 16-aligned) receives QACC: L lanes 0-7 at bytes 0-19 (128 + 32 bits),
//   H lanes 8-15 at bytes 32-51, each lane a 20-bit two's complement field;
    //   engine/esp32/main/main.c unpacks it (and its unit test pins down the lane order).
    .align  4
    .global pie_colsum_q4
    .type   pie_colsum_q4, @function
pie_colsum_q4:
    entry   a1, 32
    movi    a8, .Lq4c
    ee.vldbc.8 q7, a8                   // nibble mask
    addi    a8, a8, 1
    ee.vldbc.8 q6, a8                   // code bias
    ssai    4
    ee.zero.qacc
    beqz    a4, .Lc_store
    bnez    a6, .Lc_hi
.Lc_lo:
    l32i    a9, a2, 0
    addi    a2, a2, 4
    add     a9, a9, a5
    ee.vld.128.ip q0, a9, 0
    ee.andq     q0, q0, q7
    ee.vsubs.s8 q0, q0, q6
    ee.vldbc.8.ip q1, a3, 1
    ee.vmulas.s8.qacc q0, q1
    addi    a4, a4, -1
    bnez    a4, .Lc_lo
    j       .Lc_store
.Lc_hi:
    l32i    a9, a2, 0
    addi    a2, a2, 4
    add     a9, a9, a5
    ee.vld.128.ip q0, a9, 0
    ee.vsr.32   q0, q0
    ee.andq     q0, q0, q7
    ee.vsubs.s8 q0, q0, q6
    ee.vldbc.8.ip q1, a3, 1
    ee.vmulas.s8.qacc q0, q1
    addi    a4, a4, -1
    bnez    a4, .Lc_hi
.Lc_store:
    // 128-bit stores align their address down to 16 bytes, so each part gets
    // its own 16-byte slot: L low at 0, L high at 16, H low at 32, H high at 48.
    ee.st.qacc_l.l.128.ip a7, 16
    ee.st.qacc_l.h.32.ip  a7, 16
    ee.st.qacc_h.l.128.ip a7, 16
    ee.st.qacc_h.h.32.ip  a7, 0
    retw.n
    .size   pie_colsum_q4, . - pie_colsum_q4

    .align  4
    .global pie_gemv_s8s
    .type   pie_gemv_s8s, @function
pie_gemv_s8s:
    entry   a1, 32
    slli    a10, a6, 4                  // bytes consumed per row
    sub     a10, a7, a10                // skip to the next row start
.Ls_row:
    ee.zero.accx
    mov     a8, a3
    loopnez a6, .Ls_col
    ee.vld.128.ip q0, a2, 16
    ee.vld.128.ip q1, a8, 16
    ee.vmulas.s8.accx q0, q1
.Ls_col:
    rur.accx_0 a9
    s32i    a9, a4, 0
    addi    a4, a4, 4
    add     a2, a2, a10
    addi    a5, a5, -1
    bnez    a5, .Ls_row
    retw.n
    .size   pie_gemv_s8s, . - pie_gemv_s8s
