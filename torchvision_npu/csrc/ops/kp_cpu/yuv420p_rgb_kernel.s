	.arch armv8-a
	.text
	.align	2
	.p2align 4,,11
	.type	kp_yuv420p_to_rgb_full._omp_fn.0, %function
kp_yuv420p_to_rgb_full._omp_fn.0:
.LFB4344:
	stp	x29, x30, [sp,#-304]!
	adrp	x1, :got:__stack_chk_guard
	mov	x29, sp
	ldr	x1, [x1, #:got_lo12:__stack_chk_guard]
	stp	x21, x22, [sp,#32]
	ldr	w21, [x0,#40]
	stp	x19, x20, [sp,#16]
	mov	x20, x0
	str	w21, [sp,#264]
	ldr	x0, [x1]
	str	x0, [sp,#296]
	mov	x0, #0x0                   	// #0
	bl	omp_get_num_threads
	mov	w19, w0
	bl	omp_get_thread_num
	sdiv	w1, w21, w19
	msub	w2, w1, w19, w21
	cmp	w0, w2
	blt	.L2
.L38:
	madd	w3, w1, w0, w2
	str	w3, [sp,#252]
	add	w0, w1, w3
	str	w0, [sp,#268]
	cmp	w3, w0
	bge	.L1
	ldp	w30, w0, [x20,#44]
	stp	x23, x24, [sp,#48]
	sub	w1, w30, #0x11
	stp	x25, x26, [sp,#64]
	add	w0, w0, w0, lsr #31
	ldp	w4, w13, [x20,#52]
	asr	w5, w0, #1
	and	w0, w1, #0xfffffff0
	stp	x27, x28, [sp,#80]
	add	w0, w0, #0x10
	mul	w1, w3, w5
	ldr	w3, [x20,#68]
	str	w3, [sp,#172]
	lsl	w2, w4, #1
	ldr	w3, [sp,#264]
	sub	w19, w30, #0x10
	str	w1, [sp,#256]
	stp	d8, d9, [sp,#96]
	sdiv	w1, w1, w3
	stp	d10, d11, [sp,#112]
	stp	d12, d13, [sp,#128]
	stp	d14, d15, [sp,#144]
	str	w0, [sp,#248]
	sxtw	x0, w2
	str	w1, [sp,#260]
	str	w5, [sp,#272]
	str	w4, [sp,#276]
	ldp	w23, w22, [x20,#60]
	ldp	w24, w27, [x20,#72]
	ldp	w25, w26, [x20,#80]
	str	x0, [sp,#240]
	movi	v23.8h, #0xe9
	sxtw	x0, w4
	movi	v22.8h, #0x66
	str	x0, [sp,#192]
	ldr	x0, [x20]
	str	x0, [sp,#232]
	ldr	x0, [x20,#8]
	str	x0, [sp,#224]
	ldr	x0, [x20,#16]
	str	x0, [sp,#216]
	mvni	v21.8h, #0xb2
	mvni	v7.8h, #0xa
	mvni	v20.8h, #0x6c
	movi	v19.8h, #0x89
	movi	v18.8h, #0xa1
	movi	v17.8h, #0xc5
	mvni	v16.8h, #0xe1
	ldr	x0, [x20,#24]
	str	x0, [sp,#208]
	ldr	x0, [x20,#32]
	str	x0, [sp,#280]
	.p2align 3,,7
.L5:
	ldr	w0, [sp,#256]
	ldr	w1, [sp,#272]
	add	w1, w0, w1
	ldr	w0, [sp,#260]
	str	w1, [sp,#256]
	lsl	w2, w0, #1
	str	w2, [sp,#200]
	ldr	w2, [sp,#252]
	add	w2, w2, #0x1
	str	w2, [sp,#252]
	ldr	w2, [sp,#264]
	sdiv	w1, w1, w2
	lsl	w2, w1, #1
	str	w2, [sp,#204]
	str	w1, [sp,#260]
	cmp	w0, w1
	bge	.L9
	ldr	w1, [sp,#200]
	ldr	w0, [sp,#276]
	mul	w0, w0, w1
	sxtw	x1, w1
	str	x1, [sp,#184]
	ldr	x1, [sp,#280]
	add	x0, x1, w0, sxtw
	str	x0, [sp,#176]
	.p2align 3,,7
.L8:
	ldp	x0, x2, [sp,#224]
	ldr	w1, [sp,#200]
	asr	w15, w1, #1
	ldp	x17, x16, [x0]
	sxtw	x15, w15
	ldr	x1, [sp,#184]
	ldr	x0, [x0,#16]
	madd	x18, x1, x17, x17
	madd	x17, x1, x17, x2
	ldr	x1, [sp,#216]
	add	x18, x2, x18
	madd	x16, x15, x16, x1
	ldr	x1, [sp,#208]
	madd	x15, x15, x0, x1
	cmp	w19, #0x0
	ble	.L45
	ldr	x2, [sp,#176]
	add	x7, x17, #0x40
	ldr	x0, [sp,#192]
	add	x6, x18, #0x40
	mov	x1, #0x0                   	// #0
	add	x3, x0, x2
	.p2align 3,,7
.L36:
	asr	w0, w1, #1
	add	x5, x1, x17
	add	x4, x1, x18
	prfm	pldl1keep, [x7,x1]
	prfm	pldl1keep, [x6,x1]
	add	x1, x1, #0x10
	ldr	d0, [x15,w0,sxtw]
	ldr	d2, [x16,w0,sxtw]
	sxtw	x0, w0
	ld2	{v10.8b, v11.8b}, [x5]
	add	x0, x0, #0x40
	uxtl	v0.8h, v0.8b
	uxtl	v2.8h, v2.8b
	ld2	{v8.8b, v9.8b}, [x4]
	mul	v3.8h, v0.8h, v23.8h
	prfm	pldl1keep, [x16,x0]
	shl	v6.8h, v2.8h, #2
	prfm	pldl1keep, [x15,x0]
	mul	v1.8h, v0.8h, v20.8h
	mul	v24.8h, v2.8h, v18.8h
	ushr	v3.8h, v3.8h, #8
	sub	v6.8h, v2.8h, v6.8h
	sshr	v1.8h, v1.8h, #8
	uxtl	v5.8h, v10.8b
	mla	v3.8h, v0.8h, v22.8h
	sshr	v6.8h, v6.8h, #8
	mla	v1.8h, v0.8h, v7.8h
	uxtl	v4.8h, v11.8b
	usra	v0.8h, v3.8h, #8
	mla	v6.8h, v2.8h, v7.8h
	sshr	v1.8h, v1.8h, #4
	uxtl	v3.8h, v8.8b
	add	v0.8h, v0.8h, v21.8h
	uxtl	v8.8h, v9.8b
	ushr	v24.8h, v24.8h, #8
	ssra	v1.8h, v6.8h, #5
	add	v11.8h, v5.8h, v0.8h
	add	v9.8h, v4.8h, v0.8h
	add	v10.8h, v3.8h, v0.8h
	mla	v24.8h, v2.8h, v17.8h
	add	v0.8h, v8.8h, v0.8h
	add	v1.8h, v1.8h, v19.8h
	sqxtun	v11.8b, v11.8h
	sqxtun	v9.8b, v9.8h
	sqxtun	v0.8b, v0.8h
	sqxtun	v10.8b, v10.8h
	usra	v2.8h, v24.8h, #8
	add	v6.8h, v1.8h, v5.8h
	add	v13.8h, v1.8h, v4.8h
	zip1	v12.8b, v11.8b, v9.8b
	zip1	v24.8b, v10.8b, v0.8b
	add	v2.8h, v2.8h, v16.8h
	zip2	v9.8b, v11.8b, v9.8b
	zip2	v10.8b, v10.8b, v0.8b
	sqxtun	v6.8b, v6.8h
	sqxtun	v13.8b, v13.8h
	mov	d12, v12.d[0]
	add	v4.8h, v2.8h, v4.8h
	mov	d0, v24.d[0]
	add	v24.8h, v2.8h, v5.8h
	add	v15.8h, v1.8h, v3.8h
	zip1	v31.8b, v6.8b, v13.8b
	mov	v12.d[1], v9.d[0]
	add	v1.8h, v1.8h, v8.8h
	add	v9.8h, v2.8h, v3.8h
	add	v2.8h, v2.8h, v8.8h
	sqxtun	v14.8b, v4.8h
	sqxtun	v24.8b, v24.8h
	mov	v0.d[1], v10.d[0]
	zip2	v6.8b, v6.8b, v13.8b
	sqxtun	v15.8b, v15.8h
	sqxtun	v1.8b, v1.8h
	mov	d10, v31.d[0]
	sqxtun	v9.8b, v9.8h
	sqxtun	v31.8b, v2.8h
	zip1	v8.8b, v24.8b, v14.8b
	zip1	v11.8b, v15.8b, v1.8b
	mov	v25.16b, v0.16b
	zip2	v24.8b, v24.8b, v14.8b
	zip2	v1.8b, v15.8b, v1.8b
	zip1	v0.8b, v9.8b, v31.8b
	zip2	v9.8b, v9.8b, v31.8b
	mov	d8, v8.d[0]
	mov	d11, v11.d[0]
	mov	v10.d[1], v6.d[0]
	mov	d0, v0.d[0]
	mov	v8.d[1], v24.d[0]
	mov	v11.d[1], v1.d[0]
	mov	v4.16b, v12.16b
	mov	v0.d[1], v9.d[0]
	mov	v5.16b, v10.16b
	mov	v6.16b, v8.16b
	mov	v1.16b, v25.16b
	mov	v2.16b, v11.16b
	mov	v3.16b, v0.16b
	st3	{v4.16b-v6.16b}, [x2], #48
	st3	{v1.16b-v3.16b}, [x3], #48
	cmp	w19, w1
	bgt	.L36
	ldr	w9, [sp,#248]
.L37:
	cmp	w30, w9
	ble	.L34
	add	w10, w9, w9, lsl #1
	add	x21, x17, #0x1
	ldr	x0, [sp,#192]
	sxtw	x10, w10
	sxtw	x9, w9
	add	x20, x18, #0x1
	add	x11, x0, x10
	ldr	x0, [sp,#176]
	add	x10, x0, x10
	add	x11, x0, x11
	.p2align 3,,7
.L35:
	asr	w1, w9, #1
	ldr	w6, [sp,#172]
	ldrb	w4, [x17,x9]
	ldrb	w5, [x21,x9]
	ldrb	w0, [x15,w1,sxtw]
	ldrb	w1, [x16,w1,sxtw]
	mul	w4, w4, w13
	ldrb	w3, [x18,x9]
	mul	w5, w5, w13
	ldrb	w14, [x20,x9]
	mul	w2, w23, w0
	mul	w6, w6, w1
	mul	w0, w24, w0
	asr	w2, w2, #16
	mul	w1, w22, w1
	asr	w6, w6, #16
	mul	w3, w3, w13
	add	w0, w6, w0, asr #16
	mul	w14, w14, w13
	madd	w2, w2, w13, w27
	asr	w1, w1, #16
	madd	w0, w0, w13, w25
	add	w7, w2, w4
	madd	w1, w1, w13, w26
	add	w12, w2, w5
	add	w6, w0, w4
	asr	w7, w7, #16
	mvn	w8, w7
	add	w4, w1, w4
	asr	w6, w6, #16
	tst	w7, #0xffffff00
	mvn	w28, w6
	asr	w8, w8, #31
	and	w8, w8, #0xff
	and	w7, w7, #0xff
	asr	w28, w28, #31
	csel	w7, w7, w8, eq
	asr	w4, w4, #16
	tst	w6, #0xffffff00
	and	w28, w28, #0xff
	and	w6, w6, #0xff
	csel	w6, w6, w28, eq
	mvn	w28, w4
	add	w8, w0, w5
	asr	w12, w12, #16
	asr	w28, w28, #31
	tst	w4, #0xffffff00
	asr	w8, w8, #16
	and	w28, w28, #0xff
	and	w4, w4, #0xff
	strb	w7, [x10]
	mvn	w7, w12
	csel	w4, w4, w28, eq
	mvn	w28, w8
	add	w5, w1, w5
	strb	w6, [x10,#1]
	asr	w6, w7, #31
	tst	w12, #0xffffff00
	and	w6, w6, #0xff
	and	w12, w12, #0xff
	asr	w28, w28, #31
	asr	w5, w5, #16
	csel	w12, w12, w6, eq
	and	w28, w28, #0xff
	tst	w8, #0xffffff00
	and	w8, w8, #0xff
	add	w7, w2, w3
	csel	w8, w8, w28, eq
	mvn	w28, w5
	add	w6, w0, w3
	asr	w7, w7, #16
	add	w3, w1, w3
	asr	w28, w28, #31
	asr	w6, w6, #16
	tst	w5, #0xffffff00
	add	w2, w2, w14
	add	w0, w0, w14
	and	w5, w5, #0xff
	add	w1, w1, w14
	and	w28, w28, #0xff
	mvn	w14, w7
	asr	w3, w3, #16
	csel	w28, w5, w28, eq
	mvn	w5, w6
	asr	w14, w14, #31
	strb	w4, [x10,#2]
	mvn	w4, w3
	tst	w7, #0xffffff00
	and	w14, w14, #0xff
	and	w7, w7, #0xff
	asr	w5, w5, #31
	asr	w2, w2, #16
	csel	w7, w7, w14, eq
	and	w5, w5, #0xff
	tst	w6, #0xffffff00
	asr	w4, w4, #31
	and	w6, w6, #0xff
	asr	w0, w0, #16
	csel	w6, w6, w5, eq
	and	w4, w4, #0xff
	tst	w3, #0xffffff00
	mvn	w5, w2
	and	w3, w3, #0xff
	csel	w3, w3, w4, eq
	mvn	w4, w0
	asr	w5, w5, #31
	tst	w2, #0xffffff00
	and	w5, w5, #0xff
	and	w2, w2, #0xff
	asr	w4, w4, #31
	csel	w2, w2, w5, eq
	and	w4, w4, #0xff
	tst	w0, #0xffffff00
	and	w0, w0, #0xff
	strb	w12, [x10,#3]
	strb	w8, [x10,#4]
	csel	w0, w0, w4, eq
	strb	w28, [x10,#5]
	asr	w1, w1, #16
	strb	w7, [x11]
	strb	w6, [x11,#1]
	strb	w3, [x11,#2]
	strb	w2, [x11,#3]
	strb	w0, [x11,#4]
	tst	w1, #0xffffff00
	beq	.L32
	mvn	w1, w1
	add	x9, x9, #0x2
	add	x10, x10, #0x6
	add	x11, x11, #0x6
	asr	w1, w1, #31
	sturb	w1, [x11,#-1]
	cmp	w30, w9
	bgt	.L35
.L34:
	ldr	x1, [sp,#184]
	ldr	x2, [sp,#240]
	add	x1, x1, #0x2
	str	x1, [sp,#184]
	ldr	x1, [sp,#176]
	ldr	w0, [sp,#200]
	add	x1, x1, x2
	str	x1, [sp,#176]
	ldr	w1, [sp,#204]
	add	w0, w0, #0x2
	str	w0, [sp,#200]
	cmp	w1, w0
	bgt	.L8
.L9:
	ldr	w1, [sp,#252]
	ldr	w0, [sp,#268]
	cmp	w0, w1
	bne	.L5
	ldp	x23, x24, [sp,#48]
	ldp	x25, x26, [sp,#64]
	ldp	x27, x28, [sp,#80]
	ldp	d8, d9, [sp,#96]
	ldp	d10, d11, [sp,#112]
	ldp	d12, d13, [sp,#128]
	ldp	d14, d15, [sp,#144]
.L1:
	adrp	x0, :got:__stack_chk_guard
	ldr	x0, [x0, #:got_lo12:__stack_chk_guard]
	ldr	x2, [sp,#296]
	ldr	x1, [x0]
	subs	x2, x2, x1
	mov	x1, #0x0                   	// #0
	bne	.L46
	ldp	x19, x20, [sp,#16]
	ldp	x21, x22, [sp,#32]
	ldp	x29, x30, [sp],#304
	ret
	.p2align 2,,3
.L32:
	add	x9, x9, #0x2
	strb	w1, [x11,#5]
	add	x10, x10, #0x6
	add	x11, x11, #0x6
	cmp	w30, w9
	bgt	.L35
	b	.L34
	.p2align 2,,3
.L45:
	mov	w9, #0x0                   	// #0
	b	.L37
.L2:
	add	w1, w1, #0x1
	mov	w2, #0x0                   	// #0
	b	.L38
.L46:
	stp	x23, x24, [sp,#48]
	stp	x25, x26, [sp,#64]
	stp	x27, x28, [sp,#80]
	stp	d8, d9, [sp,#96]
	stp	d10, d11, [sp,#112]
	stp	d12, d13, [sp,#128]
	stp	d14, d15, [sp,#144]
	bl	__stack_chk_fail
.LFE4344:
	.size	kp_yuv420p_to_rgb_full._omp_fn.0, .-kp_yuv420p_to_rgb_full._omp_fn.0
	.align	2
	.p2align 4,,11
	.type	kp_yuv420p_to_rgb_limit._omp_fn.0, %function
kp_yuv420p_to_rgb_limit._omp_fn.0:
.LFB4345:
	stp	x29, x30, [sp,#-304]!
	adrp	x1, :got:__stack_chk_guard
	mov	x29, sp
	ldr	x1, [x1, #:got_lo12:__stack_chk_guard]
	stp	x21, x22, [sp,#32]
	ldr	w21, [x0,#40]
	stp	x19, x20, [sp,#16]
	mov	x20, x0
	str	w21, [sp,#264]
	ldr	x0, [x1]
	str	x0, [sp,#296]
	mov	x0, #0x0                   	// #0
	bl	omp_get_num_threads
	mov	w19, w0
	bl	omp_get_thread_num
	sdiv	w1, w21, w19
	msub	w2, w1, w19, w21
	cmp	w0, w2
	blt	.L48
.L84:
	madd	w3, w1, w0, w2
	str	w3, [sp,#252]
	add	w0, w1, w3
	str	w0, [sp,#268]
	cmp	w3, w0
	bge	.L47
	ldp	w30, w0, [x20,#44]
	stp	x23, x24, [sp,#48]
	sub	w1, w30, #0x11
	stp	x25, x26, [sp,#64]
	add	w0, w0, w0, lsr #31
	ldp	w4, w13, [x20,#52]
	asr	w5, w0, #1
	and	w0, w1, #0xfffffff0
	stp	x27, x28, [sp,#80]
	add	w0, w0, #0x10
	mul	w1, w3, w5
	ldr	w3, [x20,#68]
	str	w3, [sp,#172]
	lsl	w2, w4, #1
	ldr	w3, [sp,#264]
	sub	w19, w30, #0x10
	str	w1, [sp,#256]
	stp	d8, d9, [sp,#96]
	sdiv	w1, w1, w3
	stp	d10, d11, [sp,#112]
	stp	d12, d13, [sp,#128]
	stp	d14, d15, [sp,#144]
	str	w0, [sp,#248]
	sxtw	x0, w2
	str	w1, [sp,#260]
	str	w5, [sp,#272]
	str	w4, [sp,#276]
	ldp	w23, w22, [x20,#60]
	ldp	w24, w27, [x20,#72]
	ldp	w25, w26, [x20,#80]
	str	x0, [sp,#240]
	movi	v6.8h, #0x15
	adrp	x0, .LC1
	ldr	q31, [x0, #:lo12:.LC1]
	sxtw	x0, w4
	str	x0, [sp,#192]
	ldr	x0, [x20]
	str	x0, [sp,#232]
	ldr	x0, [x20,#8]
	str	x0, [sp,#224]
	ldr	x0, [x20,#16]
	str	x0, [sp,#216]
	ldr	x0, [x20,#24]
	str	x0, [sp,#208]
	ldr	x0, [x20,#32]
	str	x0, [sp,#280]
	.p2align 3,,7
.L51:
	ldr	w0, [sp,#256]
	ldr	w1, [sp,#272]
	add	w1, w0, w1
	ldr	w0, [sp,#260]
	str	w1, [sp,#256]
	lsl	w2, w0, #1
	str	w2, [sp,#200]
	ldr	w2, [sp,#252]
	add	w2, w2, #0x1
	str	w2, [sp,#252]
	ldr	w2, [sp,#264]
	sdiv	w1, w1, w2
	lsl	w2, w1, #1
	str	w2, [sp,#204]
	str	w1, [sp,#260]
	cmp	w0, w1
	bge	.L55
	ldr	w1, [sp,#200]
	mvni	v11.8h, #0x1d
	ldr	w0, [sp,#276]
	mul	w0, w0, w1
	sxtw	x1, w1
	str	x1, [sp,#184]
	adrp	x1, .LC2
	ldr	q10, [x1, #:lo12:.LC2]
	ldr	x1, [sp,#280]
	add	x0, x1, w0, sxtw
	str	x0, [sp,#176]
	adrp	x0, .LC3
	ldr	q5, [x0, #:lo12:.LC3]
	.p2align 3,,7
.L54:
	ldr	w0, [sp,#200]
	ldr	x1, [sp,#184]
	asr	w15, w0, #1
	ldp	x0, x2, [sp,#224]
	sxtw	x15, w15
	ldp	x17, x16, [x0]
	ldr	x0, [x0,#16]
	madd	x18, x1, x17, x17
	madd	x17, x1, x17, x2
	ldr	x1, [sp,#216]
	add	x18, x2, x18
	madd	x16, x15, x16, x1
	ldr	x1, [sp,#208]
	madd	x15, x15, x0, x1
	cmp	w19, #0x0
	ble	.L90
	ldr	x2, [sp,#176]
	add	x7, x17, #0x40
	ldr	x0, [sp,#192]
	add	x6, x18, #0x40
	mov	x1, #0x0                   	// #0
	add	x3, x0, x2
	.p2align 3,,7
.L82:
	asr	w0, w1, #1
	add	x5, x1, x17
	movi	v1.8h, #0x73
	add	x4, x1, x18
	mov	v20.16b, v11.16b
	prfm	pldl1keep, [x7,x1]
	ldr	d13, [x15,w0,sxtw]
	ldr	d0, [x16,w0,sxtw]
	sxtw	x0, w0
	ld2	{v22.8b, v23.8b}, [x5]
	add	x0, x0, #0x40
	uxtl	v13.8h, v13.8b
	prfm	pldl1keep, [x6,x1]
	uxtl	v0.8h, v0.8b
	add	x1, x1, #0x10
	uxtl	v3.8h, v22.8b
	prfm	pldl1keep, [x16,x0]
	uxtl	v12.8h, v23.8b
	prfm	pldl1keep, [x15,x0]
	mul	v15.8h, v13.8h, v1.8h
	adrp	x0, .LC4
	mvni	v1.8h, #0x20
	shl	v4.8h, v0.8h, #1
	ld2	{v16.8b, v17.8b}, [x4]
	mul	v2.8h, v0.8h, v1.8h
	mvni	v1.8h, #0x2e
	ushr	v15.8h, v15.8h, #8
	add	v4.8h, v4.8h, v0.8h
	sshr	v2.8h, v2.8h, #9
	mul	v14.8h, v13.8h, v1.8h
	movi	v1.8h, #0x2f
	ushr	v4.8h, v4.8h, #2
	uxtl	v23.8h, v16.8b
	sshr	v14.8h, v14.8h, #7
	mla	v15.8h, v13.8h, v1.8h
	mvni	v1.8h, #0x2a
	uxtl	v22.8h, v17.8b
	mla	v2.8h, v0.8h, v1.8h
	mov	v1.16b, v13.16b
	sshr	v2.8h, v2.8h, #7
	usra	v1.8h, v15.8h, #7
	mvni	v15.8h, #0x58
	add	v30.8h, v12.8h, v1.8h
	mla	v14.8h, v13.8h, v15.8h
	movi	v13.8h, #0x5d
	add	v15.8h, v3.8h, v1.8h
	mul	v25.8h, v30.8h, v6.8h
	ssra	v2.8h, v14.8h, #7
	mla	v4.8h, v0.8h, v13.8h
	mul	v26.8h, v15.8h, v6.8h
	add	v14.8h, v23.8h, v1.8h
	add	v17.8h, v2.8h, v3.8h
	usra	v0.8h, v4.8h, #7
	ldr	q4, [x0, #:lo12:.LC4]
	add	v16.8h, v2.8h, v12.8h
	add	v1.8h, v22.8h, v1.8h
	add	v13.8h, v26.8h, v4.8h
	mov	v4.16b, v31.16b
	mul	v24.8h, v14.8h, v6.8h
	add	v3.8h, v0.8h, v3.8h
	add	v12.8h, v0.8h, v12.8h
	ssra	v4.8h, v13.8h, #9
	ldr	q13, [x0, #:lo12:.LC4]
	mul	v18.8h, v1.8h, v6.8h
	add	v26.8h, v26.8h, v4.8h
	add	v27.8h, v25.8h, v13.8h
	add	v4.8h, v0.8h, v23.8h
	add	v13.8h, v2.8h, v23.8h
	mov	v23.16b, v31.16b
	add	v2.8h, v2.8h, v22.8h
	add	v0.8h, v0.8h, v22.8h
	ldr	q22, [x0, #:lo12:.LC4]
	ssra	v23.8h, v27.8h, #9
	ssra	v15.8h, v26.8h, #7
	add	v22.8h, v24.8h, v22.8h
	ldr	q26, [x0, #:lo12:.LC4]
	adrp	x0, .LC5
	add	v25.8h, v25.8h, v23.8h
	mov	v23.16b, v31.16b
	add	v26.8h, v18.8h, v26.8h
	mul	v27.8h, v16.8h, v6.8h
	ssra	v30.8h, v25.8h, #7
	ssra	v23.8h, v22.8h, #9
	mov	v22.16b, v31.16b
	sqxtun	v15.8b, v15.8h
	sqxtun	v30.8b, v30.8h
	add	v24.8h, v24.8h, v23.8h
	ssra	v22.8h, v26.8h, #9
	mul	v23.8h, v17.8h, v6.8h
	mul	v28.8h, v13.8h, v6.8h
	mul	v25.8h, v3.8h, v6.8h
	add	v18.8h, v18.8h, v22.8h
	mul	v26.8h, v2.8h, v6.8h
	ssra	v14.8h, v24.8h, #7
	mul	v24.8h, v12.8h, v6.8h
	ssra	v1.8h, v18.8h, #7
	ldr	q18, [x0, #:lo12:.LC5]
	adrp	x0, .LC6
	sqxtun	v14.8b, v14.8h
	ldr	q22, [x0, #:lo12:.LC6]
	adrp	x0, .LC5
	add	v18.8h, v23.8h, v18.8h
	sqxtun	v1.8b, v1.8h
	ssra	v22.8h, v18.8h, #9
	ldr	q18, [x0, #:lo12:.LC5]
	adrp	x0, .LC6
	add	v22.8h, v23.8h, v22.8h
	add	v29.8h, v27.8h, v18.8h
	mul	v23.8h, v4.8h, v6.8h
	zip1	v18.8b, v15.8b, v30.8b
	zip2	v15.8b, v15.8b, v30.8b
	ldr	q30, [x0, #:lo12:.LC6]
	adrp	x0, .LC5
	ssra	v17.8h, v22.8h, #7
	ldr	q22, [x0, #:lo12:.LC5]
	adrp	x0, .LC6
	ssra	v30.8h, v29.8h, #9
	mov	d18, v18.d[0]
	add	v22.8h, v28.8h, v22.8h
	ldr	q29, [x0, #:lo12:.LC6]
	adrp	x0, .LC5
	mov	v18.d[1], v15.d[0]
	add	v30.8h, v27.8h, v30.8h
	ssra	v29.8h, v22.8h, #9
	ldr	q15, [x0, #:lo12:.LC5]
	adrp	x0, .LC6
	mov	v19.16b, v18.16b
	ssra	v16.8h, v30.8h, #7
	add	v18.8h, v28.8h, v29.8h
	add	v28.8h, v25.8h, v10.8h
	sqxtun	v17.8b, v17.8h
	add	v15.8h, v26.8h, v15.8h
	sqxtun	v16.8b, v16.8h
	ssra	v20.8h, v28.8h, #9
	ldr	q29, [x0, #:lo12:.LC6]
	mul	v27.8h, v0.8h, v6.8h
	zip1	v22.8b, v17.8b, v16.8b
	add	v25.8h, v25.8h, v20.8h
	ssra	v29.8h, v15.8h, #9
	mov	v20.16b, v11.16b
	zip2	v15.8b, v17.8b, v16.8b
	mov	v16.16b, v19.16b
	add	v19.8h, v24.8h, v10.8h
	add	v21.8h, v27.8h, v10.8h
	add	v26.8h, v26.8h, v29.8h
	ssra	v13.8h, v18.8h, #7
	ssra	v20.8h, v19.8h, #9
	add	v19.8h, v23.8h, v10.8h
	ssra	v2.8h, v26.8h, #7
	ssra	v3.8h, v25.8h, #7
	add	v24.8h, v24.8h, v20.8h
	mov	v20.16b, v11.16b
	sqxtun	v2.8b, v2.8h
	sqxtun	v13.8b, v13.8h
	ssra	v12.8h, v24.8h, #7
	ssra	v20.8h, v19.8h, #9
	mov	v19.16b, v11.16b
	add	v3.8h, v3.8h, v5.8h
	add	v12.8h, v12.8h, v5.8h
	add	v23.8h, v23.8h, v20.8h
	ssra	v19.8h, v21.8h, #9
	sqxtun	v3.8b, v3.8h
	sqxtun	v12.8b, v12.8h
	mov	d22, v22.d[0]
	add	v27.8h, v27.8h, v19.8h
	zip1	v19.8b, v14.8b, v1.8b
	zip2	v14.8b, v14.8b, v1.8b
	mov	v1.16b, v4.16b
	zip1	v4.8b, v13.8b, v2.8b
	zip2	v13.8b, v13.8b, v2.8b
	ssra	v0.8h, v27.8h, #7
	mov	d19, v19.d[0]
	ssra	v1.8h, v23.8h, #7
	mov	v22.d[1], v15.d[0]
	add	v0.8h, v0.8h, v5.8h
	mov	v19.d[1], v14.d[0]
	add	v1.8h, v1.8h, v5.8h
	mov	d14, v4.d[0]
	sqxtun	v0.8b, v0.8h
	zip1	v4.8b, v3.8b, v12.8b
	zip2	v3.8b, v3.8b, v12.8b
	sqxtun	v1.8b, v1.8h
	mov	v14.d[1], v13.d[0]
	mov	v17.16b, v22.16b
	mov	d4, v4.d[0]
	zip1	v2.8b, v1.8b, v0.8b
	zip2	v1.8b, v1.8b, v0.8b
	mov	v7.16b, v19.16b
	mov	v8.16b, v14.16b
	mov	v4.d[1], v3.d[0]
	mov	d0, v2.d[0]
	mov	v18.16b, v4.16b
	mov	v0.d[1], v1.d[0]
	st3	{v16.16b-v18.16b}, [x2], #48
	mov	v9.16b, v0.16b
	st3	{v7.16b-v9.16b}, [x3], #48
	cmp	w19, w1
	bgt	.L82
	ldr	w9, [sp,#248]
.L83:
	cmp	w30, w9
	ble	.L80
	add	w10, w9, w9, lsl #1
	add	x21, x17, #0x1
	ldr	x0, [sp,#192]
	sxtw	x10, w10
	sxtw	x9, w9
	add	x20, x18, #0x1
	add	x11, x0, x10
	ldr	x0, [sp,#176]
	add	x10, x0, x10
	add	x11, x0, x11
	.p2align 3,,7
.L81:
	asr	w1, w9, #1
	ldr	w6, [sp,#172]
	ldrb	w4, [x17,x9]
	ldrb	w5, [x21,x9]
	ldrb	w0, [x15,w1,sxtw]
	ldrb	w1, [x16,w1,sxtw]
	mul	w4, w4, w13
	ldrb	w3, [x18,x9]
	mul	w5, w5, w13
	ldrb	w14, [x20,x9]
	mul	w2, w23, w0
	mul	w6, w6, w1
	mul	w0, w24, w0
	asr	w2, w2, #16
	mul	w1, w22, w1
	asr	w6, w6, #16
	mul	w3, w3, w13
	add	w0, w6, w0, asr #16
	mul	w14, w14, w13
	madd	w2, w2, w13, w27
	asr	w1, w1, #16
	madd	w0, w0, w13, w25
	add	w7, w2, w4
	madd	w1, w1, w13, w26
	add	w12, w2, w5
	add	w6, w0, w4
	asr	w7, w7, #16
	mvn	w8, w7
	add	w4, w1, w4
	asr	w6, w6, #16
	tst	w7, #0xffffff00
	mvn	w28, w6
	asr	w8, w8, #31
	and	w8, w8, #0xff
	and	w7, w7, #0xff
	asr	w28, w28, #31
	csel	w7, w7, w8, eq
	asr	w4, w4, #16
	tst	w6, #0xffffff00
	and	w28, w28, #0xff
	and	w6, w6, #0xff
	csel	w6, w6, w28, eq
	mvn	w28, w4
	add	w8, w0, w5
	asr	w12, w12, #16
	asr	w28, w28, #31
	tst	w4, #0xffffff00
	asr	w8, w8, #16
	and	w28, w28, #0xff
	and	w4, w4, #0xff
	strb	w7, [x10]
	mvn	w7, w12
	csel	w4, w4, w28, eq
	mvn	w28, w8
	add	w5, w1, w5
	strb	w6, [x10,#1]
	asr	w6, w7, #31
	tst	w12, #0xffffff00
	and	w6, w6, #0xff
	and	w12, w12, #0xff
	asr	w28, w28, #31
	asr	w5, w5, #16
	csel	w12, w12, w6, eq
	and	w28, w28, #0xff
	tst	w8, #0xffffff00
	and	w8, w8, #0xff
	add	w7, w2, w3
	csel	w8, w8, w28, eq
	mvn	w28, w5
	add	w6, w0, w3
	asr	w7, w7, #16
	add	w3, w1, w3
	asr	w28, w28, #31
	asr	w6, w6, #16
	tst	w5, #0xffffff00
	add	w2, w2, w14
	add	w0, w0, w14
	and	w5, w5, #0xff
	add	w1, w1, w14
	and	w28, w28, #0xff
	mvn	w14, w7
	asr	w3, w3, #16
	csel	w28, w5, w28, eq
	mvn	w5, w6
	asr	w14, w14, #31
	strb	w4, [x10,#2]
	mvn	w4, w3
	tst	w7, #0xffffff00
	and	w14, w14, #0xff
	and	w7, w7, #0xff
	asr	w5, w5, #31
	asr	w2, w2, #16
	csel	w7, w7, w14, eq
	and	w5, w5, #0xff
	tst	w6, #0xffffff00
	asr	w4, w4, #31
	and	w6, w6, #0xff
	asr	w0, w0, #16
	csel	w6, w6, w5, eq
	and	w4, w4, #0xff
	tst	w3, #0xffffff00
	mvn	w5, w2
	and	w3, w3, #0xff
	csel	w3, w3, w4, eq
	mvn	w4, w0
	asr	w5, w5, #31
	tst	w2, #0xffffff00
	and	w5, w5, #0xff
	and	w2, w2, #0xff
	asr	w4, w4, #31
	csel	w2, w2, w5, eq
	and	w4, w4, #0xff
	tst	w0, #0xffffff00
	and	w0, w0, #0xff
	strb	w12, [x10,#3]
	strb	w8, [x10,#4]
	csel	w0, w0, w4, eq
	strb	w28, [x10,#5]
	asr	w1, w1, #16
	strb	w7, [x11]
	strb	w6, [x11,#1]
	strb	w3, [x11,#2]
	strb	w2, [x11,#3]
	strb	w0, [x11,#4]
	tst	w1, #0xffffff00
	beq	.L78
	mvn	w1, w1
	add	x9, x9, #0x2
	add	x10, x10, #0x6
	add	x11, x11, #0x6
	asr	w1, w1, #31
	sturb	w1, [x11,#-1]
	cmp	w30, w9
	bgt	.L81
.L80:
	ldr	x1, [sp,#184]
	ldr	x2, [sp,#240]
	add	x1, x1, #0x2
	str	x1, [sp,#184]
	ldr	x1, [sp,#176]
	ldr	w0, [sp,#200]
	add	x1, x1, x2
	str	x1, [sp,#176]
	ldr	w1, [sp,#204]
	add	w0, w0, #0x2
	str	w0, [sp,#200]
	cmp	w1, w0
	bgt	.L54
.L55:
	ldr	w1, [sp,#252]
	ldr	w0, [sp,#268]
	cmp	w0, w1
	bne	.L51
	ldp	x23, x24, [sp,#48]
	ldp	x25, x26, [sp,#64]
	ldp	x27, x28, [sp,#80]
	ldp	d8, d9, [sp,#96]
	ldp	d10, d11, [sp,#112]
	ldp	d12, d13, [sp,#128]
	ldp	d14, d15, [sp,#144]
.L47:
	adrp	x0, :got:__stack_chk_guard
	ldr	x0, [x0, #:got_lo12:__stack_chk_guard]
	ldr	x2, [sp,#296]
	ldr	x1, [x0]
	subs	x2, x2, x1
	mov	x1, #0x0                   	// #0
	bne	.L91
	ldp	x19, x20, [sp,#16]
	ldp	x21, x22, [sp,#32]
	ldp	x29, x30, [sp],#304
	ret
	.p2align 2,,3
.L78:
	add	x9, x9, #0x2
	strb	w1, [x11,#5]
	add	x10, x10, #0x6
	add	x11, x11, #0x6
	cmp	w30, w9
	bgt	.L81
	b	.L80
	.p2align 2,,3
.L90:
	mov	w9, #0x0                   	// #0
	b	.L83
.L48:
	add	w1, w1, #0x1
	mov	w2, #0x0                   	// #0
	b	.L84
.L91:
	stp	x23, x24, [sp,#48]
	stp	x25, x26, [sp,#64]
	stp	x27, x28, [sp,#80]
	stp	d8, d9, [sp,#96]
	stp	d10, d11, [sp,#112]
	stp	d12, d13, [sp,#128]
	stp	d14, d15, [sp,#144]
	bl	__stack_chk_fail
.LFE4345:
	.size	kp_yuv420p_to_rgb_limit._omp_fn.0, .-kp_yuv420p_to_rgb_limit._omp_fn.0
	.align	2
	.p2align 4,,11
	.global	kp_yuv420p_to_rgb_full
	.hidden	kp_yuv420p_to_rgb_full
	.type	kp_yuv420p_to_rgb_full, %function
kp_yuv420p_to_rgb_full:
.LFB4341:
	fmov	s1, w0
	stp	x29, x30, [sp,#-112]!
	add	w9, w1, w1, lsl #1
	mov	v0.16b, v1.16b
	adrp	x0, :got:__stack_chk_guard
	mov	x29, sp
	ldr	x0, [x0, #:got_lo12:__stack_chk_guard]
	mov	x8, x3
	mov	v0.s[1], w1
	adrp	x1, .LC7
	mov	w3, #0x0                   	// #0
	ldr	q3, [x1, #:lo12:.LC7]
	adrp	x1, .LC8
	mov	v0.s[2], w2
	fmov	w2, s1
	ldr	q2, [x1, #:lo12:.LC8]
	adrp	x1, kp_yuv420p_to_rgb_full._omp_fn.0
	mov	v0.s[3], w9
	ldr	x9, [x0]
	str	x9, [sp,#104]
	mov	x9, #0x0                   	// #0
	add	x0, x1, :lo12:kp_yuv420p_to_rgb_full._omp_fn.0
	add	x1, sp, #0x10
	stp	x8, x4, [sp,#16]
	stp	x5, x6, [sp,#32]
	str	x7, [sp,#48]
	stur	q0, [sp,#56]
	stur	q3, [sp,#72]
	stur	q2, [sp,#88]
	bl	GOMP_parallel
	adrp	x0, :got:__stack_chk_guard
	ldr	x0, [x0, #:got_lo12:__stack_chk_guard]
	ldr	x2, [sp,#104]
	ldr	x1, [x0]
	subs	x2, x2, x1
	mov	x1, #0x0                   	// #0
	bne	.L95
	ldp	x29, x30, [sp],#112
	ret
.L95:
	bl	__stack_chk_fail
.LFE4341:
	.size	kp_yuv420p_to_rgb_full, .-kp_yuv420p_to_rgb_full
	.align	2
	.p2align 4,,11
	.global	kp_yuv420p_to_rgb_limit
	.hidden	kp_yuv420p_to_rgb_limit
	.type	kp_yuv420p_to_rgb_limit, %function
kp_yuv420p_to_rgb_limit:
.LFB4342:
	fmov	s1, w0
	stp	x29, x30, [sp,#-112]!
	add	w9, w1, w1, lsl #1
	mov	v0.16b, v1.16b
	adrp	x0, :got:__stack_chk_guard
	mov	x29, sp
	ldr	x0, [x0, #:got_lo12:__stack_chk_guard]
	mov	x8, x3
	mov	v0.s[1], w1
	adrp	x1, .LC9
	mov	w3, #0x0                   	// #0
	ldr	q3, [x1, #:lo12:.LC9]
	adrp	x1, .LC10
	mov	v0.s[2], w2
	fmov	w2, s1
	ldr	q2, [x1, #:lo12:.LC10]
	adrp	x1, kp_yuv420p_to_rgb_limit._omp_fn.0
	mov	v0.s[3], w9
	ldr	x9, [x0]
	str	x9, [sp,#104]
	mov	x9, #0x0                   	// #0
	add	x0, x1, :lo12:kp_yuv420p_to_rgb_limit._omp_fn.0
	add	x1, sp, #0x10
	stp	x8, x4, [sp,#16]
	stp	x5, x6, [sp,#32]
	str	x7, [sp,#48]
	stur	q0, [sp,#56]
	stur	q3, [sp,#72]
	stur	q2, [sp,#88]
	bl	GOMP_parallel
	adrp	x0, :got:__stack_chk_guard
	ldr	x0, [x0, #:got_lo12:__stack_chk_guard]
	ldr	x2, [sp,#104]
	ldr	x1, [x0]
	subs	x2, x2, x1
	mov	x1, #0x0                   	// #0
	bne	.L99
	ldp	x29, x30, [sp],#112
	ret
.L99:
	bl	__stack_chk_fail
.LFE4342:
	.size	kp_yuv420p_to_rgb_limit, .-kp_yuv420p_to_rgb_limit
	.section	.rodata.cst16,"aM",@progbits,16
	.align	4
.LC1:
	.hword	-28630
	.hword	-28630
	.hword	-28630
	.hword	-28630
	.hword	-28630
	.hword	-28630
	.hword	-28630
	.hword	-28630
	.align	4
.LC2:
	.hword	-355
	.hword	-355
	.hword	-355
	.hword	-355
	.hword	-355
	.hword	-355
	.hword	-355
	.hword	-355
	.align	4
.LC3:
	.hword	-277
	.hword	-277
	.hword	-277
	.hword	-277
	.hword	-277
	.hword	-277
	.hword	-277
	.hword	-277
	.align	4
.LC4:
	.hword	-413
	.hword	-413
	.hword	-413
	.hword	-413
	.hword	-413
	.hword	-413
	.hword	-413
	.hword	-413
	.align	4
.LC5:
	.hword	444
	.hword	444
	.hword	444
	.hword	444
	.hword	444
	.hword	444
	.hword	444
	.hword	444
	.align	4
.LC6:
	.hword	17422
	.hword	17422
	.hword	17422
	.hword	17422
	.hword	17422
	.hword	17422
	.hword	17422
	.hword	17422
	.align	4
.LC7:
	.word	65536
	.word	91881
	.word	116129
	.word	-22552
	.align	4
.LC8:
	.word	-46800
	.word	-11698176
	.word	9011200
	.word	-14778368
	.align	4
.LC9:
	.word	76309
	.word	89830
	.word	113537
	.word	-22049
	.align	4
.LC10:
	.word	-45756
	.word	-14658973
	.word	8920508
	.word	-18169187
