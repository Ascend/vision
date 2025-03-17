	.arch armv8-a
	.text
	.align	2
	.p2align 4,,11
	.global	kp_memcpy_fast
	.hidden	kp_memcpy_fast
	.type	kp_memcpy_fast, %function
kp_memcpy_fast:
.LFB5209:
	stp	x29, x30, [sp,#-64]!
	adrp	x3, :got:__stack_chk_guard
	mov	x29, sp
	ldr	x3, [x3, #:got_lo12:__stack_chk_guard]
	stp	x19, x20, [sp,#16]
	mov	x19, x1
	stp	x21, x22, [sp,#32]
	mov	x21, x0
	mov	x22, x2
	ldr	x0, [x3]
	str	x0, [sp,#56]
	mov	x0, #0x0                   	// #0
	cmp	x2, #0x80
	bhi	.L2
	sub	x3, x2, #0x1
	add	x19, x1, x2
	add	x1, x21, x2
	cmp	x3, #0x7f
	bhi	.L3
	cmp	w3, #0x7f
	bls	.L477
.L3:
.L278:
	adrp	x1, :got:__stack_chk_guard
	ldr	x1, [x1, #:got_lo12:__stack_chk_guard]
	ldr	x0, [sp,#56]
	ldr	x2, [x1]
	subs	x0, x0, x2
	mov	x2, #0x0                   	// #0
	bne	.L478
	mov	x0, x21
	ldp	x19, x20, [sp,#16]
	ldp	x21, x22, [sp,#32]
	ldp	x29, x30, [sp],#64
	ret
.L2:
	neg	x1, x21
	mov	x20, x21
	ands	x1, x1, #0xf
	beq	.L133
	ldr	q0, [x19]
	sub	x22, x2, x1
	add	x20, x21, x1
	add	x19, x19, x1
	str	q0, [x21]
.L133:
	mov	x0, #0x200                 	// #512
	bl	_Znwm
	mov	x1, x0
	movi	v0.4s, #0x0
	add	x3, x0, #0x200
	str	q0, [x1],#16
	cmp	x1, x3
	beq	.L134
	add	x1, x0, #0x20
	str	q0, [x0,#16]
	.p2align 3,,7
.L135:
	ldr	q0, [x0]
	str	q0, [x1],#16
	cmp	x1, x3
	bne	.L135
.L134:
	cmp	x22, #0x180, lsl #12
	bls	.L411
	prfm	pldl1strm, [x19]
.L411:
	cmp	x22, #0x1ff
	bls	.L479
	.p2align 3,,7
.L140:
	mov	x3, #0x0                   	// #0
	.p2align 3,,7
.L138:
	ldr	q0, [x19,x3]
	str	q0, [x0,x3]
	add	x3, x3, #0x10
	cmp	x3, #0x200
	bne	.L138
	prfm	pldl1strm, [x19,#1024]
	add	x19, x19, #0x200
	mov	x3, #0x0                   	// #0
	.p2align 3,,7
.L139:
	ldr	q0, [x0,x3]
	str	q0, [x20,x3]
	add	x3, x3, #0x10
	cmp	x3, #0x200
	bne	.L139
	sub	x22, x22, #0x200
	add	x20, x20, #0x200
	cmp	x22, #0x1ff
	bhi	.L140
.L479:
	cmp	x22, #0x7f
	bls	.L141
	ldr	q0, [x19]
	add	x1, x19, #0x100
	sub	x2, x22, #0x80
	add	x3, x19, #0x80
	add	x4, x20, #0x80
	prfm	pldl1strm, [x1]
	str	q0, [x0]
	ldr	q1, [x19,#16]
	str	q1, [x0,#16]
	ldr	q1, [x19,#32]
	str	q1, [x0,#32]
	ldr	q1, [x19,#48]
	str	q1, [x0,#48]
	ldr	q1, [x19,#64]
	str	q1, [x0,#64]
	ldr	q1, [x19,#80]
	str	q1, [x0,#80]
	ldr	q1, [x19,#96]
	str	q1, [x0,#96]
	ldr	q1, [x19,#112]
	str	q1, [x0,#112]
	str	q0, [x20]
	ldr	q0, [x0,#16]
	str	q0, [x20,#16]
	ldr	q0, [x0,#32]
	str	q0, [x20,#32]
	ldr	q0, [x0,#48]
	str	q0, [x20,#48]
	ldr	q0, [x0,#64]
	str	q0, [x20,#64]
	ldr	q0, [x0,#80]
	str	q0, [x20,#80]
	ldr	q0, [x0,#96]
	str	q0, [x20,#96]
	ldr	q0, [x0,#112]
	str	q0, [x20,#112]
	cmp	x2, #0x7f
	bls	.L277
	ldr	q0, [x19,#128]
	add	x2, x19, #0x180
	sub	x3, x22, #0x100
	add	x4, x20, #0x100
	prfm	pldl1strm, [x2]
	str	q0, [x0]
	ldr	q1, [x19,#144]
	str	q1, [x0,#16]
	ldr	q1, [x19,#160]
	str	q1, [x0,#32]
	ldr	q1, [x19,#176]
	str	q1, [x0,#48]
	ldr	q1, [x19,#192]
	str	q1, [x0,#64]
	ldr	q1, [x19,#208]
	str	q1, [x0,#80]
	ldr	q1, [x19,#224]
	str	q1, [x0,#96]
	ldr	q1, [x19,#240]
	str	q1, [x0,#112]
	str	q0, [x20,#128]
	ldr	q0, [x0,#16]
	str	q0, [x20,#144]
	ldr	q0, [x0,#32]
	str	q0, [x20,#160]
	ldr	q0, [x0,#48]
	str	q0, [x20,#176]
	ldr	q0, [x0,#64]
	str	q0, [x20,#192]
	ldr	q0, [x0,#80]
	str	q0, [x20,#208]
	ldr	q0, [x0,#96]
	str	q0, [x20,#224]
	ldr	q0, [x0,#112]
	str	q0, [x20,#240]
	cmp	x3, #0x7f
	bls	.L480
	ldr	q0, [x1]
	sub	x22, x22, #0x180
	prfm	pldl1strm, [x19,#512]
	add	x20, x20, #0x180
	str	q0, [x0]
	ldr	q1, [x19,#272]
	str	q1, [x0,#16]
	ldr	q1, [x19,#288]
	str	q1, [x0,#32]
	ldr	q1, [x19,#304]
	str	q1, [x0,#48]
	ldr	q1, [x19,#320]
	str	q1, [x0,#64]
	ldr	q1, [x19,#336]
	str	q1, [x0,#80]
	ldr	q1, [x19,#352]
	str	q1, [x0,#96]
	ldr	q1, [x19,#368]
	mov	x19, x2
	str	q1, [x0,#112]
	stur	q0, [x20,#-128]
	ldr	q0, [x0,#16]
	stur	q0, [x20,#-112]
	ldr	q0, [x0,#32]
	stur	q0, [x20,#-96]
	ldr	q0, [x0,#48]
	stur	q0, [x20,#-80]
	ldr	q0, [x0,#64]
	stur	q0, [x20,#-64]
	ldr	q0, [x0,#80]
	stur	q0, [x20,#-48]
	ldr	q0, [x0,#96]
	stur	q0, [x20,#-32]
	ldr	q0, [x0,#112]
	stur	q0, [x20,#-16]
.L141:
	sub	x1, x22, #0x1
	add	x20, x20, x22
	add	x19, x19, x22
	cmp	x1, #0x7e
	bhi	.L145
	cmp	w1, #0x7e
	bhi	.L145
	adrp	x2, .L147
	add	x2, x2, :lo12:.L147
	ldrh	w2, [x2,w1,uxtw #1]
	adr	x1, .Lrtx147
	add	x2, x1, w2, sxth #2
	br	x2
.Lrtx147:
	.section	.rodata
	.align	0
	.align	2
.L147:
	.2byte	(.L412 - .Lrtx147) / 4
	.2byte	(.L272 - .Lrtx147) / 4
	.2byte	(.L271 - .Lrtx147) / 4
	.2byte	(.L413 - .Lrtx147) / 4
	.2byte	(.L269 - .Lrtx147) / 4
	.2byte	(.L268 - .Lrtx147) / 4
	.2byte	(.L267 - .Lrtx147) / 4
	.2byte	(.L418 - .Lrtx147) / 4
	.2byte	(.L265 - .Lrtx147) / 4
	.2byte	(.L264 - .Lrtx147) / 4
	.2byte	(.L263 - .Lrtx147) / 4
	.2byte	(.L262 - .Lrtx147) / 4
	.2byte	(.L261 - .Lrtx147) / 4
	.2byte	(.L260 - .Lrtx147) / 4
	.2byte	(.L259 - .Lrtx147) / 4
	.2byte	(.L425 - .Lrtx147) / 4
	.2byte	(.L257 - .Lrtx147) / 4
	.2byte	(.L256 - .Lrtx147) / 4
	.2byte	(.L255 - .Lrtx147) / 4
	.2byte	(.L254 - .Lrtx147) / 4
	.2byte	(.L253 - .Lrtx147) / 4
	.2byte	(.L252 - .Lrtx147) / 4
	.2byte	(.L251 - .Lrtx147) / 4
	.2byte	(.L250 - .Lrtx147) / 4
	.2byte	(.L249 - .Lrtx147) / 4
	.2byte	(.L248 - .Lrtx147) / 4
	.2byte	(.L247 - .Lrtx147) / 4
	.2byte	(.L246 - .Lrtx147) / 4
	.2byte	(.L245 - .Lrtx147) / 4
	.2byte	(.L244 - .Lrtx147) / 4
	.2byte	(.L439 - .Lrtx147) / 4
	.2byte	(.L242 - .Lrtx147) / 4
	.2byte	(.L241 - .Lrtx147) / 4
	.2byte	(.L240 - .Lrtx147) / 4
	.2byte	(.L239 - .Lrtx147) / 4
	.2byte	(.L238 - .Lrtx147) / 4
	.2byte	(.L237 - .Lrtx147) / 4
	.2byte	(.L236 - .Lrtx147) / 4
	.2byte	(.L235 - .Lrtx147) / 4
	.2byte	(.L234 - .Lrtx147) / 4
	.2byte	(.L233 - .Lrtx147) / 4
	.2byte	(.L232 - .Lrtx147) / 4
	.2byte	(.L231 - .Lrtx147) / 4
	.2byte	(.L230 - .Lrtx147) / 4
	.2byte	(.L229 - .Lrtx147) / 4
	.2byte	(.L228 - .Lrtx147) / 4
	.2byte	(.L227 - .Lrtx147) / 4
	.2byte	(.L226 - .Lrtx147) / 4
	.2byte	(.L225 - .Lrtx147) / 4
	.2byte	(.L224 - .Lrtx147) / 4
	.2byte	(.L223 - .Lrtx147) / 4
	.2byte	(.L222 - .Lrtx147) / 4
	.2byte	(.L221 - .Lrtx147) / 4
	.2byte	(.L220 - .Lrtx147) / 4
	.2byte	(.L219 - .Lrtx147) / 4
	.2byte	(.L218 - .Lrtx147) / 4
	.2byte	(.L217 - .Lrtx147) / 4
	.2byte	(.L216 - .Lrtx147) / 4
	.2byte	(.L215 - .Lrtx147) / 4
	.2byte	(.L214 - .Lrtx147) / 4
	.2byte	(.L213 - .Lrtx147) / 4
	.2byte	(.L212 - .Lrtx147) / 4
	.2byte	(.L211 - .Lrtx147) / 4
	.2byte	(.L210 - .Lrtx147) / 4
	.2byte	(.L209 - .Lrtx147) / 4
	.2byte	(.L208 - .Lrtx147) / 4
	.2byte	(.L207 - .Lrtx147) / 4
	.2byte	(.L206 - .Lrtx147) / 4
	.2byte	(.L205 - .Lrtx147) / 4
	.2byte	(.L204 - .Lrtx147) / 4
	.2byte	(.L203 - .Lrtx147) / 4
	.2byte	(.L202 - .Lrtx147) / 4
	.2byte	(.L201 - .Lrtx147) / 4
	.2byte	(.L200 - .Lrtx147) / 4
	.2byte	(.L199 - .Lrtx147) / 4
	.2byte	(.L198 - .Lrtx147) / 4
	.2byte	(.L197 - .Lrtx147) / 4
	.2byte	(.L196 - .Lrtx147) / 4
	.2byte	(.L195 - .Lrtx147) / 4
	.2byte	(.L194 - .Lrtx147) / 4
	.2byte	(.L193 - .Lrtx147) / 4
	.2byte	(.L192 - .Lrtx147) / 4
	.2byte	(.L191 - .Lrtx147) / 4
	.2byte	(.L190 - .Lrtx147) / 4
	.2byte	(.L189 - .Lrtx147) / 4
	.2byte	(.L188 - .Lrtx147) / 4
	.2byte	(.L187 - .Lrtx147) / 4
	.2byte	(.L186 - .Lrtx147) / 4
	.2byte	(.L185 - .Lrtx147) / 4
	.2byte	(.L184 - .Lrtx147) / 4
	.2byte	(.L183 - .Lrtx147) / 4
	.2byte	(.L182 - .Lrtx147) / 4
	.2byte	(.L181 - .Lrtx147) / 4
	.2byte	(.L180 - .Lrtx147) / 4
	.2byte	(.L179 - .Lrtx147) / 4
	.2byte	(.L178 - .Lrtx147) / 4
	.2byte	(.L177 - .Lrtx147) / 4
	.2byte	(.L176 - .Lrtx147) / 4
	.2byte	(.L175 - .Lrtx147) / 4
	.2byte	(.L174 - .Lrtx147) / 4
	.2byte	(.L173 - .Lrtx147) / 4
	.2byte	(.L172 - .Lrtx147) / 4
	.2byte	(.L171 - .Lrtx147) / 4
	.2byte	(.L170 - .Lrtx147) / 4
	.2byte	(.L169 - .Lrtx147) / 4
	.2byte	(.L168 - .Lrtx147) / 4
	.2byte	(.L167 - .Lrtx147) / 4
	.2byte	(.L166 - .Lrtx147) / 4
	.2byte	(.L165 - .Lrtx147) / 4
	.2byte	(.L164 - .Lrtx147) / 4
	.2byte	(.L163 - .Lrtx147) / 4
	.2byte	(.L162 - .Lrtx147) / 4
	.2byte	(.L161 - .Lrtx147) / 4
	.2byte	(.L160 - .Lrtx147) / 4
	.2byte	(.L159 - .Lrtx147) / 4
	.2byte	(.L158 - .Lrtx147) / 4
	.2byte	(.L157 - .Lrtx147) / 4
	.2byte	(.L156 - .Lrtx147) / 4
	.2byte	(.L155 - .Lrtx147) / 4
	.2byte	(.L154 - .Lrtx147) / 4
	.2byte	(.L153 - .Lrtx147) / 4
	.2byte	(.L152 - .Lrtx147) / 4
	.2byte	(.L151 - .Lrtx147) / 4
	.2byte	(.L150 - .Lrtx147) / 4
	.2byte	(.L149 - .Lrtx147) / 4
	.2byte	(.L148 - .Lrtx147) / 4
	.2byte	(.L146 - .Lrtx147) / 4
	.text
.L477:
	adrp	x0, .L5
	add	x0, x0, :lo12:.L5
	ldrh	w0, [x0,w3,uxtw #1]
	adr	x3, .Lrtx5
	add	x0, x3, w0, sxth #2
	br	x0
.Lrtx5:
	.section	.rodata
	.align	0
	.align	2
.L5:
	.2byte	(.L447 - .Lrtx5) / 4
	.2byte	(.L131 - .Lrtx5) / 4
	.2byte	(.L450 - .Lrtx5) / 4
	.2byte	(.L448 - .Lrtx5) / 4
	.2byte	(.L128 - .Lrtx5) / 4
	.2byte	(.L127 - .Lrtx5) / 4
	.2byte	(.L451 - .Lrtx5) / 4
	.2byte	(.L452 - .Lrtx5) / 4
	.2byte	(.L124 - .Lrtx5) / 4
	.2byte	(.L123 - .Lrtx5) / 4
	.2byte	(.L122 - .Lrtx5) / 4
	.2byte	(.L121 - .Lrtx5) / 4
	.2byte	(.L120 - .Lrtx5) / 4
	.2byte	(.L119 - .Lrtx5) / 4
	.2byte	(.L459 - .Lrtx5) / 4
	.2byte	(.L460 - .Lrtx5) / 4
	.2byte	(.L116 - .Lrtx5) / 4
	.2byte	(.L115 - .Lrtx5) / 4
	.2byte	(.L463 - .Lrtx5) / 4
	.2byte	(.L113 - .Lrtx5) / 4
	.2byte	(.L112 - .Lrtx5) / 4
	.2byte	(.L475 - .Lrtx5) / 4
	.2byte	(.L466 - .Lrtx5) / 4
	.2byte	(.L109 - .Lrtx5) / 4
	.2byte	(.L108 - .Lrtx5) / 4
	.2byte	(.L107 - .Lrtx5) / 4
	.2byte	(.L106 - .Lrtx5) / 4
	.2byte	(.L105 - .Lrtx5) / 4
	.2byte	(.L104 - .Lrtx5) / 4
	.2byte	(.L103 - .Lrtx5) / 4
	.2byte	(.L474 - .Lrtx5) / 4
	.2byte	(.L101 - .Lrtx5) / 4
	.2byte	(.L100 - .Lrtx5) / 4
	.2byte	(.L99 - .Lrtx5) / 4
	.2byte	(.L98 - .Lrtx5) / 4
	.2byte	(.L97 - .Lrtx5) / 4
	.2byte	(.L96 - .Lrtx5) / 4
	.2byte	(.L95 - .Lrtx5) / 4
	.2byte	(.L94 - .Lrtx5) / 4
	.2byte	(.L93 - .Lrtx5) / 4
	.2byte	(.L92 - .Lrtx5) / 4
	.2byte	(.L91 - .Lrtx5) / 4
	.2byte	(.L90 - .Lrtx5) / 4
	.2byte	(.L89 - .Lrtx5) / 4
	.2byte	(.L88 - .Lrtx5) / 4
	.2byte	(.L87 - .Lrtx5) / 4
	.2byte	(.L86 - .Lrtx5) / 4
	.2byte	(.L85 - .Lrtx5) / 4
	.2byte	(.L84 - .Lrtx5) / 4
	.2byte	(.L83 - .Lrtx5) / 4
	.2byte	(.L82 - .Lrtx5) / 4
	.2byte	(.L81 - .Lrtx5) / 4
	.2byte	(.L80 - .Lrtx5) / 4
	.2byte	(.L79 - .Lrtx5) / 4
	.2byte	(.L78 - .Lrtx5) / 4
	.2byte	(.L77 - .Lrtx5) / 4
	.2byte	(.L76 - .Lrtx5) / 4
	.2byte	(.L75 - .Lrtx5) / 4
	.2byte	(.L74 - .Lrtx5) / 4
	.2byte	(.L73 - .Lrtx5) / 4
	.2byte	(.L72 - .Lrtx5) / 4
	.2byte	(.L71 - .Lrtx5) / 4
	.2byte	(.L70 - .Lrtx5) / 4
	.2byte	(.L69 - .Lrtx5) / 4
	.2byte	(.L68 - .Lrtx5) / 4
	.2byte	(.L67 - .Lrtx5) / 4
	.2byte	(.L66 - .Lrtx5) / 4
	.2byte	(.L65 - .Lrtx5) / 4
	.2byte	(.L64 - .Lrtx5) / 4
	.2byte	(.L63 - .Lrtx5) / 4
	.2byte	(.L62 - .Lrtx5) / 4
	.2byte	(.L61 - .Lrtx5) / 4
	.2byte	(.L60 - .Lrtx5) / 4
	.2byte	(.L59 - .Lrtx5) / 4
	.2byte	(.L58 - .Lrtx5) / 4
	.2byte	(.L57 - .Lrtx5) / 4
	.2byte	(.L56 - .Lrtx5) / 4
	.2byte	(.L55 - .Lrtx5) / 4
	.2byte	(.L54 - .Lrtx5) / 4
	.2byte	(.L53 - .Lrtx5) / 4
	.2byte	(.L52 - .Lrtx5) / 4
	.2byte	(.L51 - .Lrtx5) / 4
	.2byte	(.L50 - .Lrtx5) / 4
	.2byte	(.L49 - .Lrtx5) / 4
	.2byte	(.L48 - .Lrtx5) / 4
	.2byte	(.L47 - .Lrtx5) / 4
	.2byte	(.L46 - .Lrtx5) / 4
	.2byte	(.L45 - .Lrtx5) / 4
	.2byte	(.L44 - .Lrtx5) / 4
	.2byte	(.L43 - .Lrtx5) / 4
	.2byte	(.L42 - .Lrtx5) / 4
	.2byte	(.L41 - .Lrtx5) / 4
	.2byte	(.L40 - .Lrtx5) / 4
	.2byte	(.L39 - .Lrtx5) / 4
	.2byte	(.L38 - .Lrtx5) / 4
	.2byte	(.L37 - .Lrtx5) / 4
	.2byte	(.L36 - .Lrtx5) / 4
	.2byte	(.L35 - .Lrtx5) / 4
	.2byte	(.L34 - .Lrtx5) / 4
	.2byte	(.L33 - .Lrtx5) / 4
	.2byte	(.L32 - .Lrtx5) / 4
	.2byte	(.L31 - .Lrtx5) / 4
	.2byte	(.L30 - .Lrtx5) / 4
	.2byte	(.L29 - .Lrtx5) / 4
	.2byte	(.L28 - .Lrtx5) / 4
	.2byte	(.L27 - .Lrtx5) / 4
	.2byte	(.L26 - .Lrtx5) / 4
	.2byte	(.L25 - .Lrtx5) / 4
	.2byte	(.L24 - .Lrtx5) / 4
	.2byte	(.L23 - .Lrtx5) / 4
	.2byte	(.L22 - .Lrtx5) / 4
	.2byte	(.L21 - .Lrtx5) / 4
	.2byte	(.L20 - .Lrtx5) / 4
	.2byte	(.L19 - .Lrtx5) / 4
	.2byte	(.L18 - .Lrtx5) / 4
	.2byte	(.L17 - .Lrtx5) / 4
	.2byte	(.L16 - .Lrtx5) / 4
	.2byte	(.L15 - .Lrtx5) / 4
	.2byte	(.L14 - .Lrtx5) / 4
	.2byte	(.L13 - .Lrtx5) / 4
	.2byte	(.L12 - .Lrtx5) / 4
	.2byte	(.L11 - .Lrtx5) / 4
	.2byte	(.L10 - .Lrtx5) / 4
	.2byte	(.L9 - .Lrtx5) / 4
	.2byte	(.L8 - .Lrtx5) / 4
	.2byte	(.L7 - .Lrtx5) / 4
	.2byte	(.L6 - .Lrtx5) / 4
	.2byte	(.L4 - .Lrtx5) / 4
	.text
.L175:
	ldur	q3, [x19,#-99]
	ldur	q2, [x19,#-83]
	ldur	q1, [x19,#-67]
	ldur	q0, [x19,#-51]
	stur	q3, [x20,#-99]
	stur	q2, [x20,#-83]
	stur	q1, [x20,#-67]
	stur	q0, [x20,#-51]
.L239:
	ldur	q1, [x19,#-35]
	ldur	q0, [x19,#-19]
	stur	q1, [x20,#-35]
.L429:
	stur	q0, [x20,#-19]
.L271:
	ldurh	w1, [x19,#-3]
	sturh	w1, [x20,#-3]
.L412:
	ldurb	w1, [x19,#-1]
	sturb	w1, [x20,#-1]
.L145:
	mov	x1, #0x200                 	// #512
	bl	_ZdlPvm
	b	.L3
.L480:
	mov	x22, x3
	mov	x20, x4
	mov	x19, x1
	b	.L141
.L277:
	mov	x22, x2
	mov	x20, x4
	mov	x19, x3
	b	.L141
.L18:
	ldur	q3, [x19,#-115]
	ldur	q2, [x19,#-99]
	ldur	q1, [x19,#-83]
	ldur	q0, [x19,#-67]
	stur	q3, [x1,#-115]
	stur	q2, [x1,#-99]
	stur	q1, [x1,#-83]
	stur	q0, [x1,#-67]
.L82:
	ldur	q1, [x19,#-51]
	ldur	q0, [x19,#-35]
	stur	q1, [x1,#-51]
	stur	q0, [x1,#-35]
.L463:
	ldur	q0, [x19,#-19]
.L446:
	stur	q0, [x1,#-19]
.L450:
	ldurh	w0, [x19,#-3]
	sturh	w0, [x1,#-3]
.L447:
	ldurb	w0, [x19,#-1]
	sturb	w0, [x1,#-1]
	b	.L3
.L15:
	ldur	q3, [x19,#-118]
	ldur	q2, [x19,#-102]
	ldur	q1, [x19,#-86]
	ldur	q0, [x19,#-70]
	stur	q3, [x1,#-118]
	stur	q2, [x1,#-102]
	stur	q1, [x1,#-86]
	stur	q0, [x1,#-70]
.L79:
	ldur	q1, [x19,#-54]
	ldur	q0, [x19,#-38]
	stur	q1, [x1,#-54]
	stur	q0, [x1,#-38]
.L475:
	ldur	q0, [x19,#-22]
.L465:
	stur	q0, [x1,#-22]
.L127:
	ldurh	w0, [x19,#-2]
	ldur	w2, [x19,#-6]
	stur	w2, [x1,#-6]
	sturh	w0, [x1,#-2]
	b	.L3
.L171:
	ldur	q3, [x19,#-103]
	ldur	q2, [x19,#-87]
	ldur	q1, [x19,#-71]
	ldur	q0, [x19,#-55]
	stur	q3, [x20,#-103]
	stur	q2, [x20,#-87]
	stur	q1, [x20,#-71]
	stur	q0, [x20,#-55]
.L235:
	ldur	q1, [x19,#-39]
	ldur	q0, [x19,#-23]
	stur	q1, [x20,#-39]
.L431:
	stur	q0, [x20,#-23]
.L267:
	ldur	w1, [x19,#-7]
	stur	w1, [x20,#-7]
.L413:
	ldur	w1, [x19,#-4]
	stur	w1, [x20,#-4]
	b	.L145
.L14:
	ldur	q3, [x19,#-119]
	ldur	q2, [x19,#-103]
	ldur	q1, [x19,#-87]
	ldur	q0, [x19,#-71]
	stur	q3, [x1,#-119]
	stur	q2, [x1,#-103]
	stur	q1, [x1,#-87]
	stur	q0, [x1,#-71]
.L78:
	ldur	q1, [x19,#-55]
	ldur	q0, [x19,#-39]
	stur	q1, [x1,#-55]
	stur	q0, [x1,#-39]
.L466:
	ldur	q0, [x19,#-23]
.L442:
	stur	q0, [x1,#-23]
.L451:
	ldur	w0, [x19,#-7]
	stur	w0, [x1,#-7]
.L448:
	ldur	w0, [x19,#-4]
	stur	w0, [x1,#-4]
	b	.L3
.L172:
	ldur	q3, [x19,#-102]
	ldur	q2, [x19,#-86]
	ldur	q1, [x19,#-70]
	ldur	q0, [x19,#-54]
	stur	q3, [x20,#-102]
	stur	q2, [x20,#-86]
	stur	q1, [x20,#-70]
	stur	q0, [x20,#-54]
.L236:
	ldur	q1, [x19,#-38]
	ldur	q0, [x19,#-22]
	stur	q1, [x20,#-38]
.L416:
	stur	q0, [x20,#-22]
.L268:
	ldurh	w1, [x19,#-2]
	ldur	w2, [x19,#-6]
	stur	w2, [x20,#-6]
	sturh	w1, [x20,#-2]
	b	.L145
.L6:
	ldur	q3, [x19,#-127]
	ldur	q2, [x19,#-111]
	ldur	q1, [x19,#-95]
	ldur	q0, [x19,#-79]
	stur	q3, [x1,#-127]
	stur	q2, [x1,#-111]
	stur	q1, [x1,#-95]
	stur	q0, [x1,#-79]
.L70:
	ldur	q1, [x19,#-63]
	ldur	q0, [x19,#-47]
	stur	q1, [x1,#-63]
	stur	q0, [x1,#-47]
.L474:
	ldur	q0, [x19,#-31]
	stur	q0, [x1,#-31]
.L460:
	ldur	q0, [x19,#-16]
	stur	q0, [x1,#-16]
	b	.L3
.L146:
	ldur	q3, [x19,#-127]
	ldur	q2, [x19,#-111]
	ldur	q1, [x19,#-95]
	ldur	q0, [x19,#-79]
	stur	q3, [x20,#-127]
	stur	q2, [x20,#-111]
	stur	q1, [x20,#-95]
	stur	q0, [x20,#-79]
.L211:
	ldur	q1, [x19,#-63]
	ldur	q0, [x19,#-47]
	stur	q1, [x20,#-63]
	stur	q0, [x20,#-47]
.L439:
	ldur	q0, [x19,#-31]
	stur	q0, [x20,#-31]
.L425:
	ldur	q0, [x19,#-16]
	stur	q0, [x20,#-16]
	b	.L145
.L164:
	ldur	q3, [x19,#-110]
	ldur	q2, [x19,#-94]
	ldur	q1, [x19,#-78]
	ldur	q0, [x19,#-62]
	stur	q3, [x20,#-110]
	stur	q2, [x20,#-94]
	stur	q1, [x20,#-78]
	stur	q0, [x20,#-62]
.L228:
	ldur	q1, [x19,#-46]
	ldur	q0, [x19,#-30]
.L426:
	stur	q1, [x20,#-46]
	stur	q0, [x20,#-30]
.L260:
	ldur	x1, [x19,#-14]
	stur	x1, [x20,#-14]
.L418:
	ldur	x1, [x19,#-8]
	stur	x1, [x20,#-8]
	b	.L145
.L22:
	ldur	q3, [x19,#-111]
	ldur	q2, [x19,#-95]
	ldur	q1, [x19,#-79]
	ldur	q0, [x19,#-63]
	stur	q3, [x1,#-111]
	stur	q2, [x1,#-95]
	stur	q1, [x1,#-79]
	stur	q0, [x1,#-63]
.L86:
	ldur	q1, [x19,#-47]
	ldur	q0, [x19,#-31]
	stur	q1, [x1,#-47]
	stur	q0, [x1,#-31]
.L459:
	ldur	x0, [x19,#-15]
	stur	x0, [x1,#-15]
.L452:
	ldur	x0, [x19,#-8]
	stur	x0, [x1,#-8]
	b	.L3
.L173:
	ldur	q3, [x19,#-101]
	ldur	q2, [x19,#-85]
	ldur	q1, [x19,#-69]
	ldur	q0, [x19,#-53]
	stur	q3, [x20,#-101]
	stur	q2, [x20,#-85]
	stur	q1, [x20,#-69]
	stur	q0, [x20,#-53]
.L237:
	ldur	q1, [x19,#-37]
	ldur	q0, [x19,#-21]
	stur	q1, [x20,#-37]
.L414:
	stur	q0, [x20,#-21]
.L269:
	ldur	w1, [x19,#-5]
	stur	w1, [x20,#-5]
	ldurb	w1, [x19,#-1]
	sturb	w1, [x20,#-1]
	b	.L145
.L163:
	ldur	q3, [x19,#-111]
	ldur	q2, [x19,#-95]
	ldur	q1, [x19,#-79]
	ldur	q0, [x19,#-63]
	stur	q3, [x20,#-111]
	stur	q2, [x20,#-95]
	stur	q1, [x20,#-79]
	stur	q0, [x20,#-63]
.L227:
	ldur	q1, [x19,#-47]
	ldur	q0, [x19,#-31]
.L424:
	stur	q1, [x20,#-47]
	stur	q0, [x20,#-31]
.L259:
	ldur	x1, [x19,#-15]
	stur	x1, [x20,#-15]
	ldur	x1, [x19,#-8]
	stur	x1, [x20,#-8]
	b	.L145
.L48:
	ldur	q3, [x19,#-85]
	ldur	q2, [x19,#-69]
	ldur	q1, [x19,#-53]
	ldur	q0, [x19,#-37]
	stur	q3, [x1,#-85]
	stur	q2, [x1,#-69]
	stur	q1, [x1,#-53]
	stur	q0, [x1,#-37]
.L112:
	ldur	q0, [x19,#-21]
	stur	q0, [x1,#-21]
.L128:
	ldur	w0, [x19,#-5]
	stur	w0, [x1,#-5]
	ldurb	w0, [x19,#-1]
	sturb	w0, [x1,#-1]
	b	.L3
.L67:
	ldur	q3, [x19,#-66]
	ldur	q2, [x19,#-50]
	ldur	q1, [x19,#-34]
	ldur	q0, [x19,#-18]
	stur	q3, [x1,#-66]
	stur	q2, [x1,#-50]
	stur	q1, [x1,#-34]
	stur	q0, [x1,#-18]
.L131:
	ldurh	w0, [x19,#-2]
	sturh	w0, [x1,#-2]
	b	.L3
.L10:
	ldur	q3, [x19,#-123]
	ldur	q2, [x19,#-107]
	ldur	q1, [x19,#-91]
	ldur	q0, [x19,#-75]
	stur	q3, [x1,#-123]
	stur	q2, [x1,#-107]
	stur	q1, [x1,#-91]
	stur	q0, [x1,#-75]
.L74:
	ldur	q1, [x19,#-59]
	ldur	q0, [x19,#-43]
.L470:
	stur	q1, [x1,#-59]
	stur	q0, [x1,#-43]
.L106:
	ldur	q0, [x19,#-27]
	stur	q0, [x1,#-27]
	ldur	q0, [x19,#-16]
	stur	q0, [x1,#-16]
	b	.L3
.L9:
	ldur	q3, [x19,#-124]
	ldur	q2, [x19,#-108]
	ldur	q1, [x19,#-92]
	ldur	q0, [x19,#-76]
	stur	q3, [x1,#-124]
	stur	q2, [x1,#-108]
	stur	q1, [x1,#-92]
	stur	q0, [x1,#-76]
.L73:
	ldur	q1, [x19,#-60]
	ldur	q0, [x19,#-44]
.L471:
	stur	q1, [x1,#-60]
	stur	q0, [x1,#-44]
.L105:
	ldur	q0, [x19,#-28]
	stur	q0, [x1,#-28]
	ldur	q0, [x19,#-16]
	stur	q0, [x1,#-16]
	b	.L3
.L8:
	ldur	q3, [x19,#-125]
	ldur	q2, [x19,#-109]
	ldur	q1, [x19,#-93]
	ldur	q0, [x19,#-77]
	stur	q3, [x1,#-125]
	stur	q2, [x1,#-109]
	stur	q1, [x1,#-93]
	stur	q0, [x1,#-77]
.L72:
	ldur	q1, [x19,#-61]
	ldur	q0, [x19,#-45]
.L472:
	stur	q1, [x1,#-61]
	stur	q0, [x1,#-45]
.L104:
	ldur	q0, [x19,#-29]
	stur	q0, [x1,#-29]
	ldur	q0, [x19,#-16]
	stur	q0, [x1,#-16]
	b	.L3
.L7:
	ldur	q3, [x19,#-126]
	ldur	q2, [x19,#-110]
	ldur	q1, [x19,#-94]
	ldur	q0, [x19,#-78]
	stur	q3, [x1,#-126]
	stur	q2, [x1,#-110]
	stur	q1, [x1,#-94]
	stur	q0, [x1,#-78]
.L71:
	ldur	q1, [x19,#-62]
	ldur	q0, [x19,#-46]
.L473:
	stur	q1, [x1,#-62]
	stur	q0, [x1,#-46]
.L103:
	ldur	q0, [x19,#-30]
	stur	q0, [x1,#-30]
	ldur	q0, [x19,#-16]
	stur	q0, [x1,#-16]
	b	.L3
.L26:
	ldur	q3, [x19,#-107]
	ldur	q2, [x19,#-91]
	ldur	q1, [x19,#-75]
	ldur	q0, [x19,#-59]
	stur	q3, [x1,#-107]
	stur	q2, [x1,#-91]
	stur	q1, [x1,#-75]
	stur	q0, [x1,#-59]
.L90:
	ldur	q1, [x19,#-43]
	ldur	q0, [x19,#-27]
.L455:
	stur	q1, [x1,#-43]
	stur	q0, [x1,#-27]
.L122:
	ldur	w0, [x19,#-4]
	ldur	x2, [x19,#-11]
	stur	x2, [x1,#-11]
	stur	w0, [x1,#-4]
	b	.L3
.L25:
	ldur	q3, [x19,#-108]
	ldur	q2, [x19,#-92]
	ldur	q1, [x19,#-76]
	ldur	q0, [x19,#-60]
	stur	q3, [x1,#-108]
	stur	q2, [x1,#-92]
	stur	q1, [x1,#-76]
	stur	q0, [x1,#-60]
.L89:
	ldur	q1, [x19,#-44]
	ldur	q0, [x19,#-28]
.L456:
	stur	q1, [x1,#-44]
	stur	q0, [x1,#-28]
.L121:
	ldur	w0, [x19,#-4]
	ldur	x2, [x19,#-12]
	stur	x2, [x1,#-12]
	stur	w0, [x1,#-4]
	b	.L3
.L24:
	ldur	q3, [x19,#-109]
	ldur	q2, [x19,#-93]
	ldur	q1, [x19,#-77]
	ldur	q0, [x19,#-61]
	stur	q3, [x1,#-109]
	stur	q2, [x1,#-93]
	stur	q1, [x1,#-77]
	stur	q0, [x1,#-61]
.L88:
	ldur	q1, [x19,#-45]
	ldur	q0, [x19,#-29]
.L457:
	stur	q1, [x1,#-45]
	stur	q0, [x1,#-29]
.L120:
	ldur	w0, [x19,#-5]
	ldur	x2, [x19,#-13]
	stur	x2, [x1,#-13]
	stur	w0, [x1,#-5]
	ldurb	w0, [x19,#-1]
	sturb	w0, [x1,#-1]
	b	.L3
.L23:
	ldur	q3, [x19,#-110]
	ldur	q2, [x19,#-94]
	ldur	q1, [x19,#-78]
	ldur	q0, [x19,#-62]
	stur	q3, [x1,#-110]
	stur	q2, [x1,#-94]
	stur	q1, [x1,#-78]
	stur	q0, [x1,#-62]
.L87:
	ldur	q1, [x19,#-46]
	ldur	q0, [x19,#-30]
.L458:
	stur	q1, [x1,#-46]
	stur	q0, [x1,#-30]
.L119:
	ldur	x0, [x19,#-14]
	stur	x0, [x1,#-14]
	ldur	x0, [x19,#-8]
	stur	x0, [x1,#-8]
	b	.L3
.L13:
	ldur	q3, [x19,#-120]
	ldur	q2, [x19,#-104]
	ldur	q1, [x19,#-88]
	ldur	q0, [x19,#-72]
	stur	q3, [x1,#-120]
	stur	q2, [x1,#-104]
	stur	q1, [x1,#-88]
	stur	q0, [x1,#-72]
.L77:
	ldur	q1, [x19,#-56]
	ldur	q0, [x19,#-40]
.L467:
	stur	q1, [x1,#-56]
	stur	q0, [x1,#-40]
.L109:
	ldur	q0, [x19,#-24]
	stur	q0, [x1,#-24]
	ldur	q0, [x19,#-16]
	stur	q0, [x1,#-16]
	b	.L3
.L12:
	ldur	q3, [x19,#-121]
	ldur	q2, [x19,#-105]
	ldur	q1, [x19,#-89]
	ldur	q0, [x19,#-73]
	stur	q3, [x1,#-121]
	stur	q2, [x1,#-105]
	stur	q1, [x1,#-89]
	stur	q0, [x1,#-73]
.L76:
	ldur	q1, [x19,#-57]
	ldur	q0, [x19,#-41]
.L468:
	stur	q1, [x1,#-57]
	stur	q0, [x1,#-41]
.L108:
	ldur	q0, [x19,#-25]
	stur	q0, [x1,#-25]
	ldur	q0, [x19,#-16]
	stur	q0, [x1,#-16]
	b	.L3
.L11:
	ldur	q3, [x19,#-122]
	ldur	q2, [x19,#-106]
	ldur	q1, [x19,#-90]
	ldur	q0, [x19,#-74]
	stur	q3, [x1,#-122]
	stur	q2, [x1,#-106]
	stur	q1, [x1,#-90]
	stur	q0, [x1,#-74]
.L75:
	ldur	q1, [x19,#-58]
	ldur	q0, [x19,#-42]
.L469:
	stur	q1, [x1,#-58]
	stur	q0, [x1,#-42]
.L107:
	ldur	q0, [x19,#-26]
	stur	q0, [x1,#-26]
	ldur	q0, [x19,#-16]
	stur	q0, [x1,#-16]
	b	.L3
.L160:
	ldur	q3, [x19,#-114]
	ldur	q2, [x19,#-98]
	ldur	q1, [x19,#-82]
	ldur	q0, [x19,#-66]
	stur	q3, [x20,#-114]
	stur	q2, [x20,#-98]
	stur	q1, [x20,#-82]
	stur	q0, [x20,#-66]
.L224:
	ldur	q1, [x19,#-50]
	ldur	q0, [x19,#-34]
.L428:
	stur	q1, [x20,#-50]
	stur	q0, [x20,#-34]
.L256:
	ldur	q0, [x19,#-18]
	stur	q0, [x20,#-18]
	ldurh	w1, [x19,#-2]
	sturh	w1, [x20,#-2]
	b	.L145
.L169:
	ldur	q3, [x19,#-105]
	ldur	q2, [x19,#-89]
	ldur	q1, [x19,#-73]
	ldur	q0, [x19,#-57]
	stur	q3, [x20,#-105]
	stur	q2, [x20,#-89]
	stur	q1, [x20,#-73]
	stur	q0, [x20,#-57]
.L233:
	ldur	q1, [x19,#-41]
	ldur	q0, [x19,#-25]
.L419:
	stur	q1, [x20,#-41]
	stur	q0, [x20,#-25]
.L265:
	ldur	x1, [x19,#-9]
	stur	x1, [x20,#-9]
	ldurb	w1, [x19,#-1]
	sturb	w1, [x20,#-1]
	b	.L145
.L168:
	ldur	q3, [x19,#-106]
	ldur	q2, [x19,#-90]
	ldur	q1, [x19,#-74]
	ldur	q0, [x19,#-58]
	stur	q3, [x20,#-106]
	stur	q2, [x20,#-90]
	stur	q1, [x20,#-74]
	stur	q0, [x20,#-58]
.L232:
	ldur	q1, [x19,#-42]
	ldur	q0, [x19,#-26]
.L420:
	stur	q1, [x20,#-42]
	stur	q0, [x20,#-26]
.L264:
	ldurh	w1, [x19,#-2]
	ldur	x2, [x19,#-10]
	stur	x2, [x20,#-10]
	sturh	w1, [x20,#-2]
	b	.L145
.L37:
	ldp	q3, q2, [x19,#-96]
	ldp	q1, q0, [x19,#-64]
	stp	q3, q2, [x1,#-96]
	stp	q1, q0, [x1,#-64]
.L101:
	ldp	q1, q0, [x19,#-32]
	stp	q1, q0, [x1,#-32]
	b	.L3
.L36:
	ldur	q3, [x19,#-97]
	ldur	q2, [x19,#-81]
	ldur	q1, [x19,#-65]
	ldur	q0, [x19,#-49]
	stur	q3, [x1,#-97]
	stur	q2, [x1,#-81]
	stur	q1, [x1,#-65]
	stur	q0, [x1,#-49]
.L100:
	ldur	q1, [x19,#-33]
	ldur	q0, [x19,#-17]
	stur	q1, [x1,#-33]
	stur	q0, [x1,#-17]
	ldurb	w0, [x19,#-1]
	sturb	w0, [x1,#-1]
	b	.L3
.L35:
	ldur	q3, [x19,#-98]
	ldur	q2, [x19,#-82]
	ldur	q1, [x19,#-66]
	ldur	q0, [x19,#-50]
	stur	q3, [x1,#-98]
	stur	q2, [x1,#-82]
	stur	q1, [x1,#-66]
	stur	q0, [x1,#-50]
.L99:
	ldur	q1, [x19,#-34]
	ldur	q0, [x19,#-18]
	stur	q1, [x1,#-34]
	stur	q0, [x1,#-18]
	ldurh	w0, [x19,#-2]
	sturh	w0, [x1,#-2]
	b	.L3
.L34:
	ldur	q3, [x19,#-99]
	ldur	q2, [x19,#-83]
	ldur	q1, [x19,#-67]
	ldur	q0, [x19,#-51]
	stur	q3, [x1,#-99]
	stur	q2, [x1,#-83]
	stur	q1, [x1,#-67]
	stur	q0, [x1,#-51]
.L98:
	ldur	q1, [x19,#-35]
	ldur	q0, [x19,#-19]
	stur	q1, [x1,#-35]
	stur	q0, [x1,#-19]
	b	.L450
.L33:
	ldur	q3, [x19,#-100]
	ldur	q2, [x19,#-84]
	ldur	q1, [x19,#-68]
	ldur	q0, [x19,#-52]
	stur	q3, [x1,#-100]
	stur	q2, [x1,#-84]
	stur	q1, [x1,#-68]
	stur	q0, [x1,#-52]
.L97:
	ldur	q1, [x19,#-36]
	ldur	q0, [x19,#-20]
	stur	q1, [x1,#-36]
	stur	q0, [x1,#-20]
	ldur	w0, [x19,#-4]
	stur	w0, [x1,#-4]
	b	.L3
.L32:
	ldur	q3, [x19,#-101]
	ldur	q2, [x19,#-85]
	ldur	q1, [x19,#-69]
	ldur	q0, [x19,#-53]
	stur	q3, [x1,#-101]
	stur	q2, [x1,#-85]
	stur	q1, [x1,#-69]
	stur	q0, [x1,#-53]
.L96:
	ldur	q1, [x19,#-37]
	ldur	q0, [x19,#-21]
	stur	q1, [x1,#-37]
.L444:
	stur	q0, [x1,#-21]
	ldur	w0, [x19,#-5]
	stur	w0, [x1,#-5]
	b	.L447
.L31:
	ldur	q3, [x19,#-102]
	ldur	q2, [x19,#-86]
	ldur	q1, [x19,#-70]
	ldur	q0, [x19,#-54]
	stur	q3, [x1,#-102]
	stur	q2, [x1,#-86]
	stur	q1, [x1,#-70]
	stur	q0, [x1,#-54]
.L95:
	ldur	q1, [x19,#-38]
	ldur	q0, [x19,#-22]
	stur	q1, [x1,#-38]
	stur	q0, [x1,#-22]
	b	.L127
.L30:
	ldur	q3, [x19,#-103]
	ldur	q2, [x19,#-87]
	ldur	q1, [x19,#-71]
	ldur	q0, [x19,#-55]
	stur	q3, [x1,#-103]
	stur	q2, [x1,#-87]
	stur	q1, [x1,#-71]
	stur	q0, [x1,#-55]
.L94:
	ldur	q1, [x19,#-39]
	ldur	q0, [x19,#-23]
	stur	q1, [x1,#-39]
	stur	q0, [x1,#-23]
	b	.L451
.L29:
	ldur	q3, [x19,#-104]
	ldur	q2, [x19,#-88]
	ldur	q1, [x19,#-72]
	ldur	q0, [x19,#-56]
	stur	q3, [x1,#-104]
	stur	q2, [x1,#-88]
	stur	q1, [x1,#-72]
	stur	q0, [x1,#-56]
.L93:
	ldur	q1, [x19,#-40]
	ldur	q0, [x19,#-24]
	stur	q1, [x1,#-40]
	stur	q0, [x1,#-24]
	ldur	x0, [x19,#-8]
	stur	x0, [x1,#-8]
	b	.L3
.L28:
	ldur	q3, [x19,#-105]
	ldur	q2, [x19,#-89]
	ldur	q1, [x19,#-73]
	ldur	q0, [x19,#-57]
	stur	q3, [x1,#-105]
	stur	q2, [x1,#-89]
	stur	q1, [x1,#-73]
	stur	q0, [x1,#-57]
.L92:
	ldur	q1, [x19,#-41]
	ldur	q0, [x19,#-25]
.L453:
	stur	q1, [x1,#-41]
	stur	q0, [x1,#-25]
.L124:
	ldur	x0, [x19,#-9]
	stur	x0, [x1,#-9]
	ldurb	w0, [x19,#-1]
	sturb	w0, [x1,#-1]
	b	.L3
.L27:
	ldur	q3, [x19,#-106]
	ldur	q2, [x19,#-90]
	ldur	q1, [x19,#-74]
	ldur	q0, [x19,#-58]
	stur	q3, [x1,#-106]
	stur	q2, [x1,#-90]
	stur	q1, [x1,#-74]
	stur	q0, [x1,#-58]
.L91:
	ldur	q1, [x19,#-42]
	ldur	q0, [x19,#-26]
.L454:
	stur	q1, [x1,#-42]
	stur	q0, [x1,#-26]
.L123:
	ldurh	w0, [x19,#-2]
	ldur	x2, [x19,#-10]
	stur	x2, [x1,#-10]
	sturh	w0, [x1,#-2]
	b	.L3
.L20:
	ldur	q3, [x19,#-113]
	ldur	q2, [x19,#-97]
	ldur	q1, [x19,#-81]
	ldur	q0, [x19,#-65]
	stur	q3, [x1,#-113]
	stur	q2, [x1,#-97]
	stur	q1, [x1,#-81]
	stur	q0, [x1,#-65]
.L84:
	ldur	q1, [x19,#-49]
	ldur	q0, [x19,#-33]
.L461:
	stur	q1, [x1,#-49]
	stur	q0, [x1,#-33]
.L116:
	ldur	q0, [x19,#-17]
	stur	q0, [x1,#-17]
	ldurb	w0, [x19,#-1]
	sturb	w0, [x1,#-1]
	b	.L3
.L148:
	ldur	q3, [x19,#-126]
	ldur	q2, [x19,#-110]
	ldur	q1, [x19,#-94]
	ldur	q0, [x19,#-78]
	stur	q3, [x20,#-126]
	stur	q2, [x20,#-110]
	stur	q1, [x20,#-94]
	stur	q0, [x20,#-78]
.L212:
	ldur	q1, [x19,#-62]
	ldur	q0, [x19,#-46]
.L440:
	stur	q1, [x20,#-62]
	stur	q0, [x20,#-46]
.L244:
	ldur	q0, [x19,#-30]
	stur	q0, [x20,#-30]
	ldur	q0, [x19,#-16]
	stur	q0, [x20,#-16]
	b	.L145
.L191:
	ldur	q3, [x19,#-83]
	ldur	q2, [x19,#-67]
	ldur	q1, [x19,#-51]
	ldur	q0, [x19,#-35]
	stur	q3, [x20,#-83]
	stur	q2, [x20,#-67]
	stur	q1, [x20,#-51]
	stur	q0, [x20,#-35]
.L255:
	ldur	q0, [x19,#-19]
	stur	q0, [x20,#-19]
	b	.L271
.L21:
	ldp	q3, q2, [x19,#-112]
	ldp	q1, q0, [x19,#-80]
	stp	q3, q2, [x1,#-112]
	stp	q1, q0, [x1,#-80]
.L85:
	ldp	q1, q0, [x19,#-48]
	stp	q1, q0, [x1,#-48]
	ldur	q0, [x19,#-16]
	stur	q0, [x1,#-16]
	b	.L3
.L167:
	ldur	q3, [x19,#-107]
	ldur	q2, [x19,#-91]
	ldur	q1, [x19,#-75]
	ldur	q0, [x19,#-59]
	stur	q3, [x20,#-107]
	stur	q2, [x20,#-91]
	stur	q1, [x20,#-75]
	stur	q0, [x20,#-59]
.L231:
	ldur	q1, [x19,#-43]
	ldur	q0, [x19,#-27]
.L421:
	stur	q1, [x20,#-43]
	stur	q0, [x20,#-27]
.L263:
	ldur	w1, [x19,#-4]
	ldur	x2, [x19,#-11]
	stur	x2, [x20,#-11]
	stur	w1, [x20,#-4]
	b	.L145
.L19:
	ldur	q3, [x19,#-114]
	ldur	q2, [x19,#-98]
	ldur	q1, [x19,#-82]
	ldur	q0, [x19,#-66]
	stur	q3, [x1,#-114]
	stur	q2, [x1,#-98]
	stur	q1, [x1,#-82]
	stur	q0, [x1,#-66]
.L83:
	ldur	q1, [x19,#-50]
	ldur	q0, [x19,#-34]
.L462:
	stur	q1, [x1,#-50]
	stur	q0, [x1,#-34]
.L115:
	ldur	q0, [x19,#-18]
	stur	q0, [x1,#-18]
	ldurh	w0, [x19,#-2]
	sturh	w0, [x1,#-2]
	b	.L3
.L208:
	ldur	q3, [x19,#-66]
	ldur	q2, [x19,#-50]
	ldur	q1, [x19,#-34]
	ldur	q0, [x19,#-18]
	stur	q3, [x20,#-66]
	stur	q2, [x20,#-50]
	stur	q1, [x20,#-34]
	stur	q0, [x20,#-18]
.L272:
	ldurh	w1, [x19,#-2]
	sturh	w1, [x20,#-2]
	b	.L145
.L166:
	ldur	q3, [x19,#-108]
	ldur	q2, [x19,#-92]
	ldur	q1, [x19,#-76]
	ldur	q0, [x19,#-60]
	stur	q3, [x20,#-108]
	stur	q2, [x20,#-92]
	stur	q1, [x20,#-76]
	stur	q0, [x20,#-60]
.L230:
	ldur	q1, [x19,#-44]
	ldur	q0, [x19,#-28]
.L422:
	stur	q1, [x20,#-44]
	stur	q0, [x20,#-28]
.L262:
	ldur	w1, [x19,#-4]
	ldur	x2, [x19,#-12]
	stur	x2, [x20,#-12]
	stur	w1, [x20,#-4]
	b	.L145
.L165:
	ldur	q3, [x19,#-109]
	ldur	q2, [x19,#-93]
	ldur	q1, [x19,#-77]
	ldur	q0, [x19,#-61]
	stur	q3, [x20,#-109]
	stur	q2, [x20,#-93]
	stur	q1, [x20,#-77]
	stur	q0, [x20,#-61]
.L229:
	ldur	q1, [x19,#-45]
	ldur	q0, [x19,#-29]
.L423:
	stur	q1, [x20,#-45]
	stur	q0, [x20,#-29]
.L261:
	ldur	w1, [x19,#-5]
	ldur	x2, [x19,#-13]
	stur	x2, [x20,#-13]
	stur	w1, [x20,#-5]
	ldurb	w1, [x19,#-1]
	sturb	w1, [x20,#-1]
	b	.L145
.L152:
	ldur	q3, [x19,#-122]
	ldur	q2, [x19,#-106]
	ldur	q1, [x19,#-90]
	ldur	q0, [x19,#-74]
	stur	q3, [x20,#-122]
	stur	q2, [x20,#-106]
	stur	q1, [x20,#-90]
	stur	q0, [x20,#-74]
.L216:
	ldur	q1, [x19,#-58]
	ldur	q0, [x19,#-42]
.L436:
	stur	q1, [x20,#-58]
	stur	q0, [x20,#-42]
.L248:
	ldur	q0, [x19,#-26]
	stur	q0, [x20,#-26]
	ldur	q0, [x19,#-16]
	stur	q0, [x20,#-16]
	b	.L145
.L151:
	ldur	q3, [x19,#-123]
	ldur	q2, [x19,#-107]
	ldur	q1, [x19,#-91]
	ldur	q0, [x19,#-75]
	stur	q3, [x20,#-123]
	stur	q2, [x20,#-107]
	stur	q1, [x20,#-91]
	stur	q0, [x20,#-75]
.L215:
	ldur	q1, [x19,#-59]
	ldur	q0, [x19,#-43]
.L437:
	stur	q1, [x20,#-59]
	stur	q0, [x20,#-43]
.L247:
	ldur	q0, [x19,#-27]
	stur	q0, [x20,#-27]
	ldur	q0, [x19,#-16]
	stur	q0, [x20,#-16]
	b	.L145
.L150:
	ldur	q3, [x19,#-124]
	ldur	q2, [x19,#-108]
	ldur	q1, [x19,#-92]
	ldur	q0, [x19,#-76]
	stur	q3, [x20,#-124]
	stur	q2, [x20,#-108]
	stur	q1, [x20,#-92]
	stur	q0, [x20,#-76]
.L214:
	ldur	q1, [x19,#-60]
	ldur	q0, [x19,#-44]
.L438:
	stur	q1, [x20,#-60]
	stur	q0, [x20,#-44]
.L246:
	ldur	q0, [x19,#-28]
	stur	q0, [x20,#-28]
	ldur	q0, [x19,#-16]
	stur	q0, [x20,#-16]
	b	.L145
.L17:
	ldur	q3, [x19,#-116]
	ldur	q2, [x19,#-100]
	ldur	q1, [x19,#-84]
	ldur	q0, [x19,#-68]
	stur	q3, [x1,#-116]
	stur	q2, [x1,#-100]
	stur	q1, [x1,#-84]
	stur	q0, [x1,#-68]
.L81:
	ldur	q1, [x19,#-52]
	ldur	q0, [x19,#-36]
.L464:
	stur	q1, [x1,#-52]
	stur	q0, [x1,#-36]
.L113:
	ldur	q0, [x19,#-20]
	stur	q0, [x1,#-20]
	ldur	w0, [x19,#-4]
	stur	w0, [x1,#-4]
	b	.L3
.L154:
	ldur	q3, [x19,#-120]
	ldur	q2, [x19,#-104]
	ldur	q1, [x19,#-88]
	ldur	q0, [x19,#-72]
	stur	q3, [x20,#-120]
	stur	q2, [x20,#-104]
	stur	q1, [x20,#-88]
	stur	q0, [x20,#-72]
.L218:
	ldur	q1, [x19,#-56]
	ldur	q0, [x19,#-40]
.L433:
	stur	q1, [x20,#-56]
	stur	q0, [x20,#-40]
.L250:
	ldur	q0, [x19,#-24]
	stur	q0, [x20,#-24]
	ldur	q0, [x19,#-16]
	stur	q0, [x20,#-16]
	b	.L145
.L153:
	ldur	q3, [x19,#-121]
	ldur	q2, [x19,#-105]
	ldur	q1, [x19,#-89]
	ldur	q0, [x19,#-73]
	stur	q3, [x20,#-121]
	stur	q2, [x20,#-105]
	stur	q1, [x20,#-89]
	stur	q0, [x20,#-73]
.L217:
	ldur	q1, [x19,#-57]
	ldur	q0, [x19,#-41]
.L434:
	stur	q1, [x20,#-57]
	stur	q0, [x20,#-41]
.L249:
	ldur	q0, [x19,#-25]
	stur	q0, [x20,#-25]
	ldur	q0, [x19,#-16]
	stur	q0, [x20,#-16]
	b	.L145
.L187:
	ldur	q3, [x19,#-87]
	ldur	q2, [x19,#-71]
	ldur	q1, [x19,#-55]
	ldur	q0, [x19,#-39]
	stur	q3, [x20,#-87]
	stur	q2, [x20,#-71]
	stur	q1, [x20,#-55]
	stur	q0, [x20,#-39]
.L251:
	ldur	q0, [x19,#-23]
	stur	q0, [x20,#-23]
	b	.L267
.L149:
	ldur	q3, [x19,#-125]
	ldur	q2, [x19,#-109]
	ldur	q1, [x19,#-93]
	ldur	q0, [x19,#-77]
	stur	q3, [x20,#-125]
	stur	q2, [x20,#-109]
	stur	q1, [x20,#-93]
	stur	q0, [x20,#-77]
.L213:
	ldur	q1, [x19,#-61]
	ldur	q0, [x19,#-45]
.L441:
	stur	q1, [x20,#-61]
	stur	q0, [x20,#-45]
.L245:
	ldur	q0, [x19,#-29]
	stur	q0, [x20,#-29]
	ldur	q0, [x19,#-16]
	stur	q0, [x20,#-16]
	b	.L145
.L158:
	ldur	q3, [x19,#-116]
	ldur	q2, [x19,#-100]
	ldur	q1, [x19,#-84]
	ldur	q0, [x19,#-68]
	stur	q3, [x20,#-116]
	stur	q2, [x20,#-100]
	stur	q1, [x20,#-84]
	stur	q0, [x20,#-68]
.L222:
	ldur	q1, [x19,#-52]
	ldur	q0, [x19,#-36]
.L430:
	stur	q1, [x20,#-52]
	stur	q0, [x20,#-36]
.L254:
	ldur	q0, [x19,#-20]
	stur	q0, [x20,#-20]
	ldur	w1, [x19,#-4]
	stur	w1, [x20,#-4]
	b	.L145
.L189:
	ldur	q3, [x19,#-85]
	ldur	q2, [x19,#-69]
	ldur	q1, [x19,#-53]
	ldur	q0, [x19,#-37]
	stur	q3, [x20,#-85]
	stur	q2, [x20,#-69]
	stur	q1, [x20,#-53]
	stur	q0, [x20,#-37]
.L253:
	ldur	q0, [x19,#-21]
	stur	q0, [x20,#-21]
	b	.L269
.L188:
	ldur	q3, [x19,#-86]
	ldur	q2, [x19,#-70]
	ldur	q1, [x19,#-54]
	ldur	q0, [x19,#-38]
	stur	q3, [x20,#-86]
	stur	q2, [x20,#-70]
	stur	q1, [x20,#-54]
	stur	q0, [x20,#-38]
.L252:
	ldur	q0, [x19,#-22]
	stur	q0, [x20,#-22]
	b	.L268
.L161:
	ldur	q3, [x19,#-113]
	ldur	q2, [x19,#-97]
	ldur	q1, [x19,#-81]
	ldur	q0, [x19,#-65]
	stur	q3, [x20,#-113]
	stur	q2, [x20,#-97]
	stur	q1, [x20,#-81]
	stur	q0, [x20,#-65]
.L225:
	ldur	q1, [x19,#-49]
	ldur	q0, [x19,#-33]
.L427:
	stur	q1, [x20,#-49]
	stur	q0, [x20,#-33]
.L257:
	ldur	q0, [x19,#-17]
	stur	q0, [x20,#-17]
	ldurb	w1, [x19,#-1]
	sturb	w1, [x20,#-1]
	b	.L145
.L178:
	ldp	q3, q2, [x19,#-96]
	ldp	q1, q0, [x19,#-64]
	stp	q3, q2, [x20,#-96]
	stp	q1, q0, [x20,#-64]
.L242:
	ldp	q1, q0, [x19,#-32]
	stp	q1, q0, [x20,#-32]
	b	.L145
.L177:
	ldur	q3, [x19,#-97]
	ldur	q2, [x19,#-81]
	ldur	q1, [x19,#-65]
	ldur	q0, [x19,#-49]
	stur	q3, [x20,#-97]
	stur	q2, [x20,#-81]
	stur	q1, [x20,#-65]
	stur	q0, [x20,#-49]
.L241:
	ldur	q1, [x19,#-33]
	ldur	q0, [x19,#-17]
	stur	q1, [x20,#-33]
	stur	q0, [x20,#-17]
	ldurb	w1, [x19,#-1]
	sturb	w1, [x20,#-1]
	b	.L145
.L176:
	ldur	q3, [x19,#-98]
	ldur	q2, [x19,#-82]
	ldur	q1, [x19,#-66]
	ldur	q0, [x19,#-50]
	stur	q3, [x20,#-98]
	stur	q2, [x20,#-82]
	stur	q1, [x20,#-66]
	stur	q0, [x20,#-50]
.L240:
	ldur	q1, [x19,#-34]
	ldur	q0, [x19,#-18]
	stur	q1, [x20,#-34]
	stur	q0, [x20,#-18]
	ldurh	w1, [x19,#-2]
	sturh	w1, [x20,#-2]
	b	.L145
.L174:
	ldur	q3, [x19,#-100]
	ldur	q2, [x19,#-84]
	ldur	q1, [x19,#-68]
	ldur	q0, [x19,#-52]
	stur	q3, [x20,#-100]
	stur	q2, [x20,#-84]
	stur	q1, [x20,#-68]
	stur	q0, [x20,#-52]
.L238:
	ldur	q1, [x19,#-36]
	ldur	q0, [x19,#-20]
	stur	q1, [x20,#-36]
	stur	q0, [x20,#-20]
	ldur	w1, [x19,#-4]
	stur	w1, [x20,#-4]
	b	.L145
.L170:
	ldur	q3, [x19,#-104]
	ldur	q2, [x19,#-88]
	ldur	q1, [x19,#-72]
	ldur	q0, [x19,#-56]
	stur	q3, [x20,#-104]
	stur	q2, [x20,#-88]
	stur	q1, [x20,#-72]
	stur	q0, [x20,#-56]
.L234:
	ldur	q1, [x19,#-40]
	ldur	q0, [x19,#-24]
	stur	q1, [x20,#-40]
	stur	q0, [x20,#-24]
	ldur	x1, [x19,#-8]
	stur	x1, [x20,#-8]
	b	.L145
.L162:
	ldp	q3, q2, [x19,#-112]
	ldp	q1, q0, [x19,#-80]
	stp	q3, q2, [x20,#-112]
	stp	q1, q0, [x20,#-80]
.L226:
	ldp	q1, q0, [x19,#-48]
	stp	q1, q0, [x20,#-48]
	ldur	q0, [x19,#-16]
	stur	q0, [x20,#-16]
	b	.L145
.L159:
	ldur	q3, [x19,#-115]
	ldur	q2, [x19,#-99]
	ldur	q1, [x19,#-83]
	ldur	q0, [x19,#-67]
	stur	q3, [x20,#-115]
	stur	q2, [x20,#-99]
	stur	q1, [x20,#-83]
	stur	q0, [x20,#-67]
.L223:
	ldur	q1, [x19,#-51]
	ldur	q0, [x19,#-35]
	stur	q1, [x20,#-51]
	stur	q0, [x20,#-35]
	ldur	q0, [x19,#-19]
	stur	q0, [x20,#-19]
	b	.L271
.L157:
	ldur	q3, [x19,#-117]
	ldur	q2, [x19,#-101]
	ldur	q1, [x19,#-85]
	ldur	q0, [x19,#-69]
	stur	q3, [x20,#-117]
	stur	q2, [x20,#-101]
	stur	q1, [x20,#-85]
	stur	q0, [x20,#-69]
.L221:
	ldur	q1, [x19,#-53]
	ldur	q0, [x19,#-37]
	stur	q1, [x20,#-53]
	stur	q0, [x20,#-37]
	ldur	q0, [x19,#-21]
	stur	q0, [x20,#-21]
	b	.L269
.L156:
	ldur	q3, [x19,#-118]
	ldur	q2, [x19,#-102]
	ldur	q1, [x19,#-86]
	ldur	q0, [x19,#-70]
	stur	q3, [x20,#-118]
	stur	q2, [x20,#-102]
	stur	q1, [x20,#-86]
	stur	q0, [x20,#-70]
.L220:
	ldur	q1, [x19,#-54]
	ldur	q0, [x19,#-38]
	stur	q1, [x20,#-54]
	stur	q0, [x20,#-38]
	ldur	q0, [x19,#-22]
	stur	q0, [x20,#-22]
	b	.L268
.L155:
	ldur	q3, [x19,#-119]
	ldur	q2, [x19,#-103]
	ldur	q1, [x19,#-87]
	ldur	q0, [x19,#-71]
	stur	q3, [x20,#-119]
	stur	q2, [x20,#-103]
	stur	q1, [x20,#-87]
	stur	q0, [x20,#-71]
.L219:
	ldur	q1, [x19,#-55]
	ldur	q0, [x19,#-39]
	stur	q1, [x20,#-55]
	stur	q0, [x20,#-39]
	ldur	q0, [x19,#-23]
	stur	q0, [x20,#-23]
	b	.L267
.L16:
	ldur	q3, [x19,#-117]
	ldur	q2, [x19,#-101]
	ldur	q1, [x19,#-85]
	ldur	q0, [x19,#-69]
	stur	q3, [x1,#-117]
	stur	q2, [x1,#-101]
	stur	q1, [x1,#-85]
	stur	q0, [x1,#-69]
.L80:
	ldur	q1, [x19,#-53]
	ldur	q0, [x19,#-37]
	stur	q1, [x1,#-53]
	stur	q0, [x1,#-37]
	ldur	q0, [x19,#-21]
	b	.L444
.L38:
	ldur	q3, [x19,#-95]
	ldur	q2, [x19,#-79]
	ldur	q1, [x19,#-63]
	ldur	q0, [x19,#-47]
	stur	q3, [x1,#-95]
	stur	q2, [x1,#-79]
	stur	q1, [x1,#-63]
	stur	q0, [x1,#-47]
	b	.L474
.L210:
	ldp	q3, q2, [x19,#-64]
	ldp	q1, q0, [x19,#-32]
	stp	q3, q2, [x20,#-64]
	stp	q1, q0, [x20,#-32]
	b	.L145
.L69:
	ldp	q3, q2, [x19,#-64]
	ldp	q1, q0, [x19,#-32]
	stp	q3, q2, [x1,#-64]
	stp	q1, q0, [x1,#-32]
	b	.L3
.L54:
	ldur	q3, [x19,#-79]
	ldur	q2, [x19,#-63]
	ldur	q1, [x19,#-47]
	ldur	q0, [x19,#-31]
	stur	q3, [x1,#-79]
	stur	q2, [x1,#-63]
	stur	q1, [x1,#-47]
	stur	q0, [x1,#-31]
	b	.L459
.L53:
	ldp	q3, q2, [x19,#-80]
	ldp	q1, q0, [x19,#-48]
	stp	q3, q2, [x1,#-80]
	stp	q1, q0, [x1,#-48]
	b	.L460
.L52:
	ldur	q3, [x19,#-81]
	ldur	q2, [x19,#-65]
	ldur	q1, [x19,#-49]
	ldur	q0, [x19,#-33]
	stur	q3, [x1,#-81]
	stur	q2, [x1,#-65]
	b	.L461
.L51:
	ldur	q3, [x19,#-82]
	ldur	q2, [x19,#-66]
	ldur	q1, [x19,#-50]
	ldur	q0, [x19,#-34]
	stur	q3, [x1,#-82]
	stur	q2, [x1,#-66]
	b	.L462
.L50:
	ldur	q3, [x19,#-83]
	ldur	q2, [x19,#-67]
	ldur	q1, [x19,#-51]
	ldur	q0, [x19,#-35]
	stur	q3, [x1,#-83]
	stur	q2, [x1,#-67]
	stur	q1, [x1,#-51]
	stur	q0, [x1,#-35]
	ldur	q0, [x19,#-19]
	b	.L446
.L49:
	ldur	q3, [x19,#-84]
	ldur	q2, [x19,#-68]
	ldur	q1, [x19,#-52]
	ldur	q0, [x19,#-36]
	stur	q3, [x1,#-84]
	stur	q2, [x1,#-68]
	b	.L464
.L47:
	ldur	q3, [x19,#-86]
	ldur	q2, [x19,#-70]
	ldur	q1, [x19,#-54]
	ldur	q0, [x19,#-38]
	stur	q3, [x1,#-86]
	stur	q2, [x1,#-70]
	stur	q1, [x1,#-54]
	stur	q0, [x1,#-38]
	ldur	q0, [x19,#-22]
	b	.L465
.L46:
	ldur	q3, [x19,#-87]
	ldur	q2, [x19,#-71]
	ldur	q1, [x19,#-55]
	ldur	q0, [x19,#-39]
	stur	q3, [x1,#-87]
	stur	q2, [x1,#-71]
	stur	q1, [x1,#-55]
	stur	q0, [x1,#-39]
	ldur	q0, [x19,#-23]
	b	.L442
.L45:
	ldur	q3, [x19,#-88]
	ldur	q2, [x19,#-72]
	ldur	q1, [x19,#-56]
	ldur	q0, [x19,#-40]
	stur	q3, [x1,#-88]
	stur	q2, [x1,#-72]
	b	.L467
.L44:
	ldur	q3, [x19,#-89]
	ldur	q2, [x19,#-73]
	ldur	q1, [x19,#-57]
	ldur	q0, [x19,#-41]
	stur	q3, [x1,#-89]
	stur	q2, [x1,#-73]
	b	.L468
.L43:
	ldur	q3, [x19,#-90]
	ldur	q2, [x19,#-74]
	ldur	q1, [x19,#-58]
	ldur	q0, [x19,#-42]
	stur	q3, [x1,#-90]
	stur	q2, [x1,#-74]
	b	.L469
.L42:
	ldur	q3, [x19,#-91]
	ldur	q2, [x19,#-75]
	ldur	q1, [x19,#-59]
	ldur	q0, [x19,#-43]
	stur	q3, [x1,#-91]
	stur	q2, [x1,#-75]
	b	.L470
.L41:
	ldur	q3, [x19,#-92]
	ldur	q2, [x19,#-76]
	ldur	q1, [x19,#-60]
	ldur	q0, [x19,#-44]
	stur	q3, [x1,#-92]
	stur	q2, [x1,#-76]
	b	.L471
.L40:
	ldur	q3, [x19,#-93]
	ldur	q2, [x19,#-77]
	ldur	q1, [x19,#-61]
	ldur	q0, [x19,#-45]
	stur	q3, [x1,#-93]
	stur	q2, [x1,#-77]
	b	.L472
.L39:
	ldur	q3, [x19,#-94]
	ldur	q2, [x19,#-78]
	ldur	q1, [x19,#-62]
	ldur	q0, [x19,#-46]
	stur	q3, [x1,#-94]
	stur	q2, [x1,#-78]
	b	.L473
.L62:
	ldur	q3, [x19,#-71]
	ldur	q2, [x19,#-55]
	ldur	q1, [x19,#-39]
	ldur	q0, [x19,#-23]
	stur	q3, [x1,#-71]
	stur	q2, [x1,#-55]
	stur	q1, [x1,#-39]
	stur	q0, [x1,#-23]
	b	.L451
.L61:
	ldur	q3, [x19,#-72]
	ldur	q2, [x19,#-56]
	ldur	q1, [x19,#-40]
	ldur	q0, [x19,#-24]
	stur	q3, [x1,#-72]
	stur	q2, [x1,#-56]
	stur	q1, [x1,#-40]
	stur	q0, [x1,#-24]
	b	.L452
.L60:
	ldur	q3, [x19,#-73]
	ldur	q2, [x19,#-57]
	ldur	q1, [x19,#-41]
	ldur	q0, [x19,#-25]
	stur	q3, [x1,#-73]
	stur	q2, [x1,#-57]
	b	.L453
.L59:
	ldur	q3, [x19,#-74]
	ldur	q2, [x19,#-58]
	ldur	q1, [x19,#-42]
	ldur	q0, [x19,#-26]
	stur	q3, [x1,#-74]
	stur	q2, [x1,#-58]
	b	.L454
.L58:
	ldur	q3, [x19,#-75]
	ldur	q2, [x19,#-59]
	ldur	q1, [x19,#-43]
	ldur	q0, [x19,#-27]
	stur	q3, [x1,#-75]
	stur	q2, [x1,#-59]
	b	.L455
.L57:
	ldur	q3, [x19,#-76]
	ldur	q2, [x19,#-60]
	ldur	q1, [x19,#-44]
	ldur	q0, [x19,#-28]
	stur	q3, [x1,#-76]
	stur	q2, [x1,#-60]
	b	.L456
.L56:
	ldur	q3, [x19,#-77]
	ldur	q2, [x19,#-61]
	ldur	q1, [x19,#-45]
	ldur	q0, [x19,#-29]
	stur	q3, [x1,#-77]
	stur	q2, [x1,#-61]
	b	.L457
.L55:
	ldur	q3, [x19,#-78]
	ldur	q2, [x19,#-62]
	ldur	q1, [x19,#-46]
	ldur	q0, [x19,#-30]
	stur	q3, [x1,#-78]
	stur	q2, [x1,#-62]
	b	.L458
.L66:
	ldur	q3, [x19,#-67]
	ldur	q2, [x19,#-51]
	ldur	q1, [x19,#-35]
	ldur	q0, [x19,#-19]
	stur	q3, [x1,#-67]
	stur	q2, [x1,#-51]
	stur	q1, [x1,#-35]
	stur	q0, [x1,#-19]
	b	.L450
.L65:
	ldur	q3, [x19,#-68]
	ldur	q2, [x19,#-52]
	ldur	q1, [x19,#-36]
	ldur	q0, [x19,#-20]
	stur	q3, [x1,#-68]
	stur	q2, [x1,#-52]
	stur	q1, [x1,#-36]
	stur	q0, [x1,#-20]
	b	.L448
.L64:
	ldur	q3, [x19,#-69]
	ldur	q2, [x19,#-53]
	ldur	q1, [x19,#-37]
	ldur	q0, [x19,#-21]
	stur	q3, [x1,#-69]
	stur	q2, [x1,#-53]
	stur	q1, [x1,#-37]
	stur	q0, [x1,#-21]
	b	.L128
.L63:
	ldur	q3, [x19,#-70]
	ldur	q2, [x19,#-54]
	ldur	q1, [x19,#-38]
	ldur	q0, [x19,#-22]
	stur	q3, [x1,#-70]
	stur	q2, [x1,#-54]
	stur	q1, [x1,#-38]
	stur	q0, [x1,#-22]
	b	.L127
.L68:
	ldur	q3, [x19,#-65]
	ldur	q2, [x19,#-49]
	ldur	q1, [x19,#-33]
	ldur	q0, [x19,#-17]
	stur	q3, [x1,#-65]
	stur	q2, [x1,#-49]
	stur	q1, [x1,#-33]
	stur	q0, [x1,#-17]
	b	.L447
.L478:
	bl	__stack_chk_fail
	.p2align 2,,3
.L4:
	ldp	q7, q6, [x19,#-128]
	ldp	q5, q4, [x19,#-96]
	ldp	q3, q2, [x19,#-64]
	ldp	q1, q0, [x19,#-32]
	stp	q7, q6, [x1,#-128]
	stp	q5, q4, [x1,#-96]
	stp	q3, q2, [x1,#-64]
	stp	q1, q0, [x1,#-32]
	b	.L3
.L179:
	ldur	q3, [x19,#-95]
	ldur	q2, [x19,#-79]
	ldur	q1, [x19,#-63]
	ldur	q0, [x19,#-47]
	stur	q3, [x20,#-95]
	stur	q2, [x20,#-79]
	stur	q1, [x20,#-63]
	stur	q0, [x20,#-47]
	b	.L439
.L195:
	ldur	q3, [x19,#-79]
	ldur	q2, [x19,#-63]
	ldur	q1, [x19,#-47]
	ldur	q0, [x19,#-31]
	stur	q3, [x20,#-79]
	stur	q2, [x20,#-63]
	b	.L424
.L194:
	ldp	q3, q2, [x19,#-80]
	ldp	q1, q0, [x19,#-48]
	stp	q3, q2, [x20,#-80]
	stp	q1, q0, [x20,#-48]
	b	.L425
.L193:
	ldur	q3, [x19,#-81]
	ldur	q2, [x19,#-65]
	ldur	q1, [x19,#-49]
	ldur	q0, [x19,#-33]
	stur	q3, [x20,#-81]
	stur	q2, [x20,#-65]
	b	.L427
.L192:
	ldur	q3, [x19,#-82]
	ldur	q2, [x19,#-66]
	ldur	q1, [x19,#-50]
	ldur	q0, [x19,#-34]
	stur	q3, [x20,#-82]
	stur	q2, [x20,#-66]
	b	.L428
.L190:
	ldur	q3, [x19,#-84]
	ldur	q2, [x19,#-68]
	ldur	q1, [x19,#-52]
	ldur	q0, [x19,#-36]
	stur	q3, [x20,#-84]
	stur	q2, [x20,#-68]
	b	.L430
.L186:
	ldur	q3, [x19,#-88]
	ldur	q2, [x19,#-72]
	ldur	q1, [x19,#-56]
	ldur	q0, [x19,#-40]
	stur	q3, [x20,#-88]
	stur	q2, [x20,#-72]
	b	.L433
.L185:
	ldur	q3, [x19,#-89]
	ldur	q2, [x19,#-73]
	ldur	q1, [x19,#-57]
	ldur	q0, [x19,#-41]
	stur	q3, [x20,#-89]
	stur	q2, [x20,#-73]
	b	.L434
.L184:
	ldur	q3, [x19,#-90]
	ldur	q2, [x19,#-74]
	ldur	q1, [x19,#-58]
	ldur	q0, [x19,#-42]
	stur	q3, [x20,#-90]
	stur	q2, [x20,#-74]
	b	.L436
.L183:
	ldur	q3, [x19,#-91]
	ldur	q2, [x19,#-75]
	ldur	q1, [x19,#-59]
	ldur	q0, [x19,#-43]
	stur	q3, [x20,#-91]
	stur	q2, [x20,#-75]
	b	.L437
.L182:
	ldur	q3, [x19,#-92]
	ldur	q2, [x19,#-76]
	ldur	q1, [x19,#-60]
	ldur	q0, [x19,#-44]
	stur	q3, [x20,#-92]
	stur	q2, [x20,#-76]
	b	.L438
.L181:
	ldur	q3, [x19,#-93]
	ldur	q2, [x19,#-77]
	ldur	q1, [x19,#-61]
	ldur	q0, [x19,#-45]
	stur	q3, [x20,#-93]
	stur	q2, [x20,#-77]
	b	.L441
.L180:
	ldur	q3, [x19,#-94]
	ldur	q2, [x19,#-78]
	ldur	q1, [x19,#-62]
	ldur	q0, [x19,#-46]
	stur	q3, [x20,#-94]
	stur	q2, [x20,#-78]
	b	.L440
.L203:
	ldur	q3, [x19,#-71]
	ldur	q2, [x19,#-55]
	ldur	q1, [x19,#-39]
	ldur	q0, [x19,#-23]
	stur	q3, [x20,#-71]
	stur	q2, [x20,#-55]
	stur	q1, [x20,#-39]
	b	.L431
.L202:
	ldur	q3, [x19,#-72]
	ldur	q2, [x19,#-56]
	ldur	q1, [x19,#-40]
	ldur	q0, [x19,#-24]
	stur	q3, [x20,#-72]
	stur	q2, [x20,#-56]
	stur	q1, [x20,#-40]
	stur	q0, [x20,#-24]
	b	.L418
.L201:
	ldur	q3, [x19,#-73]
	ldur	q2, [x19,#-57]
	ldur	q1, [x19,#-41]
	ldur	q0, [x19,#-25]
	stur	q3, [x20,#-73]
	stur	q2, [x20,#-57]
	b	.L419
.L200:
	ldur	q3, [x19,#-74]
	ldur	q2, [x19,#-58]
	ldur	q1, [x19,#-42]
	ldur	q0, [x19,#-26]
	stur	q3, [x20,#-74]
	stur	q2, [x20,#-58]
	b	.L420
.L199:
	ldur	q3, [x19,#-75]
	ldur	q2, [x19,#-59]
	ldur	q1, [x19,#-43]
	ldur	q0, [x19,#-27]
	stur	q3, [x20,#-75]
	stur	q2, [x20,#-59]
	b	.L421
.L198:
	ldur	q3, [x19,#-76]
	ldur	q2, [x19,#-60]
	ldur	q1, [x19,#-44]
	ldur	q0, [x19,#-28]
	stur	q3, [x20,#-76]
	stur	q2, [x20,#-60]
	b	.L422
.L197:
	ldur	q3, [x19,#-77]
	ldur	q2, [x19,#-61]
	ldur	q1, [x19,#-45]
	ldur	q0, [x19,#-29]
	stur	q3, [x20,#-77]
	stur	q2, [x20,#-61]
	b	.L423
.L196:
	ldur	q3, [x19,#-78]
	ldur	q2, [x19,#-62]
	ldur	q1, [x19,#-46]
	ldur	q0, [x19,#-30]
	stur	q3, [x20,#-78]
	stur	q2, [x20,#-62]
	b	.L426
.L207:
	ldur	q3, [x19,#-67]
	ldur	q2, [x19,#-51]
	ldur	q1, [x19,#-35]
	ldur	q0, [x19,#-19]
	stur	q3, [x20,#-67]
	stur	q2, [x20,#-51]
	stur	q1, [x20,#-35]
	b	.L429
.L206:
	ldur	q3, [x19,#-68]
	ldur	q2, [x19,#-52]
	ldur	q1, [x19,#-36]
	ldur	q0, [x19,#-20]
	stur	q3, [x20,#-68]
	stur	q2, [x20,#-52]
	stur	q1, [x20,#-36]
	stur	q0, [x20,#-20]
	b	.L413
.L205:
	ldur	q3, [x19,#-69]
	ldur	q2, [x19,#-53]
	ldur	q1, [x19,#-37]
	ldur	q0, [x19,#-21]
	stur	q3, [x20,#-69]
	stur	q2, [x20,#-53]
	stur	q1, [x20,#-37]
	b	.L414
.L204:
	ldur	q3, [x19,#-70]
	ldur	q2, [x19,#-54]
	ldur	q1, [x19,#-38]
	ldur	q0, [x19,#-22]
	stur	q3, [x20,#-70]
	stur	q2, [x20,#-54]
	stur	q1, [x20,#-38]
	b	.L416
.L209:
	ldur	q3, [x19,#-65]
	ldur	q2, [x19,#-49]
	ldur	q1, [x19,#-33]
	ldur	q0, [x19,#-17]
	stur	q3, [x20,#-65]
	stur	q2, [x20,#-49]
	stur	q1, [x20,#-33]
	stur	q0, [x20,#-17]
	b	.L412
