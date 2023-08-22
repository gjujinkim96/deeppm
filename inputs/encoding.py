op_dict = {0: 'OP_INVALID', 1: 'OP_UNDECODED', 2: 'OP_CONTD', 3: 'OP_LABEL', 4: 'OP_add', 5: 'OP_or', 6: 'OP_adc', 7: 'OP_sbb', 8: 'OP_and', 9: 'OP_daa', 10: 'OP_sub', 11: 'OP_das', 12: 'OP_xor', 13: 'OP_aaa', 14: 'OP_cmp', 15: 'OP_aas', 16: 'OP_inc', 17: 'OP_dec', 18: 'OP_push', 19: 'OP_push_imm', 20: 'OP_pop', 21: 'OP_pusha', 22: 'OP_popa', 23: 'OP_bound', 24: 'OP_arpl', 25: 'OP_imul', 26: 'OP_jo_short', 27: 'OP_jno_short', 28: 'OP_jb_short', 29: 'OP_jnb_short', 30: 'OP_jz_short', 31: 'OP_jnz_short', 32: 'OP_jbe_short', 33: 'OP_jnbe_short', 34: 'OP_js_short', 35: 'OP_jns_short', 36: 'OP_jp_short', 37: 'OP_jnp_short', 38: 'OP_jl_short', 39: 'OP_jnl_short', 40: 'OP_jle_short', 41: 'OP_jnle_short', 42: 'OP_call', 43: 'OP_call_ind', 44: 'OP_call_far', 45: 'OP_call_far_ind', 46: 'OP_jmp', 47: 'OP_jmp_short', 48: 'OP_jmp_ind', 49: 'OP_jmp_far', 50: 'OP_jmp_far_ind', 51: 'OP_loopne', 52: 'OP_loope', 53: 'OP_loop', 54: 'OP_jecxz', 55: 'OP_mov_ld', 56: 'OP_mov_st', 57: 'OP_mov_imm', 58: 'OP_mov_seg', 59: 'OP_mov_priv', 60: 'OP_test', 61: 'OP_lea', 62: 'OP_xchg', 63: 'OP_cwde', 64: 'OP_cdq', 65: 'OP_fwait', 66: 'OP_pushf', 67: 'OP_popf', 68: 'OP_sahf', 69: 'OP_lahf', 70: 'OP_ret', 71: 'OP_ret_far', 72: 'OP_les', 73: 'OP_lds', 74: 'OP_enter', 75: 'OP_leave', 76: 'OP_int3', 77: 'OP_int', 78: 'OP_into', 79: 'OP_iret', 80: 'OP_aam', 81: 'OP_aad', 82: 'OP_xlat', 83: 'OP_in', 84: 'OP_out', 85: 'OP_hlt', 86: 'OP_cmc', 87: 'OP_clc', 88: 'OP_stc', 89: 'OP_cli', 90: 'OP_sti', 91: 'OP_cld', 92: 'OP_std', 93: 'OP_lar', 94: 'OP_lsl', 95: 'OP_syscall', 96: 'OP_clts', 97: 'OP_sysret', 98: 'OP_invd', 99: 'OP_wbinvd', 100: 'OP_ud2a', 101: 'OP_nop_modrm', 102: 'OP_movntps', 103: 'OP_movntpd', 104: 'OP_wrmsr', 105: 'OP_rdtsc', 106: 'OP_rdmsr', 107: 'OP_rdpmc', 108: 'OP_sysenter', 109: 'OP_sysexit', 110: 'OP_cmovo', 111: 'OP_cmovno', 112: 'OP_cmovb', 113: 'OP_cmovnb', 114: 'OP_cmovz', 115: 'OP_cmovnz', 116: 'OP_cmovbe', 117: 'OP_cmovnbe', 118: 'OP_cmovs', 119: 'OP_cmovns', 120: 'OP_cmovp', 121: 'OP_cmovnp', 122: 'OP_cmovl', 123: 'OP_cmovnl', 124: 'OP_cmovle', 125: 'OP_cmovnle', 126: 'OP_punpcklbw', 127: 'OP_punpcklwd', 128: 'OP_punpckldq', 129: 'OP_packsswb', 130: 'OP_pcmpgtb', 131: 'OP_pcmpgtw', 132: 'OP_pcmpgtd', 133: 'OP_packuswb', 134: 'OP_punpckhbw', 135: 'OP_punpckhwd', 136: 'OP_punpckhdq', 137: 'OP_packssdw', 138: 'OP_punpcklqdq', 139: 'OP_punpckhqdq', 140: 'OP_movd', 141: 'OP_movq', 142: 'OP_movdqu', 143: 'OP_movdqa', 144: 'OP_pshufw', 145: 'OP_pshufd', 146: 'OP_pshufhw', 147: 'OP_pshuflw', 148: 'OP_pcmpeqb', 149: 'OP_pcmpeqw', 150: 'OP_pcmpeqd', 151: 'OP_emms', 152: 'OP_jo', 153: 'OP_jno', 154: 'OP_jb', 155: 'OP_jnb', 156: 'OP_jz', 157: 'OP_jnz', 158: 'OP_jbe', 159: 'OP_jnbe', 160: 'OP_js', 161: 'OP_jns', 162: 'OP_jp', 163: 'OP_jnp', 164: 'OP_jl', 165: 'OP_jnl', 166: 'OP_jle', 167: 'OP_jnle', 168: 'OP_seto', 169: 'OP_setno', 170: 'OP_setb', 171: 'OP_setnb', 172: 'OP_setz', 173: 'OP_setnz', 174: 'OP_setbe', 175: 'OP_setnbe', 176: 'OP_sets', 177: 'OP_setns', 178: 'OP_setp', 179: 'OP_setnp', 180: 'OP_setl', 181: 'OP_setnl', 182: 'OP_setle', 183: 'OP_setnle', 184: 'OP_cpuid', 185: 'OP_bt', 186: 'OP_shld', 187: 'OP_rsm', 188: 'OP_bts', 189: 'OP_shrd', 190: 'OP_cmpxchg', 191: 'OP_lss', 192: 'OP_btr', 193: 'OP_lfs', 194: 'OP_lgs', 195: 'OP_movzx', 196: 'OP_ud2b', 197: 'OP_btc', 198: 'OP_bsf', 199: 'OP_bsr', 200: 'OP_movsx', 201: 'OP_xadd', 202: 'OP_movnti', 203: 'OP_pinsrw', 204: 'OP_pextrw', 205: 'OP_bswap', 206: 'OP_psrlw', 207: 'OP_psrld', 208: 'OP_psrlq', 209: 'OP_paddq', 210: 'OP_pmullw', 211: 'OP_pmovmskb', 212: 'OP_psubusb', 213: 'OP_psubusw', 214: 'OP_pminub', 215: 'OP_pand', 216: 'OP_paddusb', 217: 'OP_paddusw', 218: 'OP_pmaxub', 219: 'OP_pandn', 220: 'OP_pavgb', 221: 'OP_psraw', 222: 'OP_psrad', 223: 'OP_pavgw', 224: 'OP_pmulhuw', 225: 'OP_pmulhw', 226: 'OP_movntq', 227: 'OP_movntdq', 228: 'OP_psubsb', 229: 'OP_psubsw', 230: 'OP_pminsw', 231: 'OP_por', 232: 'OP_paddsb', 233: 'OP_paddsw', 234: 'OP_pmaxsw', 235: 'OP_pxor', 236: 'OP_psllw', 237: 'OP_pslld', 238: 'OP_psllq', 239: 'OP_pmuludq', 240: 'OP_pmaddwd', 241: 'OP_psadbw', 242: 'OP_maskmovq', 243: 'OP_maskmovdqu', 244: 'OP_psubb', 245: 'OP_psubw', 246: 'OP_psubd', 247: 'OP_psubq', 248: 'OP_paddb', 249: 'OP_paddw', 250: 'OP_paddd', 251: 'OP_psrldq', 252: 'OP_pslldq', 253: 'OP_rol', 254: 'OP_ror', 255: 'OP_rcl', 256: 'OP_rcr', 257: 'OP_shl', 258: 'OP_shr', 259: 'OP_sar', 260: 'OP_not', 261: 'OP_neg', 262: 'OP_mul', 263: 'OP_div', 264: 'OP_idiv', 265: 'OP_sldt', 266: 'OP_str', 267: 'OP_lldt', 268: 'OP_ltr', 269: 'OP_verr', 270: 'OP_verw', 271: 'OP_sgdt', 272: 'OP_sidt', 273: 'OP_lgdt', 274: 'OP_lidt', 275: 'OP_smsw', 276: 'OP_lmsw', 277: 'OP_invlpg', 278: 'OP_cmpxchg8b', 279: 'OP_fxsave32', 280: 'OP_fxrstor32', 281: 'OP_ldmxcsr', 282: 'OP_stmxcsr', 283: 'OP_lfence', 284: 'OP_mfence', 285: 'OP_clflush', 286: 'OP_sfence', 287: 'OP_prefetchnta', 288: 'OP_prefetcht0', 289: 'OP_prefetcht1', 290: 'OP_prefetcht2', 291: 'OP_prefetch', 292: 'OP_prefetchw', 293: 'OP_movups', 294: 'OP_movss', 295: 'OP_movupd', 296: 'OP_movsd', 297: 'OP_movlps', 298: 'OP_movlpd', 299: 'OP_unpcklps', 300: 'OP_unpcklpd', 301: 'OP_unpckhps', 302: 'OP_unpckhpd', 303: 'OP_movhps', 304: 'OP_movhpd', 305: 'OP_movaps', 306: 'OP_movapd', 307: 'OP_cvtpi2ps', 308: 'OP_cvtsi2ss', 309: 'OP_cvtpi2pd', 310: 'OP_cvtsi2sd', 311: 'OP_cvttps2pi', 312: 'OP_cvttss2si', 313: 'OP_cvttpd2pi', 314: 'OP_cvttsd2si', 315: 'OP_cvtps2pi', 316: 'OP_cvtss2si', 317: 'OP_cvtpd2pi', 318: 'OP_cvtsd2si', 319: 'OP_ucomiss', 320: 'OP_ucomisd', 321: 'OP_comiss', 322: 'OP_comisd', 323: 'OP_movmskps', 324: 'OP_movmskpd', 325: 'OP_sqrtps', 326: 'OP_sqrtss', 327: 'OP_sqrtpd', 328: 'OP_sqrtsd', 329: 'OP_rsqrtps', 330: 'OP_rsqrtss', 331: 'OP_rcpps', 332: 'OP_rcpss', 333: 'OP_andps', 334: 'OP_andpd', 335: 'OP_andnps', 336: 'OP_andnpd', 337: 'OP_orps', 338: 'OP_orpd', 339: 'OP_xorps', 340: 'OP_xorpd', 341: 'OP_addps', 342: 'OP_addss', 343: 'OP_addpd', 344: 'OP_addsd', 345: 'OP_mulps', 346: 'OP_mulss', 347: 'OP_mulpd', 348: 'OP_mulsd', 349: 'OP_cvtps2pd', 350: 'OP_cvtss2sd', 351: 'OP_cvtpd2ps', 352: 'OP_cvtsd2ss', 353: 'OP_cvtdq2ps', 354: 'OP_cvttps2dq', 355: 'OP_cvtps2dq', 356: 'OP_subps', 357: 'OP_subss', 358: 'OP_subpd', 359: 'OP_subsd', 360: 'OP_minps', 361: 'OP_minss', 362: 'OP_minpd', 363: 'OP_minsd', 364: 'OP_divps', 365: 'OP_divss', 366: 'OP_divpd', 367: 'OP_divsd', 368: 'OP_maxps', 369: 'OP_maxss', 370: 'OP_maxpd', 371: 'OP_maxsd', 372: 'OP_cmpps', 373: 'OP_cmpss', 374: 'OP_cmppd', 375: 'OP_cmpsd', 376: 'OP_shufps', 377: 'OP_shufpd', 378: 'OP_cvtdq2pd', 379: 'OP_cvttpd2dq', 380: 'OP_cvtpd2dq', 381: 'OP_nop', 382: 'OP_pause', 383: 'OP_ins', 384: 'OP_rep_ins', 385: 'OP_outs', 386: 'OP_rep_outs', 387: 'OP_movs', 388: 'OP_rep_movs', 389: 'OP_stos', 390: 'OP_rep_stos', 391: 'OP_lods', 392: 'OP_rep_lods', 393: 'OP_cmps', 394: 'OP_rep_cmps', 395: 'OP_repne_cmps', 396: 'OP_scas', 397: 'OP_rep_scas', 398: 'OP_repne_scas', 399: 'OP_fadd', 400: 'OP_fmul', 401: 'OP_fcom', 402: 'OP_fcomp', 403: 'OP_fsub', 404: 'OP_fsubr', 405: 'OP_fdiv', 406: 'OP_fdivr', 407: 'OP_fld', 408: 'OP_fst', 409: 'OP_fstp', 410: 'OP_fldenv', 411: 'OP_fldcw', 412: 'OP_fnstenv', 413: 'OP_fnstcw', 414: 'OP_fiadd', 415: 'OP_fimul', 416: 'OP_ficom', 417: 'OP_ficomp', 418: 'OP_fisub', 419: 'OP_fisubr', 420: 'OP_fidiv', 421: 'OP_fidivr', 422: 'OP_fild', 423: 'OP_fist', 424: 'OP_fistp', 425: 'OP_frstor', 426: 'OP_fnsave', 427: 'OP_fnstsw', 428: 'OP_fbld', 429: 'OP_fbstp', 430: 'OP_fxch', 431: 'OP_fnop', 432: 'OP_fchs', 433: 'OP_fabs', 434: 'OP_ftst', 435: 'OP_fxam', 436: 'OP_fld1', 437: 'OP_fldl2t', 438: 'OP_fldl2e', 439: 'OP_fldpi', 440: 'OP_fldlg2', 441: 'OP_fldln2', 442: 'OP_fldz', 443: 'OP_f2xm1', 444: 'OP_fyl2x', 445: 'OP_fptan', 446: 'OP_fpatan', 447: 'OP_fxtract', 448: 'OP_fprem1', 449: 'OP_fdecstp', 450: 'OP_fincstp', 451: 'OP_fprem', 452: 'OP_fyl2xp1', 453: 'OP_fsqrt', 454: 'OP_fsincos', 455: 'OP_frndint', 456: 'OP_fscale', 457: 'OP_fsin', 458: 'OP_fcos', 459: 'OP_fcmovb', 460: 'OP_fcmove', 461: 'OP_fcmovbe', 462: 'OP_fcmovu', 463: 'OP_fucompp', 464: 'OP_fcmovnb', 465: 'OP_fcmovne', 466: 'OP_fcmovnbe', 467: 'OP_fcmovnu', 468: 'OP_fnclex', 469: 'OP_fninit', 470: 'OP_fucomi', 471: 'OP_fcomi', 472: 'OP_ffree', 473: 'OP_fucom', 474: 'OP_fucomp', 475: 'OP_faddp', 476: 'OP_fmulp', 477: 'OP_fcompp', 478: 'OP_fsubrp', 479: 'OP_fsubp', 480: 'OP_fdivrp', 481: 'OP_fdivp', 482: 'OP_fucomip', 483: 'OP_fcomip', 484: 'OP_fisttp', 485: 'OP_haddpd', 486: 'OP_haddps', 487: 'OP_hsubpd', 488: 'OP_hsubps', 489: 'OP_addsubpd', 490: 'OP_addsubps', 491: 'OP_lddqu', 492: 'OP_monitor', 493: 'OP_mwait', 494: 'OP_movsldup', 495: 'OP_movshdup', 496: 'OP_movddup', 497: 'OP_femms', 498: 'OP_unknown_3dnow', 499: 'OP_pavgusb', 500: 'OP_pfadd', 501: 'OP_pfacc', 502: 'OP_pfcmpge', 503: 'OP_pfcmpgt', 504: 'OP_pfcmpeq', 505: 'OP_pfmin', 506: 'OP_pfmax', 507: 'OP_pfmul', 508: 'OP_pfrcp', 509: 'OP_pfrcpit1', 510: 'OP_pfrcpit2', 511: 'OP_pfrsqrt', 512: 'OP_pfrsqit1', 513: 'OP_pmulhrw', 514: 'OP_pfsub', 515: 'OP_pfsubr', 516: 'OP_pi2fd', 517: 'OP_pf2id', 518: 'OP_pi2fw', 519: 'OP_pf2iw', 520: 'OP_pfnacc', 521: 'OP_pfpnacc', 522: 'OP_pswapd', 523: 'OP_pshufb', 524: 'OP_phaddw', 525: 'OP_phaddd', 526: 'OP_phaddsw', 527: 'OP_pmaddubsw', 528: 'OP_phsubw', 529: 'OP_phsubd', 530: 'OP_phsubsw', 531: 'OP_psignb', 532: 'OP_psignw', 533: 'OP_psignd', 534: 'OP_pmulhrsw', 535: 'OP_pabsb', 536: 'OP_pabsw', 537: 'OP_pabsd', 538: 'OP_palignr', 539: 'OP_popcnt', 540: 'OP_movntss', 541: 'OP_movntsd', 542: 'OP_extrq', 543: 'OP_insertq', 544: 'OP_lzcnt', 545: 'OP_pblendvb', 546: 'OP_blendvps', 547: 'OP_blendvpd', 548: 'OP_ptest', 549: 'OP_pmovsxbw', 550: 'OP_pmovsxbd', 551: 'OP_pmovsxbq', 552: 'OP_pmovsxwd', 553: 'OP_pmovsxwq', 554: 'OP_pmovsxdq', 555: 'OP_pmuldq', 556: 'OP_pcmpeqq', 557: 'OP_movntdqa', 558: 'OP_packusdw', 559: 'OP_pmovzxbw', 560: 'OP_pmovzxbd', 561: 'OP_pmovzxbq', 562: 'OP_pmovzxwd', 563: 'OP_pmovzxwq', 564: 'OP_pmovzxdq', 565: 'OP_pcmpgtq', 566: 'OP_pminsb', 567: 'OP_pminsd', 568: 'OP_pminuw', 569: 'OP_pminud', 570: 'OP_pmaxsb', 571: 'OP_pmaxsd', 572: 'OP_pmaxuw', 573: 'OP_pmaxud', 574: 'OP_pmulld', 575: 'OP_phminposuw', 576: 'OP_crc32', 577: 'OP_pextrb', 578: 'OP_pextrd', 579: 'OP_extractps', 580: 'OP_roundps', 581: 'OP_roundpd', 582: 'OP_roundss', 583: 'OP_roundsd', 584: 'OP_blendps', 585: 'OP_blendpd', 586: 'OP_pblendw', 587: 'OP_pinsrb', 588: 'OP_insertps', 589: 'OP_pinsrd', 590: 'OP_dpps', 591: 'OP_dppd', 592: 'OP_mpsadbw', 593: 'OP_pcmpestrm', 594: 'OP_pcmpestri', 595: 'OP_pcmpistrm', 596: 'OP_pcmpistri', 597: 'OP_movsxd', 598: 'OP_swapgs', 599: 'OP_vmcall', 600: 'OP_vmlaunch', 601: 'OP_vmresume', 602: 'OP_vmxoff', 603: 'OP_vmptrst', 604: 'OP_vmptrld', 605: 'OP_vmxon', 606: 'OP_vmclear', 607: 'OP_vmread', 608: 'OP_vmwrite', 609: 'OP_int1', 610: 'OP_salc', 611: 'OP_ffreep', 612: 'OP_vmrun', 613: 'OP_vmmcall', 614: 'OP_vmload', 615: 'OP_vmsave', 616: 'OP_stgi', 617: 'OP_clgi', 618: 'OP_skinit', 619: 'OP_invlpga', 620: 'OP_rdtscp', 621: 'OP_invept', 622: 'OP_invvpid', 623: 'OP_pclmulqdq', 624: 'OP_aesimc', 625: 'OP_aesenc', 626: 'OP_aesenclast', 627: 'OP_aesdec', 628: 'OP_aesdeclast', 629: 'OP_aeskeygenassist', 630: 'OP_movbe', 631: 'OP_xgetbv', 632: 'OP_xsetbv', 633: 'OP_xsave32', 634: 'OP_xrstor32', 635: 'OP_xsaveopt32', 636: 'OP_vmovss', 637: 'OP_vmovsd', 638: 'OP_vmovups', 639: 'OP_vmovupd', 640: 'OP_vmovlps', 641: 'OP_vmovsldup', 642: 'OP_vmovlpd', 643: 'OP_vmovddup', 644: 'OP_vunpcklps', 645: 'OP_vunpcklpd', 646: 'OP_vunpckhps', 647: 'OP_vunpckhpd', 648: 'OP_vmovhps', 649: 'OP_vmovshdup', 650: 'OP_vmovhpd', 651: 'OP_vmovaps', 652: 'OP_vmovapd', 653: 'OP_vcvtsi2ss', 654: 'OP_vcvtsi2sd', 655: 'OP_vmovntps', 656: 'OP_vmovntpd', 657: 'OP_vcvttss2si', 658: 'OP_vcvttsd2si', 659: 'OP_vcvtss2si', 660: 'OP_vcvtsd2si', 661: 'OP_vucomiss', 662: 'OP_vucomisd', 663: 'OP_vcomiss', 664: 'OP_vcomisd', 665: 'OP_vmovmskps', 666: 'OP_vmovmskpd', 667: 'OP_vsqrtps', 668: 'OP_vsqrtss', 669: 'OP_vsqrtpd', 670: 'OP_vsqrtsd', 671: 'OP_vrsqrtps', 672: 'OP_vrsqrtss', 673: 'OP_vrcpps', 674: 'OP_vrcpss', 675: 'OP_vandps', 676: 'OP_vandpd', 677: 'OP_vandnps', 678: 'OP_vandnpd', 679: 'OP_vorps', 680: 'OP_vorpd', 681: 'OP_vxorps', 682: 'OP_vxorpd', 683: 'OP_vaddps', 684: 'OP_vaddss', 685: 'OP_vaddpd', 686: 'OP_vaddsd', 687: 'OP_vmulps', 688: 'OP_vmulss', 689: 'OP_vmulpd', 690: 'OP_vmulsd', 691: 'OP_vcvtps2pd', 692: 'OP_vcvtss2sd', 693: 'OP_vcvtpd2ps', 694: 'OP_vcvtsd2ss', 695: 'OP_vcvtdq2ps', 696: 'OP_vcvttps2dq', 697: 'OP_vcvtps2dq', 698: 'OP_vsubps', 699: 'OP_vsubss', 700: 'OP_vsubpd', 701: 'OP_vsubsd', 702: 'OP_vminps', 703: 'OP_vminss', 704: 'OP_vminpd', 705: 'OP_vminsd', 706: 'OP_vdivps', 707: 'OP_vdivss', 708: 'OP_vdivpd', 709: 'OP_vdivsd', 710: 'OP_vmaxps', 711: 'OP_vmaxss', 712: 'OP_vmaxpd', 713: 'OP_vmaxsd', 714: 'OP_vpunpcklbw', 715: 'OP_vpunpcklwd', 716: 'OP_vpunpckldq', 717: 'OP_vpacksswb', 718: 'OP_vpcmpgtb', 719: 'OP_vpcmpgtw', 720: 'OP_vpcmpgtd', 721: 'OP_vpackuswb', 722: 'OP_vpunpckhbw', 723: 'OP_vpunpckhwd', 724: 'OP_vpunpckhdq', 725: 'OP_vpackssdw', 726: 'OP_vpunpcklqdq', 727: 'OP_vpunpckhqdq', 728: 'OP_vmovd', 729: 'OP_vpshufhw', 730: 'OP_vpshufd', 731: 'OP_vpshuflw', 732: 'OP_vpcmpeqb', 733: 'OP_vpcmpeqw', 734: 'OP_vpcmpeqd', 735: 'OP_vmovq', 736: 'OP_vcmpps', 737: 'OP_vcmpss', 738: 'OP_vcmppd', 739: 'OP_vcmpsd', 740: 'OP_vpinsrw', 741: 'OP_vpextrw', 742: 'OP_vshufps', 743: 'OP_vshufpd', 744: 'OP_vpsrlw', 745: 'OP_vpsrld', 746: 'OP_vpsrlq', 747: 'OP_vpaddq', 748: 'OP_vpmullw', 749: 'OP_vpmovmskb', 750: 'OP_vpsubusb', 751: 'OP_vpsubusw', 752: 'OP_vpminub', 753: 'OP_vpand', 754: 'OP_vpaddusb', 755: 'OP_vpaddusw', 756: 'OP_vpmaxub', 757: 'OP_vpandn', 758: 'OP_vpavgb', 759: 'OP_vpsraw', 760: 'OP_vpsrad', 761: 'OP_vpavgw', 762: 'OP_vpmulhuw', 763: 'OP_vpmulhw', 764: 'OP_vcvtdq2pd', 765: 'OP_vcvttpd2dq', 766: 'OP_vcvtpd2dq', 767: 'OP_vmovntdq', 768: 'OP_vpsubsb', 769: 'OP_vpsubsw', 770: 'OP_vpminsw', 771: 'OP_vpor', 772: 'OP_vpaddsb', 773: 'OP_vpaddsw', 774: 'OP_vpmaxsw', 775: 'OP_vpxor', 776: 'OP_vpsllw', 777: 'OP_vpslld', 778: 'OP_vpsllq', 779: 'OP_vpmuludq', 780: 'OP_vpmaddwd', 781: 'OP_vpsadbw', 782: 'OP_vmaskmovdqu', 783: 'OP_vpsubb', 784: 'OP_vpsubw', 785: 'OP_vpsubd', 786: 'OP_vpsubq', 787: 'OP_vpaddb', 788: 'OP_vpaddw', 789: 'OP_vpaddd', 790: 'OP_vpsrldq', 791: 'OP_vpslldq', 792: 'OP_vmovdqu', 793: 'OP_vmovdqa', 794: 'OP_vhaddpd', 795: 'OP_vhaddps', 796: 'OP_vhsubpd', 797: 'OP_vhsubps', 798: 'OP_vaddsubpd', 799: 'OP_vaddsubps', 800: 'OP_vlddqu', 801: 'OP_vpshufb', 802: 'OP_vphaddw', 803: 'OP_vphaddd', 804: 'OP_vphaddsw', 805: 'OP_vpmaddubsw', 806: 'OP_vphsubw', 807: 'OP_vphsubd', 808: 'OP_vphsubsw', 809: 'OP_vpsignb', 810: 'OP_vpsignw', 811: 'OP_vpsignd', 812: 'OP_vpmulhrsw', 813: 'OP_vpabsb', 814: 'OP_vpabsw', 815: 'OP_vpabsd', 816: 'OP_vpalignr', 817: 'OP_vpblendvb', 818: 'OP_vblendvps', 819: 'OP_vblendvpd', 820: 'OP_vptest', 821: 'OP_vpmovsxbw', 822: 'OP_vpmovsxbd', 823: 'OP_vpmovsxbq', 824: 'OP_vpmovsxwd', 825: 'OP_vpmovsxwq', 826: 'OP_vpmovsxdq', 827: 'OP_vpmuldq', 828: 'OP_vpcmpeqq', 829: 'OP_vmovntdqa', 830: 'OP_vpackusdw', 831: 'OP_vpmovzxbw', 832: 'OP_vpmovzxbd', 833: 'OP_vpmovzxbq', 834: 'OP_vpmovzxwd', 835: 'OP_vpmovzxwq', 836: 'OP_vpmovzxdq', 837: 'OP_vpcmpgtq', 838: 'OP_vpminsb', 839: 'OP_vpminsd', 840: 'OP_vpminuw', 841: 'OP_vpminud', 842: 'OP_vpmaxsb', 843: 'OP_vpmaxsd', 844: 'OP_vpmaxuw', 845: 'OP_vpmaxud', 846: 'OP_vpmulld', 847: 'OP_vphminposuw', 848: 'OP_vaesimc', 849: 'OP_vaesenc', 850: 'OP_vaesenclast', 851: 'OP_vaesdec', 852: 'OP_vaesdeclast', 853: 'OP_vpextrb', 854: 'OP_vpextrd', 855: 'OP_vextractps', 856: 'OP_vroundps', 857: 'OP_vroundpd', 858: 'OP_vroundss', 859: 'OP_vroundsd', 860: 'OP_vblendps', 861: 'OP_vblendpd', 862: 'OP_vpblendw', 863: 'OP_vpinsrb', 864: 'OP_vinsertps', 865: 'OP_vpinsrd', 866: 'OP_vdpps', 867: 'OP_vdppd', 868: 'OP_vmpsadbw', 869: 'OP_vpcmpestrm', 870: 'OP_vpcmpestri', 871: 'OP_vpcmpistrm', 872: 'OP_vpcmpistri', 873: 'OP_vpclmulqdq', 874: 'OP_vaeskeygenassist', 875: 'OP_vtestps', 876: 'OP_vtestpd', 877: 'OP_vzeroupper', 878: 'OP_vzeroall', 879: 'OP_vldmxcsr', 880: 'OP_vstmxcsr', 881: 'OP_vbroadcastss', 882: 'OP_vbroadcastsd', 883: 'OP_vbroadcastf128', 884: 'OP_vmaskmovps', 885: 'OP_vmaskmovpd', 886: 'OP_vpermilps', 887: 'OP_vpermilpd', 888: 'OP_vperm2f128', 889: 'OP_vinsertf128', 890: 'OP_vextractf128', 891: 'OP_vcvtph2ps', 892: 'OP_vcvtps2ph', 893: 'OP_vfmadd132ps', 894: 'OP_vfmadd132pd', 895: 'OP_vfmadd213ps', 896: 'OP_vfmadd213pd', 897: 'OP_vfmadd231ps', 898: 'OP_vfmadd231pd', 899: 'OP_vfmadd132ss', 900: 'OP_vfmadd132sd', 901: 'OP_vfmadd213ss', 902: 'OP_vfmadd213sd', 903: 'OP_vfmadd231ss', 904: 'OP_vfmadd231sd', 905: 'OP_vfmaddsub132ps', 906: 'OP_vfmaddsub132pd', 907: 'OP_vfmaddsub213ps', 908: 'OP_vfmaddsub213pd', 909: 'OP_vfmaddsub231ps', 910: 'OP_vfmaddsub231pd', 911: 'OP_vfmsubadd132ps', 912: 'OP_vfmsubadd132pd', 913: 'OP_vfmsubadd213ps', 914: 'OP_vfmsubadd213pd', 915: 'OP_vfmsubadd231ps', 916: 'OP_vfmsubadd231pd', 917: 'OP_vfmsub132ps', 918: 'OP_vfmsub132pd', 919: 'OP_vfmsub213ps', 920: 'OP_vfmsub213pd', 921: 'OP_vfmsub231ps', 922: 'OP_vfmsub231pd', 923: 'OP_vfmsub132ss', 924: 'OP_vfmsub132sd', 925: 'OP_vfmsub213ss', 926: 'OP_vfmsub213sd', 927: 'OP_vfmsub231ss', 928: 'OP_vfmsub231sd', 929: 'OP_vfnmadd132ps', 930: 'OP_vfnmadd132pd', 931: 'OP_vfnmadd213ps', 932: 'OP_vfnmadd213pd', 933: 'OP_vfnmadd231ps', 934: 'OP_vfnmadd231pd', 935: 'OP_vfnmadd132ss', 936: 'OP_vfnmadd132sd', 937: 'OP_vfnmadd213ss', 938: 'OP_vfnmadd213sd', 939: 'OP_vfnmadd231ss', 940: 'OP_vfnmadd231sd', 941: 'OP_vfnmsub132ps', 942: 'OP_vfnmsub132pd', 943: 'OP_vfnmsub213ps', 944: 'OP_vfnmsub213pd', 945: 'OP_vfnmsub231ps', 946: 'OP_vfnmsub231pd', 947: 'OP_vfnmsub132ss', 948: 'OP_vfnmsub132sd', 949: 'OP_vfnmsub213ss', 950: 'OP_vfnmsub213sd', 951: 'OP_vfnmsub231ss', 952: 'OP_vfnmsub231sd', 953: 'OP_movq2dq', 954: 'OP_movdq2q', 955: 'OP_fxsave64', 956: 'OP_fxrstor64', 957: 'OP_xsave64', 958: 'OP_xrstor64', 959: 'OP_xsaveopt64', 960: 'OP_rdrand', 961: 'OP_rdfsbase', 962: 'OP_rdgsbase', 963: 'OP_wrfsbase', 964: 'OP_wrgsbase', 965: 'OP_rdseed', 966: 'OP_vfmaddsubps', 967: 'OP_vfmaddsubpd', 968: 'OP_vfmsubaddps', 969: 'OP_vfmsubaddpd', 970: 'OP_vfmaddps', 971: 'OP_vfmaddpd', 972: 'OP_vfmaddss', 973: 'OP_vfmaddsd', 974: 'OP_vfmsubps', 975: 'OP_vfmsubpd', 976: 'OP_vfmsubss', 977: 'OP_vfmsubsd', 978: 'OP_vfnmaddps', 979: 'OP_vfnmaddpd', 980: 'OP_vfnmaddss', 981: 'OP_vfnmaddsd', 982: 'OP_vfnmsubps', 983: 'OP_vfnmsubpd', 984: 'OP_vfnmsubss', 985: 'OP_vfnmsubsd', 986: 'OP_vfrczps', 987: 'OP_vfrczpd', 988: 'OP_vfrczss', 989: 'OP_vfrczsd', 990: 'OP_vpcmov', 991: 'OP_vpcomb', 992: 'OP_vpcomw', 993: 'OP_vpcomd', 994: 'OP_vpcomq', 995: 'OP_vpcomub', 996: 'OP_vpcomuw', 997: 'OP_vpcomud', 998: 'OP_vpcomuq', 999: 'OP_vpermil2pd', 1000: 'OP_vpermil2ps', 1001: 'OP_vphaddbw', 1002: 'OP_vphaddbd', 1003: 'OP_vphaddbq', 1004: 'OP_vphaddwd', 1005: 'OP_vphaddwq', 1006: 'OP_vphadddq', 1007: 'OP_vphaddubw', 1008: 'OP_vphaddubd', 1009: 'OP_vphaddubq', 1010: 'OP_vphadduwd', 1011: 'OP_vphadduwq', 1012: 'OP_vphaddudq', 1013: 'OP_vphsubbw', 1014: 'OP_vphsubwd', 1015: 'OP_vphsubdq', 1016: 'OP_vpmacssww', 1017: 'OP_vpmacsswd', 1018: 'OP_vpmacssdql', 1019: 'OP_vpmacssdd', 1020: 'OP_vpmacssdqh', 1021: 'OP_vpmacsww', 1022: 'OP_vpmacswd', 1023: 'OP_vpmacsdql', 1024: 'OP_vpmacsdd', 1025: 'OP_vpmacsdqh', 1026: 'OP_vpmadcsswd', 1027: 'OP_vpmadcswd', 1028: 'OP_vpperm', 1029: 'OP_vprotb', 1030: 'OP_vprotw', 1031: 'OP_vprotd', 1032: 'OP_vprotq', 1033: 'OP_vpshlb', 1034: 'OP_vpshlw', 1035: 'OP_vpshld', 1036: 'OP_vpshlq', 1037: 'OP_vpshab', 1038: 'OP_vpshaw', 1039: 'OP_vpshad', 1040: 'OP_vpshaq', 1041: 'OP_bextr', 1042: 'OP_blcfill', 1043: 'OP_blci', 1044: 'OP_blcic', 1045: 'OP_blcmsk', 1046: 'OP_blcs', 1047: 'OP_blsfill', 1048: 'OP_blsic', 1049: 'OP_t1mskc', 1050: 'OP_tzmsk', 1051: 'OP_llwpcb', 1052: 'OP_slwpcb', 1053: 'OP_lwpins', 1054: 'OP_lwpval', 1055: 'OP_andn', 1056: 'OP_blsr', 1057: 'OP_blsmsk', 1058: 'OP_blsi', 1059: 'OP_tzcnt', 1060: 'OP_bzhi', 1061: 'OP_pext', 1062: 'OP_pdep', 1063: 'OP_sarx', 1064: 'OP_shlx', 1065: 'OP_shrx', 1066: 'OP_rorx', 1067: 'OP_mulx', 1068: 'OP_getsec', 1069: 'OP_vmfunc', 1070: 'OP_invpcid', 1071: 'OP_xabort', 1072: 'OP_xbegin', 1073: 'OP_xend', 1074: 'OP_xtest', 1075: 'OP_vpgatherdd', 1076: 'OP_vpgatherdq', 1077: 'OP_vpgatherqd', 1078: 'OP_vpgatherqq', 1079: 'OP_vgatherdps', 1080: 'OP_vgatherdpd', 1081: 'OP_vgatherqps', 1082: 'OP_vgatherqpd', 1083: 'OP_vbroadcasti128', 1084: 'OP_vinserti128', 1085: 'OP_vextracti128', 1086: 'OP_vpmaskmovd', 1087: 'OP_vpmaskmovq', 1088: 'OP_vperm2i128', 1089: 'OP_vpermd', 1090: 'OP_vpermps', 1091: 'OP_vpermq', 1092: 'OP_vpermpd', 1093: 'OP_vpblendd', 1094: 'OP_vpsllvd', 1095: 'OP_vpsllvq', 1096: 'OP_vpsravd', 1097: 'OP_vpsrlvd', 1098: 'OP_vpsrlvq', 1099: 'OP_vpbroadcastb', 1100: 'OP_vpbroadcastw', 1101: 'OP_vpbroadcastd', 1102: 'OP_vpbroadcastq', 1103: 'OP_xsavec32', 1104: 'OP_xsavec64'}
reg_dict = {'REG_NULL': 0, 'REG_RAX': 1, 'REG_RCX': 2, 'REG_RDX': 3, 'REG_RBX': 4, 'REG_RSP': 5, 'REG_RBP': 6, 'REG_RSI': 7, 'REG_RDI': 8, 'REG_R8': 9, 'REG_R9': 10, 'REG_R10': 11, 'REG_R11': 12, 'REG_R12': 13, 'REG_R13': 14, 'REG_R14': 15, 'REG_R15': 16, 'REG_EAX': 17, 'REG_ECX': 18, 'REG_EDX': 19, 'REG_EBX': 20, 'REG_ESP': 21, 'REG_EBP': 22, 'REG_ESI': 23, 'REG_EDI': 24, 'REG_R8D': 25, 'REG_R9D': 26, 'REG_R10D': 27, 'REG_R11D': 28, 'REG_R12D': 29, 'REG_R13D': 30, 'REG_R14D': 31, 'REG_R15D': 32, 'REG_AX': 33, 'REG_CX': 34, 'REG_DX': 35, 'REG_BX': 36, 'REG_SP': 37, 'REG_BP': 38, 'REG_SI': 39, 'REG_DI': 40, 'REG_R8W': 41, 'REG_R9W': 42, 'REG_R10W': 43, 'REG_R11W': 44, 'REG_R12W': 45, 'REG_R13W': 46, 'REG_R14W': 47, 'REG_R15W': 48, 'REG_AL': 49, 'REG_CL': 50, 'REG_DL': 51, 'REG_BL': 52, 'REG_AH': 53, 'REG_CH': 54, 'REG_DH': 55, 'REG_BH': 56, 'REG_R8L': 57, 'REG_R9L': 58, 'REG_R10L': 59, 'REG_R11L': 60, 'REG_R12L': 61, 'REG_R13L': 62, 'REG_R14L': 63, 'REG_R15L': 64, 'REG_SPL': 65, 'REG_BPL': 66, 'REG_SIL': 67, 'REG_DIL': 68, 'REG_MM0': 69, 'REG_MM1': 70, 'REG_MM2': 71, 'REG_MM3': 72, 'REG_MM4': 73, 'REG_MM5': 74, 'REG_MM6': 75, 'REG_MM7': 76, 'REG_XMM0': 77, 'REG_XMM1': 78, 'REG_XMM2': 79, 'REG_XMM3': 80, 'REG_XMM4': 81, 'REG_XMM5': 82, 'REG_XMM6': 83, 'REG_XMM7': 84, 'REG_XMM8': 85, 'REG_XMM9': 86, 'REG_XMM10': 87, 'REG_XMM11': 88, 'REG_XMM12': 89, 'REG_XMM13': 90, 'REG_XMM14': 91, 'REG_XMM15': 92, 'REG_ST0': 93, 'REG_ST1': 94, 'REG_ST2': 95, 'REG_ST3': 96, 'REG_ST4': 97, 'REG_ST5': 98, 'REG_ST6': 99, 'REG_ST7': 100, 'SEG_ES': 101, 'SEG_CS': 102, 'SEG_SS': 103, 'SEG_DS': 104, 'SEG_FS': 105, 'SEG_GS': 106, 'REG_DR0': 107, 'REG_DR1': 108, 'REG_DR2': 109, 'REG_DR3': 110, 'REG_DR4': 111, 'REG_DR5': 112, 'REG_DR6': 113, 'REG_DR7': 114, 'REG_DR8': 115, 'REG_DR9': 116, 'REG_DR10': 117, 'REG_DR11': 118, 'REG_DR12': 119, 'REG_DR13': 120, 'REG_DR14': 121, 'REG_DR15': 122, 'REG_CR0': 123, 'REG_CR1': 124, 'REG_CR2': 125, 'REG_CR3': 126, 'REG_CR4': 127, 'REG_CR5': 128, 'REG_CR6': 129, 'REG_CR7': 130, 'REG_CR8': 131, 'REG_CR9': 132, 'REG_CR10': 133, 'REG_CR11': 134, 'REG_CR12': 135, 'REG_CR13': 136, 'REG_CR14': 137, 'REG_CR15': 138, 'REG_INVALID': 139, 'REG_YMM0': 140, 'REG_YMM1': 141, 'REG_YMM2': 142, 'REG_YMM3': 143, 'REG_YMM4': 144, 'REG_YMM5': 145, 'REG_YMM6': 146, 'REG_YMM7': 147, 'REG_YMM8': 148, 'REG_YMM9': 149, 'REG_YMM10': 150, 'REG_YMM11': 151, 'REG_YMM12': 152, 'REG_YMM13': 153, 'REG_YMM14': 154, 'REG_YMM15': 155}