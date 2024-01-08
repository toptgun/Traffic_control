.syntax unified
.cpu cortex-m4
.thumb

.word 0x20000400
.word 0x080000ed
.space 0xe4

.equ RCC_AHB1ENR, 0x40023830
.equ GPIOB_MODER, 0x40020400
.equ GPIOB_ODR, 0x40020414
.equ LED_PIN, 9
.equ LED_PIN_1, 15
.equ LED_PIN_2, 13
.equ DELAY_COUNT_ON, 1000000
.equ DELAY_COUNT_OFF, 50000000
.equ LED_MODE, (1 << (LED_PIN * 2))
.equ LED_MODE_1, (1 << (LED_PIN_1 * 2))
.equ LED_MODE_2, (1 << (LED_PIN_2 * 2))

.global _start
_start:
    ldr r7, =0
    @ Enable GPIOB clock
    ldr r0, =RCC_AHB1ENR
    ldr r1, [r0]
    mov r2, #(1 << 1)
    orr r1, r1, r2
    str r1, [r0]

    @ Set GPIOB_MODER
    ldr r0, =GPIOB_MODER
    ldr r1, [r0]

    @ Set PB7 (LED) as output
    ldr r2, =LED_MODE
    orr r1, r1, r2

    @ Set PB15 (LED_1) as output
    ldr r2, =LED_MODE_1
    orr r1, r1, r2

    @ Set PB13 (LED_2) as output
    ldr r2, =LED_MODE_2
    orr r1, r1, r2

    str r1, [r0]

    b loop

loop:
    @ LED ON
    ldr r0, =GPIOB_ODR
    mov r1, #(1 << LED_PIN)    
    str r1, [r0]
    bl delay
    bl delay
    bl delay

    @ LED OFF
    ldr r0, =GPIOB_ODR
    mov r1, #(0 << LED_PIN)    
    str r1, [r0]
    bl delay
    bl delay
    bl delay

    @ LED_1 ON
    ldr r0, =GPIOB_ODR
    mov r1, #(1 << LED_PIN_1)    
    str r1, [r0]
    bl delay
    bl delay
    bl delay

    @ LED_1 OFF
    ldr r0, =GPIOB_ODR
    mov r1, #(0 << LED_PIN_1)    
    str r1, [r0]
    bl delay
    bl delay
    bl delay

    @ LED_2 ON
    ldr r0, =GPIOB_ODR
    mov r1, #(1 << LED_PIN_2)    
    str r1, [r0]
    bl delay
    bl delay
    bl delay

    @ LED_2 OFF
    ldr r0, =GPIOB_ODR
    mov r1, #(0 << LED_PIN_2)    
    str r1, [r0]
    bl delay
    bl delay
    bl delay

    b loop

delay:
    ldr r2, =DELAY_COUNT_ON
delay_loop_on:
    sub r2, r2, #1
    cmp r7, r2
    bne delay_loop_on
    bx lr
