.syntax unified
.cpu cortex-m4
.thumb

.word 0x20000000
.word 0x080000ed
.space 0xe4

.equ RCC_AHB1ENR, 0x40023830
.equ RCC_APB1ENR, 0x40023840
.equ TIM2_CR1, 0x40000000
.equ TIM2_PSC, 0x40000028
.equ TIM2_ARR, 0x4000002C
.equ TIM2_DIER, 0x4000000C
.equ TIM2_SR, 0x40000010
.equ TIM2_CNT, 0x40000024
.equ NVIC_ISER, 0xE000E100
.equ NVIC_IPR, 0xE000E400
/*
@flash memory 범위 0x08000000 ~ 0x081FFFFF
.equ Flash_interface_register 0x40023C00
.equ FLASH_KEYR 0x40023C04
.equ Write_flash_KEY1 = 0x45670123
.equ Write_flash_KEY2 = 0xCDEF89AB
*/
.equ GPIOB_MODER, 0x40020400
.equ GPIOB_ODR, 0x40020414
.equ LED_PIN, 9
.equ LED_PIN_1, 15
.equ LED_PIN_2, 13
.equ LED_MODE, (1 << (LED_PIN * 2))
.equ LED_MODE_1, (1 << (LED_PIN_1 * 2))
.equ LED_MODE_2, (1 << (LED_PIN_2 * 2))

.equ TIM_CR1_CEN, 1 << 0
.equ TIM_DIER_UIE, 1 << 0
.equ TIM_SR_UIF, 1 << 0


.global _start
_start:
    bl gpio_set
    bl tim2
    b loop

loop:
    ldr r0, =GPIOB_ODR
    mov r1, #(1 << LED_PIN_1)
    str r1, [r0]
    
    bl tim2_handler
    
    b loop


tim2:
    @tim2 셋팅
    @ Configure TIM2
    ldr r0, =RCC_APB1ENR
    ldr r1, [r0]
    mov r2, #(1 << 0)  @ Enable TIM2 clock
    @orr r1, r1, r2
    str r1, [r0]
    
    
    ldr r0, =TIM2_CR1
    ldr r2, [r0]
    mov r2, #(TIM_CR1_CEN)
    @orr r2, r2, #(TIM_CR1_CEN) 타이머 활성화
    str r2, [r0]

    /*
    Prescaler (PSC)
    주기= (PSC+1 ×(ARR+1))/APB1 Timer clocks
    ex. 
    psc=1, arr=999,  APB1 Timer clocks=84Mhz, 0.00002381ms당 1회
    psc=999, arr=9999,  APB1 Timer clocks=84Mhz, 0.119047619ms당 1회
    psc=64999, arr=10,  APB1 Timer clocks=84Mhz, 0.008511905ms당 1회
    */

    ldr r0, =TIM2_PSC
    mov r1, #29999 @ Prescaler value (adjust based on your requirements)
    str r1, [r0]

    ldr r0, =TIM2_ARR
    mov r1, #29999  @ Auto-reload value (adjust based on your requirements)
    str r1, [r0]

    ldr r0, =TIM2_DIER
    mov r1, #(TIM_DIER_UIE)  @ Enable update interrupt
    str r1, [r0]
    
    ldr r0, =TIM2_SR
    mov r1, #(TIM_SR_UIF)  @ Enable update interrupt
    str r1, [r0]
    
    @ Enable TIM2 interrupt in NVIC
    ldr r0, =NVIC_ISER
    ldr r1, [r0]
    mov r2, #(1 << 28)  @ TIM2 interrupt is IRQ 28
    orr r1, r1, r2
    str r1, [r0]

    bx lr

@gpio seting값
gpio_set:
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

    bx lr

ex:
    ldr r0, =GPIOB_ODR
    mov r2, #(1 << LED_PIN)
    orr r1, r1, r2
    str r1, [r0]
    b loop
ei:
    ldr r0, =GPIOB_ODR
    mov r2, #(1 << LED_PIN_2)
    orr r1, r1, r2
    str r1, [r0]
    bx lr

tim2_handler:
    @이벤트 측정 TIM2_SR
    @TIM2_SR의 register 0번째의 값이 UIF(update flag)값
    ldr r0, =TIM2_SR
    cmp r0, #1
    
    @만약 이벤트 측정이 안될시 exit로 이동
    beq ei
    bne ex


    b loop


